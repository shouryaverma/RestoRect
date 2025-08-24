import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

from functools import partial
from tqdm import tqdm

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.util2 import extract_into_tensor, noise_like


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class RectifiedFlow(nn.Module):
    """Rectified Flow implementation for image generation.
    
    Uses straight-line paths between noise and data instead of complex diffusion schedules.
    Much simpler than DDPM with better sampling efficiency.
    
    Args:
        denoise: The velocity prediction network
        condition: The conditioning network  
        timesteps (int): Number of time steps. Default: 1000
        image_size (int): Image size. Default: 256
        n_feats (int): Number of feature channels. Default: 128
        clip_denoised (bool): Whether to clip predictions. Default: False
        parameterization (str): Always "v" for velocity prediction
    """

    def __init__(self,
                 denoise,
                 condition, 
                 timesteps=1000,
                 image_size=256,
                 n_feats=128,
                 clip_denoised=False,
                 parameterization="v",  # Rectified flow always predicts velocity
                 ):
        super().__init__()
        assert parameterization == "v", 'Rectified Flow only supports velocity prediction mode'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in velocity-prediction mode")
        
        self.clip_denoised = clip_denoised
        self.image_size = image_size
        self.channels = n_feats
        self.model = denoise
        self.condition = condition
        
        self.num_timesteps = int(timesteps)
        
        # For rectified flow, we just need simple time steps - no complex scheduling
        timesteps_array = np.linspace(0, 1, self.num_timesteps)
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('timesteps_array', to_torch(timesteps_array))

    def q_sample(self, x_start, t, x_end=None):
        """Forward process: Linear interpolation between x_start and x_end
        
        x_t = (1-t) * x_start + t * x_end
        
        Args:
            x_start: Clean data (x_0)
            t: Time steps
            x_end: Target noise (x_1), defaults to random noise
        """
        if x_end is None:
            x_end = torch.randn_like(x_start)
            
        # Extract time values for this batch
        t_expanded = extract_into_tensor(self.timesteps_array, t, x_start.shape)
        
        # Linear interpolation: x_t = (1-t) * x_0 + t * x_1
        x_t = (1 - t_expanded) * x_start + t_expanded * x_end
        
        return x_t, x_end

    def compute_velocity(self, x_start, x_end):
        """Compute the velocity field: v = x_end - x_start
        
        This is the target for our velocity prediction network.
        """
        return x_end - x_start

    def predict_x0_from_velocity(self, x_t, t, velocity):
        """Predict x_0 from velocity: x_0 = x_t - t * velocity"""
        t_expanded = extract_into_tensor(self.timesteps_array, t, x_t.shape)
        x_0_pred = x_t - t_expanded * velocity
        return x_0_pred

    def predict_x1_from_velocity(self, x_t, t, velocity):
        """Predict x_1 from velocity: x_1 = x_t + (1-t) * velocity"""
        t_expanded = extract_into_tensor(self.timesteps_array, t, x_t.shape)
        x_1_pred = x_t + (1 - t_expanded) * velocity
        return x_1_pred

    def p_mean_variance(self, x, t, c, clip_denoised: bool):
        """Compute the mean for the posterior p(x_{t-1} | x_t)
        
        For rectified flow, this is much simpler than DDPM.
        """
        velocity = self.model(x, t, c)
        
        # For rectified flow, we can predict x_0 directly
        x_recon = self.predict_x0_from_velocity(x, t, velocity)
        
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
            
        # For rectified flow, the "mean" is just a step along the straight line
        # We'll implement ODE stepping in the sampling functions
        return x_recon, None, None, velocity

    def p_sample(self, x, t, c, clip_denoised=True, repeat_noise=False):
        """Single sampling step using ODE integration
        
        For rectified flow, we use Euler's method for ODE solving.
        """
        b, *_, device = *x.shape, x.device
        velocity = self.model(x, t, c)
        
        if clip_denoised:
            # Predict x_0 and clip it
            x_0_pred = self.predict_x0_from_velocity(x, t, velocity)
            x_0_pred.clamp_(-1., 1.)
            # Recompute velocity with clipped x_0
            x_1_pred = self.predict_x1_from_velocity(x, t, velocity)
            velocity = x_1_pred - x_0_pred
        
        return x, velocity

    def ode_sample_step(self, x, t_current, t_next, c, clip_denoised=True):
        """Single ODE step from t_current to t_next
        
        Uses Euler's method: x_{t_next} = x_{t_current} + (t_next - t_current) * velocity
        """
        # Get velocity at current time
        t_current_tensor = torch.full((x.shape[0],), t_current, device=x.device, dtype=torch.long)
        velocity = self.model(x, t_current_tensor, c)
        
        if clip_denoised:
            x_0_pred = self.predict_x0_from_velocity(x, t_current_tensor, velocity)
            x_0_pred.clamp_(-1., 1.)
            x_1_pred = self.predict_x1_from_velocity(x, t_current_tensor, velocity)
            velocity = x_1_pred - x_0_pred
        
        # Euler step
        dt = t_next - t_current
        x_next = x + dt * velocity
        
        return x_next

    def p_sample_loop(self, shape, return_intermediates=False, num_steps=None):
        """Full sampling loop using ODE integration
        
        Args:
            shape: Shape of samples to generate
            return_intermediates: Whether to return intermediate steps
            num_steps: Number of ODE steps (can be much smaller than training steps)
        """
        device = self.timesteps_array.device
        b = shape[0]
        
        # Start from pure noise (t=1)
        img = torch.randn(shape, device=device)
        intermediates = [img] if return_intermediates else []
        
        # Use fewer steps for sampling if specified
        if num_steps is None:
            num_steps = self.num_timesteps
            
        # Create time schedule for sampling
        time_schedule = np.linspace(1.0, 0.0, num_steps + 1)
        
        for i in tqdm(range(num_steps), desc='RF Sampling'):
            t_current = time_schedule[i]
            t_next = time_schedule[i + 1]
            
            # Convert to tensor indices for conditioning
            t_idx = int(t_current * (self.num_timesteps - 1))
            t_tensor = torch.full((b,), t_idx, device=device, dtype=torch.long)
            
            # Get conditioning (this should be provided externally in real usage)
            c = torch.zeros(b, 256, device=device)  # Placeholder
            
            img = self.ode_sample_step(img, t_current, t_next, c, self.clip_denoised)
            
            if return_intermediates:
                intermediates.append(img)
                
        if return_intermediates:
            return img, intermediates
        return img

    def sample(self, batch_size=16, return_intermediates=False, num_steps=20):
        """Generate samples using rectified flow
        
        Args:
            batch_size: Number of samples
            return_intermediates: Return intermediate steps
            num_steps: Number of ODE steps (typically 1-10 for rectified flow)
        """
        image_size = self.image_size
        channels = self.channels
        shape = (batch_size, channels, image_size, image_size)
        return self.p_sample_loop(shape, return_intermediates, num_steps)

    def training_losses(self, x_start, t, c=None, noise=None):
        """Compute training losses for rectified flow
        
        The loss is simply: ||velocity_pred - velocity_true||^2
        where velocity_true = noise - x_start
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Forward process: get x_t
        x_t, x_end = self.q_sample(x_start=x_start, t=t, x_end=noise)
        
        # True velocity
        velocity_true = self.compute_velocity(x_start, x_end)
        
        # Predicted velocity
        velocity_pred = self.model(x_t, t, c)
        
        # Simple MSE loss
        loss = F.mse_loss(velocity_pred, velocity_true, reduction='none')
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        
        return loss.mean(), velocity_pred, velocity_true

    def forward(self, input_features, target_features=None):
        """Forward pass for training/inference
        
        Args:
            input_features: Input for conditioning network
            target_features: Target features from S1 (for training only)
        """
        device = self.timesteps_array.device
        b = input_features.shape[0]
        
        if self.training and target_features is not None:
            # Training mode: learn to generate target_features
            pred_deg_list = []
            
            # Get conditioning from input
            c = self.condition(input_features)
            
            # Use target_features as x_0 (clean data)
            x_start = target_features
            
            # Sample random time steps for velocity training
            t = torch.randint(0, self.num_timesteps, (b,), device=device, dtype=torch.long)
            
            # Forward process: interpolate between clean and noise
            x_t, x_end = self.q_sample(x_start=x_start, t=t, x_end=None)
            
            # Predict velocity (this is where learning happens)
            velocity_pred = self.model(x_t, t, c)
            
            # Compute velocity loss internally (this trains the velocity predictor)
            true_velocity = self.compute_velocity(x_start, x_end)
            self.velocity_loss = F.mse_loss(velocity_pred, true_velocity)
            
            # Now generate features using a quick RF sampling for KD loss
            # Start from noise and do a few steps to generate features
            x_noise = torch.randn_like(x_start)
            deg_prep = x_noise
            
            # Quick sampling to generate features
            num_quick_steps = max(20, self.num_timesteps)
            time_schedule = np.linspace(1.0, 0.0, num_quick_steps + 1)
            
            for i in range(num_quick_steps):
                t_current = time_schedule[i]
                t_next = time_schedule[i + 1]
                
                # Convert to tensor index
                t_idx = int(t_current * (self.num_timesteps - 1))
                t_tensor = torch.full((b,), t_idx, device=device, dtype=torch.long)
                
                # ODE step
                velocity = self.model(deg_prep, t_tensor, c)
                dt = t_next - t_current
                deg_prep = deg_prep + dt * velocity
                
                pred_deg_list.append(deg_prep)
            
            # Fill remaining slots for compatibility
            while len(pred_deg_list) < self.num_timesteps:
                pred_deg_list.append(deg_prep)
            
            return deg_prep, pred_deg_list
            
        else:
            # Inference mode: sample from noise to generate features
            shape = (b, self.channels * 4)  # Match original interface 
            x_noisy = torch.randn(shape, device=device)
            c = self.condition(input_features)
            
            # Use fast sampling
            num_steps = 20
            time_schedule = np.linspace(1.0, 0.0, num_steps + 1)
            
            x_current = x_noisy
            for i in range(num_steps):
                t_current = time_schedule[i]
                t_next = time_schedule[i + 1]
                x_current = self.ode_sample_step(x_current, t_current, t_next, c, self.clip_denoised)
                
            return x_current


# For compatibility with existing code that might import DDIMSampler
class RFSampler:
    """Rectified Flow sampler - much simpler than DDIM"""
    
    def __init__(self, model, schedule="linear"):
        self.model = model
        
    def sample(self, steps, shape, conditioning=None, verbose=True):
        """Sample using rectified flow ODE"""
        return self.model.p_sample_loop(shape, num_steps=steps)
        
    def make_schedule(self, ddim_num_steps, verbose=True):
        """Create sampling schedule (much simpler for RF)"""
        self.ddim_timesteps = np.linspace(1.0, 0.0, ddim_num_steps + 1)
        if verbose:
            print(f"RF Sampler: Using {ddim_num_steps} sampling steps")