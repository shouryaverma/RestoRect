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
    """Optimized Rectified Flow implementation for image generation.
    
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

    def ode_sample_step(self, x, t_tensor, c, dt):
        """Optimized single ODE step with cached conditioning"""
        velocity = self.model(x, t_tensor, c)
        return x - dt * velocity  # Negative for noise-to-clean direction

    def inference_generation(self, input_features, num_steps=5):
        """Separate fast inference generation
        
        Args:
            input_features: Input for conditioning network
            num_steps: Number of ODE steps (optimized for quality-speed tradeoff)
        """
        device = self.timesteps_array.device
        b = input_features.shape[0]
        
        shape = (b, self.channels * 4)
        x_current = torch.randn(shape, device=device)
        c = self.condition(input_features)  # Cache conditioning
        
        # Optimized n-step generation
        dt = 1.0 / num_steps
        
        with torch.no_grad():  # Ensure no gradient computation during inference
            for i in range(num_steps):
                t_current = 1.0 - i * dt
                t_idx = int(t_current * (self.num_timesteps - 1))
                t_tensor = torch.full((b,), t_idx, device=device, dtype=torch.long)
                
                x_current = self.ode_sample_step(x_current, t_tensor, c, dt)
        
        # Return in expected format - no wasteful padding
        pred_deg_list = [x_current] * min(num_steps, self.num_timesteps)
        
        return x_current, pred_deg_list

    def single_step_inference(self, input_features):
        """Ultra-fast single-step inference for well-trained models"""
        device = self.timesteps_array.device
        b = input_features.shape[0]
        
        shape = (b, self.channels * 4)
        x_1 = torch.randn(shape, device=device)  # Start from noise
        c = self.condition(input_features)
        
        # Single prediction at t=1.0 (maximum noise)
        t_max = torch.full((b,), self.num_timesteps - 1, device=device, dtype=torch.long)
        
        with torch.no_grad():
            velocity = self.model(x_1, t_max, c)
            # Direct jump to t=0 (clean data)
            x_0 = x_1 - velocity  # Since velocity = x_1 - x_0
        
        return x_0

    def p_sample_loop(self, shape, return_intermediates=False, num_steps=5):
        """Full sampling loop using ODE integration
        
        Args:
            shape: Shape of samples to generate
            return_intermediates: Whether to return intermediate steps
            num_steps: Number of ODE steps (default 5 for quality-speed balance)
        """
        device = self.timesteps_array.device
        b = shape[0]
        
        # Start from pure noise (t=1)
        img = torch.randn(shape, device=device)
        intermediates = [img] if return_intermediates else []
        
        # Create time schedule for sampling
        time_schedule = np.linspace(1.0, 0.0, num_steps + 1)
        dt = 1.0 / num_steps
        
        # Placeholder conditioning - should be provided externally
        c = torch.zeros(b, 256, device=device)
        
        with torch.no_grad():
            for i in range(num_steps):
                t_current = time_schedule[i]
                t_idx = int(t_current * (self.num_timesteps - 1))
                t_tensor = torch.full((b,), t_idx, device=device, dtype=torch.long)
                
                img = self.ode_sample_step(img, t_tensor, c, dt)
                
                if return_intermediates:
                    intermediates.append(img)
                    
        if return_intermediates:
            return img, intermediates
        return img

    def sample(self, batch_size=16, return_intermediates=False, num_steps=5):
        """Generate samples using rectified flow
        
        Args:
            batch_size: Number of samples
            return_intermediates: Return intermediate steps
            num_steps: Number of ODE steps (5 for optimal quality-speed balance)
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
            # Training mode: ONLY do velocity learning, no generation
            c = self.condition(input_features)
            x_start = target_features
            
            # Sample random time steps for velocity training
            t = torch.randint(0, self.num_timesteps, (b,), device=device, dtype=torch.long)
            
            # Forward process: interpolate between clean and noise
            x_t, x_end = self.q_sample(x_start=x_start, t=t, x_end=None)
            
            # Predict velocity (this is where learning happens)
            velocity_pred = self.model(x_t, t, c)
            
            # Compute velocity loss internally
            true_velocity = self.compute_velocity(x_start, x_end)
            self.velocity_loss = F.mse_loss(velocity_pred, true_velocity)
            
            # Return ONLY what's needed for KD loss - do generation separately
            return self.inference_generation(input_features)
            
        else:
            # Inference mode: use optimized generation
            result, _ = self.inference_generation(input_features)
            return result


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