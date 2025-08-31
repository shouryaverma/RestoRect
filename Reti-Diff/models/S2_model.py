import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
from collections import OrderedDict
from models import lr_scheduler as lr_scheduler
from torch import nn
from basicsr.archs import build_network
from basicsr.utils import get_root_logger
from basicsr.losses import build_loss
import os

from losses.flex_loss import FlexLoss

class AnisotropicDiffusion(nn.Module):
    def __init__(self, sensitivity_param=0.1):
        super().__init__()
        self.s = nn.Parameter(torch.tensor(sensitivity_param))
        
    def forward(self, image):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
        
        B, C, H, W = image.shape
        diffused = torch.zeros_like(image)
        
        # Clamp sensitivity parameter to prevent numerical instability
        s_clamped = torch.clamp(self.s, min=0.01, max=1.0)
        
        for c in range(C):
            img_c = image[:, c:c+1, :, :]
            grad_x = F.conv2d(img_c, sobel_x, padding=1)
            grad_y = F.conv2d(img_c, sobel_y, padding=1)
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
            
            # Prevent explosion by clamping the exponent
            exponent = torch.clamp(-(grad_mag / s_clamped)**2, min=-10.0, max=0.0)
            diffusion_coeff = torch.exp(exponent)
            
            diffused_x = diffusion_coeff * grad_x
            diffused_y = diffusion_coeff * grad_y
            kernel = torch.ones(1, 1, 3, 3, device=image.device) / 9
            diffused[:, c:c+1, :, :] = F.conv2d(diffused_x + diffused_y, kernel, padding=1)
            
        return diffused

class GradientAwareWeighting(nn.Module):
    def forward(self, illumination_map):
        grad_x = torch.abs(illumination_map[:, :, :, 1:] - illumination_map[:, :, :, :-1])
        grad_y = torch.abs(illumination_map[:, :, 1:, :] - illumination_map[:, :, :-1, :])
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        weights = 1.0 / torch.exp(grad_mag)
        return weights

class PolarizedHVIColorSpace(nn.Module):
    def __init__(self, learnable_params=True):
        super().__init__()
        self.eps = 1e-6
        if learnable_params:
            # Constrain k parameter to reasonable range
            self.k = nn.Parameter(torch.tensor(1.0))
        else:
            self.k = 1.0
    
    def rgb_to_hvi(self, rgb_img):
        """Polarized transformation eliminates red discontinuity"""
        # Clamp RGB values to valid range
        rgb_img = torch.clamp(rgb_img, 0.0, 1.0)
        
        I_max = torch.max(rgb_img, dim=1, keepdim=True)[0]
        hsv = self.rgb_to_hsv(rgb_img)
        H, S, V = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
        
        # Polarized coordinates (eliminates red discontinuity)
        h_polar = torch.cos(3.14159 * H / 3)
        v_polar = torch.sin(3.14159 * H / 3)
        
        # Adaptive intensity collapse with clamping
        k_clamped = torch.clamp(self.k, min=0.1, max=5.0) if isinstance(self.k, nn.Parameter) else self.k
        C_k = k_clamped * torch.sin(3.14159 * I_max / 2) + self.eps
        
        # Final HV maps
        H_hv = C_k * S * h_polar
        V_hv = C_k * S * v_polar
        
        return torch.cat([H_hv, V_hv, I_max], dim=1)
    
    def rgb_to_hsv(self, rgb):
        """Standard RGB to HSV conversion with numerical stability"""
        max_val, max_idx = torch.max(rgb, dim=1, keepdim=True)
        min_val = torch.min(rgb, dim=1, keepdim=True)[0]
        diff = max_val - min_val
        
        hue = torch.zeros_like(max_val)
        
        # Use epsilon to prevent division by zero
        safe_diff = torch.clamp(diff, min=1e-6)
        
        # Red maximum
        mask = (max_idx == 0) & (diff > 1e-6)
        hue[mask] = (rgb[:, 1:2] - rgb[:, 2:3])[mask] / safe_diff[mask]
        # Green maximum
        mask = (max_idx == 1) & (diff > 1e-6)
        hue[mask] = 2.0 + (rgb[:, 2:3] - rgb[:, 0:1])[mask] / safe_diff[mask]
        # Blue maximum
        mask = (max_idx == 2) & (diff > 1e-6)
        hue[mask] = 4.0 + (rgb[:, 0:1] - rgb[:, 1:2])[mask] / safe_diff[mask]
        
        hue = hue / 6.0
        hue[hue < 0] += 1.0
        hue = torch.clamp(hue, 0.0, 1.0)
        
        # Prevent division by zero in saturation calculation
        safe_max = torch.clamp(max_val, min=1e-6)
        saturation = torch.where(max_val > 1e-6, diff / safe_max, torch.zeros_like(max_val))
        value = max_val
        
        return torch.cat([hue, saturation, value], dim=1)

class PolarizedHVIColorLoss(nn.Module):
    def __init__(self, weight=0.05):
        super().__init__()
        self.hvi_transform = PolarizedHVIColorSpace(learnable_params=True)
        self.weight = weight
        
    def forward(self, pred, gt):
        # Check for NaN inputs
        if torch.isnan(pred).any() or torch.isnan(gt).any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        pred_hvi = self.hvi_transform.rgb_to_hvi(pred)
        gt_hvi = self.hvi_transform.rgb_to_hvi(gt)
        
        # Check for NaN in HVI conversion
        if torch.isnan(pred_hvi).any() or torch.isnan(gt_hvi).any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        h_loss = F.l1_loss(pred_hvi[:, 0:1], gt_hvi[:, 0:1])
        v_loss = F.l1_loss(pred_hvi[:, 1:2], gt_hvi[:, 1:2])
        i_loss = F.l1_loss(pred_hvi[:, 2:3], gt_hvi[:, 2:3])
        
        total_loss = h_loss + v_loss + i_loss
        
        # Additional safety check
        if torch.isnan(total_loss):
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        return self.weight * total_loss

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


def get_batchnorm_layer(opts):
    if opts.norm_layer == "batch":
        norm_layer = nn.BatchNorm2d
    elif opts.layer == "spectral_instance":
        norm_layer = nn.InstanceNorm2d
    else:
        print("not implemented")
        exit()
    return norm_layer

def get_conv2d_layer(in_c, out_c, k, s, p=0, dilation=1, groups=1):
    return nn.Conv2d(in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=k,
                    stride=s,
                    padding=p,dilation=dilation, groups=groups)

def get_deconv2d_layer(in_c, out_c, k=1, s=1, p=1):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear"),
        nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=k,
            stride=s,
            padding=p
        )
    )

class Decom(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom = nn.Sequential(
            get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=4, k=3, s=1, p=1),
            nn.ReLU()
        )

        self.anisotropic_diffusion = AnisotropicDiffusion()
        self.gradient_weighting = GradientAwareWeighting()

    def forward(self, input):
        decom = self.decom(input)
        R = decom[:, 0:3, :, :]
        L = decom[:, 3:4, :, :].expand(-1, 3, -1, -1)
        
        # Compute explicit constraints
        A_input = self.anisotropic_diffusion(input)
        A_reflectance = self.anisotropic_diffusion(R)
        texture_consistency_loss = F.l1_loss(A_input, A_reflectance)
        
        weights = self.gradient_weighting(L)
        grad_L_x = L[:, :, :, 1:] - L[:, :, :, :-1]
        grad_L_y = L[:, :, 1:, :] - L[:, :, :-1, :]
        weights_x = weights[:, :, :, 1:]
        weights_y = weights[:, :, 1:, :]
        weighted_grad_loss = torch.mean((weights_x * grad_L_x)**2) + torch.mean((weights_y * grad_L_y)**2)
        
        reconstruction_loss = F.mse_loss(R * L, input)
        
        constraint_losses = {
            'texture_consistency': 0.05 * texture_consistency_loss,
            'illumination_smoothing': 0.2 * weighted_grad_loss,
            'reconstruction': 0.05 * reconstruction_loss
        }
        
        return R, L[:, 0:1, :, :], constraint_losses

def aux_load_initialize(model, decom_model_path):
    if os.path.exists(decom_model_path):
        checkpoint_Decom_low = torch.load(decom_model_path)
        pretrained_state_dict = checkpoint_Decom_low['state_dict']['model_R']
        
        # Get current model state dict
        model_state_dict = model.state_dict()
        
        # Filter pretrained weights to only include existing keys
        filtered_state_dict = {}
        for key, value in pretrained_state_dict.items():
            if key in model_state_dict:
                filtered_state_dict[key] = value
            else:
                print(f"Skipping key not in enhanced model: {key}")
        
        # Load only the compatible weights
        model.load_state_dict(filtered_state_dict, strict=False)
        
        # Print which new components were initialized randomly
        missing_keys = set(model_state_dict.keys()) - set(filtered_state_dict.keys())
        if missing_keys:
            print(f"New enhancement components initialized randomly: {missing_keys}")
        
        # Freeze the base decomposition parameters (keep existing behavior)
        for name, param in model.named_parameters():
            if name in filtered_state_dict:  # Only freeze loaded parameters
                param.requires_grad = False
            # New enhancement components remain trainable
        
        return model
    else:
        print("pretrained Initialize Model does not exist, check ---> %s " % decom_model_path)
        exit()

@MODEL_REGISTRY.register()
class RestoRect_S2Model(SRModel):
    """
    Updated RestoRect Stage 2 model using Rectified Flow instead of DDPM.
    
    Key improvements:
    1. Faster sampling with rectified flow (1-4 steps vs 100+ DDPM steps)
    2. More stable training with velocity prediction
    3. Simpler mathematics with straight-line paths
    """

    def __init__(self, opt):
        self.use_flex = opt.get('use_flex', True)
        super(RestoRect_S2Model, self).__init__(opt)
        if self.is_train:
            self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
            if self.mixing_flag:
                print("-----------------------mixup on-----------------------")
                mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
                use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
                self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)
        self.net_g_S1 = build_network(opt['network_S1'])
        self.net_g_S1 = self.model_to_device(self.net_g_S1)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_S1', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g_S1, load_path, True, param_key)
        
        self.net_g_S1.eval()
        if self.opt['dist']:
            self.model_Es1_rex = self.net_g_S1.module.E_rex
            self.model_Es1_img = self.net_g_S1.module.E_img
        else:
            self.model_Es1_rex = self.net_g_S1.E_rex
            self.model_Es1_img = self.net_g_S1.E_img
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        if self.is_train:
            self.encoder_iter = opt["train"]["encoder_iter"]
            self.lr_encoder = opt["train"]["lr_encoder"]
            self.lr_sr = opt["train"]["lr_sr"]
            self.gamma_encoder = opt["train"]["gamma_encoder"]
            self.gamma_sr = opt["train"]["gamma_sr"]
            self.lr_decay_encoder = opt["train"]["lr_decay_encoder"]
            self.lr_decay_sr = opt["train"]["lr_decay_sr"]

        self.Decom_l = Decom().cuda()
        self.Decom_l = aux_load_initialize(self.Decom_l, opt['pretrain_decomnet_low'])
        self.Decom_l.eval()

        self.Decom_h = Decom().cuda()
        self.Decom_h = aux_load_initialize(self.Decom_h, opt['pretrain_decomnet_high'])
        self.Decom_h.eval()

        # Add FLEX option parsing
        if self.use_flex and self.is_train:
            print("----------------------- FLEX enabled -----------------------")

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized in the second stage.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # Parameters for rectified flow velocity predictors
        parms=[]
        for k,v in self.net_g.named_parameters():
            if "rex_denoise" in k or "rex_condition" in k or "img_denoise" in k or "img_condition" in k or"denoise" in k or "condition" in k:
                parms.append(v)
        self.optimizer_e = self.get_optimizer(optim_type, parms, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_e)


    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepRestartLR(optimizer,
                                                    **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartLR(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingWarmupRestarts':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingWarmupRestarts(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartCyclicLR(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'TrueCosineAnnealingLR':
            print('..', 'cosineannealingLR')
            for optimizer in self.optimizers:
                self.schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingLRWithRestart':
            print('..', 'CosineAnnealingLR_With_Restart')
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLRWithRestart(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'LinearLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.LinearLR(
                        optimizer, train_opt['total_iter']))
        elif scheduler_type == 'VibrateLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.VibrateLR(
                        optimizer, train_opt['total_iter']))
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('kd_opt'):
            self.cri_kd = build_loss(train_opt['kd_opt']).to(self.device)
        else:
            self.cri_kd = None

        if train_opt.get('recon_opt'):
            self.cri_recon = build_loss(train_opt['recon_opt']).to(self.device)
        else:
            self.cri_recon = None

        # Add velocity loss for rectified flow training
        if train_opt.get('velocity_opt'):
            self.cri_velocity = build_loss(train_opt['velocity_opt']).to(self.device)
        else:
            self.cri_velocity = nn.MSELoss()  # Default MSE for velocity prediction

        print(f"Tried to initialize FLEX loss")
        # Add FLEX loss initialization
        if hasattr(self, 'use_flex') and self.use_flex:
            train_opt = self.opt['train']
            flex_opt = train_opt.get('flex_opt', {})

            # Use default target layers for RestoRectS2
            target_layers = flex_opt.get('target_layers', [
                'decoder_level3', 'decoder_level2', 'decoder_level1', 'img_refinement'
            ])
            
            print(f"Initializing FLEX with target layers: {target_layers}")

            try:
                self.cri_flex = FlexLoss(
                    target_layers=target_layers,
                    layer_weights=flex_opt.get('layer_weights', None),
                    loss_weight=flex_opt.get('loss_weight', 0.15),
                    snr_threshold=flex_opt.get('snr_threshold', 0.4),
                    outlier_threshold=flex_opt.get('outlier_threshold', 2.5),
                    kernel_size=flex_opt.get('kernel_size', 3),
                    apply_cross_norm=flex_opt.get('apply_cross_norm', True),
                    apply_outlier_detection=flex_opt.get('apply_outlier_detection', True)
                ).to(self.device)

                print(f"FLEX initialized successfully with:")
                print(f"  - Loss weight: {flex_opt.get('loss_weight', 0.15)}")
                print(f"  - SNR threshold: {flex_opt.get('snr_threshold', 0.4)}")
                print(f"  - Target layers: {target_layers}")
                
            except Exception as e:
                print(f"FLEX initialization failed: {e}")
                self.cri_flex = None
        else:
            print(f"FLEX loss not initialized - use_flex: {getattr(self, 'use_flex', 'not set')}")
            self.cri_flex = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_recon is None:
            raise ValueError('All losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.is_train and self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RestoRect_S2Model, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def pad_test(self, window_size):        
        # scale = self.opt.get('scale', 1)
        scale = 1
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        lq = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        gt = F.pad(self.gt, (0, mod_pad_w*scale, 0, mod_pad_h*scale), 'reflect')
        return lq,gt,mod_pad_h,mod_pad_w

    def test(self):
        window_size = self.opt['val'].get('window_size', 0)
        if window_size:
            lq,gt,mod_pad_h,mod_pad_w=self.pad_test(window_size)
        else:
            lq=self.lq

        with torch.no_grad():
            r_lq, i_lq, _ = self.Decom_l(lq)

        retinex_lq = torch.cat([r_lq, i_lq], dim=1)

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(lq, retinex_lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(lq, retinex_lq)
            self.net_g.train()
        if window_size:
            scale = self.opt.get('scale', 1)
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def compute_velocity_loss(self, pred_features, target_features, clean_features):
        """Compute velocity loss for rectified flow training
        
        Velocity = target - clean (noise - original_features)
        Loss = ||predicted_velocity - true_velocity||^2
        """
        # True velocity: from clean features to target features
        true_velocity = target_features - clean_features
        
        # Predicted velocity should match this
        velocity_loss = self.cri_velocity(pred_features, true_velocity)
        
        return velocity_loss

    def compute_trajectory_consistency_loss(self, pred_list, target_feature):
        """Ensure RF trajectory is smooth and consistent"""
        if len(pred_list) < 2:
            return torch.tensor(0.0, device=target_feature.device)
        
        consistency_loss = 0.0
        
        # Smooth trajectory constraint - consecutive predictions should be similar
        for i in range(len(pred_list) - 1):
            diff = pred_list[i+1] - pred_list[i]
            consistency_loss += torch.mean(diff ** 2) * 0.1
        
        # Final prediction should be close to target S1 feature
        if len(pred_list) > 0:
            final_loss = F.mse_loss(pred_list[-1], target_feature)
            consistency_loss += final_loss * 0.5
        
        return consistency_loss

    def optimize_parameters(self, current_iter):
        # Clear cache at start
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            self.r_gt, self.i_gt, _ = self.Decom_h(self.gt)
            self.r_lq, self.i_lq, _ = self.Decom_l(self.lq)

        self.retinex_gt = torch.cat([self.r_gt, self.i_gt], dim=1)
        self.retinex_lq = torch.cat([self.r_lq, self.i_lq], dim=1)

        # Learning rate scheduling
        if current_iter < self.encoder_iter:
            lr_encoder = self.lr_encoder * (self.gamma_encoder ** ((current_iter) // self.lr_decay_encoder))
            for param_group in self.optimizer_e.param_groups:
                param_group['lr'] = lr_encoder
        else:
            lr = self.lr_sr * (self.gamma_sr ** ((current_iter - self.encoder_iter) // self.lr_decay_sr))
            for param_group in self.optimizer_g.param_groups:
                param_group['lr'] = lr 
        
        l_total = 0
        loss_dict = OrderedDict()
        
        # Get S1 features WITH no_grad to prevent gradient accumulation
        with torch.no_grad():
            _, S1_IPR_rex = self.model_Es1_rex(self.retinex_lq, self.retinex_gt)
            _, S1_IPR_img = self.model_Es1_img(self.lq, self.gt)
            # Detach to ensure no gradients
            S1_IPR_rex = [f.detach() for f in S1_IPR_rex]
            S1_IPR_img = [f.detach() for f in S1_IPR_img]

        if current_iter < self.encoder_iter:
            # Phase 1: Train only velocity predictors
            self.optimizer_e.zero_grad()
            
            if self.opt['dist']:
                rex_diffusion = self.net_g.module.rex_diffusion
                img_diffusion = self.net_g.module.img_diffusion
            else:
                rex_diffusion = self.net_g.rex_diffusion
                img_diffusion = self.net_g.img_diffusion
            
            _, pred_IPR_list_rex = rex_diffusion(self.retinex_lq, S1_IPR_rex[0])
            _, pred_IPR_list_img = img_diffusion(self.lq, S1_IPR_img[0])

            i_rex = len(pred_IPR_list_rex) - 1
            i_img = len(pred_IPR_list_img) - 1

            S2_IPR_rex = [pred_IPR_list_rex[i_rex]]
            S2_IPR_img = [pred_IPR_list_img[i_img]]

            # Knowledge distillation losses
            l_kd_r, l_abs_r = self.cri_kd(S1_IPR_rex, S2_IPR_rex)
            l_kd_i, l_abs_i = self.cri_kd(S1_IPR_img, S2_IPR_img)

            # Velocity losses
            if hasattr(rex_diffusion, 'velocity_loss'):
                l_velocity_rex = rex_diffusion.velocity_loss
                l_total += l_velocity_rex * 0.1
                loss_dict['l_velocity_rex'] = l_velocity_rex
                    
            if hasattr(img_diffusion, 'velocity_loss'):
                l_velocity_img = img_diffusion.velocity_loss
                l_total += l_velocity_img * 0.1
                loss_dict['l_velocity_img'] = l_velocity_img

            l_total += l_abs_r + l_abs_i
            loss_dict['r_l_kd_%d'%(i_rex)] = l_kd_r
            loss_dict['r_l_abs_%d'%(i_rex)] = l_abs_r
            loss_dict['i_l_kd_%d' % (i_img)] = l_kd_i
            loss_dict['i_l_abs_%d' % (i_img)] = l_abs_i

            l_total.backward()
            self.optimizer_e.step()

            # Clear intermediate tensors
            del pred_IPR_list_rex, pred_IPR_list_img, S2_IPR_rex, S2_IPR_img
            torch.cuda.empty_cache()

        else:
            # Phase 2: Train full network
            self.optimizer_g.zero_grad()
            
            S1_IPR = [S1_IPR_rex[0], S1_IPR_img[0]]
            self.output, pred_IPR_list, output_rex = self.net_g(self.lq, self.retinex_lq, S1_IPR)
            output_decom_img = output_rex[0]
            output_decom_mat = output_rex[1]

            # 1. Main reconstruction loss
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

            # 2. Decomposition consistency losses
            l_recon_in = self.cri_pix(output_decom_img, self.lq)
            l_total += l_recon_in
            loss_dict['l_recon_in'] = l_recon_in

            l_recon_out = self.cri_pix(output_decom_mat, self.retinex_gt)
            l_total += l_recon_out
            loss_dict['l_recon_out'] = l_recon_out

            # 3. Knowledge distillation losses
            i_rex = len(pred_IPR_list[0]) - 1
            i_img = len(pred_IPR_list[1]) - 1

            S2_IPR_rex = [pred_IPR_list[0][i_rex]]
            S2_IPR_img = [pred_IPR_list[1][i_img]]

            l_kd_r, l_abs_r = self.cri_kd(S1_IPR_rex, S2_IPR_rex)
            l_kd_i, l_abs_i = self.cri_kd(S1_IPR_img, S2_IPR_img)

            l_total += l_abs_r + l_abs_i
            loss_dict['r_l_kd_%d'%(i_rex)] = l_kd_r
            loss_dict['r_l_abs_%d'%(i_rex)] = l_abs_r
            loss_dict['i_l_kd_%d' % (i_img)] = l_kd_i
            loss_dict['i_l_abs_%d' % (i_img)] = l_abs_i

            # 4. Efficient FLEX Loss - core innovation preserved
            if self.cri_flex is not None:
                # Use key features from each stream
                teacher_features = [S1_IPR_rex[0], S1_IPR_img[0]]  # Key teacher features
                student_features = [pred_IPR_list[0][i_rex], pred_IPR_list[1][i_img]]  # Student features
                
                # Apply FLEX with SNR thresholding
                timestep = 0.2  # High SNR regime for RF
                l_flex = self.cri_flex(teacher_features, student_features, 
                                    timestep=timestep, max_timestep=1.0)

                l_total += l_flex
                loss_dict['l_flex'] = l_flex

            # 5. Velocity losses (lower weight in Phase 2)
            if self.opt.get('dist', False):
                rex_diffusion = self.net_g.module.rex_diffusion
                img_diffusion = self.net_g.module.img_diffusion
            else:
                rex_diffusion = self.net_g.rex_diffusion
                img_diffusion = self.net_g.img_diffusion
                
            if hasattr(rex_diffusion, 'velocity_loss'):
                l_velocity_rex = rex_diffusion.velocity_loss
                l_total += l_velocity_rex * 0.05
                loss_dict['l_velocity_rex'] = l_velocity_rex
                
            if hasattr(img_diffusion, 'velocity_loss'):
                l_velocity_img = img_diffusion.velocity_loss
                l_total += l_velocity_img * 0.05
                loss_dict['l_velocity_img'] = l_velocity_img

            l_total.backward()
            self.optimizer_g.step()

            # Clear large intermediate tensors after backward pass
            del output_decom_img, output_decom_mat, pred_IPR_list, S2_IPR_rex, S2_IPR_img
            torch.cuda.empty_cache()

        # Clear S1 features at the end
        del S1_IPR_rex, S1_IPR_img
        
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)