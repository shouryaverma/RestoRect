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
import torch.nn as nn
import os

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
        
        for c in range(C):
            img_c = image[:, c:c+1, :, :]
            grad_x = F.conv2d(img_c, sobel_x, padding=1)
            grad_y = F.conv2d(img_c, sobel_y, padding=1)
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
            diffusion_coeff = torch.exp(-(grad_mag / self.s)**2)
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
        self.eps = 1e-8
        if learnable_params:
            self.k = nn.Parameter(torch.tensor(1.0))
        else:
            self.k = 1.0
    
    def rgb_to_hvi(self, rgb_img):
        """Polarized transformation eliminates red discontinuity"""
        I_max = torch.max(rgb_img, dim=1, keepdim=True)[0]
        hsv = self.rgb_to_hsv(rgb_img)
        H, S, V = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
        
        # Polarized coordinates (eliminates red discontinuity)
        h_polar = torch.cos(3.14159 * H / 3)  # Orthogonal horizontal
        v_polar = torch.sin(3.14159 * H / 3)  # Orthogonal vertical
        
        # Adaptive intensity collapse
        C_k = self.k * torch.sin(3.14159 * I_max / 2) + self.eps
        
        # Final HV maps
        H_hv = C_k * S * h_polar
        V_hv = C_k * S * v_polar
        
        return torch.cat([H_hv, V_hv, I_max], dim=1)
    
    def rgb_to_hsv(self, rgb):
        """Standard RGB to HSV conversion"""
        max_val, max_idx = torch.max(rgb, dim=1, keepdim=True)
        min_val = torch.min(rgb, dim=1, keepdim=True)[0]
        diff = max_val - min_val
        
        hue = torch.zeros_like(max_val)
        # Red maximum
        mask = (max_idx == 0) & (diff > 0)
        hue[mask] = (rgb[:, 1:2] - rgb[:, 2:3])[mask] / diff[mask]
        # Green maximum
        mask = (max_idx == 1) & (diff > 0)
        hue[mask] = 2.0 + (rgb[:, 2:3] - rgb[:, 0:1])[mask] / diff[mask]
        # Blue maximum
        mask = (max_idx == 2) & (diff > 0)
        hue[mask] = 4.0 + (rgb[:, 0:1] - rgb[:, 1:2])[mask] / diff[mask]
        
        hue = hue / 6.0
        hue[hue < 0] += 1.0
        
        saturation = torch.where(max_val > 0, diff / max_val, torch.zeros_like(max_val))
        value = max_val
        
        return torch.cat([hue, saturation, value], dim=1)

class PolarizedHVIColorLoss(nn.Module):
    def __init__(self, weight=0.05):
        super().__init__()
        self.hvi_transform = PolarizedHVIColorSpace(learnable_params=True)
        self.weight = weight
        
    def forward(self, pred, gt):
        pred_hvi = self.hvi_transform.rgb_to_hvi(pred)
        gt_hvi = self.hvi_transform.rgb_to_hvi(gt)
        
        h_loss = F.l1_loss(pred_hvi[:, 0:1], gt_hvi[:, 0:1])  # Horizontal
        v_loss = F.l1_loss(pred_hvi[:, 1:2], gt_hvi[:, 1:2])  # Vertical
        i_loss = F.l1_loss(pred_hvi[:, 2:3], gt_hvi[:, 2:3])  # Intensity
        
        return self.weight * (h_loss + v_loss + i_loss)
        
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
            'illumination_smoothing': 0.05 * weighted_grad_loss,
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
class RetiDiff_S1Model(SRModel):
    """
    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RetiDiff_S1Model, self).__init__(opt)
        if self.is_train:
            self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
            if self.mixing_flag:
                mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
                use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
                self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.Decom_l = Decom().cuda()
        self.Decom_l = aux_load_initialize(self.Decom_l,opt['pretrain_decomnet_low'])
        self.Decom_l.eval()

        self.Decom_h = Decom().cuda()
        self.Decom_h = aux_load_initialize(self.Decom_h,opt['pretrain_decomnet_high'])
        self.Decom_h.eval()

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepRestartLR(optimizer,
                                     []               **train_opt['scheduler']))
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
        # Call parent class init_training_settings first
        super().init_training_settings()
        
        # Add polarized color loss
        self.cri_hvi = PolarizedHVIColorLoss(weight=0.05).to(self.device)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.is_train and self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RetiDiff_S1Model, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
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
            gt=self.gt

        with torch.no_grad():
            r_gt, i_gt, _ = self.Decom_h(gt)
            r_lq, i_lq, _ = self.Decom_l(lq)

        retinex = [(r_lq, i_lq),(r_gt, i_gt)]

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(lq, gt, retinex)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq, self.gt, self.retinex)
            self.net_g.train()

        if window_size:
            scale = self.opt.get('scale', 1)
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def optimize_parameters(self, current_iter):
        with torch.no_grad():
            r_gt, i_gt, constraint_losses_gt = self.Decom_h(self.gt)
            r_lq, i_lq, constraint_losses_lq = self.Decom_l(self.lq)

        self.retinex = [(r_lq, i_lq),(r_gt, i_gt)]

        self.optimizer_g.zero_grad()
        self.output, self.recon = self.net_g(self.lq, self.gt, self.retinex)
        self.gt_r_i = torch.cat((r_gt, i_gt), dim=1)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
            l_recon_out = self.cri_pix(self.recon[1], self.gt_r_i)
            l_total += l_recon_out
            loss_dict['l_recon_out'] = l_recon_out
            l_recon_in = self.cri_pix(self.recon[0], self.lq)
            l_total += l_recon_in
            loss_dict['l_recon_in'] = l_recon_in
            l_hvi = self.cri_hvi(self.output, self.gt)
            l_total += l_hvi
            loss_dict['l_hvi'] = l_hvi

            # Add constraint losses
            for name, loss_val in constraint_losses_gt.items():
                l_total += loss_val
                loss_dict[f'gt_{name}'] = loss_val

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
