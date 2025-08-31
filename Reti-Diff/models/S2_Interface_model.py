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
from basicsr.metrics import calculate_metric

from torch import nn
from basicsr.archs import build_network
from basicsr.utils import get_root_logger
from basicsr.losses import build_loss
import os
from basicsr.utils import get_root_logger, imwrite, tensor2img
from os import path as osp


from basicsr.utils.dist_util import master_only
from torch.nn.parallel import DataParallel, DistributedDataParallel
import thop
from tqdm import tqdm

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
        checkpoint_Decom_low = torch.load(decom_model_path, map_location='cpu')
        
        # Handle different checkpoint formats robustly
        pretrained_state_dict = None
        
        # Try different possible formats
        if isinstance(checkpoint_Decom_low, dict):
            if 'state_dict' in checkpoint_Decom_low:
                if 'model_R' in checkpoint_Decom_low['state_dict']:
                    pretrained_state_dict = checkpoint_Decom_low['state_dict']['model_R']
                else:
                    pretrained_state_dict = checkpoint_Decom_low['state_dict']
            elif 'model' in checkpoint_Decom_low:
                pretrained_state_dict = checkpoint_Decom_low['model']
            else:
                # Assume the entire checkpoint is the state dict
                pretrained_state_dict = checkpoint_Decom_low
        else:
            # If it's not a dict, treat it as the state dict directly
            pretrained_state_dict = checkpoint_Decom_low
        
        if pretrained_state_dict is None:
            raise ValueError(f"Could not extract state dict from checkpoint: {decom_model_path}")
        
        # Get current model state dict
        model_state_dict = model.state_dict()
        
        # Filter pretrained weights to only include existing keys
        filtered_state_dict = {}
        for key, value in pretrained_state_dict.items():
            if key in model_state_dict:
                filtered_state_dict[key] = value
            else:
                print(f"Skipping key not in current model: {key}")
        
        # Load only the compatible weights
        model.load_state_dict(filtered_state_dict, strict=False)
        
        # Print which new components were initialized randomly
        missing_keys = set(model_state_dict.keys()) - set(filtered_state_dict.keys())
        if missing_keys:
            print(f"New components initialized randomly: {missing_keys}")
        
        # Print what was successfully loaded
        print(f"Successfully loaded {len(filtered_state_dict)} parameters from {decom_model_path}")
        
        # Freeze the base decomposition parameters
        for name, param in model.named_parameters():
            if name in filtered_state_dict:  # Only freeze loaded parameters
                param.requires_grad = False
        
        return model
    else:
        print("pretrained Initialize Model does not exist, check ---> %s " % decom_model_path)
        exit()

@MODEL_REGISTRY.register()
class S2_Interface_Model(SRModel):
    """
    RetiDiff S2 Interface Model for testing with RectifiedFlow
    Updated to support RectifiedFlow instead of DDPM
    """

    def __init__(self, opt):
        super(S2_Interface_Model, self).__init__(opt)
        if self.is_train:
            self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
            if self.mixing_flag:
                print("-----------------------mixup on-----------------------")
                mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
                use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
                self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.num_gpu = opt['num_gpu']

        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        if self.is_train:
            self.encoder_iter = opt["train"]["encoder_iter"]
            self.lr_encoder = opt["train"]["lr_encoder"]
            self.lr_sr = opt["train"]["lr_sr"]
            self.gamma_encoder = opt["train"]["gamma_encoder"]
            self.gamma_sr = opt["train"]["gamma_sr"]
            self.lr_decay_encoder = opt["train"]["lr_decay_encoder"]
            self.lr_decay_sr = opt["train"]["lr_decay_sr"]
        if self.num_gpu != 0:
            self.Decom_l = Decom().cuda()
        else:
            self.Decom_l = Decom()

        # Use the existing aux_load_initialize function that handles checkpoint format correctly
        self.Decom_l = aux_load_initialize(self.Decom_l, opt['pretrain_decomnet_low'])
        self.Decom_l.eval()


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
        self.is_train = False
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', True)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            img_ext = osp.splitext(osp.basename(val_data['lq_path'][0]))[1]
            print("img_name:", img_name + img_ext)
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}{img_ext}')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        self.is_train = True


    @master_only
    def print_network(self, net):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = f'{net.__class__.__name__} - {net.module.__class__.__name__}'
        else:
            net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger = get_root_logger()

        logger.info(f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        logger.info(net_str)

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
        return lq,mod_pad_h,mod_pad_w

    def test(self):
        """Testing with RectifiedFlow - much faster than DDPM"""
        window_size = self.opt['val'].get('window_size', 0)
        if window_size:
            lq,mod_pad_h,mod_pad_w=self.pad_test(window_size)
        else:
            lq=self.lq
        with torch.no_grad():
            r_lq, i_lq, _ = self.Decom_l(lq)

        retinex_lq = torch.cat([r_lq, i_lq], dim=1)

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                # RectifiedFlow inference is much simpler and faster
                self.output = self.net_g_ema(lq, retinex_lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                # RectifiedFlow inference is much simpler and faster
                self.output = self.net_g(lq, retinex_lq)
            self.net_g.train()
        if window_size:
            scale = self.opt.get('scale', 1)
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]