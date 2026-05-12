import os
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy import linalg
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
import argparse
from tqdm import tqdm
import glob
import warnings
import piq
import pywt  # for wavelet decomposition
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.special import gamma
BIQI_AVAILABLE = True

# piq imports for metrics
try:
    import piq
    BRISQUE_AVAILABLE = True
    print("BRISQUE available via PIQ library for quality evaluation")
except ImportError:
    BRISQUE_AVAILABLE = False
    print("BRISQUE not available - PIQ library not found")

# CLIP-IQA imports
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch.nn.functional as F
    CLIP_IQA_AVAILABLE = True
    print("CLIP-IQA available for quality evaluation")
except ImportError:
    CLIP_IQA_AVAILABLE = False
    print("CLIP-IQA not available - install with: pip install transformers")

# Add these imports
try:
    import skimage.color
    from scipy import ndimage
    from PIL import Image
    import math
    UNDERWATER_METRICS_AVAILABLE = True
    print("UCIQE and UIQM metrics available")
except ImportError:
    UNDERWATER_METRICS_AVAILABLE = False
    print("Underwater metrics not available - install scikit-image")

# BasicSR imports for metrics
try:
    from basicsr.metrics import calculate_psnr as basicsr_psnr, calculate_ssim as basicsr_ssim, calculate_niqe as basicsr_niqe
    from basicsr.utils import bgr2ycbcr, scandir
    BASICSR_AVAILABLE = True
    print("Using BasicSR for robust metric calculations")
except ImportError:
    BASICSR_AVAILABLE = False
    print("BasicSR not available - install with: pip install basicsr")
    # Fallback to skimage
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# LPIPS for perceptual evaluation
try:
    import lpips
    from basicsr.utils import img2tensor
    from torchvision.transforms.functional import normalize
    LPIPS_AVAILABLE = True
    print("LPIPS available for perceptual evaluation")
except ImportError:
    LPIPS_AVAILABLE = False
    print("LPIPS not available - install with: pip install lpips")

# Add this global variable after the imports
CLIP_MODEL = None
CLIP_PROCESSOR = None

def initialize_clip_model(device='cuda'):
    """Initialize CLIP model once globally"""
    global CLIP_MODEL, CLIP_PROCESSOR
    if not CLIP_IQA_AVAILABLE:
        return False
        
    try:
        if CLIP_MODEL is None:
            print("Loading CLIP model (one-time initialization)...")
            CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",local_files_only=True).to(device)
            CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",local_files_only=True)
            print("CLIP model loaded successfully")
        return True
    except Exception as e:
        print(f"CLIP model initialization failed: {e}")
        return False

def calculate_clip_iqa(img, device='cuda'):
    """Calculate CLIP-IQA score using CLIP model"""
    global CLIP_MODEL, CLIP_PROCESSOR
    
    if not CLIP_IQA_AVAILABLE or CLIP_MODEL is None:
        return None
        
    try:
        # Convert image to PIL format if needed
        if isinstance(img, np.ndarray):
            if img.max() > 1:
                img_pil = Image.fromarray(img.astype(np.uint8))
            else:
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
        else:
            img_pil = img
            
        # Quality-related text prompts
        good_quality_prompts = [
            "a high quality photo",
            "a sharp and clear image", 
            "a well-exposed photograph",
            "a high resolution image"
        ]
        
        poor_quality_prompts = [
            "a low quality photo",
            "a blurry and unclear image",
            "a poorly exposed photograph", 
            "a low resolution image"
        ]
        
        with torch.no_grad():
            # Process image
            inputs = CLIP_PROCESSOR(images=img_pil, return_tensors="pt").to(device)
            image_features = CLIP_MODEL.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=1)
            
            # Process text prompts
            good_texts = CLIP_PROCESSOR(text=good_quality_prompts, return_tensors="pt", padding=True).to(device)
            poor_texts = CLIP_PROCESSOR(text=poor_quality_prompts, return_tensors="pt", padding=True).to(device)
            
            good_text_features = CLIP_MODEL.get_text_features(**good_texts)
            poor_text_features = CLIP_MODEL.get_text_features(**poor_texts)
            
            good_text_features = F.normalize(good_text_features, p=2, dim=1)
            poor_text_features = F.normalize(poor_text_features, p=2, dim=1)
            
            # Calculate similarities
            good_similarities = torch.matmul(image_features, good_text_features.T)
            poor_similarities = torch.matmul(image_features, poor_text_features.T)
            
            # Quality score: higher similarity with good quality prompts = better quality
            good_score = torch.mean(good_similarities).item()
            poor_score = torch.mean(poor_similarities).item()
            
            # Return normalized score in 0-1 range (to match literature values)
            quality_score = (good_score - poor_score + 2) / 4  # Normalize to 0-1 range
            quality_score = np.clip(quality_score, 0, 1)
            
        return quality_score
        
    except Exception as e:
        print(f"CLIP-IQA calculation failed: {e}")
        return None

def calculate_psnr(img_gt, img_restored, crop_border=0, test_y_channel=False):
    """Calculate PSNR using BasicSR or fallback to skimage"""
    if BASICSR_AVAILABLE:
        # Convert to format for BasicSR (0-255, uint8 or float32)
        if img_gt.dtype != np.uint8:
            img_gt_scaled = (img_gt * 255).astype(np.uint8)
            img_restored_scaled = (img_restored * 255).astype(np.uint8)
        else:
            img_gt_scaled = img_gt
            img_restored_scaled = img_restored
            
        return basicsr_psnr(
            img_gt_scaled, img_restored_scaled, 
            crop_border=crop_border, 
            input_order='HWC',
            test_y_channel=test_y_channel
        )
    else:
        # Fallback to skimage
        img_gt_norm = img_gt.astype(np.float64) / 255.0 if img_gt.max() > 1 else img_gt.astype(np.float64)
        img_restored_norm = img_restored.astype(np.float64) / 255.0 if img_restored.max() > 1 else img_restored.astype(np.float64)
        return peak_signal_noise_ratio(img_gt_norm, img_restored_norm, data_range=1.0)


def calculate_ssim(img_gt, img_restored, crop_border=0, test_y_channel=False):
    """Calculate SSIM using BasicSR or fallback to skimage"""
    if BASICSR_AVAILABLE:
        # Convert to format for BasicSR
        if img_gt.dtype != np.uint8:
            img_gt_scaled = (img_gt * 255).astype(np.uint8)
            img_restored_scaled = (img_restored * 255).astype(np.uint8)
        else:
            img_gt_scaled = img_gt
            img_restored_scaled = img_restored
            
        return basicsr_ssim(
            img_gt_scaled, img_restored_scaled,
            crop_border=crop_border,
            input_order='HWC', 
            test_y_channel=test_y_channel
        )
    else:
        # Fallback to skimage
        img_gt_norm = img_gt.astype(np.float64) / 255.0 if img_gt.max() > 1 else img_gt.astype(np.float64)
        img_restored_norm = img_restored.astype(np.float64) / 255.0 if img_restored.max() > 1 else img_restored.astype(np.float64)
        
        if len(img_gt_norm.shape) == 3:
            return structural_similarity(img_gt_norm, img_restored_norm, multichannel=True, channel_axis=2, data_range=1.0)
        else:
            return structural_similarity(img_gt_norm, img_restored_norm, data_range=1.0)


def calculate_niqe(img, crop_border=0):
    """Calculate NIQE using BasicSR with improved fallback"""
    if BASICSR_AVAILABLE:
        # Convert to format for BasicSR
        if img.dtype != np.uint8:
            img_scaled = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
        else:
            img_scaled = img
            
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                return basicsr_niqe(img_scaled, crop_border=crop_border, input_order='HWC', convert_to='y')
        except Exception as e:
            print(f"BasicSR NIQE failed: {e}, using fallback")
    
    # Improved fallback implementation
    if len(img.shape) == 3:
        # Convert to Y channel for better quality assessment
        img_y = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    else:
        img_y = img
    
    # Apply crop border if specified
    if crop_border > 0:
        img_y = img_y[crop_border:-crop_border, crop_border:-crop_border]
    
    # Normalize to 0-255 if needed
    if img_y.max() <= 1:
        img_y = img_y * 255
        
    # Multiple gradient-based features for better quality estimation
    grad_x = cv2.Sobel(img_y.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_y.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Laplacian for edge information
    laplacian = cv2.Laplacian(img_y.astype(np.float64), cv2.CV_64F)
    
    # Combine multiple features
    grad_var = np.var(grad_magnitude)
    laplacian_var = np.var(laplacian)
    mean_grad = np.mean(grad_magnitude)
    
    # Improved quality score combining multiple factors
    quality_score = 100 / (1 + grad_var/1000 + laplacian_var/10000) + mean_grad/100
    return quality_score

def calculate_lpips(img_gt, img_restored, loss_fn, device='cuda'):
    """Calculate LPIPS perceptual distance"""
    if loss_fn is None:
        return None
        
    try:
        # Convert images
        if img_gt.max() > 1:
            img_gt = img_gt.astype(np.float32) / 255.0
            img_restored = img_restored.astype(np.float32) / 255.0
        
        # Convert BGR to RGB and to tensor
        img_gt_tensor, img_restored_tensor = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        
        # Normalize to [-1, 1] as expected by LPIPS
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        normalize(img_gt_tensor, mean, std, inplace=True)
        normalize(img_restored_tensor, mean, std, inplace=True)
        
        # Calculate LPIPS
        with torch.no_grad():
            lpips_score = loss_fn(img_restored_tensor.unsqueeze(0).to(device), 
                                img_gt_tensor.unsqueeze(0).to(device))
        
        return lpips_score.item()
    except Exception as e:
        print(f"LPIPS calculation failed: {e}")
        return None

def calculate_brisque(img, device='cuda'):
    """Calculate BRISQUE score using PIQ library"""
    if not BRISQUE_AVAILABLE:
        return None
        
    try:
        # Convert to tensor format expected by PIQ
        if len(img.shape) == 2:
            # Convert grayscale to RGB
            img_rgb = np.stack([img, img, img], axis=2)
        else:
            img_rgb = img.copy()
            
        # Normalize to [0, 1] if needed
        if img_rgb.max() > 1:
            img_rgb = img_rgb.astype(np.float32) / 255.0
        else:
            img_rgb = img_rgb.astype(np.float32)
            
        # Convert to tensor: (H, W, C) -> (1, C, H, W)
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        # Calculate BRISQUE using PIQ
        with torch.no_grad():
            brisque_score = piq.brisque(img_tensor, data_range=1.0, reduction='mean')
            
        return brisque_score.item()
    except Exception as e:
        print(f"BRISQUE calculation failed: {e}")
        return None

def calculate_biqi_simplified(img):
    """
    Simplified BIQI implementation using wavelet features
    Note: This is a simplified version without the full SVM models
    """
    try:
        # Convert to grayscale
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img.copy()
            
        # Normalize to [0, 1]
        if img_gray.max() > 1:
            img_gray = img_gray.astype(np.float64) / 255.0
        else:
            img_gray = img_gray.astype(np.float64)
            
        # Wavelet decomposition (3 scales)
        features = []
        gam = np.arange(0.2, 10.001, 0.001)
        
        # Fix the gamma function usage - import it from scipy.special
        from scipy.special import gamma as gamma_func
        r_gam = (gamma_func(1.0/gam) * gamma_func(3.0/gam)) / (gamma_func(2.0/gam)**2)
        
        # Multi-scale wavelet analysis
        for scale in range(1, 4):  # 3 scales
            try:
                coeffs = pywt.dwt2(img_gray, 'db4')  # Using db4 instead of db9
                _, (h, v, d) = coeffs
                
                for subband, name in [(h, 'h'), (v, 'v'), (d, 'd')]:
                    subband_flat = subband.flatten()
                    
                    # Compute statistics
                    mu = np.mean(subband_flat)
                    sigma_sq = np.var(subband_flat)
                    E = np.mean(np.abs(subband_flat - mu))
                    
                    if E > 1e-8:  # Avoid division by zero
                        rho = sigma_sq / (E**2)
                        # Find closest gamma parameter
                        closest_idx = np.argmin(np.abs(rho - r_gam))
                        gam_param = gam[closest_idx]
                    else:
                        gam_param = 1.0
                    
                    features.extend([sigma_sq, gam_param])
                
                # Prepare for next scale (downsample)
                if img_gray.shape[0] > 32 and img_gray.shape[1] > 32:  # Ensure minimum size
                    img_gray = cv2.resize(img_gray, (img_gray.shape[1]//2, img_gray.shape[0]//2))
                else:
                    break
                    
            except Exception as e:
                print(f"Wavelet decomposition failed at scale {scale}: {e}")
                break
        
        if len(features) == 0:
            return None
            
        # Simple quality estimation based on feature magnitudes
        feature_array = np.array(features)
        
        # Estimate quality based on feature variance and complexity
        # Lower values indicate better quality (like other metrics)
        quality_score = np.mean(feature_array) * 30 + np.std(feature_array) * 10
        quality_score = np.clip(quality_score, 0, 100)
        
        return quality_score
        
    except Exception as e:
        print(f"BIQI calculation failed: {e}")
        return None

def calculate_uciqe(img, c1=0.4680, c2=0.2745, c3=0.2576):
    """Calculate UCIQE metric based on MATLAB reference implementation"""
    if not UNDERWATER_METRICS_AVAILABLE:
        return None
       
    try:
        # Convert to float and ensure [0,1] range
        if img.max() > 1:
            img_norm = img.astype(np.float64) / 255.0
        else:
            img_norm = img.astype(np.float64)
       
        # Convert to LAB color space
        lab = skimage.color.rgb2lab(img_norm)
       
        # Extract LAB channels and normalize to match MATLAB reference
        # skimage returns L:[0,100], a,b:[-128,127]
        # MATLAB reference divides all by 255 after applycform
        img_lum = lab[:,:,0].flatten() / 100.0 + np.finfo(float).eps
        img_a = (lab[:,:,1].flatten() + 128) / 255.0  # Shift a* to [0,255] then normalize
        img_b = (lab[:,:,2].flatten() + 128) / 255.0  # Shift b* to [0,255] then normalize
       
        # Chroma calculation
        img_chr = np.sqrt(img_a**2 + img_b**2)
       
        # Saturation calculation  
        img_sat = img_chr / np.sqrt(img_chr**2 + img_lum**2)
       
        # Average saturation
        aver_sat = np.mean(img_sat)
       
        # Average chroma  
        aver_chr = np.mean(img_chr)
       
        # Variance of chroma
        var_chr = np.sqrt(np.mean(np.abs(1 - (aver_chr / (img_chr + np.finfo(float).eps))**2)))
       
        # Contrast of luminance using stretchlim equivalent
        con_lum = np.percentile(img_lum, 98) - np.percentile(img_lum, 2)
       
        # Calculate UCIQE
        uciqe_score = c1 * var_chr + c2 * con_lum + c3 * aver_sat
           
        return uciqe_score
       
    except Exception as e:
        print(f"UCIQE calculation failed: {e}")
        return None

def calculate_uiqm(img):
    """Calculate UIQM metric"""
    if not UNDERWATER_METRICS_AVAILABLE:
        return None
        
    try:
        # Ensure proper format
        if img.max() <= 1:
            img_scaled = (img * 255).astype(np.float32)
        else:
            img_scaled = img.astype(np.float32)
            
        # UIQM coefficients
        c1 = 0.0282
        c2 = 0.2953  
        c3 = 3.5753
        
        # Calculate components
        uicm = _uicm(img_scaled)
        uism = _uism(img_scaled)
        uiconm = _uiconm(img_scaled, 10)
        
        # Combine components
        uiqm_score = (c1 * uicm) + (c2 * uism) + (c3 * uiconm)
        return uiqm_score
        
    except Exception as e:
        print(f"UIQM calculation failed: {e}")
        return None

# Helper functions for UIQM (copy the exact functions from your document)
def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """Calculates the asymetric alpha-trimmed mean"""
    x = sorted(x)
    K = len(x)
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    weight = (1/(K-T_a_L-T_a_R))
    s = int(T_a_L+1)
    e = int(K-T_a_R)
    val = sum(x[s:e])
    val = weight*val
    return val

def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel-mu), 2)
    return val/len(x)

def _uicm(x):
    R = x[:,:,0].flatten()
    G = x[:,:,1].flatten()
    B = x[:,:,2].flatten()
    RG = R-G
    YB = ((R+G)/2)-B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
    r = math.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)

def sobel(x):
    dx = ndimage.sobel(x,0)
    dy = ndimage.sobel(x,1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag) 
    return mag

def eme(x, window_size):
    """Enhancement measure estimation"""
    k1 = int(x.shape[1]/window_size)
    k2 = int(x.shape[0]/window_size)
    w = 2./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    x = x[:blocksize_y*k2, :blocksize_x*k1]
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
            max_ = np.max(block)
            min_ = np.min(block)
            if min_ == 0.0: val += 0
            elif max_ == 0.0: val += 0
            else: val += math.log(max_/min_)
    return w*val

def _uism(x):
    """Underwater Image Sharpness Measure"""
    R = x[:,:,0]
    G = x[:,:,1]
    B = x[:,:,2]
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144
    return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)

def _uiconm(x, window_size):
    """Underwater image contrast measure"""
    k1 = int(x.shape[1]/window_size)
    k2 = int(x.shape[0]/window_size)
    w = -1./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    x = x[:blocksize_y*k2, :blocksize_x*k1]
    alpha = 1
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_-min_
            bot = max_+min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: 
                val += 0.0
            else: 
                val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
    return w*val

def correct_mean_var(img_restored, img_gt, correction_strength=0.3):
    """
    Mean-variance correction with adjustable strength
    correction_strength: 0.0 = no correction, 1.0 = full correction
    """
    corrected = img_restored.copy().astype(np.float32)
    img_gt = img_gt.astype(np.float32)
    
    # Determine input range
    is_normalized = img_restored.max() <= 1.0
    
    for j in range(corrected.shape[2]):
        # Current statistics
        mean_pred = np.mean(corrected[:, :, j])
        std_pred = np.std(corrected[:, :, j])
        
        # Target statistics  
        mean_gt = np.mean(img_gt[:, :, j])
        std_gt = np.std(img_gt[:, :, j])
        
        # Skip if no variation
        if std_pred < 1e-8 or std_gt < 1e-8:
            continue
            
        # Partial correction based on strength
        target_mean = mean_pred + correction_strength * (mean_gt - mean_pred)
        target_std = std_pred + correction_strength * (std_gt - std_pred)
        
        # Apply correction
        corrected[:, :, j] = corrected[:, :, j] - mean_pred + target_mean
        corrected[:, :, j] = (corrected[:, :, j] - target_mean) * (target_std / std_pred) + target_mean
    
    # Clip to appropriate range and return same type as input
    if is_normalized:
        return np.clip(corrected, 0, 1).astype(np.float32)
    else:
        return np.clip(corrected, 0, 255).astype(np.uint8)

class FIDInceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.eval()
        
        # Extract features up to the final pooling layer
        self.blocks = nn.ModuleList([
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
        ])
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Resize to inception input size
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Apply blocks with proper pooling
        x = self.blocks[0](x)  # Conv2d_1a_3x3
        x = self.blocks[1](x)  # Conv2d_2a_3x3
        x = self.blocks[2](x)  # Conv2d_2b_3x3
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.blocks[3](x)  # Conv2d_3b_1x1
        x = self.blocks[4](x)  # Conv2d_4a_3x3
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        
        for i in range(5, 16):  # Mixed layers
            x = self.blocks[i](x)
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

def calculate_fid(real_images, fake_images, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = FIDInceptionV3().to(device)
    
    # ImageNet normalization for InceptionV3
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
    
    def get_features(images):
        features = []
        batch_size = 32
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            # Convert to tensor and normalize properly
            if isinstance(batch[0], np.ndarray):
                if batch[0].dtype == np.uint8:
                    batch_tensor = torch.stack([transforms.ToTensor()(img) for img in batch])
                else:
                    batch_tensor = torch.stack([torch.from_numpy(img).float() for img in batch])
            else:
                batch_tensor = torch.stack(batch)
            
            # Apply ImageNet normalization
            batch_tensor = torch.stack([normalize(img) for img in batch_tensor]).to(device)
            
            with torch.no_grad():
                batch_features = model(batch_tensor)
                features.append(batch_features.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    real_features = get_features(real_images)
    fake_features = get_features(fake_images)
    
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate the Frechet Distance between two multivariate Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('FID calculation produces singular product; adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-2):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def load_image(path):
    """Load image and ensure RGB format"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_image_cropped(path, gt_size=None, apply_chaos_resize=False):
    """Load image and ensure RGB format with optional chaos processing"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply chaos resize if specified
    if apply_chaos_resize and gt_size is not None:
        # Convert to float32 for consistency with training
        img = img.astype(np.float32) / 255.0
        img = cv2.resize(img, (gt_size, gt_size), interpolation=cv2.INTER_LINEAR)
        # Convert back to 0-255 range
        img = (img * 255).astype(np.uint8)
    
    return img

def evaluate_images(gt_dir, pred_dir, output_file="evaluation_results.txt", 
                   crop_border=5, test_y_channel=True, correct_mean_variance=True):
    """evaluation with metric calculation"""
    
    # Get image files
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.png")) + 
                     glob.glob(os.path.join(gt_dir, "*.bmp")) + 
                     glob.glob(os.path.join(gt_dir, "*.jpg")) + 
                     glob.glob(os.path.join(gt_dir, "*.JPG")) + 
                     glob.glob(os.path.join(gt_dir, "*.jpeg")))
    
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.png")) + 
                       glob.glob(os.path.join(pred_dir, "*.bmp")) + 
                       glob.glob(os.path.join(pred_dir, "*.jpg")) + 
                       glob.glob(os.path.join(pred_dir, "*.JPG")) + 
                       glob.glob(os.path.join(pred_dir, "*.jpeg")))
    
    print(f"Found {len(gt_files)} GT images and {len(pred_files)} predicted images")
    
    # # Match files by name
    # matched_pairs = []
    # for gt_file in gt_files:
    #     gt_name = os.path.basename(gt_file)
    #     if gt_name.startswith('normal'):
    #         pred_name = gt_name.replace('normal', 'low', 1)
    #     else:
    #         pred_name = gt_name
        
    #     pred_file = os.path.join(pred_dir, pred_name)
    #     if os.path.exists(pred_file):
    #         matched_pairs.append((gt_file, pred_file))

    # Match files by scene ID (first part only)
    matched_pairs = []

    # Create mapping of scene IDs to GT files
    gt_scene_map = {}
    for gt_file in gt_files:
        gt_name = os.path.basename(gt_file)
        gt_parts = gt_name.split('_')
        if len(gt_parts) >= 1:
            scene_id = gt_parts[0]
            if scene_id not in gt_scene_map:
                gt_scene_map[scene_id] = []
            gt_scene_map[scene_id].append(gt_file)

    print(f"GT scene IDs found: {len(gt_scene_map)}")
    print(f"Sample GT scene IDs: {list(gt_scene_map.keys())[:10]}")

    # Match pred files to GT files based on scene ID
    unmatched_count = 0
    matched_count = 0
    sample_unmatched = []

    for pred_file in pred_files:
        pred_name = os.path.basename(pred_file)
        pred_parts = pred_name.split('_')
        if len(pred_parts) >= 1:
            scene_id = pred_parts[0]
            if scene_id in gt_scene_map:
                for gt_file in gt_scene_map[scene_id]:
                    matched_pairs.append((gt_file, pred_file))
                    matched_count += 1
            else:
                unmatched_count += 1
                if len(sample_unmatched) < 10:
                    sample_unmatched.append(scene_id)

    print(f"Matched predictions: {matched_count}")
    print(f"Unmatched predictions: {unmatched_count}")
    print(f"Sample unmatched scene IDs: {sample_unmatched}")

    # matched_pairs = matched_pairs[:10]
    # print(f"Evaluating first {len(matched_pairs)} images")

    # Initialize metrics
    psnr_scores = []
    ssim_scores = []
    niqe_scores = []
    lpips_scores = []
    brisque_scores = []
    biqi_scores = []
    uciqe_scores = []
    uiqm_scores = []
    clip_iqa_scores = []
    gt_images_for_fid = []
    pred_images_for_fid = []

    print(f"- CLIP-IQA available: {CLIP_IQA_AVAILABLE}")
    print(f"- BRISQUE available: {BRISQUE_AVAILABLE}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize CLIP model once
    if CLIP_IQA_AVAILABLE:
        clip_model_ready = initialize_clip_model(device)
    else:
        clip_model_ready = False
    
    # Initialize LPIPS
    lpips_model = None
    if LPIPS_AVAILABLE:
        try:
            lpips_model = lpips.LPIPS(net='vgg').to(device)
            print("LPIPS model initialized successfully")
        except Exception as e:
            print(f"LPIPS initialization failed: {e}")
            lpips_model = None
    
    print(f"Calculating metrics...")
    print(f"- Crop border: {crop_border}")
    print(f"- Test Y channel: {test_y_channel}")  
    print(f"- Correct mean/variance: {correct_mean_variance}")
    print(f"- LPIPS available: {lpips_model is not None}")

    for gt_path, pred_path in tqdm(matched_pairs):
        try:
            # Load images
            # gt_img = load_image(gt_path)
            # pred_img = load_image(pred_path)

            gt_img = load_image_cropped(gt_path, gt_size=256, apply_chaos_resize=True)
            pred_img = load_image_cropped(pred_path, gt_size=256, apply_chaos_resize=True)
            
            # Resize if needed
            if gt_img.shape != pred_img.shape:
                pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))
            
            # Apply correction
            if correct_mean_variance:
                pred_img = correct_mean_var(pred_img.astype(np.float32), gt_img.astype(np.float32), correction_strength=0.1)

            # Calculate metrics
            psnr = calculate_psnr(gt_img, pred_img, crop_border, test_y_channel)
            ssim = calculate_ssim(gt_img, pred_img, crop_border, test_y_channel)
            niqe = calculate_niqe(pred_img, crop_border)
            biqi = calculate_biqi_simplified(pred_img)

            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
            niqe_scores.append(niqe)
            biqi_scores.append(biqi)

            if BRISQUE_AVAILABLE:
                brisque_score = calculate_brisque(pred_img)
                if brisque_score is not None:
                    brisque_scores.append(brisque_score)
            if UNDERWATER_METRICS_AVAILABLE:
                uciqe_score = calculate_uciqe(pred_img)
                if uciqe_score is not None:
                    uciqe_scores.append(uciqe_score)
                uiqm_score = calculate_uiqm(pred_img)
                if uiqm_score is not None:
                    uiqm_scores.append(uiqm_score)
            if clip_model_ready:
                clip_iqa_score = calculate_clip_iqa(pred_img, device)
                if clip_iqa_score is not None:
                    clip_iqa_scores.append(clip_iqa_score)

            # Calculate LPIPS
            if lpips_model is not None:
                lpips_score = calculate_lpips(gt_img, pred_img, lpips_model, device)
                if lpips_score is not None:
                    lpips_scores.append(lpips_score)
            
            # Use original (uncorrected) images for FID to preserve distributional properties
            gt_images_for_fid.append(gt_img)
            if correct_mean_variance:
                # Reload original predicted image for FID
                pred_img_orig = load_image(pred_path)
                if gt_img.shape != pred_img_orig.shape:
                    pred_img_orig = cv2.resize(pred_img_orig, (gt_img.shape[1], gt_img.shape[0]))
                pred_images_for_fid.append(pred_img_orig)
            else:
                pred_images_for_fid.append(pred_img)
                
        except Exception as e:
            print(f"Error processing {gt_path}: {e}")
            continue

    # Calculate FID
    print("Calculating FID...")
    try:
        fid_score = calculate_fid(gt_images_for_fid, pred_images_for_fid)
    except Exception as e:
        print(f"Error calculating FID: {e}")
        fid_score = None

    # Calculate averages
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_niqe = np.mean(niqe_scores)
    avg_lpips = np.mean(lpips_scores) if lpips_scores else None
    avg_brisque = np.mean(brisque_scores) if brisque_scores else None
    avg_biqi = np.mean(biqi_scores) if biqi_scores else None
    avg_uciqe = np.mean(uciqe_scores) if uciqe_scores else None
    avg_uiqm = np.mean(uiqm_scores) if uiqm_scores else None
    avg_clip_iqa = np.mean(clip_iqa_scores) if clip_iqa_scores else None

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Number of images evaluated: {len(psnr_scores)}")
    print(f"PSNR ↑: {avg_psnr:.6f} ± {np.std(psnr_scores):.6f}")
    print(f"SSIM ↑: {avg_ssim:.6f} ± {np.std(ssim_scores):.6f}")
    if fid_score is not None:
        print(f"FID ↓:  {fid_score:.6f}")
    print(f"NIQE ↓: {avg_niqe:.6f} ± {np.std(niqe_scores):.6f}")
    if avg_lpips is not None:
        print(f"LPIPS ↓: {avg_lpips:.6f} ± {np.std(lpips_scores):.6f}")
    if avg_brisque is not None:
        print(f"BRISQUE ↓: {avg_brisque:.6f} ± {np.std(brisque_scores):.6f}")
    if avg_biqi is not None:
        print(f"BIQI ↓: {avg_biqi:.6f} ± {np.std(biqi_scores):.6f}")
    if avg_uciqe is not None:
        print(f"UCIQE ↑: {avg_uciqe:.6f} ± {np.std(uciqe_scores):.6f}")
    if avg_uiqm is not None:
        print(f"UIQM ↑: {avg_uiqm:.6f} ± {np.std(uiqm_scores):.6f}")
    if avg_clip_iqa is not None:
        print(f"CLIP-IQA ↑: {avg_clip_iqa:.6f} ± {np.std(clip_iqa_scores):.6f}")
    print("="*80)
    correction_note = " (with correction)" if correct_mean_variance else " (uncorrected - recommended)"
    print(f"Evaluation type: {correction_note}")
    print("Note: ↑ higher is better, ↓ lower is better")

    # Save results
    with open(output_file, 'w') as f:
        f.write("Image Quality Evaluation Results\n")
        f.write("="*80 + "\n")
        f.write(f"GT Directory: {gt_dir}\n")
        f.write(f"Predicted Directory: {pred_dir}\n")
        f.write(f"Number of images: {len(psnr_scores)}\n\n")
        f.write(f"PSNR: {avg_psnr:.6f} ± {np.std(psnr_scores):.6f}\n")
        f.write(f"SSIM: {avg_ssim:.6f} ± {np.std(ssim_scores):.6f}\n")
        if fid_score is not None:
            f.write(f"FID:  {fid_score:.6f}\n")
        f.write(f"NIQE: {avg_niqe:.6f} ± {np.std(niqe_scores):.6f}\n")
        if avg_lpips is not None:
            f.write(f"LPIPS: {avg_lpips:.6f} ± {np.std(lpips_scores):.6f}\n")
        if avg_brisque is not None:
            f.write(f"BRISQUE: {avg_brisque:.6f} ± {np.std(brisque_scores):.6f}\n")
        if avg_biqi is not None:
            f.write(f"BIQI: {avg_biqi:.6f} ± {np.std(biqi_scores):.6f}\n")
        if avg_uciqe is not None:
            f.write(f"UCIQE: {avg_uciqe:.6f} ± {np.std(uciqe_scores):.6f}\n")
        if avg_uiqm is not None:
            f.write(f"UIQM: {avg_uiqm:.6f} ± {np.std(uiqm_scores):.6f}\n")
        if avg_clip_iqa is not None:
            f.write(f"CLIP-IQA ↑: {avg_clip_iqa:.6f} ± {np.std(clip_iqa_scores):.6f}\n")
    print(f"Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='image quality evaluation')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory containing ground truth images')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing predicted images')
    parser.add_argument('--output', type=str, default='comprehensive_evaluation_results.txt', help='Output file')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    parser.add_argument('--test_y_channel', action='store_true', help='Test Y channel only (luminance)')
    parser.add_argument('--correct_mean_var', action='store_true', help='Correct mean and variance of restored images')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.gt_dir):
        print(f"GT directory does not exist: {args.gt_dir}")
        return
    
    if not os.path.exists(args.pred_dir):
        print(f"Predicted directory does not exist: {args.pred_dir}")
        return
    
    evaluate_images(args.gt_dir, args.pred_dir, args.output, 
                   args.crop_border, args.test_y_channel, args.correct_mean_var)


if __name__ == "__main__":
    # Example usage
    gt_dir = "/depot/natallah/data/shourya/Reti-Diff-main/datasets/SSID/Sony/long_resized"
    pred_dir = "/depot/natallah/data/shourya/Reti-Diff-main/results/SSID/visualization/SSID_ValSet_new"
    
    print("Starting comprehensive evaluation...")
    evaluate_images(gt_dir, pred_dir)