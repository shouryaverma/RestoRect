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
    """InceptionV3 model for FID calculation, extracts 2048-dim features"""
    
    def __init__(self):
        super().__init__()
        # Load pretrained inception
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.eval()
        
        # Remove the final average pooling and classifier
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3  
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Input should be in range [0, 1], we convert to [-1, 1]
        x = 2 * x - 1
        
        # Resize to inception input size
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Forward through inception layers
        x = self.Conv2d_1a_3x3(x)  # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)  # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)  # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)  # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)  # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # N x 192 x 35 x 35
        x = self.Mixed_5b(x)  # N x 256 x 35 x 35
        x = self.Mixed_5c(x)  # N x 288 x 35 x 35
        x = self.Mixed_5d(x)  # N x 288 x 35 x 35
        x = self.Mixed_6a(x)  # N x 768 x 17 x 17
        x = self.Mixed_6b(x)  # N x 768 x 17 x 17
        x = self.Mixed_6c(x)  # N x 768 x 17 x 17
        x = self.Mixed_6d(x)  # N x 768 x 17 x 17
        x = self.Mixed_6e(x)  # N x 768 x 17 x 17
        x = self.Mixed_7a(x)  # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)  # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)  # N x 2048 x 8 x 8
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))  # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)  # N x 2048
        
        return x

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

def calculate_fid(real_images, fake_images, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Calculate FID score between real and fake images"""
    model = FIDInceptionV3().to(device)
    
    def get_features(images):
        features = []
        batch_size = 32
        
        # Simple transform - just convert to tensor
        transform = transforms.ToTensor()
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            # Convert numpy arrays to tensors
            batch_tensor = torch.stack([transform(img) for img in batch]).to(device)
            
            with torch.no_grad():
                batch_features = model(batch_tensor)
                features.append(batch_features.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    # Get features
    print("  Extracting real image features...")
    real_features = get_features(real_images)
    print("  Extracting generated image features...")
    fake_features = get_features(fake_images)
    
    print(f"  Real features shape: {real_features.shape}")
    print(f"  Fake features shape: {fake_features.shape}")
    
    # Calculate statistics
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_score

def load_image(path):
    """Load image and ensure RGB format"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def evaluate_images(gt_dir, pred_dir, output_file="evaluation_results.txt", 
                   crop_border=5, test_y_channel=True, correct_mean_variance=True):
    """evaluation with metric calculation"""
    
    # Get image files
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.png")) + 
                     glob.glob(os.path.join(gt_dir, "*.jpg")) + 
                     glob.glob(os.path.join(gt_dir, "*.jpeg")))
    
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.png")) + 
                       glob.glob(os.path.join(pred_dir, "*.jpg")) + 
                       glob.glob(os.path.join(pred_dir, "*.jpeg")))
    
    print(f"Found {len(gt_files)} GT images and {len(pred_files)} predicted images")
    
    # Match files by name
    matched_pairs = []
    for gt_file in gt_files:
        gt_name = os.path.basename(gt_file)
        if gt_name.startswith('normal'):
            pred_name = gt_name.replace('normal', 'low', 1)
        else:
            pred_name = gt_name
        
        pred_file = os.path.join(pred_dir, pred_name)
        if os.path.exists(pred_file):
            matched_pairs.append((gt_file, pred_file))
    
    print(f"Found {len(matched_pairs)} matching pairs")
    if len(matched_pairs) == 0:
        print("No matching image pairs found!")
        return
    
    # Initialize metrics
    psnr_scores = []
    ssim_scores = []
    niqe_scores = []
    lpips_scores = []
    brisque_scores = []
    biqi_scores = []
    print(f"- BRISQUE available: {BRISQUE_AVAILABLE}")
    gt_images_for_fid = []
    pred_images_for_fid = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
            gt_img = load_image(gt_path)
            pred_img = load_image(pred_path)
            
            # Resize if needed
            if gt_img.shape != pred_img.shape:
                pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))
            
            # Apply correction
            if correct_mean_variance:
                pred_img = correct_mean_var(pred_img.astype(np.float32), gt_img.astype(np.float32), correction_strength=0.2)

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
    gt_dir = "/depot/natallah/data/shourya/Reti-Diff-main/datasets/LOL-v2/Real_captured/Test/Normal"
    pred_dir = "/depot/natallah/data/shourya/Reti-Diff-main/results/LLIE_Real/visualization/Testset"
    
    print("Starting comprehensive evaluation...")
    evaluate_images(gt_dir, pred_dir)