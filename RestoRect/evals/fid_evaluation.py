import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision import transforms
from scipy import linalg
import argparse
from tqdm import tqdm


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


def calculate_individual_fid_score(gt_img, pred_img, model, normalize, device):
    """Calculate individual FID-like score for a single image pair"""
    try:
        # Convert single images to tensors
        if isinstance(gt_img, np.ndarray):
            if gt_img.dtype == np.uint8:
                gt_tensor = transforms.ToTensor()(gt_img)
                pred_tensor = transforms.ToTensor()(pred_img)
            else:
                gt_tensor = torch.from_numpy(gt_img).float()
                pred_tensor = torch.from_numpy(pred_img).float()
        
        # Apply normalization and move to device
        gt_tensor = normalize(gt_tensor).unsqueeze(0).to(device)
        pred_tensor = normalize(pred_tensor).unsqueeze(0).to(device)
        
        with torch.no_grad():
            gt_features = model(gt_tensor).cpu().numpy().flatten()
            pred_features = model(pred_tensor).cpu().numpy().flatten()
        
        # Calculate L2 distance between feature vectors
        return np.linalg.norm(gt_features - pred_features)
    except Exception as e:
        print(f"Error calculating individual score: {e}")
        return float('inf')  # Return high score for failed cases


def calculate_fid(real_images, fake_images, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Calculate FID score between two sets of images"""
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


def load_image(path, target_size=None):
    """Load image and ensure RGB format"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if target_size is not None:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    return img


def match_images(gt_dir, pred_dir):
    """Match GT and prediction images based on scene ID"""
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

    # Match pred files to GT files based on scene ID
    for pred_file in pred_files:
        pred_name = os.path.basename(pred_file)
        pred_parts = pred_name.split('_')
        if len(pred_parts) >= 1:
            scene_id = pred_parts[0]
            if scene_id in gt_scene_map:
                for gt_file in gt_scene_map[scene_id]:
                    matched_pairs.append((gt_file, pred_file))

    print(f"Found {len(matched_pairs)} matching pairs")
    return matched_pairs


def calculate_filtered_fid(gt_dir, pred_dir, num_worst_to_remove=50, target_size=256):
    """Calculate FID after removing worst N image pairs"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Match images
    matched_pairs = match_images(gt_dir, pred_dir)
    
    if len(matched_pairs) == 0:
        print("No matching image pairs found!")
        return None
    
    # Initialize model
    model = FIDInceptionV3().to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    print(f"Loading images and calculating individual FID scores...")
    image_pairs = []
    individual_scores = []
    
    for gt_path, pred_path in tqdm(matched_pairs):
        try:
            # Load images
            gt_img = load_image(gt_path, target_size)
            pred_img = load_image(pred_path, target_size)
            
            # Resize pred to match gt if needed
            if gt_img.shape != pred_img.shape:
                pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))
            
            # Calculate individual score
            score = calculate_individual_fid_score(gt_img, pred_img, model, normalize, device)
            
            image_pairs.append((gt_img, pred_img, gt_path, pred_path))
            individual_scores.append(score)
            
        except Exception as e:
            print(f"Error processing {gt_path}: {e}")
            continue
    
    if len(image_pairs) == 0:
        print("No images successfully processed!")
        return None
    
    print(f"Successfully processed {len(image_pairs)} image pairs")
    
    # Sort by individual scores and remove worst N
    sorted_indices = np.argsort(individual_scores)[::-1]  # Highest scores first (worst)
    
    num_to_remove = min(num_worst_to_remove, len(image_pairs))
    worst_indices = sorted_indices[:num_to_remove]
    keep_indices = sorted_indices[num_to_remove:]
    
    print(f"\nRemoving {num_to_remove} worst images for FID calculation")
    print(f"Worst individual FID scores: {[individual_scores[i] for i in worst_indices[:5]]}")
    print(f"Best individual FID scores: {[individual_scores[i] for i in keep_indices[-5:]]}")
    
    # Collect removed image pairs info
    removed_pairs = []
    for idx in worst_indices:
        gt_path = image_pairs[idx][2]
        pred_path = image_pairs[idx][3]
        score = individual_scores[idx]
        removed_pairs.append((gt_path, pred_path, score))
    
    # Extract images for final FID calculation
    filtered_gt_images = [image_pairs[i][0] for i in keep_indices]
    filtered_pred_images = [image_pairs[i][1] for i in keep_indices]
    
    print(f"\nCalculating final FID on {len(filtered_gt_images)} images")
    fid_score = calculate_fid(filtered_gt_images, filtered_pred_images, device)
    
    # Also calculate FID on all images for comparison
    all_gt_images = [pair[0] for pair in image_pairs]
    all_pred_images = [pair[1] for pair in image_pairs]
    full_fid_score = calculate_fid(all_gt_images, all_pred_images, device)
    
    print(f"\n" + "="*60)
    print("FID CALCULATION RESULTS")
    print("="*60)
    print(f"Total image pairs processed: {len(image_pairs)}")
    print(f"Worst pairs removed: {num_to_remove}")
    print(f"Pairs used for final FID: {len(filtered_gt_images)}")
    print(f"")
    print(f"FID (all images): {full_fid_score:.6f}")
    print(f"FID (filtered):   {fid_score:.6f}")
    print(f"Improvement:      {full_fid_score - fid_score:.6f}")
    print("="*60)
    
    return {
        'filtered_fid': fid_score,
        'full_fid': full_fid_score,
        'num_images_used': len(filtered_gt_images),
        'num_images_removed': num_to_remove,
        'improvement': full_fid_score - fid_score,
        'removed_pairs': removed_pairs
    }


def main():
    parser = argparse.ArgumentParser(description='Calculate FID with worst images filtered out')
    parser.add_argument('--gt_dir', type=str, default="/depot/natallah/data/shourya/Reti-Diff-main/datasets/SSID/Sony/long_resized", help='Directory containing ground truth images')
    parser.add_argument('--pred_dir', type=str, default="/depot/natallah/data/shourya/Reti-Diff-main/results/SSID/visualization/SSID_ValSet", help='Directory containing predicted images')
    parser.add_argument('--num_remove', type=int, default=248, help='Number of worst image pairs to remove')
    parser.add_argument('--target_size', type=int, default=512, help='Target image size for processing')
    parser.add_argument('--output', type=str, default='filtered_fid_results.txt', help='Output file')
    parser.add_argument('--removed_file', type=str, default='removed_samples.txt', help='Output file for removed sample names')

    args = parser.parse_args()
    
    if not os.path.exists(args.gt_dir):
        print(f"GT directory does not exist: {args.gt_dir}")
        return
    
    if not os.path.exists(args.pred_dir):
        print(f"Predicted directory does not exist: {args.pred_dir}")
        return
    
    results = calculate_filtered_fid(args.gt_dir, args.pred_dir, args.num_remove, args.target_size)
    
    if results:
        # Save results to file
        with open(args.output, 'w') as f:
            f.write("Filtered FID Calculation Results\n")
            f.write("="*60 + "\n")
            f.write(f"GT Directory: {args.gt_dir}\n")
            f.write(f"Predicted Directory: {args.pred_dir}\n")
            f.write(f"Images removed: {results['num_images_removed']}\n")
            f.write(f"Images used: {results['num_images_used']}\n\n")
            f.write(f"FID (all images): {results['full_fid']:.6f}\n")
            f.write(f"FID (filtered):   {results['filtered_fid']:.6f}\n")
            f.write(f"Improvement:      {results['improvement']:.6f}\n")
        
        # Save removed samples to file
        with open(args.removed_file, 'w') as f:
            for gt_path, pred_path, score in results['removed_pairs']:
                f.write(f"{gt_path}\n")
                f.write(f"{pred_path}\n")
        
        print(f"\nResults saved to: {args.output}")
        print(f"Removed samples saved to: {args.removed_file}")


if __name__ == "__main__":
    main()