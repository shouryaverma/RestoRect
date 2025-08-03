import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from scipy import ndimage
from basicsr.utils.registry import LOSS_REGISTRY


class FeatureExtractor:
    """Hook-based feature extractor for RGFormer transformer levels"""
    
    def __init__(self, model, target_layers):
        """
        Args:
            model: RGFormer model
            target_layers: List of layer names to extract features from
                          e.g., ['decoder_level3', 'decoder_level2', 'decoder_level1', 'img_refinement']
        """
        self.model = model
        self.target_layers = target_layers
        self.features = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on target layers"""
        
        def get_hook_fn(layer_name):
            def hook_fn(module, input, output):
                # For transformer blocks that return [x, k_v], we want x
                if isinstance(output, list) and len(output) == 2:
                    self.features[layer_name] = output[0]  # x is the feature tensor
                else:
                    self.features[layer_name] = output
            return hook_fn
        
        # Register hooks on the specified layers
        for layer_name in self.target_layers:
            if hasattr(self.model, layer_name):
                layer = getattr(self.model, layer_name)
                hook = layer.register_forward_hook(get_hook_fn(layer_name))
                self.hooks.append(hook)
            else:
                print(f"Warning: Layer {layer_name} not found in model")
    
    def extract_features(self, *args, **kwargs):
        """Run forward pass and extract features"""
        self.features.clear()
        _ = self.model(*args, **kwargs)
        return self.features.copy()
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class FeatureNormalizer:
    """Cross-normalization and outlier detection for LPL"""
    
    def __init__(self, outlier_threshold=2.0, morphology_kernel_size=3):
        """
        Args:
            outlier_threshold: Threshold for outlier detection (in standard deviations)
            morphology_kernel_size: Kernel size for morphological operations
        """
        self.outlier_threshold = outlier_threshold
        self.kernel_size = morphology_kernel_size
    
    def cross_normalize(self, clean_features, pred_features):
        """
        Cross-normalize features using statistics from predicted features
        
        Args:
            clean_features: Clean/teacher features (B, C, H, W)
            pred_features: Predicted/student features (B, C, H, W)
            
        Returns:
            normalized_clean, normalized_pred: Cross-normalized features
        """
        # Compute statistics from predicted features
        pred_mean = pred_features.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
        pred_std = pred_features.std(dim=[2, 3], keepdim=True) + 1e-8  # (B, C, 1, 1)
        
        # Normalize both using predicted statistics
        normalized_clean = (clean_features - pred_mean) / pred_std
        normalized_pred = (pred_features - pred_mean) / pred_std
        
        return normalized_clean, normalized_pred
    
    def detect_outliers(self, features):
        """
        Detect outliers in feature maps using morphological operations
        
        Args:
            features: Feature tensor (B, C, H, W)
            
        Returns:
            outlier_mask: Binary mask (B, C, H, W) where 1 = keep, 0 = outlier
        """
        B, C, H, W = features.shape
        device = features.device
        
        # Initialize mask
        outlier_mask = torch.ones_like(features)
        
        # Process each sample and channel
        for b in range(B):
            for c in range(C):
                feature_map = features[b, c].detach().cpu().numpy()
                
                # Compute local statistics using morphological operations
                kernel = np.ones((self.kernel_size, self.kernel_size))
                
                # Local mean and std
                local_mean = ndimage.uniform_filter(feature_map, size=self.kernel_size)
                local_var = ndimage.uniform_filter(feature_map**2, size=self.kernel_size) - local_mean**2
                local_std = np.sqrt(np.maximum(local_var, 1e-8))
                
                # Detect outliers
                z_scores = np.abs(feature_map - local_mean) / (local_std + 1e-8)
                outliers = z_scores > self.outlier_threshold
                
                # Apply morphological closing to fill small gaps
                outliers = ndimage.binary_closing(outliers, structure=kernel)
                
                # Convert to keep mask (1 = keep, 0 = remove)
                keep_mask = ~outliers
                outlier_mask[b, c] = torch.from_numpy(keep_mask.astype(np.float32)).to(device)
        
        return outlier_mask


@LOSS_REGISTRY.register()
class LatentPerceptualLoss(nn.Module):
    """
    Latent Perceptual Loss for RetiDiff transformer features
    
    Based on "Perceptual Losses Improve Diffusion-based Image Quality" (ICLR 2025)
    Adapted for transformer decoder features instead of autoencoder decoder features.
    """
    
    def __init__(self, 
                 target_layers=None,
                 layer_weights=None,
                 loss_weight=1.0,
                 snr_threshold=0.3,
                 outlier_threshold=2.0,
                 kernel_size=3,
                 apply_cross_norm=True,
                 apply_outlier_detection=True):
        """
        Args:
            target_layers: List of RGFormer layer names to extract features from
            layer_weights: Weights for each layer (auto-computed if None)
            loss_weight: Overall loss weight
            snr_threshold: Only apply loss when t < this threshold (for RF: low t = high SNR)
            outlier_threshold: Threshold for outlier detection
            kernel_size: Kernel size for morphological operations
            apply_cross_norm: Whether to apply cross-normalization
            apply_outlier_detection: Whether to apply outlier detection
        """
        super().__init__()
        
        # Default layers for RGFormer
        if target_layers is None:
            target_layers = [
                'decoder_level3',  # 192 channels, 1/4 resolution
                'decoder_level2',  # 96 channels, 1/2 resolution  
                'decoder_level1',  # 96 channels, full resolution
                'img_refinement'   # 96 channels, full resolution
            ]
        
        self.target_layers = target_layers
        self.loss_weight = loss_weight
        self.snr_threshold = snr_threshold
        self.apply_cross_norm = apply_cross_norm
        self.apply_outlier_detection = apply_outlier_detection
        
        # Feature normalizer
        self.normalizer = FeatureNormalizer(
            outlier_threshold=outlier_threshold,
            morphology_kernel_size=kernel_size
        )
        
        # Layer weights (depth-specific weighting from paper)
        if layer_weights is None:
            # Auto-compute weights based on typical RGFormer resolutions
            # Assuming resolutions: [1/4, 1/2, 1, 1] relative to input
            base_resolutions = [0.25, 0.5, 1.0, 1.0]  # Relative resolutions
            self.layer_weights = self._compute_depth_weights(base_resolutions)
        else:
            self.layer_weights = layer_weights
            
        print(f"LPL initialized with layers: {target_layers}")
        print(f"Layer weights: {self.layer_weights}")
    
    def _compute_depth_weights(self, resolutions):
        """Compute depth-specific weights: ωl = 2^(-rl/r1)"""
        r1 = resolutions[0]  # Base resolution
        weights = [2**(-r/r1) for r in resolutions]
        return weights
    
    def extract_multi_scale_features(self, model, *args, **kwargs):
        """Extract features from multiple transformer decoder levels"""
        extractor = FeatureExtractor(model, self.target_layers)
        
        try:
            features = extractor.extract_features(*args, **kwargs)
        finally:
            extractor.remove_hooks()
            
        return features
    
    def compute_layer_loss(self, clean_feat, pred_feat, layer_idx):
        """
        Compute LPL loss for a single layer
        
        Args:
            clean_feat: Clean/teacher features (B, C, H, W)
            pred_feat: Predicted/student features (B, C, H, W)
            layer_idx: Layer index for weighting
            
        Returns:
            layer_loss: Weighted loss for this layer
        """
        # Cross-normalization
        if self.apply_cross_norm:
            clean_norm, pred_norm = self.normalizer.cross_normalize(clean_feat, pred_feat)
        else:
            clean_norm, pred_norm = clean_feat, pred_feat
        
        # Outlier detection
        if self.apply_outlier_detection:
            outlier_mask = self.normalizer.detect_outliers(pred_norm)
        else:
            outlier_mask = torch.ones_like(pred_norm)
        
        # Compute per-channel loss
        B, C, H, W = clean_norm.shape
        layer_loss = 0.0
        
        for c in range(C):
            # Get features for this channel
            clean_c = clean_norm[:, c:c+1, :, :]  # (B, 1, H, W)
            pred_c = pred_norm[:, c:c+1, :, :]    # (B, 1, H, W)
            mask_c = outlier_mask[:, c:c+1, :, :] # (B, 1, H, W)
            
            # Compute masked L2 loss
            diff = (clean_c - pred_c) * mask_c
            channel_loss = torch.mean(diff ** 2)
            layer_loss += channel_loss
        
        # Average over channels and apply layer weight
        layer_loss = layer_loss / C
        weighted_loss = self.layer_weights[layer_idx] * layer_loss
        
        return weighted_loss
    
    def should_apply_loss(self, timestep=None, max_timestep=None):
        """
        Determine if LPL should be applied based on SNR threshold
        
        For Rectified Flow: t ∈ [0, 1], where t=0 is clean, t=1 is noise
        Apply loss when t < threshold (high SNR, close to clean data)
        
        Args:
            timestep: Current timestep (for RF: 0-1 range)
            max_timestep: Maximum timestep (for compatibility)
            
        Returns:
            should_apply: Boolean indicating whether to apply loss
        """
        if timestep is None:
            return True  # Always apply if no timestep info
            
        # For Rectified Flow, normalize timestep to [0, 1] if needed
        if max_timestep is not None and max_timestep > 1:
            t_normalized = timestep / max_timestep
        else:
            t_normalized = timestep
            
        # Apply loss only at high SNR (low t values for RF)
        return t_normalized < self.snr_threshold
    
    def forward(self, clean_features, pred_features, timestep=None, max_timestep=None):
        """
        Compute Latent Perceptual Loss
        
        Args:
            clean_features: Dict of clean/teacher features {layer_name: tensor}
            pred_features: Dict of predicted/student features {layer_name: tensor}
            timestep: Current timestep (optional, for SNR thresholding)
            max_timestep: Maximum timestep (optional)
            
        Returns:
            lpl_loss: Computed latent perceptual loss
        """
        # Check SNR threshold
        if not self.should_apply_loss(timestep, max_timestep):
            return torch.tensor(0.0, device=next(iter(clean_features.values())).device)
        
        total_loss = 0.0
        num_valid_layers = 0
        
        # Compute loss for each layer
        for layer_idx, layer_name in enumerate(self.target_layers):
            if layer_name in clean_features and layer_name in pred_features:
                clean_feat = clean_features[layer_name]
                pred_feat = pred_features[layer_name]
                
                # Ensure same shape
                if clean_feat.shape != pred_feat.shape:
                    print(f"Warning: Shape mismatch for {layer_name}: "
                          f"clean {clean_feat.shape} vs pred {pred_feat.shape}")
                    continue
                
                layer_loss = self.compute_layer_loss(clean_feat, pred_feat, layer_idx)
                total_loss += layer_loss
                num_valid_layers += 1
        
        # Average over valid layers and apply overall weight
        if num_valid_layers > 0:
            avg_loss = total_loss / num_valid_layers
            final_loss = self.loss_weight * avg_loss
        else:
            final_loss = torch.tensor(0.0, device=next(iter(clean_features.values())).device)
            
        return final_loss


class LPLIntegrator:
    """Helper class to integrate LPL into RetiDiff training"""
    
    def __init__(self, lpl_loss, transformer_model):
        """
        Args:
            lpl_loss: LatentPerceptualLoss instance
            transformer_model: RGFormer model
        """
        self.lpl_loss = lpl_loss
        self.transformer = transformer_model
    
    def compute_lpl_for_training(self, clean_inputs, pred_inputs, timestep=None, max_timestep=None):
        """
        Compute LPL loss during training
        
        Args:
            clean_inputs: Inputs for clean/teacher forward pass (img, k_v_clean, k_v_i_clean)
            pred_inputs: Inputs for predicted/student forward pass (img, k_v_pred, k_v_i_pred)
            timestep: Current timestep
            max_timestep: Maximum timestep
            
        Returns:
            lpl_loss: Computed LPL loss
        """
        # Extract features from clean (teacher) forward pass
        clean_features = self.lpl_loss.extract_multi_scale_features(
            self.transformer, *clean_inputs
        )
        
        # Extract features from predicted (student) forward pass
        pred_features = self.lpl_loss.extract_multi_scale_features(
            self.transformer, *pred_inputs
        )
        
        # Compute LPL loss
        lpl_loss = self.lpl_loss(clean_features, pred_features, timestep, max_timestep)
        
        return lpl_loss
    
    def compute_lpl_with_caching(self, clean_inputs, pred_inputs, timestep=None, max_timestep=None):
        """
        Compute LPL with feature caching to avoid double forward passes
        
        This is more efficient when you need both the final output and LPL loss.
        """
        # First forward pass: clean features
        clean_features = self.lpl_loss.extract_multi_scale_features(
            self.transformer, *clean_inputs
        )
        
        # Second forward pass: predicted features  
        pred_features = self.lpl_loss.extract_multi_scale_features(
            self.transformer, *pred_inputs
        )
        
        # Compute LPL
        lpl_loss = self.lpl_loss(clean_features, pred_features, timestep, max_timestep)
        
        # Return both features and loss for potential reuse
        return {
            'lpl_loss': lpl_loss,
            'clean_features': clean_features,
            'pred_features': pred_features
        }


# Factory function for easy instantiation
def create_retidiff_lpl(loss_weight=0.1, snr_threshold=0.3, **kwargs):
    """
    Factory function to create LPL for RetiDiff
    
    Args:
        loss_weight: Overall LPL loss weight
        snr_threshold: SNR threshold for applying loss
        **kwargs: Additional arguments for LatentPerceptualLoss
        
    Returns:
        LatentPerceptualLoss instance configured for RetiDiff
    """
    return LatentPerceptualLoss(
        loss_weight=loss_weight,
        snr_threshold=snr_threshold,
        **kwargs
    )