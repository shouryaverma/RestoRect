import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register() 
class FlexLoss(nn.Module):
    """
    FLEX Loss with LPL-inspired optimizations:
    - Resolution-aware layer weighting (no more manual weights)
    - Fast percentile-based outlier detection (no convolutions)
    - Cross-normalization
    - Reduced memory overhead
    """
    
    def __init__(self, 
                 target_layers=None,
                 layer_weights=None,
                 loss_weight=0.15,
                 snr_threshold=0.4,
                 outlier_percentile=95.0,
                 apply_cross_norm=True,
                 apply_outlier_detection=True):
        super().__init__()

        if target_layers is None:
            target_layers = [
                'decoder_level3',  
                'decoder_level2',   
                'decoder_level1',   
            ]
        
        self.target_layers = target_layers
        self.loss_weight = loss_weight
        self.snr_threshold = snr_threshold
        self.apply_cross_norm = apply_cross_norm
        self.apply_outlier_detection = apply_outlier_detection
        self.outlier_percentile = outlier_percentile
        
        # Resolution-aware weighting
        if layer_weights is None:
            # Auto-compute weights based on expected resolution scaling
            # Assuming typical U-Net structure: [1/4, 1/2, 1, 1]
            base_resolutions = [4, 2, 1, 1]  # Relative to final output
            # Weight inversely to resolution (higher res = lower weight)
            self.layer_weights = torch.tensor([1.0 / (r ** 0.5) for r in base_resolutions])
        else:
            self.layer_weights = torch.tensor(layer_weights)
            
        self.layer_weights = nn.Parameter(self.layer_weights, requires_grad=False)

        print(f"FLEX: {len(target_layers)} layers, resolution-aware weights, fast outlier detection")
    
    def cross_normalize(self, teacher_feat, student_feat):
        """
        Cross-normalization using student statistics
        More stable than original implementation
        """
        if student_feat.dim() == 4:
            spatial_dims = [2, 3]
        elif student_feat.dim() == 3:
            spatial_dims = [2]
        elif student_feat.dim() == 2:
            spatial_dims = [1]
        else:
            spatial_dims = list(range(1, student_feat.dim()))
        
        # Use student feature statistics for both (core FLEX innovation)
        student_mean = student_feat.mean(dim=spatial_dims, keepdim=True)
        student_std = student_feat.std(dim=spatial_dims, keepdim=True, unbiased=False) + 1e-8
        
        # More stable normalization
        teacher_norm = (teacher_feat - student_mean) / student_std
        student_norm = (student_feat - student_mean) / student_std
        
        return teacher_norm, student_norm
    
    def fast_outlier_detection(self, features):
        """
        Percentile-based outlier detection
        Uses channel-wise percentiles
        """
        if features.dim() != 4:
            return torch.ones_like(features)
        
        B, C, H, W = features.shape
        
        # Reshape for channel-wise percentile computation
        feat_reshaped = features.view(B, C, -1)
        
        # Compute per-channel percentile thresholds (much faster than convolutions)
        percentile_vals = torch.quantile(feat_reshaped.abs(), 
                                       self.outlier_percentile / 100.0, 
                                       dim=2, keepdim=True)
        
        # Create mask: 1 for normal values, 0 for outliers
        percentile_vals = percentile_vals.unsqueeze(-1)  # [B, C, 1, 1]
        outlier_mask = (features.abs() <= percentile_vals).float()
        
        return outlier_mask
    
    def compute_resolution_weight(self, feat_shape):
        """
        Dynamically compute resolution-based weight
        Higher resolution features get lower weight
        """
        if len(feat_shape) >= 4:
            h, w = feat_shape[-2:]
            resolution_factor = h * w
            # Normalize by some baseline (e.g., 64x64)
            baseline_res = 64 * 64
            weight = (baseline_res / resolution_factor) ** 0.25
            return max(weight, 0.1)  # Clamp to avoid extremely small weights
        return 1.0
    
    def compute_flex_loss(self, teacher_feat, student_feat, layer_idx):
        """
        FLEX loss computation with automatic resolution weighting
        """
        # 1. Cross-normalization (core innovation)
        if self.apply_cross_norm:
            teacher_norm, student_norm = self.cross_normalize(teacher_feat, student_feat)
        else:
            teacher_norm, student_norm = teacher_feat, student_feat
        
        # 2. Fast outlier detection (only for 4D spatial features)
        if self.apply_outlier_detection and student_norm.dim() == 4:
            reliable_mask = self.fast_outlier_detection(student_norm)
        else:
            reliable_mask = torch.ones_like(student_norm)
        
        # 3. Dynamic resolution-based weighting
        resolution_weight = self.compute_resolution_weight(student_norm.shape)
        
        # 4. Get predefined layer weight
        if layer_idx < len(self.layer_weights):
            layer_weight = self.layer_weights[layer_idx].item()
        else:
            layer_weight = 1.0
            
        # Combine weights
        total_weight = layer_weight * resolution_weight
        
        # 5. Compute masked loss - fully vectorized
        diff = teacher_norm - student_norm
        masked_diff = diff * reliable_mask
        
        # Normalize by number of valid (non-outlier) elements
        loss_numerator = torch.sum(masked_diff ** 2)
        loss_denominator = torch.sum(reliable_mask) + 1e-8
        
        layer_loss = loss_numerator / loss_denominator
        weighted_loss = total_weight * layer_loss
        
        return weighted_loss
    
    def should_apply_loss(self, timestep=None, max_timestep=None):
        """SNR-aware loss application"""
        if timestep is None:
            return True
            
        if max_timestep is not None and max_timestep > 1:
            t_norm = timestep / max_timestep
        else:
            t_norm = timestep
            
        return t_norm < self.snr_threshold
    
    def forward(self, teacher_features, student_features, timestep=None, max_timestep=None):
        """
        FLEX loss computation with streaming approach
        """
        # SNR threshold check
        if not self.should_apply_loss(timestep, max_timestep):
            if isinstance(teacher_features, (list, tuple)):
                device = teacher_features[0].device
            else:
                device = next(iter(teacher_features.values())).device
            return torch.tensor(0.0, device=device, dtype=torch.float32)
        
        total_loss = 0.0
        valid_layers = 0
        
        # Handle list input (most common case) - process in streaming fashion
        if isinstance(teacher_features, (list, tuple)) and isinstance(student_features, (list, tuple)):
            n_layers = min(len(teacher_features), len(student_features))
            
            # Stream processing to save memory
            for i in range(n_layers):
                teacher_feat = teacher_features[i]
                student_feat = student_features[i] 
                
                # Shape compatibility check
                if teacher_feat.shape != student_feat.shape:
                    continue
                
                # Compute layer loss (automatically handles weighting)
                layer_loss = self.compute_flex_loss(teacher_feat, student_feat, i)
                total_loss = total_loss + layer_loss  # Avoid +=, create new tensor
                valid_layers += 1
                
                # Clear references immediately to save memory
                del teacher_feat, student_feat, layer_loss
        
        # Handle dict input (less common)
        elif isinstance(teacher_features, dict) and isinstance(student_features, dict):
            for i, layer_name in enumerate(self.target_layers):
                if layer_name not in teacher_features or layer_name not in student_features:
                    continue
                    
                teacher_feat = teacher_features[layer_name]
                student_feat = student_features[layer_name]
                
                if teacher_feat.shape != student_feat.shape:
                    continue
                
                layer_loss = self.compute_flex_loss(teacher_feat, student_feat, i)
                total_loss = total_loss + layer_loss
                valid_layers += 1
        
        # Final loss computation
        if valid_layers > 0:
            avg_loss = total_loss / valid_layers
            final_loss = self.loss_weight * avg_loss
        else:
            if isinstance(teacher_features, (list, tuple)):
                device = teacher_features[0].device
            else:
                device = next(iter(teacher_features.values())).device
            final_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            
        return final_loss