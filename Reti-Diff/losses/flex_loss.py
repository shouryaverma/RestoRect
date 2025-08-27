import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY

class GPUOutlierDetector(nn.Module):
    """GPU-native outlier detection using local statistics"""
    def __init__(self, kernel_size=3, threshold=2.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        
        # Create kernel for local mean computation
        self.register_buffer('kernel', torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2))
        
    def forward(self, features):
        """
        Detect outliers using GPU-native operations
        Args:
            features: (B, C, H, W) tensor
        Returns:
            mask: (B, C, H, W) binary mask where 1=keep, 0=outlier
        """
        B, C, H, W = features.shape
        device = features.device
        
        # Reshape for group convolution: (B*C, 1, H, W)
        feat_reshaped = features.view(B * C, 1, H, W)
        
        # Compute local mean using convolution
        pad_size = self.kernel_size // 2
        local_mean = F.conv2d(feat_reshaped, self.kernel, padding=pad_size)
        
        # Compute local variance
        feat_squared = feat_reshaped ** 2
        local_mean_sq = F.conv2d(feat_squared, self.kernel, padding=pad_size)
        local_var = local_mean_sq - local_mean ** 2
        local_std = torch.sqrt(torch.clamp(local_var, min=1e-8))
        
        # Compute z-scores
        z_scores = torch.abs(feat_reshaped - local_mean) / (local_std + 1e-8)
        
        # Create outlier mask (1=keep, 0=outlier)
        mask = (z_scores <= self.threshold).float()
        
        # Reshape back to original shape
        mask = mask.view(B, C, H, W)
        
        # Apply morphological closing using max pooling + min pooling
        # This fills small gaps in the mask
        mask = self.morphological_closing(mask)
        
        return mask
    
    def morphological_closing(self, mask):
        """GPU-native morphological closing"""
        # Dilation (max pooling with negative values)
        dilated = -F.max_pool2d(-mask, kernel_size=self.kernel_size, stride=1, 
                               padding=self.kernel_size//2)
        
        # Erosion (min pooling)
        closed = F.max_pool2d(dilated, kernel_size=self.kernel_size, stride=1, 
                             padding=self.kernel_size//2)
        
        return closed

@LOSS_REGISTRY.register() 
class FlexLoss(nn.Module):
    """
    Efficient FLEX Loss maintaining core innovations:
    - Cross-normalization with predicted statistics  
    - Multi-scale transformer feature matching
    - SNR-aware application
    - GPU-native outlier detection
    """
    
    def __init__(self, 
                 target_layers=None,
                 layer_weights=None,
                 loss_weight=0.15,
                 snr_threshold=0.4,
                 outlier_threshold=2.5,
                 kernel_size=3,
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
        
        # GPU-native outlier detector
        if apply_outlier_detection:
            self.outlier_detector = GPUOutlierDetector(kernel_size, outlier_threshold)
        
        # Depth-based layer weights: deeper layers get higher weight
        if layer_weights is None:
            # Assuming resolutions: [1/4, 1/2, 1, 1] - weight by importance
            self.layer_weights = torch.tensor([2.0, 1.5, 1.0, 0.8])  
        else:
            self.layer_weights = torch.tensor(layer_weights)
            
        self.layer_weights = nn.Parameter(self.layer_weights, requires_grad=False)

        print(f"Efficient FLEX: {len(target_layers)} layers, weight={loss_weight}, SNR<{snr_threshold}")
    
    def cross_normalize(self, teacher_feat, student_feat):
        """
        Core FLEX cross-normalization using student statistics
        Handles both 4D (B,C,H,W) and 2D (B,C) feature tensors
        """
        # Dynamically handle different tensor shapes
        if student_feat.dim() == 4:
            # 4D tensor (B, C, H, W) - spatial features
            spatial_dims = [2, 3]
        elif student_feat.dim() == 3:
            # 3D tensor (B, C, L) - sequence features  
            spatial_dims = [2]
        elif student_feat.dim() == 2:
            # 2D tensor (B, C) - global features
            spatial_dims = [1]
        else:
            # Fallback - normalize over all dims except batch
            spatial_dims = list(range(1, student_feat.dim()))
        
        # Use student feature statistics for both (key innovation)
        student_mean = student_feat.mean(dim=spatial_dims, keepdim=True)
        student_std = student_feat.std(dim=spatial_dims, keepdim=True) + 1e-8
        
        # Normalize both using student statistics
        teacher_norm = (teacher_feat - student_mean) / student_std
        student_norm = (student_feat - student_mean) / student_std
        
        return teacher_norm, student_norm
    
    def compute_flex_loss(self, teacher_feat, student_feat, layer_weight):
        """
        Compute FLEX loss for single layer with all optimizations
        """
        # 1. Cross-normalization (core innovation)
        if self.apply_cross_norm:
            teacher_norm, student_norm = self.cross_normalize(teacher_feat, student_feat)
        else:
            teacher_norm, student_norm = teacher_feat, student_feat
        
        # 2. Outlier detection (only for 4D spatial features)
        if self.apply_outlier_detection and student_norm.dim() == 4:
            reliable_mask = self.outlier_detector(student_norm)
        else:
            reliable_mask = torch.ones_like(student_norm)
        
        # 3. Masked MSE loss - fully vectorized
        diff = teacher_norm - student_norm
        masked_diff = diff * reliable_mask
        
        # Compute loss with proper normalization
        loss_numerator = torch.sum(masked_diff ** 2)
        loss_denominator = torch.sum(reliable_mask) + 1e-8  # Avoid division by zero
        
        layer_loss = loss_numerator / loss_denominator
        weighted_loss = layer_weight * layer_loss
        
        return weighted_loss
    
    def should_apply_loss(self, timestep=None, max_timestep=None):
        """SNR-aware loss application (high SNR = low timestep for RF)"""
        if timestep is None:
            return True
            
        # Normalize timestep to [0, 1] 
        if max_timestep is not None and max_timestep > 1:
            t_norm = timestep / max_timestep
        else:
            t_norm = timestep
            
        # Apply only at high SNR (low t for rectified flow)
        return t_norm < self.snr_threshold
    
    def extract_multi_scale_features(self, model, *args, **kwargs):
        """
        Efficient feature extraction - modify this based on your model structure
        This is a placeholder - you should replace with direct feature access
        """
        # For now, return empty dict - we'll use features passed directly
        return {}
    
    def forward(self, teacher_features, student_features, timestep=None, max_timestep=None):
        """
        Main FLEX loss computation
        
        Args:
            teacher_features: List or dict of teacher features
            student_features: List or dict of student features  
            timestep: Current timestep for SNR thresholding
            max_timestep: Max timestep for normalization
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
        
        # Handle list input (most efficient for your use case)
        if isinstance(teacher_features, (list, tuple)) and isinstance(student_features, (list, tuple)):
            n_layers = min(len(teacher_features), len(student_features), len(self.layer_weights))
            
            for i in range(n_layers):
                teacher_feat = teacher_features[i]
                student_feat = student_features[i] 
                
                # Shape compatibility check
                if teacher_feat.shape != student_feat.shape:
                    continue
                    
                # Get layer weight
                layer_weight = self.layer_weights[i]
                
                # Compute layer loss
                layer_loss = self.compute_flex_loss(teacher_feat, student_feat, layer_weight)
                total_loss += layer_loss
                valid_layers += 1
        
        # Handle dict input
        elif isinstance(teacher_features, dict) and isinstance(student_features, dict):
            for i, layer_name in enumerate(self.target_layers):
                if layer_name not in teacher_features or layer_name not in student_features:
                    continue
                    
                teacher_feat = teacher_features[layer_name]
                student_feat = student_features[layer_name]
                
                if teacher_feat.shape != student_feat.shape:
                    continue
                
                layer_weight = self.layer_weights[i] if i < len(self.layer_weights) else 1.0
                layer_loss = self.compute_flex_loss(teacher_feat, student_feat, layer_weight)
                total_loss += layer_loss
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