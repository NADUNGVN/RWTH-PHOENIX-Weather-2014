"""
Feature Encoders for Multi-Cue CSLR
- FrameEncoder: ConvNeXt-S for full frames
- HandEncoder: ConvNeXt-T for hand crops
- PoseEncoder: MLP + TCN for pose keypoints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional


class FrameEncoder(nn.Module):
    """
    Encode full frames using ConvNeXt-Small.
    
    Input: (B, T, 3, 224, 224)
    Output: (B, T, output_dim)
    """
    
    def __init__(self, output_dim=768, pretrained=True, freeze_stages=2, dropout=0.2):
        super().__init__()
        
        # Load ConvNeXt-Small pretrained
        weights = 'IMAGENET1K_V1' if pretrained else None
        convnext = models.convnext_small(weights=weights)
        
        # Get the features (without classifier)
        self.backbone = convnext.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # ConvNeXt-Small output dim is 768
        backbone_dim = 768
        
        # Projection layer
        if backbone_dim != output_dim:
            self.projection = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(backbone_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        else:
            self.projection = nn.LayerNorm(output_dim)
        
        # Freeze early stages
        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)
        
        self.output_dim = output_dim
    
    def _freeze_stages(self, num_stages):
        """Freeze first N stages of ConvNeXt."""
        # ConvNeXt has 8 stages in features
        for i, block in enumerate(self.backbone):
            if i < num_stages:
                for param in block.parameters():
                    param.requires_grad = False
    
    def forward(self, frames):
        """
        Args:
            frames: (B, T, C, H, W) or (B*T, C, H, W)
        Returns:
            features: (B, T, output_dim)
        """
        if frames.dim() == 5:
            B, T, C, H, W = frames.shape
            x = frames.view(B * T, C, H, W)
        else:
            B, T = frames.shape[0], 1
            x = frames
        
        # Extract features
        x = self.backbone(x)  # (B*T, 768, H', W')
        x = self.pool(x)      # (B*T, 768, 1, 1)
        x = x.flatten(1)      # (B*T, 768)
        
        # Project
        x = self.projection(x)  # (B*T, output_dim)
        
        # Reshape back
        x = x.view(B, T, -1)  # (B, T, output_dim)
        
        return x


class HandEncoder(nn.Module):
    """
    Encode cropped hand regions using ConvNeXt-Tiny.
    
    Input: (B, T, 2, 3, 112, 112) - 2 hands (left, right)
    Output: (B, T, output_dim)
    """
    
    def __init__(self, output_dim=768, pretrained=True, dropout=0.2):
        super().__init__()
        
        # Load ConvNeXt-Tiny (lighter than Small)
        weights = 'IMAGENET1K_V1' if pretrained else None
        convnext = models.convnext_tiny(weights=weights)
        
        self.backbone = convnext.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # ConvNeXt-Tiny output dim is 768
        backbone_dim = 768
        
        # Projection
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Learnable attention for fusing left and right hands
        self.hand_attention = nn.Sequential(
            nn.Linear(output_dim, output_dim // 4),
            nn.ReLU(),
            nn.Linear(output_dim // 4, 1)
        )
        
        self.output_dim = output_dim
    
    def forward(self, hands, hand_mask=None):
        """
        Args:
            hands: (B, T, 2, C, H, W) - left and right hands
            hand_mask: (B, T, 2) - 1 for valid, 0 for missing hands
        Returns:
            features: (B, T, output_dim)
        """
        B, T, num_hands, C, H, W = hands.shape
        
        # Reshape: (B*T*2, C, H, W)
        x = hands.view(B * T * num_hands, C, H, W)
        
        # Extract features
        x = self.backbone(x)  # (B*T*2, 768, H', W')
        x = self.pool(x)      # (B*T*2, 768, 1, 1)
        x = x.flatten(1)      # (B*T*2, 768)
        
        # Project
        x = self.projection(x)  # (B*T*2, output_dim)
        
        # Reshape: (B, T, 2, output_dim)
        x = x.view(B, T, num_hands, -1)
        
        # Fuse left and right hands using attention
        if hand_mask is not None:
            # Mask invalid hands
            hand_mask = hand_mask.unsqueeze(-1)  # (B, T, 2, 1)
            x = x * hand_mask
        
        # Attention-based fusion
        attn_scores = self.hand_attention(x)  # (B, T, 2, 1)
        
        if hand_mask is not None:
            attn_scores = attn_scores.masked_fill(hand_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=2)  # (B, T, 2, 1)
        
        # Handle case where both hands are missing
        attn_weights = torch.nan_to_num(attn_weights, nan=0.5)
        
        # Weighted sum
        fused = (x * attn_weights).sum(dim=2)  # (B, T, output_dim)
        
        return fused


class TemporalConvNet(nn.Module):
    """1D Temporal Convolutions to capture motion patterns."""
    
    def __init__(self, channels, num_layers=3, kernel_size=5, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            x: (B, T, C)
        """
        x = x.transpose(1, 2)  # (B, C, T)
        
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        
        return x.transpose(1, 2)  # (B, T, C)


class PoseEncoder(nn.Module):
    """
    Encode pose keypoints sequence.
    
    Input: (B, T, 75, 3) - 75 keypoints x (x, y, confidence)
           - 33 body pose points
           - 21 left hand points  
           - 21 right hand points
    Output: (B, T, output_dim)
    """
    
    def __init__(self, input_dim=225, hidden_dim=512, output_dim=768, 
                 num_tcn_layers=3, kernel_size=5, dropout=0.1):
        super().__init__()
        
        # Spatial encoder: encode each frame's pose
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Temporal encoder: capture motion patterns
        self.temporal_encoder = TemporalConvNet(
            output_dim, 
            num_layers=num_tcn_layers,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        self.output_dim = output_dim
    
    def forward(self, pose):
        """
        Args:
            pose: (B, T, 75, 3)
        Returns:
            features: (B, T, output_dim)
        """
        B, T, num_points, coords = pose.shape
        
        # Flatten keypoints: (B, T, 225)
        x = pose.view(B, T, -1)
        
        # Spatial encoding (per-frame)
        x = self.spatial_encoder(x)  # (B, T, output_dim)
        
        # Temporal encoding (across frames)
        x = self.temporal_encoder(x)  # (B, T, output_dim)
        
        return x


class LightweightPoseEncoder(nn.Module):
    """
    Lightweight pose encoder for faster training.
    Uses only body pose (33 points) + hand presence signals.
    """
    
    def __init__(self, output_dim=768, dropout=0.1):
        super().__init__()
        
        # Body pose: 33 points * 3 = 99
        # Hand presence: 2 (left, right)
        input_dim = 99 + 2
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.output_dim = output_dim
    
    def forward(self, pose):
        """
        Args:
            pose: (B, T, 75, 3)
        Returns:
            features: (B, T, output_dim)
        """
        B, T = pose.shape[:2]
        
        # Extract body pose (first 33 points)
        body_pose = pose[:, :, :33, :].reshape(B, T, -1)  # (B, T, 99)
        
        # Check hand presence (sum of confidence)
        left_hand_present = (pose[:, :, 33:54, 2].sum(dim=-1, keepdim=True) > 0).float()
        right_hand_present = (pose[:, :, 54:75, 2].sum(dim=-1, keepdim=True) > 0).float()
        
        # Concatenate
        x = torch.cat([body_pose, left_hand_present, right_hand_present], dim=-1)
        
        return self.encoder(x)
