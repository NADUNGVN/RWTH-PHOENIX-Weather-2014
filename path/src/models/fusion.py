"""
Cross-Modal Fusion Module for Multi-Cue CSLR
Combines features from Frame, Hand, and Pose encoders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossModalFusion(nn.Module):
    """
    Fuse features from 3 modalities using cross-attention and gating.
    
    Input: 
        f_frame: (B, T, D)
        f_hand: (B, T, D)
        f_pose: (B, T, D)
    Output: (B, T, D)
    """
    
    def __init__(self, d_model=768, n_heads=8, dropout=0.1, num_layers=2):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Build fusion layers
        self.fusion_layers = nn.ModuleList([
            CrossModalFusionLayer(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, f_frame, f_hand, f_pose, mask=None):
        """
        Args:
            f_frame: (B, T, D) - frame features (anchor)
            f_hand: (B, T, D) - hand features
            f_pose: (B, T, D) - pose features
            mask: (B, T) - padding mask (1 for valid, 0 for padding)
        Returns:
            fused: (B, T, D)
            attn_weights: dict of attention weights for visualization
        """
        attn_weights = {}
        
        # Convert mask to attention mask format
        attn_mask = None
        if mask is not None:
            attn_mask = (mask == 0)  # True for padding positions
        
        # Apply fusion layers
        x = f_frame
        for i, layer in enumerate(self.fusion_layers):
            x, attn_hand, attn_pose = layer(x, f_hand, f_pose, attn_mask)
            attn_weights[f'layer_{i}_hand'] = attn_hand
            attn_weights[f'layer_{i}_pose'] = attn_pose
        
        return self.final_norm(x), attn_weights


class CrossModalFusionLayer(nn.Module):
    """Single layer of cross-modal fusion."""
    
    def __init__(self, d_model=768, n_heads=8, dropout=0.1):
        super().__init__()
        
        # Cross-attention: Frame attends to Hand
        self.cross_attn_hand = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_hand = nn.LayerNorm(d_model)
        self.dropout_hand = nn.Dropout(dropout)
        
        # Cross-attention: Frame attends to Pose
        self.cross_attn_pose = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_pose = nn.LayerNorm(d_model)
        self.dropout_pose = nn.Dropout(dropout)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # Fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, f_frame, f_hand, f_pose, attn_mask=None):
        """
        Fusion strategy:
        1. Frame features are the "anchor" (main representation)
        2. Cross-attend from frame -> hand to enhance with hand details
        3. Cross-attend from frame -> pose to enhance with structural info
        4. Gated combination of all features
        """
        # Cross-attention: Frame queries Hand
        f_frame_norm = self.norm_hand(f_frame)
        f_hand_norm = self.norm_hand(f_hand)
        
        f_enhanced_hand, attn_hand = self.cross_attn_hand(
            query=f_frame_norm,
            key=f_hand_norm,
            value=f_hand_norm,
            key_padding_mask=attn_mask,
            need_weights=True
        )
        f_frame = f_frame + self.dropout_hand(f_enhanced_hand)
        
        # Cross-attention: Frame queries Pose
        f_frame_norm = self.norm_pose(f_frame)
        f_pose_norm = self.norm_pose(f_pose)
        
        f_enhanced_pose, attn_pose = self.cross_attn_pose(
            query=f_frame_norm,
            key=f_pose_norm,
            value=f_pose_norm,
            key_padding_mask=attn_mask,
            need_weights=True
        )
        f_frame = f_frame + self.dropout_pose(f_enhanced_pose)
        
        # Concatenate all features
        concat_features = torch.cat([f_frame, f_hand, f_pose], dim=-1)  # (B, T, D*3)
        
        # Compute gate
        gate_weights = self.gate(concat_features)  # (B, T, D)
        
        # Fusion projection
        f_fused = self.fusion_proj(concat_features)  # (B, T, D)
        
        # Gated residual
        output = gate_weights * f_fused + (1 - gate_weights) * f_frame
        
        return self.final_norm(output), attn_hand, attn_pose


class SimpleFusion(nn.Module):
    """
    Simple fusion by concatenation and projection.
    Faster alternative to CrossModalFusion.
    """
    
    def __init__(self, d_model=768, dropout=0.1):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, f_frame, f_hand, f_pose, mask=None):
        # Concatenate
        concat = torch.cat([f_frame, f_hand, f_pose], dim=-1)
        
        # Project
        fused = self.projection(concat)
        
        return fused, {}


class AttentionPoolingFusion(nn.Module):
    """
    Attention-based pooling fusion.
    Each modality votes for the final representation.
    """
    
    def __init__(self, d_model=768, dropout=0.1):
        super().__init__()
        
        # Modality attention
        self.modality_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, f_frame, f_hand, f_pose, mask=None):
        # Stack modalities: (B, T, 3, D)
        stacked = torch.stack([f_frame, f_hand, f_pose], dim=2)
        
        # Compute attention weights: (B, T, 3, 1)
        attn_scores = self.modality_attention(stacked)
        attn_weights = F.softmax(attn_scores, dim=2)
        
        # Weighted sum: (B, T, D)
        fused = (stacked * attn_weights).sum(dim=2)
        
        return self.norm(fused), {'modality_weights': attn_weights.squeeze(-1)}


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion with learnable modality weights.
    Weights are learned per-timestep based on feature content.
    """
    
    def __init__(self, d_model=768, dropout=0.1):
        super().__init__()
        
        # Weight predictor for each modality
        self.weight_predictor = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3)  # 3 modality weights
        )
        
        # Projection after weighted sum
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, f_frame, f_hand, f_pose, mask=None):
        B, T, D = f_frame.shape
        
        # Concatenate for weight prediction
        concat = torch.cat([f_frame, f_hand, f_pose], dim=-1)  # (B, T, 3D)
        
        # Predict weights: (B, T, 3)
        weights = F.softmax(self.weight_predictor(concat), dim=-1)
        
        # Stack modalities: (B, T, 3, D)
        stacked = torch.stack([f_frame, f_hand, f_pose], dim=2)
        
        # Weighted sum
        weights = weights.unsqueeze(-1)  # (B, T, 3, 1)
        fused = (stacked * weights).sum(dim=2)  # (B, T, D)
        
        return self.projection(fused), {'modality_weights': weights.squeeze(-1)}
