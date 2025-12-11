"""
Multi-Cue CSLR Model
Combines Frame, Hand, and Pose encoders with Cross-Modal Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from .encoders import FrameEncoder, HandEncoder, PoseEncoder, LightweightPoseEncoder
from .fusion import CrossModalFusion, SimpleFusion, AdaptiveFusion
from .conformer import ConformerEncoder, PositionalEncoding, LightweightTemporalEncoder


class MultiCueCSLR(nn.Module):
    """
    Multi-Cue Continuous Sign Language Recognition Model.
    
    Architecture:
    1. Feature Extractors (Frame, Hand, Pose)
    2. Cross-Modal Fusion
    3. Temporal Encoder (Conformer)
    4. CTC Output Layer
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 8,
        n_temporal_layers: int = 6,
        n_fusion_layers: int = 2,
        conformer_kernel: int = 31,
        ff_expansion: int = 4,
        dropout: float = 0.1,
        frame_pretrained: bool = True,
        hand_pretrained: bool = True,
        frame_freeze_stages: int = 2,
        fusion_type: str = 'cross_attention',  # 'cross_attention', 'simple', 'adaptive'
        temporal_type: str = 'conformer',  # 'conformer', 'lightweight'
        use_auxiliary_ctc: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_auxiliary_ctc = use_auxiliary_ctc
        
        # Feature Encoders
        self.frame_encoder = FrameEncoder(
            output_dim=d_model,
            pretrained=frame_pretrained,
            freeze_stages=frame_freeze_stages,
            dropout=dropout
        )
        
        self.hand_encoder = HandEncoder(
            output_dim=d_model,
            pretrained=hand_pretrained,
            dropout=dropout
        )
        
        self.pose_encoder = PoseEncoder(
            output_dim=d_model,
            dropout=dropout
        )
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Cross-Modal Fusion
        if fusion_type == 'cross_attention':
            self.fusion = CrossModalFusion(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                num_layers=n_fusion_layers
            )
        elif fusion_type == 'simple':
            self.fusion = SimpleFusion(d_model=d_model, dropout=dropout)
        elif fusion_type == 'adaptive':
            self.fusion = AdaptiveFusion(d_model=d_model, dropout=dropout)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Temporal Encoder
        if temporal_type == 'conformer':
            self.temporal_encoder = ConformerEncoder(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_temporal_layers,
                conv_kernel=conformer_kernel,
                ff_expansion=ff_expansion,
                dropout=dropout
            )
        elif temporal_type == 'lightweight':
            self.temporal_encoder = LightweightTemporalEncoder(
                d_model=d_model,
                n_layers=n_temporal_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown temporal type: {temporal_type}")
        
        # Output Layers
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Auxiliary output layers for each modality (deep supervision)
        if use_auxiliary_ctc:
            self.frame_output = nn.Linear(d_model, vocab_size)
            self.hand_output = nn.Linear(d_model, vocab_size)
            self.pose_output = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize non-pretrained weights."""
        for name, p in self.named_parameters():
            if 'encoder' in name and 'backbone' in name:
                continue  # Skip pretrained backbones
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        frames: torch.Tensor,
        hands: torch.Tensor,
        poses: torch.Tensor,
        frame_lengths: Optional[list] = None,
        hand_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            frames: (B, T, 3, 224, 224) - full frames
            hands: (B, T, 2, 3, 112, 112) - left and right hand crops
            poses: (B, T, 75, 3) - pose keypoints
            frame_lengths: list of actual sequence lengths
            hand_mask: (B, T, 2) - mask for valid hands
            
        Returns:
            dict with:
                - 'fused': (T, B, vocab_size) - main output for CTC
                - 'frame': (T, B, vocab_size) - frame-only output
                - 'hand': (T, B, vocab_size) - hand-only output
                - 'pose': (T, B, vocab_size) - pose-only output
                - 'f_frame', 'f_hand', 'f_pose': features for consistency loss
        """
        B, T = frames.shape[:2]
        
        # Create padding mask
        mask = None
        if frame_lengths is not None:
            mask = torch.zeros(B, T, device=frames.device)
            for i, length in enumerate(frame_lengths):
                mask[i, :length] = 1
        
        # 1. Extract features from each modality
        f_frame = self.frame_encoder(frames)  # (B, T, D)
        f_hand = self.hand_encoder(hands, hand_mask)  # (B, T, D)
        f_pose = self.pose_encoder(poses)  # (B, T, D)
        
        # 2. Add positional encoding
        f_frame = self.pos_encoder(f_frame)
        f_hand = self.pos_encoder(f_hand)
        f_pose = self.pos_encoder(f_pose)
        
        # 3. Cross-modal fusion
        f_fused, attn_weights = self.fusion(f_frame, f_hand, f_pose, mask)  # (B, T, D)
        
        # 4. Temporal encoding
        f_temporal = self.temporal_encoder(f_fused, mask)  # (B, T, D)
        
        # 5. Output layer
        logits = self.output_layer(f_temporal)  # (B, T, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # CTC expects (T, B, vocab_size)
        output = log_probs.transpose(0, 1)
        
        outputs = {
            'fused': output,
            'f_frame': f_frame,
            'f_hand': f_hand,
            'f_pose': f_pose,
            'attn_weights': attn_weights
        }
        
        # Auxiliary outputs for deep supervision (use features directly, no extra temporal encoding)
        if self.use_auxiliary_ctc:
            # Use features directly without additional temporal encoding (faster & more stable)
            outputs['frame'] = F.log_softmax(self.frame_output(f_frame), dim=-1).transpose(0, 1)
            outputs['hand'] = F.log_softmax(self.hand_output(f_hand), dim=-1).transpose(0, 1)
            outputs['pose'] = F.log_softmax(self.pose_output(f_pose), dim=-1).transpose(0, 1)
        
        return outputs
    
    def get_features(
        self,
        frames: torch.Tensor,
        hands: torch.Tensor,
        poses: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract features without computing outputs."""
        f_frame = self.frame_encoder(frames)
        f_hand = self.hand_encoder(hands)
        f_pose = self.pose_encoder(poses)
        return f_frame, f_hand, f_pose


class MultiCueLoss(nn.Module):
    """
    Combined loss for Multi-Cue model:
    1. CTC Loss (main)
    2. Auxiliary CTC losses for each modality
    3. Cross-modal consistency loss (disabled by default - can hurt performance)
    """
    
    def __init__(self, blank_id, lambda_aux=0.3, lambda_consist=0.0):
        super().__init__()
        
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True)
        self.lambda_aux = lambda_aux
        self.lambda_consist = lambda_consist  # Set to 0 by default
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            outputs: dict from MultiCueCSLR forward
            targets: (B, S) target sequences
            input_lengths: (B,) input sequence lengths
            target_lengths: (B,) target sequence lengths
            
        Returns:
            dict with loss components
        """
        # Main CTC loss
        loss_main = self.ctc_loss(
            outputs['fused'], targets, input_lengths, target_lengths
        )
        
        total_loss = loss_main
        losses = {'main': loss_main}
        
        # Auxiliary losses (deep supervision)
        if 'frame' in outputs:
            loss_frame = self.ctc_loss(
                outputs['frame'], targets, input_lengths, target_lengths
            )
            loss_hand = self.ctc_loss(
                outputs['hand'], targets, input_lengths, target_lengths
            )
            loss_pose = self.ctc_loss(
                outputs['pose'], targets, input_lengths, target_lengths
            )
            
            loss_aux = (loss_frame + loss_hand + loss_pose) / 3
            total_loss = total_loss + self.lambda_aux * loss_aux
            
            losses['aux'] = loss_aux
            losses['frame'] = loss_frame
            losses['hand'] = loss_hand
            losses['pose'] = loss_pose
        
        # Cross-modal consistency loss
        if 'f_frame' in outputs:
            loss_consist = self._consistency_loss(
                outputs['f_frame'], outputs['f_hand'], outputs['f_pose']
            )
            total_loss = total_loss + self.lambda_consist * loss_consist
            losses['consist'] = loss_consist
        
        losses['total'] = total_loss
        return losses
    
    def _consistency_loss(self, f_frame, f_hand, f_pose):
        """
        Encourage features from different modalities to align.
        Uses cosine similarity loss.
        """
        # Normalize features
        f_frame = F.normalize(f_frame, dim=-1)
        f_hand = F.normalize(f_hand, dim=-1)
        f_pose = F.normalize(f_pose, dim=-1)
        
        # Cosine similarity should be high
        sim_fh = F.cosine_similarity(f_frame, f_hand, dim=-1).mean()
        sim_fp = F.cosine_similarity(f_frame, f_pose, dim=-1).mean()
        sim_hp = F.cosine_similarity(f_hand, f_pose, dim=-1).mean()
        
        # Loss = 1 - similarity
        loss = 3 - (sim_fh + sim_fp + sim_hp)
        
        return loss


def make_multicue_model(
    vocab_size: int,
    d_model: int = 768,
    n_heads: int = 8,
    n_temporal_layers: int = 6,
    dropout: float = 0.1,
    pretrained: bool = True,
    **kwargs
) -> MultiCueCSLR:
    """
    Factory function to create MultiCueCSLR model.
    
    Args:
        vocab_size: vocabulary size for CTC
        d_model: model dimension
        n_heads: number of attention heads
        n_temporal_layers: number of Conformer layers
        dropout: dropout rate
        pretrained: use pretrained backbones
        
    Returns:
        MultiCueCSLR model
    """
    return MultiCueCSLR(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_temporal_layers=n_temporal_layers,
        dropout=dropout,
        frame_pretrained=pretrained,
        hand_pretrained=pretrained,
        **kwargs
    )
