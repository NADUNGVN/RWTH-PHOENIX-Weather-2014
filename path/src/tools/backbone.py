"""
Backbone networks for Sign Language Recognition.
Supports: EfficientNet, ResNet, ConvNeXt
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple


class BackboneFactory:
    """Factory class to create different backbone networks."""
    
    SUPPORTED_BACKBONES = [
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
        'resnet18', 'resnet34', 'resnet50', 'resnet101',
        'convnext_tiny', 'convnext_small',
        'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'
    ]
    
    # Output feature dimensions for each backbone
    FEATURE_DIMS = {
        'efficientnet_b0': 1280,
        'efficientnet_b1': 1280,
        'efficientnet_b2': 1408,
        'efficientnet_b3': 1536,
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'convnext_tiny': 768,
        'convnext_small': 768,
        'mobilenet_v2': 1280,
        'mobilenet_v3_small': 576,
        'mobilenet_v3_large': 960,
    }
    
    @classmethod
    def create(cls, name: str, pretrained: bool = True, frozen_stages: int = 0) -> Tuple[nn.Module, int]:
        """
        Create a backbone network.
        
        Args:
            name: Name of the backbone (e.g., 'efficientnet_b0')
            pretrained: Whether to load pretrained ImageNet weights
            frozen_stages: Number of stages to freeze (0 = no freezing)
            
        Returns:
            Tuple of (backbone_module, feature_dim)
        """
        if name not in cls.SUPPORTED_BACKBONES:
            raise ValueError(f"Backbone {name} not supported. Choose from: {cls.SUPPORTED_BACKBONES}")
        
        weights = 'IMAGENET1K_V1' if pretrained else None
        
        if name.startswith('efficientnet'):
            backbone = cls._create_efficientnet(name, weights)
        elif name.startswith('resnet'):
            backbone = cls._create_resnet(name, weights)
        elif name.startswith('convnext'):
            backbone = cls._create_convnext(name, weights)
        elif name.startswith('mobilenet'):
            backbone = cls._create_mobilenet(name, weights)
        else:
            raise ValueError(f"Unknown backbone: {name}")
        
        if frozen_stages > 0:
            cls._freeze_stages(backbone, frozen_stages)
        
        return backbone, cls.FEATURE_DIMS[name]
    
    @classmethod
    def _create_efficientnet(cls, name: str, weights) -> nn.Module:
        """Create EfficientNet backbone."""
        model_fn = getattr(models, name)
        model = model_fn(weights=weights)
        # Remove classifier, keep features
        backbone = model.features
        return backbone
    
    @classmethod
    def _create_resnet(cls, name: str, weights) -> nn.Module:
        """Create ResNet backbone."""
        model_fn = getattr(models, name)
        model = model_fn(weights=weights)
        # Remove FC layer, keep conv layers
        backbone = nn.Sequential(*list(model.children())[:-2])
        return backbone
    
    @classmethod
    def _create_convnext(cls, name: str, weights) -> nn.Module:
        """Create ConvNeXt backbone."""
        model_fn = getattr(models, name)
        model = model_fn(weights=weights)
        # Remove classifier
        backbone = model.features
        return backbone
    
    @classmethod
    def _create_mobilenet(cls, name: str, weights) -> nn.Module:
        """Create MobileNet backbone."""
        model_fn = getattr(models, name)
        model = model_fn(weights=weights)
        backbone = model.features
        return backbone
    
    @staticmethod
    def _freeze_stages(backbone: nn.Module, num_stages: int):
        """Freeze first N stages of the backbone."""
        children = list(backbone.children())
        for i, child in enumerate(children[:num_stages]):
            for param in child.parameters():
                param.requires_grad = False


class SignLanguageBackbone(nn.Module):
    """
    Backbone wrapper for Sign Language Recognition.
    Handles frame-by-frame feature extraction from video sequences.
    """
    
    def __init__(self, 
                 backbone_name: str = 'efficientnet_b0',
                 pretrained: bool = True,
                 frozen_stages: int = 0,
                 output_dim: Optional[int] = None,
                 dropout: float = 0.2):
        """
        Args:
            backbone_name: Name of backbone architecture
            pretrained: Use ImageNet pretrained weights
            frozen_stages: Number of stages to freeze
            output_dim: If specified, project features to this dimension
            dropout: Dropout rate before projection
        """
        super().__init__()
        
        self.backbone, self.feature_dim = BackboneFactory.create(
            backbone_name, pretrained, frozen_stages
        )
        self.backbone_name = backbone_name
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Optional projection layer
        self.output_dim = output_dim if output_dim else self.feature_dim
        if output_dim and output_dim != self.feature_dim:
            self.projection = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.feature_dim, output_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.projection = nn.Identity()
        
        # For compatibility with original code
        self.last_channel = self.output_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """
        Forward pass for video frames.
        
        Args:
            x: Input tensor of shape (batch * seq_len, channels, height, width)
               or (batch, seq_len, channels, height, width)
               
        Returns:
            features: Pooled features (batch * seq_len, output_dim)
            feature_map: Feature maps before pooling
            grad: None (for compatibility)
        """
        # Handle 5D input (batch, seq, C, H, W)
        if x.dim() == 5:
            batch_size, seq_len = x.shape[:2]
            x = x.view(-1, *x.shape[2:])  # (batch * seq, C, H, W)
        else:
            batch_size, seq_len = None, None
        
        # Extract features
        feature_map = self.backbone(x)  # (N, C, H', W')
        
        # Global average pooling
        pooled = self.pool(feature_map)  # (N, C, 1, 1)
        pooled = pooled.flatten(1)       # (N, C)
        
        # Project features
        features = self.projection(pooled)  # (N, output_dim)
        
        return features, feature_map, None


class EfficientNetBackbone(SignLanguageBackbone):
    """EfficientNet-B0 backbone (default recommended)."""
    
    def __init__(self, pretrained: bool = True, output_dim: int = 1280, **kwargs):
        super().__init__(
            backbone_name='efficientnet_b0',
            pretrained=pretrained,
            output_dim=output_dim,
            **kwargs
        )


class ResNet50Backbone(SignLanguageBackbone):
    """ResNet-50 backbone."""
    
    def __init__(self, pretrained: bool = True, output_dim: int = 1280, **kwargs):
        super().__init__(
            backbone_name='resnet50',
            pretrained=pretrained,
            output_dim=output_dim,
            **kwargs
        )


class ConvNeXtBackbone(SignLanguageBackbone):
    """ConvNeXt-Tiny backbone (modern architecture)."""
    
    def __init__(self, pretrained: bool = True, output_dim: int = 1280, **kwargs):
        super().__init__(
            backbone_name='convnext_tiny',
            pretrained=pretrained,
            output_dim=output_dim,
            **kwargs
        )


# For backward compatibility with original mb2.py interface
def create_backbone(name: str = 'efficientnet_b0', pretrained: bool = True, 
                    channels: int = 3, output_dim: int = 1280) -> SignLanguageBackbone:
    """
    Create a backbone network.
    
    Args:
        name: Backbone name
        pretrained: Use pretrained weights
        channels: Input channels (ignored, always 3 for pretrained)
        output_dim: Output feature dimension
        
    Returns:
        SignLanguageBackbone instance
    """
    return SignLanguageBackbone(
        backbone_name=name,
        pretrained=pretrained,
        output_dim=output_dim
    )


if __name__ == '__main__':
    # Test backbones
    print("Testing backbones...")
    
    for name in ['efficientnet_b0', 'resnet50', 'convnext_tiny']:
        print(f"\n{name}:")
        backbone = SignLanguageBackbone(backbone_name=name, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        feat, fmap, _ = backbone(x)
        print(f"  Input: {x.shape}")
        print(f"  Feature: {feat.shape}")
        print(f"  Feature map: {fmap.shape}")
        print(f"  Output dim: {backbone.output_dim}")
