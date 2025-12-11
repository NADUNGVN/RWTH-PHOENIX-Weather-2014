"""
Conformer Temporal Encoder for Multi-Cue CSLR
Combines local convolution patterns with global self-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ConformerEncoder(nn.Module):
    """
    Conformer encoder for temporal modeling.
    Combines local conv patterns + global attention.
    
    Input: (B, T, D)
    Output: (B, T, D)
    """
    
    def __init__(self, d_model=768, n_heads=8, n_layers=6, 
                 conv_kernel=31, ff_expansion=4, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, n_heads, conv_kernel, ff_expansion, dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, D)
            mask: (B, T) - padding mask (1 for valid, 0 for padding)
        Returns:
            x: (B, T, D)
        """
        # Convert mask to attention mask
        attn_mask = None
        if mask is not None:
            attn_mask = (mask == 0)  # True for padding
        
        for layer in self.layers:
            x = layer(x, attn_mask)
        
        return self.final_norm(x)


class ConformerBlock(nn.Module):
    """
    Single Conformer block:
    FFN (half) -> Self-Attention -> Convolution -> FFN (half)
    """
    
    def __init__(self, d_model, n_heads, conv_kernel, ff_expansion, dropout):
        super().__init__()
        
        # Feed Forward 1 (half step)
        self.ff1 = FeedForward(d_model, expansion=ff_expansion, dropout=dropout)
        
        # Multi-Head Self Attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Convolution Module
        self.conv_module = ConvolutionModule(d_model, conv_kernel, dropout)
        
        # Feed Forward 2 (half step)
        self.ff2 = FeedForward(d_model, expansion=ff_expansion, dropout=dropout)
        
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, attn_mask=None):
        # FFN 1 with 0.5 residual weight
        x = x + 0.5 * self.ff1(x)
        
        # Self Attention
        x_norm = self.attn_norm(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm, 
            key_padding_mask=attn_mask
        )
        x = x + self.attn_dropout(attn_out)
        
        # Convolution
        x = x + self.conv_module(x, attn_mask)
        
        # FFN 2 with 0.5 residual weight
        x = x + 0.5 * self.ff2(x)
        
        return self.final_norm(x)


class ConvolutionModule(nn.Module):
    """Convolution module in Conformer."""
    
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Pointwise conv (expand)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        
        # Depthwise conv
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()  # Swish
        
        # Pointwise conv (project)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, D)
            mask: (B, T) attention mask
        Returns:
            x: (B, T, D)
        """
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (B, D, T)
        
        # Pointwise conv + GLU
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        
        # Depthwise conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # Pointwise conv + dropout
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        return x.transpose(1, 2)  # (B, T, D)


class FeedForward(nn.Module):
    """Feed Forward module with Swish activation."""
    
    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        
        hidden = d_model * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),  # Swish
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            x: (B, T, D) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for Conformer."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        self.d_model = d_model
        self.pe = nn.Embedding(2 * max_len - 1, d_model)
        
        # Initialize
        nn.init.xavier_uniform_(self.pe.weight)
    
    def forward(self, length):
        """
        Generate relative position encoding.
        
        Args:
            length: sequence length
        Returns:
            rel_pe: (length, length, d_model)
        """
        positions = torch.arange(length, device=self.pe.weight.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = relative_positions + length - 1  # Shift to positive
        
        return self.pe(relative_positions)


class LightweightTemporalEncoder(nn.Module):
    """
    Lightweight temporal encoder using 1D convolutions.
    Faster alternative to Conformer.
    """
    
    def __init__(self, d_model=768, n_layers=4, kernel_size=5, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Conv1d(d_model, d_model * 2, kernel_size, padding=kernel_size // 2),
                nn.GLU(dim=1),
                nn.Dropout(dropout)
            ))
        
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, D)
        Returns:
            x: (B, T, D)
        """
        for layer in self.layers:
            residual = x
            x = x.transpose(1, 2)  # (B, D, T)
            x = layer[0](x.transpose(1, 2)).transpose(1, 2)  # LayerNorm
            x = layer[1](x)  # Conv
            x = layer[2](x)  # GLU
            x = layer[3](x.transpose(1, 2))  # Dropout, back to (B, T, D)
            x = x + residual
        
        return self.final_norm(x)
