"""
CNN Image Encoder for modSTOVE.

Extracts spatial features from 128x128 RGB images for slot attention.
Implemented in JAX/Flax with full JIT compatibility.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Sequence
import numpy as np


class SoftPositionEmbed(nn.Module):
    """
    Soft positional embedding for spatial features.
    
    Projects normalized 2D coordinates to hidden dimension and adds to features.
    """
    hidden_dim: int
    resolution: Tuple[int, int]
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Add positional embeddings to features.
        
        Args:
            x: Features [B, H, W, C]
            
        Returns:
            Features with position embeddings [B, H, W, C]
        """
        B, H, W, C = x.shape
        
        # Build position grid
        grid = self._build_grid(H, W)
        
        # Project 4D position to hidden_dim
        pos_embed = nn.Dense(
            features=self.hidden_dim,
            name='pos_proj'
        )(grid)  # [H, W, hidden_dim]
        
        return x + pos_embed
    
    def _build_grid(self, H: int, W: int) -> jnp.ndarray:
        """Build normalized position grid."""
        x_coords = jnp.linspace(-1, 1, W)
        y_coords = jnp.linspace(-1, 1, H)
        
        yy, xx = jnp.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Stack as 4D position: (x, y, x+step, y+step)
        grid = jnp.stack([
            xx,
            yy,
            xx + 2.0 / W,
            yy + 2.0 / H,
        ], axis=-1)  # [H, W, 4]
        
        return grid


class ImageEncoder(nn.Module):
    """
    CNN Encoder for 128x128 RGB images.
    
    Architecture:
    - 5 convolutional layers with progressively increasing channels
    - Output: 32x32 or 16x16 feature map (configurable)
    - Final output shape: [batch, H*W, feature_dim]
    """
    hidden_dim: int = 64
    feature_dim: int = 128
    output_resolution: int = 32  # 32 or 16
    kernel_size: int = 4
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Extract features from input images.
        
        Args:
            x: Input images [B, H, W, C] where H=W=128, C=3
            train: Whether in training mode
            
        Returns:
            Features [B, H'*W', feature_dim] ready for slot attention
        """
        B = x.shape[0]
        
        # Determine stride pattern for desired output resolution
        # 128 -> 64 -> 32 -> 16 -> 8 -> 4 with stride 2
        # For 32x32 output: two stride-2 layers (128->64->32)
        # For 16x16 output: three stride-2 layers (128->64->32->16)
        
        if self.output_resolution == 32:
            strides = [2, 2, 1, 1, 1]
        else:  # 16x16
            strides = [2, 2, 2, 1, 1]
        
        channels = [32, 64, 128, 128, self.hidden_dim]
        
        # Layer 1
        x = nn.Conv(
            features=channels[0],
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(strides[0], strides[0]),
            padding='SAME',
            name='conv1'
        )(x)
        x = nn.relu(x)
        
        # Layer 2
        x = nn.Conv(
            features=channels[1],
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(strides[1], strides[1]),
            padding='SAME',
            name='conv2'
        )(x)
        x = nn.relu(x)
        
        # Layer 3
        x = nn.Conv(
            features=channels[2],
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(strides[2], strides[2]),
            padding='SAME',
            name='conv3'
        )(x)
        x = nn.relu(x)
        
        # Layer 4
        x = nn.Conv(
            features=channels[3],
            kernel_size=(3, 3),
            strides=(strides[3], strides[3]),
            padding='SAME',
            name='conv4'
        )(x)
        x = nn.relu(x)
        
        # Layer 5 (no activation)
        x = nn.Conv(
            features=channels[4],
            kernel_size=(3, 3),
            strides=(strides[4], strides[4]),
            padding='SAME',
            name='conv5'
        )(x)  # [B, H', W', hidden_dim]
        
        H_out, W_out = x.shape[1], x.shape[2]
        
        # Add positional embeddings
        x = SoftPositionEmbed(
            hidden_dim=self.hidden_dim,
            resolution=(H_out, W_out),
            name='pos_embed'
        )(x)
        
        # Flatten spatial dimensions
        x = x.reshape(B, H_out * W_out, self.hidden_dim)  # [B, H'*W', hidden_dim]
        
        # MLP for feature projection
        x = nn.Dense(features=self.hidden_dim, name='mlp1')(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.feature_dim, name='mlp2')(x)
        
        return x
    
    def get_output_shape(self, input_resolution: int = 128) -> Tuple[int, int]:
        """Return (num_patches, feature_dim) for given input resolution."""
        num_patches = self.output_resolution * self.output_resolution
        return (num_patches, self.feature_dim)


class ImageEncoderWithNorm(nn.Module):
    """
    Image encoder with layer normalization for improved training stability.
    """
    hidden_dim: int = 64
    feature_dim: int = 128
    output_resolution: int = 32
    kernel_size: int = 4
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Extract features from input images with normalization.
        
        Args:
            x: Input images [B, H, W, C] where H=W=128, C=3
            train: Whether in training mode
            
        Returns:
            Features [B, H'*W', feature_dim]
        """
        B = x.shape[0]
        
        if self.output_resolution == 32:
            strides = [2, 2, 1, 1, 1]
        else:
            strides = [2, 2, 2, 1, 1]
        
        channels = [32, 64, 128, 128, self.hidden_dim]
        
        # Conv blocks with group norm
        for i, (ch, stride) in enumerate(zip(channels, strides)):
            ks = self.kernel_size if i < 3 else 3
            x = nn.Conv(
                features=ch,
                kernel_size=(ks, ks),
                strides=(stride, stride),
                padding='SAME',
                name=f'conv{i+1}'
            )(x)
            if i < len(channels) - 1:
                x = nn.GroupNorm(num_groups=min(8, ch), name=f'gn{i+1}')(x)
                x = nn.relu(x)
        
        H_out, W_out = x.shape[1], x.shape[2]
        
        # Positional embeddings
        x = SoftPositionEmbed(
            hidden_dim=self.hidden_dim,
            resolution=(H_out, W_out),
            name='pos_embed'
        )(x)
        
        # Flatten and project
        x = x.reshape(B, H_out * W_out, self.hidden_dim)
        x = nn.LayerNorm(name='pre_mlp_norm')(x)
        x = nn.Dense(features=self.hidden_dim, name='mlp1')(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.feature_dim, name='mlp2')(x)
        
        return x
