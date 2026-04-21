"""
Slot Attention Encoder for modSTOVE.

Implements slot attention for object discovery with structured state representation:
- Position (3D): dims 0-2
- Velocity (3D): dims 3-5  
- Size (3D): dims 6-8
- Latent appearance (119D): dims 9-127
Total: 128D per slot
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Optional
import numpy as np


# State dimension indices
POS_SLICE = slice(0, 3)      # x, y, z position
VEL_SLICE = slice(3, 6)      # vx, vy, vz velocity
SIZE_SLICE = slice(6, 9)     # sx, sy, sz size
LATENT_SLICE = slice(9, 128) # appearance features

SLOT_DIM = 128
NUM_SLOTS = 7


class SlotAttention(nn.Module):
    """
    Slot Attention module for object-centric representation learning.
    
    Slots compete for input features through attention, enabling
    object discovery. Empty slots naturally emerge through attention competition.
    """
    num_slots: int = NUM_SLOTS
    slot_dim: int = SLOT_DIM
    num_iterations: int = 3
    hidden_dim: int = 256
    epsilon: float = 1e-8
    
    def setup(self):
        # Learnable slot initialization parameters
        self.slot_mu = self.param(
            'slot_mu',
            nn.initializers.normal(stddev=1.0),
            (1, self.num_slots, self.slot_dim)
        )
        self.slot_log_sigma = self.param(
            'slot_log_sigma',
            nn.initializers.zeros,
            (1, self.num_slots, self.slot_dim)
        )
    
    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        slots: Optional[jnp.ndarray] = None,
        num_iterations: Optional[int] = None,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Perform slot attention.
        
        Args:
            inputs: Input features [B, N, input_dim]
            slots: Optional initial slots [B, num_slots, slot_dim]
            num_iterations: Number of refinement iterations
            deterministic: If True, use mean initialization (no sampling)
            
        Returns:
            Tuple of:
                - Updated slots [B, num_slots, slot_dim]
                - Attention weights [B, num_slots, N]
        """
        B, N, D_in = inputs.shape
        
        if num_iterations is None:
            num_iterations = self.num_iterations
        
        # Initialize slots
        if slots is None:
            slots = self._initialize_slots(B, deterministic)
        
        # Normalize inputs
        inputs = nn.LayerNorm(name='norm_input')(inputs)
        
        # Project inputs to keys and values
        k = nn.Dense(self.slot_dim, use_bias=False, name='proj_k')(inputs)
        v = nn.Dense(self.slot_dim, use_bias=False, name='proj_v')(inputs)
        
        scale = self.slot_dim ** -0.5
        
        # Iterative attention
        for i in range(num_iterations):
            slots_prev = slots
            slots = nn.LayerNorm(name=f'norm_slots_{i}')(slots)
            
            # Compute attention queries
            q = nn.Dense(self.slot_dim, use_bias=False, name=f'proj_q_{i}')(slots)
            
            # Attention scores
            attn_logits = jnp.einsum('bnd,bmd->bnm', q, k) * scale  # [B, num_slots, N]
            
            # Softmax over slots (competition)
            attn_weights = jax.nn.softmax(attn_logits, axis=1)
            
            # Weighted mean of values (normalized)
            attn_normalized = attn_weights / (attn_weights.sum(axis=-1, keepdims=True) + self.epsilon)
            updates = jnp.einsum('bnm,bmd->bnd', attn_normalized, v)
            
            # GRU-style update
            slots = self._gru_update(updates, slots_prev, name=f'gru_{i}')
            
            # MLP refinement with residual
            mlp_out = self._mlp(slots, name=f'mlp_{i}')
            slots = slots + mlp_out
        
        return slots, attn_weights
    
    def _initialize_slots(self, batch_size: int, deterministic: bool = False) -> jnp.ndarray:
        """Initialize slots from learned Gaussian distribution."""
        mu = jnp.broadcast_to(self.slot_mu, (batch_size, self.num_slots, self.slot_dim))
        
        if deterministic:
            return mu
        
        sigma = jax.nn.softplus(self.slot_log_sigma)
        sigma = jnp.broadcast_to(sigma, (batch_size, self.num_slots, self.slot_dim))
        
        # Sample from Gaussian
        key = self.make_rng('sample')
        noise = jax.random.normal(key, mu.shape)
        return mu + sigma * noise
    
    def _gru_update(
        self,
        updates: jnp.ndarray,
        hidden: jnp.ndarray,
        name: str,
    ) -> jnp.ndarray:
        """GRU-style update for slots."""
        B, K, D = updates.shape
        
        # Flatten for processing
        updates_flat = updates.reshape(B * K, D)
        hidden_flat = hidden.reshape(B * K, D)
        
        # GRU gates
        concat = jnp.concatenate([updates_flat, hidden_flat], axis=-1)
        
        gates = nn.Dense(2 * D, name=f'{name}_gates')(concat)
        reset_gate, update_gate = jnp.split(jax.nn.sigmoid(gates), 2, axis=-1)
        
        candidate = nn.Dense(D, name=f'{name}_candidate')(
            jnp.concatenate([updates_flat, reset_gate * hidden_flat], axis=-1)
        )
        candidate = jnp.tanh(candidate)
        
        new_hidden = (1 - update_gate) * hidden_flat + update_gate * candidate
        
        return new_hidden.reshape(B, K, D)
    
    def _mlp(self, x: jnp.ndarray, name: str) -> jnp.ndarray:
        """MLP for slot refinement."""
        x = nn.LayerNorm(name=f'{name}_norm')(x)
        x = nn.Dense(self.hidden_dim, name=f'{name}_fc1')(x)
        x = nn.relu(x)
        x = nn.Dense(self.slot_dim, name=f'{name}_fc2')(x)
        return x


class SlotEncoder(nn.Module):
    """
    Full Slot Encoder for modSTOVE.
    
    Combines slot attention with structured state inference:
    - Position inferred from attention spatial distribution
    - Velocity as learnable per-slot projection
    - Size as learnable per-slot projection
    - Latent as remaining slot dimensions
    
    Output shape: [batch, 7, 128]
    """
    num_slots: int = NUM_SLOTS
    slot_dim: int = SLOT_DIM
    input_dim: int = 128
    num_iterations: int = 3
    hidden_dim: int = 256
    feature_resolution: int = 32  # Spatial resolution of input features
    
    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
        prev_slots: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Encode image features to structured slot representations.
        
        Args:
            features: Image features [B, N, input_dim] where N = H*W
            prev_slots: Optional previous slots for temporal consistency
            deterministic: If True, use deterministic initialization
            
        Returns:
            Tuple of:
                - Structured slots [B, num_slots, 128] with (pos, vel, size, latent)
                - Attention weights [B, num_slots, N]
                - Raw slots before structuring [B, num_slots, slot_dim]
        """
        B, N, D = features.shape
        H = W = self.feature_resolution
        
        # Project input features if needed
        if D != self.slot_dim:
            features = nn.Dense(self.slot_dim, name='input_proj')(features)
        
        # Slot attention
        slot_attn = SlotAttention(
            num_slots=self.num_slots,
            slot_dim=self.slot_dim,
            num_iterations=self.num_iterations,
            hidden_dim=self.hidden_dim,
            name='slot_attention'
        )
        
        raw_slots, attn_weights = slot_attn(
            features,
            slots=prev_slots,
            deterministic=deterministic
        )
        
        # Infer position from attention weights
        position = self._infer_position(attn_weights, H, W)  # [B, num_slots, 3]
        
        # Project slots to structured representation
        # Velocity
        velocity = nn.Dense(3, name='vel_proj')(raw_slots)  # [B, num_slots, 3]
        
        # Size (always positive via softplus)
        size_raw = nn.Dense(3, name='size_proj')(raw_slots)
        size = jax.nn.softplus(size_raw) * 0.5  # [B, num_slots, 3]
        
        # Latent appearance (remaining 119 dims)
        latent = nn.Dense(119, name='latent_proj')(raw_slots)  # [B, num_slots, 119]
        
        # Combine into structured state
        structured_slots = jnp.concatenate([
            position,  # dims 0-2
            velocity,  # dims 3-5
            size,      # dims 6-8
            latent,    # dims 9-127
        ], axis=-1)  # [B, num_slots, 128]
        
        return structured_slots, attn_weights, raw_slots
    
    def _infer_position(
        self,
        attn_weights: jnp.ndarray,
        H: int,
        W: int,
    ) -> jnp.ndarray:
        """
        Infer 3D position from 2D attention weights.
        
        Uses center of mass of attention as x, y position.
        Z position is inferred from attention entropy (spread).
        
        Args:
            attn_weights: Attention [B, num_slots, N]
            H, W: Spatial dimensions
            
        Returns:
            Position [B, num_slots, 3]
        """
        B, K, N = attn_weights.shape
        
        # Create normalized coordinate grids
        x_coords = jnp.linspace(-1, 1, W)
        y_coords = jnp.linspace(-1, 1, H)
        yy, xx = jnp.meshgrid(y_coords, x_coords, indexing='ij')
        
        coords_flat = jnp.stack([
            xx.flatten(),  # x
            yy.flatten(),  # y
        ], axis=-1)  # [N, 2]
        
        # Compute center of mass for x, y
        # attn_weights: [B, K, N], coords: [N, 2]
        xy_pos = jnp.einsum('bkn,nd->bkd', attn_weights, coords_flat)  # [B, K, 2]
        
        # Z position from attention entropy (more spread = further away)
        entropy = -jnp.sum(
            attn_weights * jnp.log(attn_weights + 1e-8),
            axis=-1,
            keepdims=True
        )
        # Normalize entropy to [-1, 1] range
        max_entropy = jnp.log(N)
        z_pos = 2.0 * (entropy / max_entropy) - 1.0  # [B, K, 1]
        
        position = jnp.concatenate([xy_pos, z_pos], axis=-1)  # [B, K, 3]
        
        return position


class TemporalSlotEncoder(nn.Module):
    """
    Slot encoder with temporal consistency for video sequences.
    
    Uses previous frame's slots to initialize current frame's slots,
    enabling tracking of objects across time.
    """
    num_slots: int = NUM_SLOTS
    slot_dim: int = SLOT_DIM
    input_dim: int = 128
    num_iterations: int = 3
    hidden_dim: int = 256
    feature_resolution: int = 32
    
    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
        prev_slots: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Encode features with temporal slot propagation.
        
        Args:
            features: [B, T, N, input_dim] or [B, N, input_dim]
            prev_slots: Optional [B, num_slots, slot_dim]
            deterministic: If True, use deterministic initialization
            
        Returns:
            For video: (slots [B, T, K, 128], attn [B, T, K, N], raw [B, T, K, D])
            For image: (slots [B, K, 128], attn [B, K, N], raw [B, K, D])
        """
        encoder = SlotEncoder(
            num_slots=self.num_slots,
            slot_dim=self.slot_dim,
            input_dim=self.input_dim,
            num_iterations=self.num_iterations,
            hidden_dim=self.hidden_dim,
            feature_resolution=self.feature_resolution,
            name='slot_encoder'
        )
        
        if features.ndim == 3:
            # Single frame
            return encoder(features, prev_slots, deterministic)
        
        # Video: [B, T, N, D]
        B, T, N, D = features.shape
        
        all_slots = []
        all_attn = []
        all_raw = []
        
        slots = prev_slots
        
        for t in range(T):
            # Use previous structured slots to initialize raw slots
            if slots is not None:
                # Project structured slots back to raw format for initialization
                init_slots = nn.Dense(self.slot_dim, name='temporal_proj')(slots)
            else:
                init_slots = None
            
            struct_slots, attn, raw_slots = encoder(
                features[:, t],
                prev_slots=init_slots,
                deterministic=deterministic
            )
            
            all_slots.append(struct_slots)
            all_attn.append(attn)
            all_raw.append(raw_slots)
            
            slots = struct_slots
        
        return (
            jnp.stack(all_slots, axis=1),
            jnp.stack(all_attn, axis=1),
            jnp.stack(all_raw, axis=1),
        )
