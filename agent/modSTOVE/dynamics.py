"""
Graph Neural Network Dynamics Model for modSTOVE.

Action-conditioned dynamics prediction using pairwise object interactions.
Key equation: z_{t+1} = f(g(z_t, a_t) + Σ α(z_o, z_o') h(z_o, z_o'))

Action space: 5 discrete actions
- FORWARD (0)
- STOP (1)  
- TURN_LEFT (2)
- TURN_RIGHT (3)
- STOP_TURNING (4)
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Optional
import numpy as np

from .slot_encoder import POS_SLICE, VEL_SLICE, SIZE_SLICE, LATENT_SLICE, SLOT_DIM, NUM_SLOTS


# Action space
NUM_ACTIONS = 5
ACTION_FORWARD = 0
ACTION_STOP = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3
ACTION_STOP_TURNING = 4


class ActionEncoder(nn.Module):
    """
    Encodes discrete actions to continuous embeddings.
    """
    num_actions: int = NUM_ACTIONS
    embed_dim: int = 32
    
    @nn.compact
    def __call__(self, action: jnp.ndarray) -> jnp.ndarray:
        """
        Encode action to embedding.
        
        Args:
            action: Action indices [B] or one-hot [B, num_actions]
            
        Returns:
            Action embedding [B, embed_dim]
        """
        if action.ndim == 1:
            # Convert indices to one-hot
            action = jax.nn.one_hot(action, self.num_actions)
        
        # MLP encoding
        x = nn.Dense(self.embed_dim, name='fc1')(action)
        x = nn.relu(x)
        x = nn.Dense(self.embed_dim, name='fc2')(x)
        
        return x


class PairwiseInteraction(nn.Module):
    """
    Computes pairwise interactions between objects using attention.
    
    Implements: α(z_o, z_o') h(z_o, z_o')
    where α is attention weight and h is interaction function.
    """
    hidden_dim: int = 128
    num_heads: int = 4
    
    @nn.compact
    def __call__(self, slots: jnp.ndarray) -> jnp.ndarray:
        """
        Compute pairwise object interactions.
        
        Args:
            slots: Object states [B, K, D]
            
        Returns:
            Interaction effects [B, K, D]
        """
        B, K, D = slots.shape
        
        # Compute pairwise features
        # [B, K, D] -> [B, K, K, 2D]
        slots_i = jnp.expand_dims(slots, axis=2)  # [B, K, 1, D]
        slots_j = jnp.expand_dims(slots, axis=1)  # [B, 1, K, D]
        
        slots_i = jnp.broadcast_to(slots_i, (B, K, K, D))
        slots_j = jnp.broadcast_to(slots_j, (B, K, K, D))
        
        pair_features = jnp.concatenate([slots_i, slots_j], axis=-1)  # [B, K, K, 2D]
        
        # Interaction function h(z_o, z_o')
        h = nn.Dense(self.hidden_dim, name='h_fc1')(pair_features)
        h = nn.relu(h)
        h = nn.Dense(D, name='h_fc2')(h)  # [B, K, K, D]
        
        # Attention weights α(z_o, z_o')
        # Using scaled dot-product attention
        q = nn.Dense(self.hidden_dim, name='attn_q')(slots)  # [B, K, H]
        k = nn.Dense(self.hidden_dim, name='attn_k')(slots)  # [B, K, H]
        
        # Attention scores
        scale = self.hidden_dim ** -0.5
        attn_logits = jnp.einsum('bih,bjh->bij', q, k) * scale  # [B, K, K]
        
        # Mask self-interactions
        mask = jnp.eye(K, dtype=bool)
        attn_logits = jnp.where(mask, -1e9, attn_logits)
        
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)  # [B, K, K]
        
        # Weighted sum of interactions
        interactions = jnp.einsum('bij,bijd->bid', attn_weights, h)  # [B, K, D]
        
        return interactions


class ObjectTransition(nn.Module):
    """
    Per-object state transition function.
    
    Implements: g(z_t, a_t) - the action-conditioned individual update.
    """
    hidden_dim: int = 128
    output_dim: int = SLOT_DIM
    
    @nn.compact
    def __call__(
        self,
        slots: jnp.ndarray,
        action_embed: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute individual object transitions.
        
        Args:
            slots: Object states [B, K, D]
            action_embed: Action embedding [B, action_dim]
            
        Returns:
            Individual transitions [B, K, D]
        """
        B, K, D = slots.shape
        
        # Broadcast action to all objects
        action_broadcast = jnp.expand_dims(action_embed, axis=1)  # [B, 1, A]
        action_broadcast = jnp.broadcast_to(
            action_broadcast, 
            (B, K, action_embed.shape[-1])
        )
        
        # Concatenate slot state with action
        x = jnp.concatenate([slots, action_broadcast], axis=-1)  # [B, K, D+A]
        
        # MLP transition
        x = nn.Dense(self.hidden_dim, name='fc1')(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, name='fc2')(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, name='fc3')(x)
        
        return x


class DynamicsModel(nn.Module):
    """
    Full Graph Neural Network Dynamics Model.
    
    Predicts next-step object states given current states and action.
    
    Architecture:
        z_{t+1} = f(g(z_t, a_t) + Σ α(z_o, z_o') h(z_o, z_o'))
        
    where:
        - g: Individual object transition conditioned on action
        - h: Pairwise interaction function
        - α: Attention-based interaction weights
        - f: Output transformation
    
    Input: slots [B, 7, 128] + action [B] or [B, 5]
    Output: next_slots [B, 7, 128]
    """
    num_slots: int = NUM_SLOTS
    slot_dim: int = SLOT_DIM
    hidden_dim: int = 256
    action_dim: int = 32
    num_interaction_layers: int = 2
    
    @nn.compact
    def __call__(
        self,
        slots: jnp.ndarray,
        action: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Predict next object states.
        
        Args:
            slots: Current object states [B, num_slots, slot_dim]
            action: Action [B] (indices) or [B, num_actions] (one-hot)
            deterministic: If True, don't apply dropout
            
        Returns:
            Predicted next states [B, num_slots, slot_dim]
        """
        B, K, D = slots.shape
        
        # Encode action
        action_embed = ActionEncoder(
            embed_dim=self.action_dim,
            name='action_encoder'
        )(action)  # [B, action_dim]
        
        # Individual transition g(z_t, a_t)
        individual_update = ObjectTransition(
            hidden_dim=self.hidden_dim,
            output_dim=self.slot_dim,
            name='object_transition'
        )(slots, action_embed)  # [B, K, D]
        
        # Pairwise interactions
        interaction_total = jnp.zeros_like(slots)
        
        for i in range(self.num_interaction_layers):
            interactions = PairwiseInteraction(
                hidden_dim=self.hidden_dim,
                name=f'interaction_{i}'
            )(slots)
            interaction_total = interaction_total + interactions
        
        # Combine individual and interaction updates
        combined = individual_update + interaction_total
        
        # Output transformation f(.)
        x = nn.LayerNorm(name='output_norm')(combined)
        x = nn.Dense(self.hidden_dim, name='output_fc1')(x)
        x = nn.relu(x)
        delta = nn.Dense(self.slot_dim, name='output_fc2')(x)  # [B, K, D]
        
        # Residual connection: predict change in state
        next_slots = slots + delta
        
        # Enforce physical constraints on structured components
        next_slots = self._enforce_constraints(next_slots)
        
        return next_slots
    
    def _enforce_constraints(self, slots: jnp.ndarray) -> jnp.ndarray:
        """
        Enforce physical constraints on predicted states.
        
        - Position: clamp to valid range
        - Size: ensure positive via softplus
        """
        pos = slots[..., POS_SLICE]  # [B, K, 3]
        vel = slots[..., VEL_SLICE]  # [B, K, 3]
        size = slots[..., SIZE_SLICE]  # [B, K, 3]
        latent = slots[..., LATENT_SLICE]  # [B, K, 119]
        
        # Clamp position to reasonable range
        pos = jnp.clip(pos, -2.0, 2.0)
        
        # Clamp velocity
        vel = jnp.clip(vel, -1.0, 1.0)
        
        # Ensure size is positive
        size = jax.nn.softplus(size) * 0.5
        
        return jnp.concatenate([pos, vel, size, latent], axis=-1)
    
    def predict_sequence(
        self,
        initial_slots: jnp.ndarray,
        actions: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Predict sequence of states given initial state and actions.
        
        Args:
            initial_slots: Initial states [B, K, D]
            actions: Action sequence [B, T] or [B, T, num_actions]
            deterministic: If True, don't apply dropout
            
        Returns:
            Predicted states [B, T+1, K, D] (includes initial state)
        """
        B = initial_slots.shape[0]
        T = actions.shape[1]
        
        all_slots = [initial_slots]
        current_slots = initial_slots
        
        for t in range(T):
            action_t = actions[:, t]
            next_slots = self(current_slots, action_t, deterministic)
            all_slots.append(next_slots)
            current_slots = next_slots
        
        return jnp.stack(all_slots, axis=1)  # [B, T+1, K, D]


class PhysicsInformedDynamics(nn.Module):
    """
    Dynamics model with explicit physics-based structure.
    
    Uses Newtonian mechanics for position updates:
        x_{t+1} = x_t + v_t * dt + 0.5 * a_t * dt^2
        v_{t+1} = v_t + a_t * dt
        
    where acceleration is predicted by the network.
    """
    num_slots: int = NUM_SLOTS
    slot_dim: int = SLOT_DIM
    hidden_dim: int = 256
    action_dim: int = 32
    dt: float = 1.0  # Time step
    
    @nn.compact
    def __call__(
        self,
        slots: jnp.ndarray,
        action: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Predict next states using physics-informed update.
        
        Args:
            slots: Current states [B, K, D]
            action: Action [B] or [B, num_actions]
            deterministic: Unused, for API compatibility
            
        Returns:
            Next states [B, K, D]
        """
        B, K, D = slots.shape
        
        # Extract structured components
        pos = slots[..., POS_SLICE]  # [B, K, 3]
        vel = slots[..., VEL_SLICE]  # [B, K, 3]
        size = slots[..., SIZE_SLICE]  # [B, K, 3]
        latent = slots[..., LATENT_SLICE]  # [B, K, 119]
        
        # Encode action
        action_embed = ActionEncoder(
            embed_dim=self.action_dim,
            name='action_encoder'
        )(action)
        
        # Predict acceleration from full state + action
        accel_input = jnp.concatenate([
            pos, vel, size, latent,
            jnp.broadcast_to(
                action_embed[:, None, :],
                (B, K, self.action_dim)
            )
        ], axis=-1)
        
        # MLP to predict acceleration
        accel = nn.Dense(self.hidden_dim, name='accel_fc1')(accel_input)
        accel = nn.relu(accel)
        accel = nn.Dense(self.hidden_dim, name='accel_fc2')(accel)
        accel = nn.relu(accel)
        accel = nn.Dense(3, name='accel_out')(accel)  # [B, K, 3]
        
        # Clamp acceleration
        accel = jnp.clip(accel, -0.5, 0.5)
        
        # Physics update
        new_pos = pos + vel * self.dt + 0.5 * accel * self.dt ** 2
        new_vel = vel + accel * self.dt
        
        # Clamp to valid ranges
        new_pos = jnp.clip(new_pos, -2.0, 2.0)
        new_vel = jnp.clip(new_vel, -1.0, 1.0)
        
        # Predict changes to size and latent (smaller updates)
        size_latent_input = jnp.concatenate([
            slots,
            jnp.broadcast_to(action_embed[:, None, :], (B, K, self.action_dim))
        ], axis=-1)
        
        delta_size = nn.Dense(3, name='delta_size')(size_latent_input)
        delta_latent = nn.Dense(119, name='delta_latent')(size_latent_input)
        
        new_size = jax.nn.softplus(size + 0.1 * delta_size) * 0.5
        new_latent = latent + 0.1 * delta_latent
        
        # Combine
        next_slots = jnp.concatenate([
            new_pos,
            new_vel,
            new_size,
            new_latent
        ], axis=-1)
        
        return next_slots
