"""
modSTOVE: Modified STOVE for nextPlayer.

Full model combining:
- ImageEncoder: CNN for 128x128 RGB images
- SlotEncoder: Slot attention for object discovery
- DynamicsModel: GNN for action-conditioned dynamics

Object State Representation (128D per slot):
- Position: 3D (x, y, z) - dims 0-2
- Velocity: 3D (vx, vy, vz) - dims 3-5
- Size: 3D (sx, sy, sz) - dims 6-8
- Latent: 119D appearance features - dims 9-127

Action Space: 5 discrete actions
- FORWARD, STOP, TURN_LEFT, TURN_RIGHT, STOP_TURNING
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Dict, Tuple, Optional, Any, Callable
from functools import partial

from .image_model import ImageEncoder, ImageEncoderWithNorm
from .slot_encoder import SlotEncoder, TemporalSlotEncoder, POS_SLICE, VEL_SLICE, SIZE_SLICE, LATENT_SLICE, NUM_SLOTS, SLOT_DIM
from .dynamics import DynamicsModel, PhysicsInformedDynamics, NUM_ACTIONS
from .decoder import SlotDecoder


class ModSTOVE(nn.Module):
    """
    Modified STOVE: Object-centric world model for action-conditioned prediction.
    
    Combines slot attention for object discovery with graph neural network
    dynamics for predicting how objects evolve under actions.
    
    Architecture:
        1. Image Encoder: 128x128 RGB -> [B, H*W, feature_dim]
        2. Slot Encoder: features -> [B, 7, 128] structured object states
        3. Dynamics Model: (slots, action) -> next_slots
    """
    # Image encoder params
    encoder_hidden_dim: int = 64
    encoder_feature_dim: int = 128
    encoder_output_resolution: int = 32
    
    # Slot encoder params
    num_slots: int = NUM_SLOTS
    slot_dim: int = SLOT_DIM
    slot_iterations: int = 3
    slot_hidden_dim: int = 256
    
    # Dynamics params
    dynamics_hidden_dim: int = 256
    dynamics_action_dim: int = 32
    num_interaction_layers: int = 2
    use_physics_dynamics: bool = False

    # Decoder params
    decoder_hidden_dim: int = 64
    decoder_init_resolution: int = 8
    decoder_output_resolution: int = 128

    # Training params
    reconstruction_weight: float = 1.0
    dynamics_weight: float = 1.0
    
    def setup(self):
        # Image encoder
        self.image_encoder = ImageEncoderWithNorm(
            hidden_dim=self.encoder_hidden_dim,
            feature_dim=self.encoder_feature_dim,
            output_resolution=self.encoder_output_resolution,
            name='image_encoder'
        )
        
        # Slot encoder
        self.slot_encoder = SlotEncoder(
            num_slots=self.num_slots,
            slot_dim=self.slot_dim,
            input_dim=self.encoder_feature_dim,
            num_iterations=self.slot_iterations,
            hidden_dim=self.slot_hidden_dim,
            feature_resolution=self.encoder_output_resolution,
            name='slot_encoder'
        )
        
        # Dynamics model
        if self.use_physics_dynamics:
            self.dynamics = PhysicsInformedDynamics(
                num_slots=self.num_slots,
                slot_dim=self.slot_dim,
                hidden_dim=self.dynamics_hidden_dim,
                action_dim=self.dynamics_action_dim,
                name='dynamics'
            )
        else:
            self.dynamics = DynamicsModel(
                num_slots=self.num_slots,
                slot_dim=self.slot_dim,
                hidden_dim=self.dynamics_hidden_dim,
                action_dim=self.dynamics_action_dim,
                num_interaction_layers=self.num_interaction_layers,
                name='dynamics'
            )

        # Image decoder (spatial-broadcast, slot-attention style)
        self.decoder = SlotDecoder(
            output_resolution=self.decoder_output_resolution,
            init_resolution=self.decoder_init_resolution,
            hidden_dim=self.decoder_hidden_dim,
            name='decoder',
        )
    
    def encode(
        self,
        images: jnp.ndarray,
        prev_slots: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Encode images to slot representations.
        
        Args:
            images: Input images [B, H, W, C] where H=W=128, C=3
            prev_slots: Optional previous slots for temporal consistency
            deterministic: If True, use deterministic slot initialization
            
        Returns:
            Tuple of:
                - Structured slots [B, num_slots, slot_dim]
                - Attention weights [B, num_slots, N]
                - Raw slots [B, num_slots, slot_dim]
        """
        # Extract CNN features
        features = self.image_encoder(images)  # [B, N, feature_dim]
        
        # Slot attention
        slots, attn_weights, raw_slots = self.slot_encoder(
            features,
            prev_slots=prev_slots,
            deterministic=deterministic
        )
        
        return slots, attn_weights, raw_slots
    
    def decode(
        self,
        slots: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Decode slots to a reconstructed image via spatial-broadcast decoder.

        Args:
            slots: [B, num_slots, slot_dim]

        Returns:
            Tuple of:
                - recon: [B, H, W, 3] alpha-composited RGB reconstruction
                - masks: [B, num_slots, H, W, 1] per-slot softmaxed alpha masks
                - rgb:   [B, num_slots, H, W, 3] per-slot RGB predictions
        """
        return self.decoder(slots)

    def predict_next(
        self,
        slots: jnp.ndarray,
        action: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Predict next-step slot states given current slots and action.
        
        Args:
            slots: Current slot states [B, num_slots, slot_dim]
            action: Action [B] (indices) or [B, num_actions] (one-hot)
            deterministic: If True, don't apply dropout
            
        Returns:
            Predicted next slots [B, num_slots, slot_dim]
        """
        return self.dynamics(slots, action, deterministic)
    
    def rollout(
        self,
        initial_slots: jnp.ndarray,
        actions: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Roll out dynamics for multiple timesteps.
        
        Args:
            initial_slots: Initial slot states [B, num_slots, slot_dim]
            actions: Action sequence [B, T] or [B, T, num_actions]
            deterministic: If True, don't apply dropout
            
        Returns:
            Predicted slot sequence [B, T+1, num_slots, slot_dim]
        """
        B = initial_slots.shape[0]
        T = actions.shape[1]
        
        all_slots = [initial_slots]
        current_slots = initial_slots
        
        for t in range(T):
            action_t = actions[:, t]
            next_slots = self.predict_next(current_slots, action_t, deterministic)
            all_slots.append(next_slots)
            current_slots = next_slots
        
        return jnp.stack(all_slots, axis=1)
    
    def __call__(
        self,
        images: jnp.ndarray,
        actions: Optional[jnp.ndarray] = None,
        next_images: Optional[jnp.ndarray] = None,
        prev_slots: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass for training or inference.
        
        Args:
            images: Input images [B, H, W, C]
            actions: Optional actions [B] or [B, num_actions]
            next_images: Optional next-frame images for dynamics supervision
            prev_slots: Optional previous slots
            deterministic: If True, use deterministic mode
            
        Returns:
            Dictionary with:
                - 'slots': Current slot states
                - 'attn_weights': Attention masks
                - 'raw_slots': Raw slot representations
                - 'pred_next_slots': Predicted next slots (if actions provided)
                - 'next_slots': Encoded next slots (if next_images provided)
        """
        results = {}
        
        # Encode current image
        slots, attn_weights, raw_slots = self.encode(
            images, 
            prev_slots=prev_slots,
            deterministic=deterministic
        )
        
        results['slots'] = slots
        results['attn_weights'] = attn_weights
        results['raw_slots'] = raw_slots

        # Decode slots to reconstructed image
        recon, masks, slot_rgb = self.decode(slots)
        results['recon'] = recon
        results['masks'] = masks
        results['slot_rgb'] = slot_rgb

        # Predict next slots if action provided
        if actions is not None:
            pred_next_slots = self.predict_next(slots, actions, deterministic)
            results['pred_next_slots'] = pred_next_slots
        
        # Encode next image if provided (for supervision)
        if next_images is not None:
            next_slots, next_attn, next_raw = self.encode(
                next_images,
                prev_slots=slots,
                deterministic=deterministic
            )
            results['next_slots'] = next_slots
            results['next_attn_weights'] = next_attn
        
        return results
    
    def loss(
        self,
        images: jnp.ndarray,
        actions: jnp.ndarray,
        next_images: jnp.ndarray,
        prev_slots: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
        dyn_weight: float = 1.0,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Compute training loss.

        Reconstruction + dynamics objective (standard slot-attention recipe):
            L = L_recon + dyn_weight * L_dynamics

        where:
            L_recon:    MSE between decoded slots and the input image
            L_dynamics: MSE between predicted and encoded next-frame slots

        Args:
            images: Current images [B, H, W, C]
            actions: Actions [B] or [B, num_actions]
            next_images: Next-frame images [B, H, W, C]
            prev_slots: Optional previous slots
            deterministic: If True, use deterministic mode
            dyn_weight: Scalar weight on the dynamics term (use a warmup
                schedule from the training loop).

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        outputs = self(
            images,
            actions=actions,
            next_images=next_images,
            prev_slots=prev_slots,
            deterministic=deterministic,
        )

        recon = outputs['recon']
        pred_next_slots = outputs['pred_next_slots']
        next_slots = outputs['next_slots']

        recon_loss = jnp.mean((recon - images) ** 2)
        dynamics_loss = jnp.mean((pred_next_slots - next_slots) ** 2)

        total_loss = (
            self.reconstruction_weight * recon_loss
            + dyn_weight * dynamics_loss
        )

        loss_dict = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'dynamics_loss': dynamics_loss,
            'dyn_weight': jnp.asarray(dyn_weight),
        }

        return total_loss, loss_dict


def create_train_state(
    rng: jax.random.PRNGKey,
    model: ModSTOVE,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
) -> train_state.TrainState:
    """
    Create a training state for the model.
    
    Args:
        rng: Random key
        model: ModSTOVE model instance
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        
    Returns:
        Initialized TrainState
    """
    # Initialize with dummy inputs
    dummy_images = jnp.zeros((1, 128, 128, 3))
    dummy_actions = jnp.zeros((1,), dtype=jnp.int32)
    dummy_next = jnp.zeros((1, 128, 128, 3))
    
    rng, init_rng, sample_rng = jax.random.split(rng, 3)
    
    variables = model.init(
        {'params': init_rng, 'sample': sample_rng},
        dummy_images,
        actions=dummy_actions,
        next_images=dummy_next,
        deterministic=False
    )
    
    params = variables['params']
    
    # Optimizer with weight decay
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


@partial(jax.jit, static_argnums=(0,))
def train_step(
    model: ModSTOVE,
    state: train_state.TrainState,
    images: jnp.ndarray,
    actions: jnp.ndarray,
    next_images: jnp.ndarray,
    rng: jax.random.PRNGKey,
    dyn_weight: jnp.ndarray,
) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray]]:
    """
    Single training step.

    Args:
        model: ModSTOVE model
        state: Current training state
        images: Batch of images [B, H, W, C]
        actions: Batch of actions [B]
        next_images: Batch of next images [B, H, W, C]
        rng: Random key
        dyn_weight: Scalar weight on the dynamics term (pass as a JAX scalar
            so it can change across steps without re-jitting).

    Returns:
        Updated state and loss dictionary
    """
    def loss_fn(params):
        loss, loss_dict = model.apply(
            {'params': params},
            images,
            actions=actions,
            next_images=next_images,
            deterministic=False,
            dyn_weight=dyn_weight,
            method=model.loss,
            rngs={'sample': rng}
        )
        return loss, loss_dict

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, loss_dict), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    return state, loss_dict


@partial(jax.jit, static_argnums=(0,))
def eval_step(
    model: ModSTOVE,
    state: train_state.TrainState,
    images: jnp.ndarray,
    actions: jnp.ndarray,
    next_images: jnp.ndarray,
    rng: jax.random.PRNGKey,
    dyn_weight: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """
    Evaluation step.

    Args:
        model: ModSTOVE model
        state: Current training state
        images: Batch of images
        actions: Batch of actions
        next_images: Batch of next images
        rng: Random key
        dyn_weight: Scalar weight on the dynamics term.

    Returns:
        Loss dictionary
    """
    loss, loss_dict = model.apply(
        {'params': state.params},
        images,
        actions=actions,
        next_images=next_images,
        deterministic=True,
        dyn_weight=dyn_weight,
        method=model.loss,
        rngs={'sample': rng}
    )

    return loss_dict


@partial(jax.jit, static_argnums=(0,))
def encode_image(
    model: ModSTOVE,
    params: Dict[str, Any],
    images: jnp.ndarray,
    rng: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Encode images to slot representations (JIT-compiled).
    
    Args:
        model: ModSTOVE model
        params: Model parameters
        images: Input images [B, H, W, C]
        rng: Random key
        
    Returns:
        Tuple of (slots, attention_weights)
    """
    slots, attn, _ = model.apply(
        {'params': params},
        images,
        deterministic=True,
        method=model.encode,
        rngs={'sample': rng}
    )
    return slots, attn


@partial(jax.jit, static_argnums=(0,))
def predict_rollout(
    model: ModSTOVE,
    params: Dict[str, Any],
    initial_slots: jnp.ndarray,
    actions: jnp.ndarray,
) -> jnp.ndarray:
    """
    Predict slot rollout given initial slots and action sequence.
    
    Args:
        model: ModSTOVE model
        params: Model parameters
        initial_slots: Initial slots [B, K, D]
        actions: Action sequence [B, T]
        
    Returns:
        Predicted slots [B, T+1, K, D]
    """
    return model.apply(
        {'params': params},
        initial_slots,
        actions,
        deterministic=True,
        method=model.rollout
    )


def create_modstove(
    num_slots: int = 7,
    use_physics_dynamics: bool = False,
    **kwargs
) -> ModSTOVE:
    """
    Factory function to create a ModSTOVE model.
    
    Args:
        num_slots: Number of object slots
        use_physics_dynamics: If True, use physics-informed dynamics
        **kwargs: Additional model arguments
        
    Returns:
        ModSTOVE model instance
    """
    return ModSTOVE(
        num_slots=num_slots,
        use_physics_dynamics=use_physics_dynamics,
        **kwargs
    )
