"""
modSTOVE Pretraining Script

Trains modSTOVE on video clips from the BallCatch environment.

Batch generation strategy:
- 50% static batches: no agent movement, pure physics observation
- 50% action batches: one random action injected mid-clip

This teaches modSTOVE both passive observation and action-conditioned prediction.
"""

import os
import glob

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import serialization
from typing import Tuple, NamedTuple, Optional
from tqdm import tqdm
import time

from nextPlayer.environment import BallCatchEnv, EnvParams
from nextPlayer.agent.modSTOVE import ModSTOVE, create_modstove

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


class TrainConfig(NamedTuple):
    """Training configuration."""
    batch_size: int = 128
    clip_length: int = 8
    learning_rate: float = 2e-4
    lr_warmup_steps: int = 1000
    lr_decay_steps: int = 100000
    weight_decay: float = 0.01
    max_steps: int = 100000
    log_interval: int = 100
    num_balls: int = 5
    seed: int = 42


class BatchData(NamedTuple):
    """A batch of video clips with optional actions."""
    images: jnp.ndarray      # [batch, clip_length, 128, 128, 3]
    actions: jnp.ndarray     # [batch, clip_length]
    action_mask: jnp.ndarray # [batch, clip_length] - 1 where action applied
    is_static: jnp.ndarray   # [batch] - 1 if static batch, 0 if action batch


def save_checkpoint(state: train_state.TrainState, step: int,
                    config: TrainConfig) -> str:
    """Save a checkpoint and return the file path."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    prefix = f"modstove_b{config.num_balls}_lr{config.learning_rate:.0e}"
    path = os.path.join(CHECKPOINT_DIR, f"{prefix}_step{step:07d}.ckpt")
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))
    return path


def cleanup_checkpoints(step: int, config: TrainConfig) -> None:
    """Apply logarithmic retention: after saving at `step`, delete old checkpoints
    that no longer fall on a retention boundary.

    Retention tiers (checked from coarsest to finest):
      - Every 100_000 steps: always kept
      - Every  10_000 steps: kept while step < 100_000
      - Every   1_000 steps: kept while step <  10_000
      - Every     100 steps: kept while step <   1_000
    Anything that doesn't match a tier for its range is deleted.
    """
    prefix = f"modstove_b{config.num_balls}_lr{config.learning_rate:.0e}"
    pattern = os.path.join(CHECKPOINT_DIR, f"{prefix}_step*.ckpt")

    for path in glob.glob(pattern):
        fname = os.path.basename(path)
        try:
            ckpt_step = int(fname.split("_step")[1].split(".ckpt")[0])
        except (IndexError, ValueError):
            continue

        if ckpt_step == step:
            continue

        keep = False
        if ckpt_step % 100_000 == 0:
            keep = True
        elif ckpt_step % 10_000 == 0 and step < 100_000:
            keep = True
        elif ckpt_step % 1_000 == 0 and step < 10_000:
            keep = True
        elif ckpt_step % 100 == 0 and step < 1_000:
            keep = True

        if not keep:
            os.remove(path)


def load_latest_checkpoint(state: train_state.TrainState,
                           config: TrainConfig) -> Tuple[train_state.TrainState, int]:
    """Load the most recent checkpoint if one exists.

    Returns:
        (restored_state, step) or (original_state, 0) if no checkpoint found.
    """
    prefix = f"modstove_b{config.num_balls}_lr{config.learning_rate:.0e}"
    pattern = os.path.join(CHECKPOINT_DIR, f"{prefix}_step*.ckpt")
    files = sorted(glob.glob(pattern))
    if not files:
        return state, 0

    latest = files[-1]
    fname = os.path.basename(latest)
    ckpt_step = int(fname.split("_step")[1].split(".ckpt")[0])

    with open(latest, "rb") as f:
        state = serialization.from_bytes(state, f.read())

    print(f"Resumed from checkpoint: {fname} (step {ckpt_step})")
    return state, ckpt_step


def create_optimizer(config: TrainConfig):
    """Create optimizer with warmup and decay."""
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.lr_warmup_steps,
        decay_steps=config.lr_decay_steps,
        end_value=config.learning_rate * 0.1,
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=config.weight_decay),
    )


def generate_static_clip(
    key: jax.random.PRNGKey,
    env: BallCatchEnv,
    clip_length: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate a static clip with no agent movement.
    
    Returns:
        images: [clip_length, 128, 128, 3]
        actions: [clip_length] all zeros (no action)
    """
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    
    images = [obs]
    actions = []
    
    for t in range(clip_length - 1):
        key, step_key = jax.random.split(key)
        # Action 1 = STOP (no movement)
        action = 1
        obs, state, _, _, _ = env.step(step_key, state, action)
        images.append(obs)
        actions.append(action)
    
    # Add final action (not used, just for shape consistency)
    actions.append(1)
    
    return jnp.stack(images), jnp.array(actions)


def generate_action_clip(
    key: jax.random.PRNGKey,
    env: BallCatchEnv,
    clip_length: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate a clip with one random action in the middle.
    
    Returns:
        images: [clip_length, 128, 128, 3]
        actions: [clip_length]
        action_mask: [clip_length] - 1 at the action frame
    """
    key, reset_key, action_key = jax.random.split(key, 3)
    obs, state = env.reset(reset_key)
    
    # Pick action frame (middle-ish)
    action_frame = clip_length // 2
    # Pick random action (0-4)
    random_action = jax.random.randint(action_key, (), 0, 5)
    
    images = [obs]
    actions = []
    action_mask = []
    
    for t in range(clip_length - 1):
        key, step_key = jax.random.split(key)
        
        if t == action_frame:
            action = random_action
            mask = 1
        else:
            action = 1  # STOP
            mask = 0
        
        obs, state, _, _, _ = env.step(step_key, state, action)
        images.append(obs)
        actions.append(action)
        action_mask.append(mask)
    
    # Final entries
    actions.append(1)
    action_mask.append(0)
    
    return jnp.stack(images), jnp.array(actions), jnp.array(action_mask)


def generate_batch(
    key: jax.random.PRNGKey,
    env: BallCatchEnv,
    batch_size: int,
    clip_length: int,
) -> BatchData:
    """
    Generate a batch of clips.
    
    50% static (no agent movement)
    50% with one random action mid-clip
    """
    keys = jax.random.split(key, batch_size + 1)
    key, batch_keys = keys[0], keys[1:]
    
    all_images = []
    all_actions = []
    all_masks = []
    all_is_static = []
    
    for i, k in enumerate(batch_keys):
        if i < batch_size // 2:
            # Static batch
            images, actions = generate_static_clip(k, env, clip_length)
            mask = jnp.zeros(clip_length)
            is_static = 1
        else:
            # Action batch
            images, actions, mask = generate_action_clip(k, env, clip_length)
            is_static = 0
        
        all_images.append(images)
        all_actions.append(actions)
        all_masks.append(mask)
        all_is_static.append(is_static)
    
    return BatchData(
        images=jnp.stack(all_images),
        actions=jnp.stack(all_actions),
        action_mask=jnp.stack(all_masks),
        is_static=jnp.array(all_is_static),
    )


def compute_loss(
    params,
    model: ModSTOVE,
    batch: BatchData,
    key: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, dict]:
    """
    Compute modSTOVE loss on a batch.
    
    Returns:
        loss: scalar
        metrics: dict of additional metrics
    """
    batch_size, clip_length = batch.images.shape[:2]
    
    # Split RNG for encode and dynamics calls
    key_encode, key_dynamics = jax.random.split(key)
    
    # Encode all frames
    # Reshape to [batch * clip_length, 128, 128, 3]
    flat_images = batch.images.reshape(-1, 128, 128, 3)
    
    # Get slots for all frames
    slots, attn, raw_slots = model.apply(
        {'params': params},
        flat_images,
        method=model.encode,
        rngs={'sample': key_encode},
    )
    
    # Reshape back to [batch, clip_length, num_slots, slot_dim]
    slots = slots.reshape(batch_size, clip_length, -1, slots.shape[-1])
    
    # Dynamics prediction loss: predict frame t+1 from frame t
    pred_losses = []
    
    for t in range(clip_length - 1):
        current_slots = slots[:, t]  # [batch, num_slots, slot_dim]
        next_slots = slots[:, t + 1]  # [batch, num_slots, slot_dim]
        action = batch.actions[:, t]  # [batch]
        
        # One-hot encode action
        action_onehot = jax.nn.one_hot(action, 5)  # [batch, 5]
        
        # Predict next slots
        key_dynamics, subkey = jax.random.split(key_dynamics)
        pred_next = model.apply(
            {'params': params},
            current_slots,
            action_onehot,
            method=model.predict_next,
            rngs={'sample': subkey},
        )
        
        # MSE loss on slot predictions
        slot_loss = jnp.mean((pred_next - next_slots) ** 2)
        pred_losses.append(slot_loss)
    
    dynamics_loss = jnp.mean(jnp.stack(pred_losses))
    
    # Image reconstruction loss (optional, via ELBO)
    # For now, just use dynamics loss
    total_loss = dynamics_loss
    
    metrics = {
        'dynamics_loss': dynamics_loss,
        'total_loss': total_loss,
    }
    
    return total_loss, metrics


@jax.jit(static_argnames=("model",))
def train_step(
    state: train_state.TrainState,
    batch: BatchData,
    key: jax.random.PRNGKey,
    *,
    model: ModSTOVE,
) -> Tuple[train_state.TrainState, dict]:
    """Single training step."""
    
    def loss_fn(params):
        return compute_loss(params, model, batch, key)
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, metrics


def train(config: TrainConfig):
    """Main training loop."""
    print("=" * 60)
    print("modSTOVE Pretraining")
    print("=" * 60)
    print(f"Config: {config}")
    
    # Initialize RNG
    key = jax.random.PRNGKey(config.seed)
    
    # Create environment with JIT-compiled step/reset
    env = BallCatchEnv(EnvParams(num_balls=config.num_balls))
    env.step = jax.jit(env.step)
    env.reset = jax.jit(env.reset)
    print(f"Environment created with {config.num_balls} balls")
    
    # Create model
    key, init_key, sample_key = jax.random.split(key, 3)
    model = create_modstove()
    
    # Initialize ALL model parameters (encoder + dynamics) via __call__
    dummy_images = jnp.zeros((1, 128, 128, 3))
    dummy_actions = jnp.zeros((1,), dtype=jnp.int32)
    dummy_next = jnp.zeros((1, 128, 128, 3))
    variables = model.init(
        {'params': init_key, 'sample': sample_key},
        dummy_images,
        actions=dummy_actions,
        next_images=dummy_next,
        deterministic=False,
    )
    params = variables['params']
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {param_count:,}")
    
    # Create optimizer and training state
    optimizer = create_optimizer(config)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )
    
    # Resume from checkpoint if available
    state, start_step = load_latest_checkpoint(state, config)
    
    # CUDA warmup: run one dummy step so the first real step isn't penalized
    print("Warming up CUDA kernels...")
    key, warmup_batch_key, warmup_step_key = jax.random.split(key, 3)
    warmup_batch = generate_batch(warmup_batch_key, env, config.batch_size, config.clip_length)
    _state, _metrics = train_step(state, warmup_batch, warmup_step_key, model=model)
    jax.block_until_ready(_metrics)
    del _state, _metrics, warmup_batch

    # Training loop
    print(f"\nStarting training from step {start_step}...")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    start_time = time.time()
    
    for step in tqdm(range(start_step, config.max_steps), desc="Training",
                     initial=start_step, total=config.max_steps):
        # Generate batch
        key, batch_key, step_key = jax.random.split(key, 3)
        batch = generate_batch(batch_key, env, config.batch_size, config.clip_length)
        
        # Train step
        state, metrics = train_step(state, batch, step_key, model=model)
        
        # Logging
        if step % config.log_interval == 0:
            elapsed = time.time() - start_time
            steps_done = step - start_step
            steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
            tqdm.write(
                f"Step {step:6d} | "
                f"Loss: {metrics['total_loss']:.4f} | "
                f"Dynamics: {metrics['dynamics_loss']:.4f} | "
                f"Steps/s: {steps_per_sec:.1f}"
            )
        
        # Save checkpoint every 100 steps, then prune old ones
        if step > 0 and step % 100 == 0:
            path = save_checkpoint(state, step, config)
            cleanup_checkpoints(step, config)
            if step % 1000 == 0:
                tqdm.write(f"Checkpoint saved: {os.path.basename(path)}")
    
    # Final checkpoint
    path = save_checkpoint(state, config.max_steps, config)
    print(f"\nTraining complete! Final checkpoint: {os.path.basename(path)}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    
    return state, model


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pretrain modSTOVE")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--clip-length', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max-steps', type=int, default=100000)
    parser.add_argument('--num-balls', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    config = TrainConfig(
        batch_size=args.batch_size,
        clip_length=args.clip_length,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        num_balls=args.num_balls,
        seed=args.seed,
    )
    
    train(config)


if __name__ == '__main__':
    main()
