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
import numpy as np
from flax.training import train_state
from flax import serialization
from typing import Tuple, NamedTuple, Optional
from tqdm import tqdm
import time

from PIL import Image, ImageDraw

from nextPlayer.environment import BallCatchEnv, EnvParams
from nextPlayer.agent.modSTOVE import ModSTOVE, create_modstove

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")

# Distinct colors for bounding-box overlays (one per slot).
_BBOX_COLORS = [
    (230,  25,  75),
    ( 60, 180,  75),
    (255, 225,  25),
    (  0, 130, 200),
    (245, 130,  48),
    (145,  30, 180),
    ( 70, 240, 240),
    (240,  50, 230),
    (210, 245,  60),
    (250, 190, 212),
]

# Seed for the deterministic sample clip used to dump debug images. Keeping
# this fixed (and separate from config.seed) means the same frames get
# visualized at every checkpoint, so you can scrub reconstructions over time.
_SAMPLE_IMAGE_SEED = 20240423


class TrainConfig(NamedTuple):
    """Training configuration."""
    batch_size: int = 64
    clip_length: int = 8
    learning_rate: float = 2e-4
    lr_warmup_steps: int = 1000
    lr_decay_steps: int = 100000
    weight_decay: float = 0.01
    max_steps: int = 100000
    log_interval: int = 100
    num_balls: int = 5
    seed: int = 42
    # Dynamics-loss warmup: recon only for the first `dyn_warmup_start` steps,
    # then linearly ramp dyn weight 0 -> 1 over `dyn_warmup_ramp` steps.
    dyn_warmup_start: int = 2000
    dyn_warmup_ramp: int = 3000
    # Run identity: lets you start a fresh namespace (e.g. "v2") without
    # colliding with previous runs' checkpoints.
    run_name: str = "v1"


class BatchData(NamedTuple):
    """A batch of video clips with optional actions."""
    images: jnp.ndarray      # [batch, clip_length, 128, 128, 3]
    actions: jnp.ndarray     # [batch, clip_length]
    action_mask: jnp.ndarray # [batch, clip_length] - 1 where action applied
    is_static: jnp.ndarray   # [batch] - 1 if static batch, 0 if action batch


def checkpoint_prefix(config: TrainConfig) -> str:
    """Filename prefix for all checkpoints of a given run."""
    return (
        f"modstove_{config.run_name}"
        f"_b{config.num_balls}_lr{config.learning_rate:.0e}"
    )


def save_checkpoint(state: train_state.TrainState, step: int,
                    config: TrainConfig) -> str:
    """Save a checkpoint and return the file path."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    prefix = checkpoint_prefix(config)
    path = os.path.join(CHECKPOINT_DIR, f"{prefix}_step{step:07d}.ckpt")
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))
    return path


def _is_permanent_checkpoint(ckpt_step: int) -> bool:
    """Whether a given step is on a *permanent* retention boundary.

    These checkpoints are never deleted, regardless of training progress:
      - steps 100, 200, ..., 1_000          (every 100 up to 1k)
      - steps 1_000, 2_000, ..., 10_000     (every 1k up to 10k)
      - steps 10_000, ..., 100_000          (every 10k up to 100k)
      - every 100_000 beyond that
    """
    if ckpt_step <= 0:
        return False
    if ckpt_step <= 1_000 and ckpt_step % 100 == 0:
        return True
    if ckpt_step <= 10_000 and ckpt_step % 1_000 == 0:
        return True
    if ckpt_step <= 100_000 and ckpt_step % 10_000 == 0:
        return True
    if ckpt_step % 100_000 == 0:
        return True
    return False


def cleanup_checkpoints(step: int, config: TrainConfig) -> None:
    """Prune old checkpoints that aren't on a permanent retention boundary.

    Permanent boundaries (see `_is_permanent_checkpoint`) are always kept.
    The checkpoint that was just saved (`step`) is also always kept.
    Everything else is deleted so the checkpoint directory doesn't grow
    without bound.
    """
    prefix = checkpoint_prefix(config)
    pattern = os.path.join(CHECKPOINT_DIR, f"{prefix}_step*.ckpt")

    for path in glob.glob(pattern):
        fname = os.path.basename(path)
        try:
            ckpt_step = int(fname.split("_step")[1].split(".ckpt")[0])
        except (IndexError, ValueError):
            continue

        if ckpt_step == step:
            continue
        if _is_permanent_checkpoint(ckpt_step):
            continue

        os.remove(path)


def load_latest_checkpoint(state: train_state.TrainState,
                           config: TrainConfig) -> Tuple[train_state.TrainState, int]:
    """Load the most recent checkpoint for this run if one exists.

    Returns:
        (restored_state, step) or (original_state, 0) if no checkpoint found.
    """
    prefix = checkpoint_prefix(config)
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


def build_sample_images(env: BallCatchEnv, clip_length: int) -> jnp.ndarray:
    """Generate a single deterministic clip of `clip_length` frames for debug dumps.

    Uses a fixed seed so the same scene is visualized at every checkpoint.
    Agent is held still (STOP) so the only motion is ball physics.
    """
    key = jax.random.PRNGKey(_SAMPLE_IMAGE_SEED)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)

    frames = [obs]
    for _ in range(clip_length - 1):
        key, step_key = jax.random.split(key)
        obs, state, _, _, _ = env.step(step_key, state, 1)  # 1 = STOP
        frames.append(obs)

    return jnp.stack(frames)  # [clip_length, 128, 128, 3]


def _to_uint8_image(img: np.ndarray) -> np.ndarray:
    """Clip a float image in [0, 1] and convert to uint8 RGB."""
    arr = np.clip(np.asarray(img), 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _mask_bbox(mask: np.ndarray, threshold: float = 0.5) -> Optional[Tuple[int, int, int, int]]:
    """Return (x0, y0, x1, y1) for pixels above `threshold` in a 2D mask, or None."""
    active = mask > threshold
    if not active.any():
        return None
    rows = np.where(active.any(axis=1))[0]
    cols = np.where(active.any(axis=0))[0]
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


def dump_sample_images(
    state: train_state.TrainState,
    model: ModSTOVE,
    sample_clip: jnp.ndarray,
    step: int,
    ckpt_basename: str,
) -> None:
    """Encode/decode a single deterministic clip and write debug images.

    `sample_clip` is a `[T, 128, 128, 3]` clip from `build_sample_images`.
    Slots are encoded frame-by-frame with `prev_slots` threaded through (same
    as training) so the saved reconstructions reflect what the model actually
    sees during dynamics rollouts.

    Writes three subdirectories per checkpoint:
      - original_images/frame_{t}.png
      - reconstructions/frame_{t}.png
      - bounding_boxes/frame_{t}.png   (original with one box per visible slot)
    """
    out_dir = os.path.join(IMAGES_DIR, ckpt_basename)
    orig_dir = os.path.join(out_dir, "original_images")
    recon_dir = os.path.join(out_dir, "reconstructions")
    bbox_dir = os.path.join(out_dir, "bounding_boxes")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)

    T = sample_clip.shape[0]
    key = jax.random.PRNGKey(_SAMPLE_IMAGE_SEED + step)

    prev_slots = None
    recon_frames = []
    masks_frames = []
    for t in range(T):
        key, key_enc, key_dec = jax.random.split(key, 3)
        slots_t, _attn_t, _raw_t = model.apply(
            {'params': state.params},
            sample_clip[t:t + 1],
            prev_slots=prev_slots,
            deterministic=True,
            method=model.encode,
            rngs={'sample': key_enc},
        )
        recon_t, masks_t, _slot_rgb_t = model.apply(
            {'params': state.params},
            slots_t,
            method=model.decode,
            rngs={'sample': key_dec},
        )
        recon_frames.append(recon_t[0])
        masks_frames.append(masks_t[0])
        prev_slots = slots_t

    orig_np = np.asarray(sample_clip)
    recon_np = np.asarray(jnp.stack(recon_frames, axis=0))
    masks_np = np.asarray(jnp.stack(masks_frames, axis=0))[..., 0]  # [T, K, H, W]

    num_slots = masks_np.shape[1]

    for t in range(T):
        orig_img = Image.fromarray(_to_uint8_image(orig_np[t]))
        recon_img = Image.fromarray(_to_uint8_image(recon_np[t]))
        orig_img.save(os.path.join(orig_dir, f"frame_{t}.png"))
        recon_img.save(os.path.join(recon_dir, f"frame_{t}.png"))

        bbox_img = orig_img.copy()
        draw = ImageDraw.Draw(bbox_img)
        for k in range(num_slots):
            box = _mask_bbox(masks_np[t, k])
            if box is None:
                continue
            color = _BBOX_COLORS[k % len(_BBOX_COLORS)]
            draw.rectangle(box, outline=color, width=2)
        bbox_img.save(os.path.join(bbox_dir, f"frame_{t}.png"))


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
    dyn_weight: jnp.ndarray,
) -> Tuple[jnp.ndarray, dict]:
    """
    Compute modSTOVE loss on a batch.

    Per-clip objective:
        L = L_recon + dyn_weight * L_dynamics

    where
        L_recon    = mean MSE between decoded slots and each input frame
        L_dynamics = mean MSE between predicted next slots and encoded next slots

    Slot identity is carried across frames by feeding each frame's encoded
    slots in as `prev_slots` for the next frame (standard temporal
    slot-attention initialization).
    """
    batch_size, clip_length = batch.images.shape[:2]

    # Encode frame-by-frame so we can thread prev_slots across time.
    all_slots = []
    all_recons = []
    prev_slots = None

    for t in range(clip_length):
        key, subkey = jax.random.split(key)
        slots_t, _attn_t, _raw_t = model.apply(
            {'params': params},
            batch.images[:, t],
            prev_slots=prev_slots,
            deterministic=False,
            method=model.encode,
            rngs={'sample': subkey},
        )

        key, subkey = jax.random.split(key)
        recon_t, _masks_t, _slot_rgb_t = model.apply(
            {'params': params},
            slots_t,
            method=model.decode,
            rngs={'sample': subkey},
        )

        all_slots.append(slots_t)
        all_recons.append(recon_t)
        prev_slots = slots_t

    slots = jnp.stack(all_slots, axis=1)   # [B, T, K, D]
    recon = jnp.stack(all_recons, axis=1)  # [B, T, H, W, 3]

    recon_loss = jnp.mean((recon - batch.images) ** 2)

    pred_losses = []
    for t in range(clip_length - 1):
        current_slots = slots[:, t]
        next_slots = slots[:, t + 1]
        action_onehot = jax.nn.one_hot(batch.actions[:, t], 5)

        key, subkey = jax.random.split(key)
        pred_next = model.apply(
            {'params': params},
            current_slots,
            action_onehot,
            method=model.predict_next,
            rngs={'sample': subkey},
        )
        pred_losses.append(jnp.mean((pred_next - next_slots) ** 2))

    dynamics_loss = jnp.mean(jnp.stack(pred_losses))

    total_loss = recon_loss + dyn_weight * dynamics_loss

    metrics = {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'dynamics_loss': dynamics_loss,
        'dyn_weight': dyn_weight,
    }

    return total_loss, metrics


@jax.jit(static_argnames=("model",))
def train_step(
    state: train_state.TrainState,
    batch: BatchData,
    key: jax.random.PRNGKey,
    dyn_weight: jnp.ndarray,
    *,
    model: ModSTOVE,
) -> Tuple[train_state.TrainState, dict]:
    """Single training step."""

    def loss_fn(params):
        return compute_loss(params, model, batch, key, dyn_weight)

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, metrics


def dyn_weight_at(step: int, config: TrainConfig) -> float:
    """Warmup schedule for the dynamics-loss weight.

    0 for the first `dyn_warmup_start` steps, then linearly ramps to 1.0
    over `dyn_warmup_ramp` steps. This lets the encoder+decoder converge
    to meaningful reconstructions before the dynamics objective starts
    pulling on the latent states.
    """
    start = config.dyn_warmup_start
    ramp = max(1, config.dyn_warmup_ramp)
    if step < start:
        return 0.0
    if step >= start + ramp:
        return 1.0
    return float(step - start) / float(ramp)


def train(config: TrainConfig, fresh: bool = False):
    """Main training loop.

    Args:
        config: Training configuration.
        fresh: If True, ignore any existing checkpoints for this run-name and
            start from step 0. (Does not delete them; use purge_checkpoints.py
            for that.)
    """
    print("=" * 60)
    print("modSTOVE Pretraining")
    print("=" * 60)
    print(f"Config: {config}")
    print(f"Fresh start: {fresh}")
    
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
    
    # Resume from checkpoint if available (unless --fresh was passed)
    if fresh:
        print(f"--fresh: ignoring any existing checkpoints for run '{config.run_name}'")
        start_step = 0
    else:
        state, start_step = load_latest_checkpoint(state, config)
    
    # CUDA warmup: run one dummy step so the first real step isn't penalized
    print("Warming up CUDA kernels...")
    key, warmup_batch_key, warmup_step_key = jax.random.split(key, 3)
    warmup_batch = generate_batch(warmup_batch_key, env, config.batch_size, config.clip_length)
    _state, _metrics = train_step(
        state, warmup_batch, warmup_step_key, jnp.float32(0.0), model=model,
    )
    jax.block_until_ready(_metrics)
    del _state, _metrics, warmup_batch

    # Build the deterministic sample clip used for checkpoint image dumps.
    sample_clip = build_sample_images(env, config.clip_length)

    # Training loop
    print(f"\nStarting training from step {start_step}...")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print(f"Sample images: {IMAGES_DIR}")
    start_time = time.time()
    
    for step in tqdm(range(start_step, config.max_steps), desc="Training",
                     initial=start_step, total=config.max_steps):
        # Generate batch
        key, batch_key, step_key = jax.random.split(key, 3)
        batch = generate_batch(batch_key, env, config.batch_size, config.clip_length)

        # Train step
        dw = jnp.float32(dyn_weight_at(step, config))
        state, metrics = train_step(state, batch, step_key, dw, model=model)

        # Logging
        if step % config.log_interval == 0:
            elapsed = time.time() - start_time
            steps_done = step - start_step
            steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
            tqdm.write(
                f"Step {step:6d} | "
                f"Loss: {metrics['total_loss']:.4f} | "
                f"Recon: {metrics['recon_loss']:.4f} | "
                f"Dyn: {metrics['dynamics_loss']:.4f} | "
                f"\u03bb_dyn: {float(metrics['dyn_weight']):.3f} | "
                f"Steps/s: {steps_per_sec:.1f}"
            )
        
        # Save checkpoint every 100 steps, then prune old ones and dump debug
        # images alongside the checkpoint we just wrote.
        if step > 0 and step % 100 == 0:
            path = save_checkpoint(state, step, config)
            cleanup_checkpoints(step, config)
            ckpt_basename = os.path.splitext(os.path.basename(path))[0]
            dump_sample_images(state, model, sample_clip, step, ckpt_basename)
            if step % 1000 == 0:
                tqdm.write(f"Checkpoint saved: {os.path.basename(path)}")

    # Final checkpoint
    path = save_checkpoint(state, config.max_steps, config)
    ckpt_basename = os.path.splitext(os.path.basename(path))[0]
    dump_sample_images(state, model, sample_clip, config.max_steps, ckpt_basename)
    print(f"\nTraining complete! Final checkpoint: {os.path.basename(path)}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    
    return state, model


def main():
    """Entry point."""
    import argparse

    _defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Pretrain modSTOVE")
    parser.add_argument('--batch-size', type=int, default=_defaults.batch_size)
    parser.add_argument('--clip-length', type=int, default=_defaults.clip_length)
    parser.add_argument('--lr', type=float, default=_defaults.learning_rate)
    parser.add_argument('--max-steps', type=int, default=_defaults.max_steps)
    parser.add_argument('--num-balls', type=int, default=_defaults.num_balls)
    parser.add_argument('--seed', type=int, default=_defaults.seed)
    parser.add_argument(
        '--run-name',
        type=str,
        default=_defaults.run_name,
        help="Identifier appended to checkpoint filenames (default: %(default)s). "
             "Use a new value (e.g. 'v2') to start a fresh checkpoint namespace.",
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help="Ignore any existing checkpoints for this --run-name and start "
             "from step 0. Does not delete them; use purge_checkpoints.py.",
    )
    args = parser.parse_args()

    config = TrainConfig(
        batch_size=args.batch_size,
        clip_length=args.clip_length,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        num_balls=args.num_balls,
        seed=args.seed,
        run_name=args.run_name,
    )

    train(config, fresh=args.fresh)


if __name__ == '__main__':
    main()
