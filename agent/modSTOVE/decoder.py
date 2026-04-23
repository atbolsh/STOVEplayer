"""
Slot-Attention spatial-broadcast decoder for modSTOVE.

Standard Locatello et al. (2020) decoder: each slot is broadcast to a small
spatial grid, decoded independently to (R, G, B, alpha), then alpha-composited
across slots to produce the final reconstructed image.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple

from .image_model import SoftPositionEmbed


class SlotDecoder(nn.Module):
    """
    Spatial-broadcast decoder.

    For each slot independently:
      1. Broadcast the slot vector to an 8x8 grid.
      2. Add a soft positional embedding.
      3. Upsample 8 -> 16 -> 32 -> 64 -> 128 with four stride-2 transposed convs.
      4. Final 3x3 conv -> 4 channels: 3 RGB + 1 alpha.

    Across slots, alpha is softmax-normalized and used to alpha-composite the
    per-slot RGB predictions into the final reconstruction.
    """
    output_resolution: int = 128
    init_resolution: int = 8
    hidden_dim: int = 64
    kernel_size: int = 5

    @nn.compact
    def __call__(
        self,
        slots: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Decode slots to a reconstructed image.

        Args:
            slots: [B, K, D] slot vectors.

        Returns:
            recon:  [B, H, W, 3]       alpha-composited RGB reconstruction.
            masks:  [B, K, H, W, 1]    per-slot softmaxed alpha masks.
            rgb:    [B, K, H, W, 3]    per-slot RGB predictions.
        """
        B, K, D = slots.shape
        H0 = W0 = self.init_resolution

        x = slots.reshape(B * K, D)
        x = jnp.broadcast_to(
            x[:, None, None, :],
            (B * K, H0, W0, D),
        )

        x = SoftPositionEmbed(
            hidden_dim=D,
            resolution=(H0, W0),
            name='pos_embed',
        )(x)

        n_upsamples = 0
        res = H0
        while res < self.output_resolution:
            res *= 2
            n_upsamples += 1
        assert res == self.output_resolution, (
            f"output_resolution ({self.output_resolution}) must be "
            f"init_resolution ({self.init_resolution}) * 2^n."
        )

        for i in range(n_upsamples):
            x = nn.ConvTranspose(
                features=self.hidden_dim,
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(2, 2),
                padding='SAME',
                name=f'deconv{i+1}',
            )(x)
            x = nn.relu(x)

        x = nn.Conv(
            features=4,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            name='out_conv',
        )(x)

        x = x.reshape(B, K, self.output_resolution, self.output_resolution, 4)
        rgb = nn.sigmoid(x[..., :3])
        alpha_logits = x[..., 3:4]
        masks = jax.nn.softmax(alpha_logits, axis=1)
        recon = jnp.sum(masks * rgb, axis=1)

        return recon, masks, rgb
