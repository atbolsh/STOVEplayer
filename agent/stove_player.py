"""
STOVEplayer: SwiGLU-based LLM wrapper for agent control.

Combines modSTOVE object representations with a transformer-based
text encoder/decoder for action selection and reasoning.

Architecture:
- SwiGLU Text Encoder (8 layers, 6 heads, 768D)
- SwiGLU Text Decoder (4 layers, 6 heads, 768D)
- Additive slot projection (physics + content -> 768D)
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple, Any
import math


class SwiGLU(nn.Module):
    """SwiGLU activation: SiLU(xW1) * (xW3) @ W2"""
    hidden_dim: int
    out_dim: int
    
    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.hidden_dim, use_bias=False, name='gate')(x)
        gate = nn.silu(gate)
        up = nn.Dense(self.hidden_dim, use_bias=False, name='up')(x)
        return nn.Dense(self.out_dim, use_bias=False, name='down')(gate * up)


class SwiGLUEncoderLayer(nn.Module):
    """Transformer encoder layer with SwiGLU feedforward."""
    embed_dim: int = 768
    num_heads: int = 6
    ff_dim: int = 3072  # 4x embed_dim
    dropout: float = 0.1
    
    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        # Pre-norm self-attention
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            dropout_rate=self.dropout,
            deterministic=deterministic,
        )(x, x, mask=mask)
        x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)
        x = residual + x
        
        # Pre-norm SwiGLU FFN
        residual = x
        x = nn.LayerNorm()(x)
        x = SwiGLU(hidden_dim=self.ff_dim, out_dim=self.embed_dim)(x)
        x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)
        x = residual + x
        
        return x


class SwiGLUDecoderLayer(nn.Module):
    """Transformer decoder layer with SwiGLU feedforward and cross-attention."""
    embed_dim: int = 768
    num_heads: int = 6
    ff_dim: int = 3072
    dropout: float = 0.1
    
    @nn.compact
    def __call__(self, x, context, mask=None, deterministic=True):
        # Pre-norm causal self-attention
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            dropout_rate=self.dropout,
            deterministic=deterministic,
        )(x, x, mask=mask)
        x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)
        x = residual + x
        
        # Pre-norm cross-attention to context (slot embeddings)
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            dropout_rate=self.dropout,
            deterministic=deterministic,
        )(x, context)
        x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)
        x = residual + x
        
        # Pre-norm SwiGLU FFN
        residual = x
        x = nn.LayerNorm()(x)
        x = SwiGLU(hidden_dim=self.ff_dim, out_dim=self.embed_dim)(x)
        x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)
        x = residual + x
        
        return x


class SlotProjection(nn.Module):
    """
    Project modSTOVE slots (128D) to LLM embedding space (768D).
    
    Uses additive projection like positional embeddings:
    - Physics (9D: pos + vel + size) -> 768D
    - Content (119D: latent) -> 768D
    - Final = physics_emb + content_emb
    
    Both use variance-normalized initialization (Xavier) for equal contribution.
    """
    embed_dim: int = 768
    physics_dim: int = 9   # position (3) + velocity (3) + size (3)
    content_dim: int = 119  # latent appearance
    
    @nn.compact
    def __call__(self, slots):
        """
        Args:
            slots: [batch, num_slots, 128] modSTOVE output
            
        Returns:
            slot_embeddings: [batch, num_slots, embed_dim]
        """
        # Split physics and content
        physics = slots[..., :self.physics_dim]   # [batch, 7, 9]
        content = slots[..., self.physics_dim:]   # [batch, 7, 119]
        
        # Project both to full embed_dim with Xavier init (variance normalized)
        # Xavier: std = 1/sqrt(fan_in), so output variance ≈ 1 for both
        physics_emb = nn.Dense(
            self.embed_dim,
            kernel_init=nn.initializers.variance_scaling(
                1.0, 'fan_in', 'truncated_normal'
            ),
            name='physics_proj'
        )(physics)
        
        content_emb = nn.Dense(
            self.embed_dim,
            kernel_init=nn.initializers.variance_scaling(
                1.0, 'fan_in', 'truncated_normal'
            ),
            name='content_proj'
        )(content)
        
        # Additive combination (like positional + token embeddings)
        return physics_emb + content_emb


class SwiGLUTextEncoder(nn.Module):
    """SwiGLU-based transformer encoder for text."""
    vocab_size: int = 32000
    embed_dim: int = 768
    num_heads: int = 6
    num_layers: int = 8
    ff_dim: int = 3072
    max_seq_len: int = 128
    dropout: float = 0.1
    
    @nn.compact
    def __call__(self, tokens, deterministic=True):
        """
        Args:
            tokens: [batch, seq_len] token IDs
            
        Returns:
            encodings: [batch, seq_len, embed_dim]
        """
        batch, seq_len = tokens.shape
        
        # Token embeddings
        x = nn.Embed(self.vocab_size, self.embed_dim)(tokens)
        
        # Positional embeddings (learned)
        positions = jnp.arange(seq_len)
        pos_emb = nn.Embed(self.max_seq_len, self.embed_dim, name='pos_embed')(positions)
        x = x + pos_emb
        
        # Dropout
        x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)
        
        # Encoder layers
        for i in range(self.num_layers):
            x = SwiGLUEncoderLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
                name=f'encoder_layer_{i}'
            )(x, deterministic=deterministic)
        
        # Final layer norm
        x = nn.LayerNorm()(x)
        
        return x


class SwiGLUTextDecoder(nn.Module):
    """SwiGLU-based transformer decoder with slot context."""
    vocab_size: int = 32000
    embed_dim: int = 768
    num_heads: int = 6
    num_layers: int = 4
    ff_dim: int = 3072
    max_seq_len: int = 128
    dropout: float = 0.1
    
    def _make_causal_mask(self, seq_len):
        """Create causal attention mask."""
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        return mask
    
    @nn.compact
    def __call__(self, tokens, context, deterministic=True):
        """
        Args:
            tokens: [batch, seq_len] token IDs
            context: [batch, num_slots, embed_dim] slot embeddings
            
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch, seq_len = tokens.shape
        
        # Token embeddings
        x = nn.Embed(self.vocab_size, self.embed_dim)(tokens)
        
        # Positional embeddings (learned)
        positions = jnp.arange(seq_len)
        pos_emb = nn.Embed(self.max_seq_len, self.embed_dim, name='pos_embed')(positions)
        x = x + pos_emb
        
        # Dropout
        x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)
        
        # Causal mask for self-attention
        causal_mask = self._make_causal_mask(seq_len)
        
        # Decoder layers
        for i in range(self.num_layers):
            x = SwiGLUDecoderLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
                name=f'decoder_layer_{i}'
            )(x, context, mask=causal_mask, deterministic=deterministic)
        
        # Final layer norm
        x = nn.LayerNorm()(x)
        
        # Output projection to vocab
        logits = nn.Dense(self.vocab_size, name='output_proj')(x)
        
        return logits


class STOVEplayer(nn.Module):
    """
    Full agent combining modSTOVE perception with SwiGLU LLM.
    
    Architecture:
    - SlotProjection: modSTOVE slots (128D) -> LLM embeddings (768D)
    - SwiGLUTextEncoder: text input -> encodings
    - SwiGLUTextDecoder: (encodings, slot_context) -> output logits
    
    The slot context is prepended to the decoder's cross-attention,
    allowing the LLM to condition on the visual scene understanding.
    """
    vocab_size: int = 32000
    embed_dim: int = 768
    num_heads: int = 6
    encoder_layers: int = 8
    decoder_layers: int = 4
    ff_dim: int = 3072
    max_seq_len: int = 128
    num_slots: int = 7
    slot_dim: int = 128
    dropout: float = 0.1
    
    @nn.compact
    def __call__(
        self,
        tokens: jnp.ndarray,
        slots: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass.
        
        Args:
            tokens: [batch, seq_len] input token IDs
            slots: [batch, num_slots, slot_dim] modSTOVE output
            
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Project slots to embedding space
        slot_context = SlotProjection(
            embed_dim=self.embed_dim,
            name='slot_projection'
        )(slots)  # [batch, num_slots, embed_dim]
        
        # Encode text
        text_encoding = SwiGLUTextEncoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.encoder_layers,
            ff_dim=self.ff_dim,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            name='text_encoder'
        )(tokens, deterministic=deterministic)
        
        # Combine slot context with text encoding for decoder context
        # Prepend slots so they're always available for cross-attention
        combined_context = jnp.concatenate([slot_context, text_encoding], axis=1)
        
        # Decode with context
        logits = SwiGLUTextDecoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.decoder_layers,
            ff_dim=self.ff_dim,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            name='text_decoder'
        )(tokens, combined_context, deterministic=deterministic)
        
        return logits
    
    def encode_text(self, tokens: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Encode text only (no slots)."""
        return SwiGLUTextEncoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.encoder_layers,
            ff_dim=self.ff_dim,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            name='text_encoder'
        )(tokens, deterministic=deterministic)
    
    def project_slots(self, slots: jnp.ndarray) -> jnp.ndarray:
        """Project slots to embedding space."""
        return SlotProjection(
            embed_dim=self.embed_dim,
            name='slot_projection'
        )(slots)


def create_stove_player(
    vocab_size: int = 32000,
    embed_dim: int = 768,
    num_heads: int = 6,
    encoder_layers: int = 8,
    decoder_layers: int = 4,
    num_slots: int = 7,
    slot_dim: int = 128,
) -> STOVEplayer:
    """Factory function for STOVEplayer."""
    return STOVEplayer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        ff_dim=embed_dim * 4,
        num_slots=num_slots,
        slot_dim=slot_dim,
    )


# JIT-compiled helpers
@jax.jit
def forward_pass(model, params, tokens, slots, deterministic=True):
    """JIT-compiled forward pass."""
    return model.apply(params, tokens, slots, deterministic=deterministic)


def init_stove_player(key, vocab_size=32000, max_seq_len=128, num_slots=7, slot_dim=128):
    """Initialize STOVEplayer with random parameters."""
    model = create_stove_player(vocab_size=vocab_size, num_slots=num_slots, slot_dim=slot_dim)
    
    # Dummy inputs for initialization
    dummy_tokens = jnp.zeros((1, max_seq_len), dtype=jnp.int32)
    dummy_slots = jnp.zeros((1, num_slots, slot_dim), dtype=jnp.float32)
    
    params = model.init(key, dummy_tokens, dummy_slots)
    return model, params
