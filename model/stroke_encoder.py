"""
Stroke Encoder — Encodes individual strokes into fixed-size embeddings.

Takes raw stroke point sequences (x, y, pressure, time, altitude, azimuth)
and produces a single embedding vector per stroke that captures
the stroke's character: its shape, speed, pressure pattern, etc.
"""

import math
import torch
import torch.nn as nn
from config import StrokeEncoderConfig


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position."""

    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)]


class StrokeEncoder(nn.Module):
    """Encodes individual strokes into fixed-size embeddings.

    Architecture:
        1. Linear projection: 6 features → hidden_dim
        2. Add sinusoidal positional encoding (position within stroke)
        3. Prepend [CLS] token
        4. 4-layer Transformer encoder with self-attention
        5. Output: [CLS] token's final hidden state → stroke embedding

    Input: (batch, num_points, 6) — raw stroke features
    Output: (batch, hidden_dim) — stroke embedding
    """

    def __init__(self, cfg: StrokeEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Project raw features to hidden dim
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)

        # Positional encoding for point position within stroke
        self.pos_encoding = SinusoidalPositionalEncoding(cfg.hidden_dim, max_len=cfg.input_dim * 32)

        # [CLS] token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers,
        )

        # Final layer norm
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        points: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of strokes.

        Args:
            points: (batch, max_points, 6) — raw stroke features (normalized)
            mask: (batch, max_points) — 1 for real points, 0 for padding

        Returns:
            (batch, hidden_dim) — stroke embedding per stroke
        """
        batch_size = points.size(0)

        # Project features
        x = self.input_proj(points)  # (batch, max_points, hidden_dim)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Prepend [CLS] token
        cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)
        x = torch.cat([cls, x], dim=1)  # (batch, 1 + max_points, hidden_dim)

        # Extend mask for [CLS] (always attended to)
        cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)
        extended_mask = torch.cat([cls_mask, mask], dim=1)  # (batch, 1 + max_points)

        # Convert mask to attention format: True = IGNORE
        attn_mask = (extended_mask == 0)  # (batch, 1 + max_points)

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Extract [CLS] token output
        cls_output = x[:, 0]  # (batch, hidden_dim)

        # Final normalization
        return self.norm(cls_output)

    def encode_page_strokes(
        self,
        stroke_points: torch.Tensor,
        stroke_masks: torch.Tensor,
        page_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode all strokes on a page efficiently.

        Only processes active (non-padding) strokes to avoid wasted compute.
        With max_strokes=512 and typical pages having ~30 strokes, this
        skips ~95% of unnecessary transformer forward passes.

        Args:
            stroke_points: (batch, max_strokes, max_points, 6)
            stroke_masks: (batch, max_strokes, max_points)
            page_mask: (batch, max_strokes) — 1 for real strokes, 0 for padding

        Returns:
            (batch, max_strokes, hidden_dim) — stroke embeddings
        """
        B, S, P, F = stroke_points.shape

        # Only encode active strokes
        flat_page_mask = page_mask.reshape(B * S)
        active_indices = flat_page_mask.nonzero(as_tuple=True)[0]

        if len(active_indices) == 0:
            return torch.zeros(B, S, self.cfg.hidden_dim, device=stroke_points.device)

        flat_points = stroke_points.reshape(B * S, P, F)
        flat_masks = stroke_masks.reshape(B * S, P)

        # Encode only active strokes
        active_embeddings = self.forward(
            flat_points[active_indices], flat_masks[active_indices]
        )

        # Scatter back into full tensor
        embeddings = torch.zeros(
            B * S, self.cfg.hidden_dim,
            device=stroke_points.device, dtype=active_embeddings.dtype,
        )
        embeddings[active_indices] = active_embeddings
        embeddings = embeddings.reshape(B, S, -1)

        return embeddings
