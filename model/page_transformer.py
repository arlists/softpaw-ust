"""
Page-Level Transformer — Contextualizes stroke embeddings across the page.

After the stroke encoder produces per-stroke embeddings, the page transformer
lets each stroke attend to every other stroke on the page. This enables
context-aware grouping and classification: a circle near a Venn diagram
is recognized differently than a circle around text.
"""

import math
import torch
import torch.nn as nn
from config import PageTransformerConfig


class SpatialPositionMLP(nn.Module):
    """Projects 2D stroke centroid positions to positional embeddings."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, centroids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            centroids: (batch, max_strokes, 2) — x, y centroid per stroke
        Returns:
            (batch, max_strokes, output_dim) — spatial position embeddings
        """
        return self.mlp(centroids)


class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal encoding based on stroke temporal order."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, order_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            order_indices: (batch, max_strokes) — temporal order index per stroke
        Returns:
            (batch, max_strokes, d_model)
        """
        # Gather positional encodings by index
        batch_size, seq_len = order_indices.shape
        # Clamp indices to valid range
        indices = order_indices.clamp(0, self.pe.size(1) - 1).long()
        return self.pe[0][indices]  # (batch, max_strokes, d_model)


class PageTransformer(nn.Module):
    """Contextualizes stroke embeddings across the entire page.

    Architecture:
        1. Project stroke embeddings (128-dim) → 256-dim
        2. Add spatial position embedding from centroid (2D → 128-dim MLP → concat → 256-dim)
        3. Add temporal positional encoding (sinusoidal, based on stroke order)
        4. 6-layer Transformer encoder with self-attention across all strokes
        5. Output: contextualized stroke embeddings (256-dim)

    The spatial position is concatenated (not added) to preserve both signals.
    The temporal encoding is added to the combined representation.
    """

    def __init__(self, cfg: PageTransformerConfig, stroke_embed_dim: int = 128):
        super().__init__()
        self.cfg = cfg

        # Project stroke embedding to match spatial dim for concatenation
        stroke_proj_dim = cfg.hidden_dim - cfg.spatial_embed_dim
        self.stroke_proj = nn.Linear(stroke_embed_dim, stroke_proj_dim)

        # Spatial position embedding
        self.spatial_embed = SpatialPositionMLP(cfg.spatial_embed_dim)

        # Temporal positional encoding (added, not concatenated)
        self.temporal_pe = TemporalPositionalEncoding(cfg.hidden_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers,
        )

        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        stroke_embeddings: torch.Tensor,
        stroke_centroids: torch.Tensor,
        stroke_temporal_order: torch.Tensor,
        page_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Contextualize stroke embeddings across the page.

        Args:
            stroke_embeddings: (batch, max_strokes, stroke_embed_dim) from StrokeEncoder
            stroke_centroids: (batch, max_strokes, 2) — centroid x, y per stroke
            stroke_temporal_order: (batch, max_strokes) — temporal order index
            page_mask: (batch, max_strokes) — 1 for real strokes, 0 for padding

        Returns:
            (batch, max_strokes, hidden_dim) — contextualized stroke embeddings
        """
        # Project stroke embeddings
        stroke_proj = self.stroke_proj(stroke_embeddings)  # (B, S, stroke_proj_dim)

        # Compute spatial position embeddings
        spatial = self.spatial_embed(stroke_centroids)  # (B, S, spatial_embed_dim)

        # Concatenate stroke content + spatial position → hidden_dim
        x = torch.cat([stroke_proj, spatial], dim=-1)  # (B, S, hidden_dim)

        # Add temporal positional encoding
        temporal = self.temporal_pe(stroke_temporal_order)  # (B, S, hidden_dim)
        x = x + temporal

        # Create attention mask: True = IGNORE
        attn_mask = (page_mask == 0)  # (B, S)

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Final normalization
        x = self.norm(x)

        # Zero out padding positions
        x = x * page_mask.unsqueeze(-1)

        return x
