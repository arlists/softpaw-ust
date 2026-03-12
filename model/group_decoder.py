"""
Group Decoder — DETR-style group prediction.

Uses learnable query embeddings that each "claim" a subset of strokes
(via predicted masks) and classify what type of group they represent.

Directly inspired by DETR and Mask2Former.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GroupDecoderConfig


class GroupDecoder(nn.Module):
    """DETR-style decoder for predicting stroke groups.

    Architecture:
        - 32 learnable query embeddings (256-dim)
        - 3-layer Transformer decoder:
            - Self-attention among queries (coordination to avoid duplicates)
            - Cross-attention to contextualized stroke embeddings
            - FFN
        - Classification head: MLP → 9 classes per query
        - Mask head: dot product between query and stroke embeddings → per-stroke assignment

    Produces intermediate outputs at each decoder layer for auxiliary losses.
    """

    def __init__(self, cfg: GroupDecoderConfig):
        super().__init__()
        self.cfg = cfg

        # Learnable query embeddings
        self.query_embed = nn.Embedding(cfg.num_queries, cfg.hidden_dim)
        nn.init.normal_(self.query_embed.weight, std=0.02)

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            GroupDecoderLayer(cfg) for _ in range(cfg.num_layers)
        ])

        # Classification head (shared across layers for aux losses)
        self.class_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

        # Mask projection — project queries and strokes to shared space for dot product
        self.mask_proj_query = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.mask_proj_stroke = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        stroke_embeddings: torch.Tensor,
        page_mask: torch.Tensor,
    ) -> dict:
        """Predict groups from contextualized stroke embeddings.

        Args:
            stroke_embeddings: (batch, max_strokes, hidden_dim) from PageTransformer
            page_mask: (batch, max_strokes) — 1 for real, 0 for padding

        Returns dict with:
            - class_logits: (batch, num_queries, num_classes) — final layer
            - mask_logits: (batch, num_queries, max_strokes) — final layer
            - query_embeddings: (batch, num_queries, hidden_dim) — final layer
            - aux_outputs: list of dicts with class_logits and mask_logits from intermediate layers
        """
        B = stroke_embeddings.size(0)

        # Initialize queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)

        # Project stroke embeddings for mask computation (shared across layers)
        stroke_for_mask = self.mask_proj_stroke(stroke_embeddings)  # (B, S, D)

        # Memory key padding mask (True = ignore)
        memory_key_padding_mask = (page_mask == 0)  # (B, S)

        # Collect intermediate outputs for auxiliary losses
        aux_outputs = []

        for layer in self.layers:
            queries = layer(
                queries=queries,
                memory=stroke_embeddings,
                memory_key_padding_mask=memory_key_padding_mask,
            )

            # Compute intermediate predictions (for aux losses)
            q_normed = self.norm(queries)
            aux_class = self.class_head(q_normed)  # (B, Q, C)
            q_for_mask = self.mask_proj_query(q_normed)  # (B, Q, D)
            aux_mask = torch.bmm(q_for_mask, stroke_for_mask.transpose(1, 2))  # (B, Q, S)

            aux_outputs.append({
                "class_logits": aux_class,
                "mask_logits": aux_mask,
            })

        # Final output is from the last layer
        final = aux_outputs[-1]
        aux_outputs = aux_outputs[:-1]  # intermediate layers only

        return {
            "class_logits": final["class_logits"],
            "mask_logits": final["mask_logits"],
            "query_embeddings": self.norm(queries),
            "aux_outputs": aux_outputs,
        }


class GroupDecoderLayer(nn.Module):
    """Single layer of the group decoder.

    Self-attention (queries ↔ queries) → Cross-attention (queries ↔ strokes) → FFN
    """

    def __init__(self, cfg: GroupDecoderConfig):
        super().__init__()

        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(cfg.hidden_dim)

        # Cross-attention: queries attend to stroke embeddings
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(cfg.hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.hidden_dim),
            nn.Dropout(cfg.dropout),
        )
        self.ffn_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            queries: (batch, num_queries, hidden_dim)
            memory: (batch, max_strokes, hidden_dim) — stroke embeddings
            memory_key_padding_mask: (batch, max_strokes) — True = ignore

        Returns:
            (batch, num_queries, hidden_dim) — updated queries
        """
        # Pre-norm self-attention
        q = self.self_attn_norm(queries)
        q2, _ = self.self_attn(q, q, q)
        queries = queries + q2

        # Pre-norm cross-attention
        q = self.cross_attn_norm(queries)
        q2, _ = self.cross_attn(
            query=q,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
        )
        queries = queries + q2

        # Pre-norm FFN
        q = self.ffn_norm(queries)
        queries = queries + self.ffn(q)

        return queries
