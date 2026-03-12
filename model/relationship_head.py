"""
Relationship Head — Predicts relationships between groups.

Given pairs of group query embeddings, classifies the relationship type:
none, precedes, emphasizes, selects, connects, deletes, spatial_proximity.
"""

import torch
import torch.nn as nn
from config import RelationshipHeadConfig


class RelationshipHead(nn.Module):
    """MLP classifier for inter-group relationships.

    Takes concatenated pairs of group embeddings and predicts
    the relationship type between them.

    For N active groups, evaluates all N*(N-1) ordered pairs.
    """

    def __init__(self, cfg: RelationshipHeadConfig):
        super().__init__()
        self.cfg = cfg

        self.mlp = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim // 2, cfg.num_relations),
        )

    def forward(
        self,
        group_embeddings: torch.Tensor,
        group_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict relationships between all pairs of active groups.

        Args:
            group_embeddings: (batch, num_queries, hidden_dim)
            group_mask: (batch, num_queries) — 1 for active groups, 0 for no_object/padding

        Returns:
            rel_logits: (batch, num_queries, num_queries, num_relations)
                — relationship prediction for each ordered pair (i, j)
        """
        B, Q, D = group_embeddings.shape

        # Create all pairs: (B, Q, Q, 2*D)
        # Expand group_i: (B, Q, 1, D) → (B, Q, Q, D)
        group_i = group_embeddings.unsqueeze(2).expand(B, Q, Q, D)
        # Expand group_j: (B, 1, Q, D) → (B, Q, Q, D)
        group_j = group_embeddings.unsqueeze(1).expand(B, Q, Q, D)

        # Concatenate pairs
        pairs = torch.cat([group_i, group_j], dim=-1)  # (B, Q, Q, 2*D)

        # Classify each pair
        rel_logits = self.mlp(pairs)  # (B, Q, Q, num_relations)

        # Mask out pairs involving inactive groups
        # Both source and target must be active
        pair_mask = group_mask.unsqueeze(2) * group_mask.unsqueeze(1)  # (B, Q, Q)
        # Self-pairs are always "none"
        diag_mask = 1 - torch.eye(Q, device=group_embeddings.device).unsqueeze(0)
        pair_mask = pair_mask * diag_mask

        # Apply mask: set inactive pairs to large negative for "none" class
        # This ensures inactive pairs predict "none" after softmax
        inactive = (pair_mask == 0).unsqueeze(-1).expand_as(rel_logits)
        rel_logits = rel_logits.masked_fill(inactive, -1e4)
        # Set "none" class (index 0) to 0 for inactive pairs
        none_boost = torch.zeros_like(rel_logits)
        none_boost[..., 0] = 1e4
        rel_logits = rel_logits + none_boost * inactive.float()

        return rel_logits
