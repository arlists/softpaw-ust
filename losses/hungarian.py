"""
Hungarian Matcher — Optimal bipartite matching between predicted and GT groups.

Uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment) to find
the optimal one-to-one matching between predicted group queries and ground
truth groups, minimizing a combined cost of classification + mask similarity.

Directly inspired by DETR's matching strategy.
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple

from config import LossConfig, NO_OBJECT_INDEX


def dice_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute soft dice score between predictions and targets.

    Args:
        pred: (N, S) — predicted mask probabilities (after sigmoid)
        target: (M, S) — ground truth binary masks

    Returns:
        (N, M) — pairwise dice scores
    """
    # pred: (N, 1, S), target: (1, M, S)
    pred = pred.unsqueeze(1)
    target = target.unsqueeze(0)

    intersection = (pred * target).sum(dim=-1)  # (N, M)
    cardinality = pred.sum(dim=-1) + target.sum(dim=-1)  # (N, M)

    return 2 * intersection / (cardinality + 1e-6)


class HungarianMatcher:
    """Bipartite matcher between predicted queries and ground truth groups.

    Finds the optimal assignment that minimizes:
        cost = λ_cls * CE(pred_class, gt_class)
             + λ_bce * BCE(pred_mask, gt_mask)
             + λ_dice * (1 - Dice(pred_mask, gt_mask))
    """

    def __init__(self, cfg: LossConfig):
        self.cost_class = cfg.match_cost_class
        self.cost_mask_bce = cfg.match_cost_mask_bce
        self.cost_mask_dice = cfg.match_cost_mask_dice

    @torch.no_grad()
    def match(
        self,
        class_logits: torch.Tensor,
        mask_logits: torch.Tensor,
        gt_classes: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Find optimal matching for a batch.

        Args:
            class_logits: (B, Q, C) — predicted class logits
            mask_logits: (B, Q, S) — predicted mask logits
            gt_classes: (B, Q) — GT class indices (no_object for unused slots)
            gt_masks: (B, Q, S) — GT binary stroke masks

        Returns:
            List of (pred_indices, gt_indices) tuples, one per batch element.
            pred_indices[i] is matched to gt_indices[i].
        """
        B, Q, C = class_logits.shape
        S = mask_logits.size(2)

        results = []

        for b in range(B):
            # Find active GT groups (not no_object)
            active_gt = (gt_classes[b] != NO_OBJECT_INDEX).nonzero(as_tuple=True)[0]
            n_gt = len(active_gt)

            if n_gt == 0:
                # No GT groups — match nothing
                results.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long),
                ))
                continue

            # Classification cost: negative log probability of GT class
            pred_probs = class_logits[b].softmax(dim=-1)  # (Q, C)
            gt_cls = gt_classes[b, active_gt]  # (n_gt,)
            cost_class = -pred_probs[:, gt_cls]  # (Q, n_gt)

            # Mask BCE cost
            pred_masks = mask_logits[b].sigmoid()  # (Q, S)
            target_masks = gt_masks[b, active_gt]  # (n_gt, S)

            # Pairwise BCE: (Q, n_gt)
            # Expand for pairwise computation
            pred_expanded = pred_masks.unsqueeze(1).expand(-1, n_gt, -1)  # (Q, n_gt, S)
            tgt_expanded = target_masks.unsqueeze(0).expand(Q, -1, -1)  # (Q, n_gt, S)
            cost_mask_bce = F.binary_cross_entropy(
                pred_expanded.reshape(-1, S),
                tgt_expanded.reshape(-1, S),
                reduction="none",
            ).reshape(Q, n_gt, S).mean(dim=-1)  # (Q, n_gt)

            # Mask Dice cost
            dice = dice_score(pred_masks, target_masks)  # (Q, n_gt)
            cost_mask_dice = 1 - dice

            # Combined cost
            cost = (
                self.cost_class * cost_class
                + self.cost_mask_bce * cost_mask_bce
                + self.cost_mask_dice * cost_mask_dice
            )

            # Solve assignment with scipy
            cost_np = cost.cpu().numpy()
            pred_idx, gt_idx = linear_sum_assignment(cost_np)

            results.append((
                torch.tensor(pred_idx, dtype=torch.long, device=class_logits.device),
                active_gt[torch.tensor(gt_idx, dtype=torch.long)].to(class_logits.device),
            ))

        return results
