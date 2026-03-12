"""
Multi-task loss for the SoftPaw Unified Stroke Transformer.

Combines:
    - Group classification loss (CE with class weighting)
    - Stroke mask loss (BCE + Dice)
    - Text recognition loss (CE, teacher forcing)
    - Math recognition loss (CE, teacher forcing)
    - Relationship classification loss (CE)

All losses computed after Hungarian matching between predictions and GT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from config import LossConfig, NO_OBJECT_INDEX, GROUP_CLASSES
from .hungarian import HungarianMatcher


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute dice loss.

    Args:
        pred: (N, S) — predicted mask probabilities (after sigmoid)
        target: (N, S) — ground truth binary masks

    Returns:
        scalar loss
    """
    intersection = (pred * target).sum(dim=-1)
    cardinality = pred.sum(dim=-1) + target.sum(dim=-1)
    loss = 1 - 2 * intersection / (cardinality + 1e-6)
    return loss.mean()


class SoftPawLoss(nn.Module):
    """Combined multi-task loss for the UST model.

    1. Hungarian match predicted groups to GT groups
    2. Compute classification loss on all queries (matched → GT class, unmatched → no_object)
    3. Compute mask loss on matched pairs only (BCE + Dice)
    4. Compute text CE on matched text groups
    5. Compute math CE on matched math groups
    6. Compute relationship CE on all active group pairs
    7. Apply auxiliary losses from intermediate decoder layers
    """

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        self.matcher = HungarianMatcher(cfg)

        # Class weights: down-weight no_object to handle imbalance
        class_weights = torch.ones(len(GROUP_CLASSES))
        class_weights[NO_OBJECT_INDEX] = cfg.no_object_weight
        self.register_buffer("class_weights", class_weights)

        self._text_class = GROUP_CLASSES.index("text")
        self._math_class = GROUP_CLASSES.index("math")

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        step: int = 0,
        recognition_start_step: int = 2000,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses.

        Args:
            outputs: model outputs dict
            batch: input batch dict
            step: current training step (for phased training)
            recognition_start_step: step at which to enable recognition losses

        Returns:
            dict with individual losses and total loss
        """
        device = outputs["class_logits"].device

        # Hungarian matching
        matches = self.matcher.match(
            class_logits=outputs["class_logits"],
            mask_logits=outputs["mask_logits"],
            gt_classes=batch["group_classes"],
            gt_masks=batch["group_masks"],
        )

        # Classification loss
        loss_class = self._classification_loss(
            outputs["class_logits"], batch["group_classes"], matches
        )

        # Mask losses
        loss_mask_bce, loss_mask_dice = self._mask_loss(
            outputs["mask_logits"], batch["group_masks"], matches
        )

        # Recognition losses (only after warmup)
        enable_recognition = step >= recognition_start_step
        if enable_recognition:
            loss_text = self._text_loss(
                outputs["text_logits"], batch["text_targets"],
                batch["text_lengths"], batch["group_classes"], matches
            )
            loss_math = self._math_loss(
                outputs["math_logits"], batch["math_targets"],
                batch["math_lengths"], batch["group_classes"], matches
            )
        else:
            loss_text = torch.tensor(0.0, device=device)
            loss_math = torch.tensor(0.0, device=device)

        # Relationship loss
        if enable_recognition:
            loss_rel = self._relationship_loss(
                outputs["rel_logits"], batch["rel_matrix"],
                batch["group_classes"], matches
            )
        else:
            loss_rel = torch.tensor(0.0, device=device)

        # Auxiliary losses from intermediate decoder layers
        loss_aux = torch.tensor(0.0, device=device)
        for aux_out in outputs.get("aux_outputs", []):
            aux_matches = self.matcher.match(
                class_logits=aux_out["class_logits"],
                mask_logits=aux_out["mask_logits"],
                gt_classes=batch["group_classes"],
                gt_masks=batch["group_masks"],
            )
            aux_cls = self._classification_loss(
                aux_out["class_logits"], batch["group_classes"], aux_matches
            )
            aux_bce, aux_dice = self._mask_loss(
                aux_out["mask_logits"], batch["group_masks"], aux_matches
            )
            loss_aux += self.cfg.aux_loss_weight * (
                self.cfg.weight_class * aux_cls
                + self.cfg.weight_mask_bce * aux_bce
                + self.cfg.weight_mask_dice * aux_dice
            )

        # Total loss
        total = (
            self.cfg.weight_class * loss_class
            + self.cfg.weight_mask_bce * loss_mask_bce
            + self.cfg.weight_mask_dice * loss_mask_dice
            + self.cfg.weight_text * loss_text
            + self.cfg.weight_math * loss_math
            + self.cfg.weight_relationship * loss_rel
            + loss_aux
        )

        return {
            "total": total,
            "class": loss_class,
            "mask_bce": loss_mask_bce,
            "mask_dice": loss_mask_dice,
            "text": loss_text,
            "math": loss_math,
            "relationship": loss_rel,
            "aux": loss_aux,
        }

    def _classification_loss(
        self,
        class_logits: torch.Tensor,
        gt_classes: torch.Tensor,
        matches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Cross-entropy on group classification after Hungarian matching.

        Matched queries → GT class. Unmatched queries → no_object.
        """
        B, Q, C = class_logits.shape
        device = class_logits.device

        # Build target tensor: all no_object by default
        targets = torch.full((B, Q), NO_OBJECT_INDEX, dtype=torch.long, device=device)

        for b, (pred_idx, gt_idx) in enumerate(matches):
            if len(pred_idx) > 0:
                targets[b, pred_idx] = gt_classes[b, gt_idx]

        # Flatten and compute CE
        loss = F.cross_entropy(
            class_logits.reshape(-1, C),
            targets.reshape(-1),
            weight=self.class_weights.to(device),
            label_smoothing=self.cfg.label_smoothing,
        )
        return loss

    def _mask_loss(
        self,
        mask_logits: torch.Tensor,
        gt_masks: torch.Tensor,
        matches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """BCE + Dice loss on stroke assignment masks for matched groups only."""
        device = mask_logits.device
        total_bce = torch.tensor(0.0, device=device)
        total_dice = torch.tensor(0.0, device=device)
        count = 0

        for b, (pred_idx, gt_idx) in enumerate(matches):
            if len(pred_idx) == 0:
                continue

            pred_logit = mask_logits[b, pred_idx]  # (n_matched, S)
            target = gt_masks[b, gt_idx]  # (n_matched, S)

            total_bce += F.binary_cross_entropy_with_logits(pred_logit, target, reduction="mean")
            pred = pred_logit.sigmoid()  # for Dice loss below
            total_dice += dice_loss(pred, target)
            count += 1

        if count == 0:
            return total_bce, total_dice

        return total_bce / count, total_dice / count

    def _text_loss(
        self,
        text_logits: torch.Tensor,
        text_targets: torch.Tensor,
        text_lengths: torch.Tensor,
        gt_classes: torch.Tensor,
        matches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """CE loss on text decoder output for matched text groups."""
        device = text_logits.device
        total_loss = torch.tensor(0.0, device=device)
        count = 0

        B = text_logits.size(0)

        for b, (pred_idx, gt_idx) in enumerate(matches):
            for pi, gi in zip(pred_idx, gt_idx):
                pi, gi = pi.item(), gi.item()
                if gt_classes[b, gi] != self._text_class:
                    continue
                seq_len = text_lengths[b, gi].item()
                if seq_len <= 1:
                    continue

                # text_logits[b, gi]: (T, V) — logits from model (indexed by GT group)
                # text_targets[b, gi]: (T,) — target IDs
                logits = text_logits[b, gi, :seq_len - 1]  # (T-1, V) — predict next token
                targets = text_targets[b, gi, 1:seq_len]  # (T-1,) — shifted targets

                total_loss += F.cross_entropy(logits, targets)
                count += 1

        if count == 0:
            return total_loss
        return total_loss / count

    def _math_loss(
        self,
        math_logits: torch.Tensor,
        math_targets: torch.Tensor,
        math_lengths: torch.Tensor,
        gt_classes: torch.Tensor,
        matches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """CE loss on math decoder output for matched math groups."""
        device = math_logits.device
        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for b, (pred_idx, gt_idx) in enumerate(matches):
            for pi, gi in zip(pred_idx, gt_idx):
                pi, gi = pi.item(), gi.item()
                if gt_classes[b, gi] != self._math_class:
                    continue
                seq_len = math_lengths[b, gi].item()
                if seq_len <= 1:
                    continue

                logits = math_logits[b, gi, :seq_len - 1]
                targets = math_targets[b, gi, 1:seq_len]

                total_loss += F.cross_entropy(logits, targets)
                count += 1

        if count == 0:
            return total_loss
        return total_loss / count

    def _relationship_loss(
        self,
        rel_logits: torch.Tensor,
        rel_matrix: torch.Tensor,
        gt_classes: torch.Tensor,
        matches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """CE loss on relationship predictions between matched groups."""
        device = rel_logits.device
        total_loss = torch.tensor(0.0, device=device)
        count = 0

        B, Q, _, R = rel_logits.shape

        for b, (pred_idx, gt_idx) in enumerate(matches):
            if len(pred_idx) < 2:
                continue

            # For all pairs of matched groups
            for i in range(len(pred_idx)):
                for j in range(len(pred_idx)):
                    if i == j:
                        continue
                    pi_i, gi_i = pred_idx[i].item(), gt_idx[i].item()
                    pi_j, gi_j = pred_idx[j].item(), gt_idx[j].item()

                    logits = rel_logits[b, pi_i, pi_j]  # (R,)
                    target = rel_matrix[b, gi_i, gi_j]  # scalar

                    total_loss += F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
                    count += 1

        if count == 0:
            return total_loss
        return total_loss / count
