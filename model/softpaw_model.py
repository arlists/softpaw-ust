"""
SoftPaw Unified Stroke Transformer (UST) — The complete model.

One model, one forward pass, complete page understanding.

Pipeline:
    1. Stroke Encoder: raw strokes → per-stroke embeddings
    2. Page Transformer: per-stroke → contextualized across page
    3. Group Decoder: contextualized strokes → group predictions (masks + classes)
    4. Text Decoder: grouped text strokes → character sequences
    5. Math Decoder: grouped math strokes → LaTeX token sequences
    6. Relationship Head: group pairs → relationship types
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from config import ModelConfig, GROUP_CLASSES
from .stroke_encoder import StrokeEncoder
from .page_transformer import PageTransformer
from .group_decoder import GroupDecoder
from .text_decoder import TextDecoder
from .math_decoder import MathDecoder
from .relationship_head import RelationshipHead


class SoftPawUST(nn.Module):
    """SoftPaw Unified Stroke Transformer.

    The complete model for page understanding from raw stroke data.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Backbone
        self.stroke_encoder = StrokeEncoder(cfg.stroke_encoder)
        self.page_transformer = PageTransformer(
            cfg.page_transformer,
            stroke_embed_dim=cfg.stroke_encoder.hidden_dim,
        )

        # Group prediction
        self.group_decoder = GroupDecoder(cfg.group_decoder)

        # Recognition heads
        self.text_decoder = TextDecoder(cfg.text_decoder)
        self.math_decoder = MathDecoder(cfg.math_decoder)

        # Relationship prediction
        self.relationship_head = RelationshipHead(cfg.relationship_head)

        # Class indices for routing
        self._text_class = GROUP_CLASSES.index("text")
        self._math_class = GROUP_CLASSES.index("math")
        self._no_object = GROUP_CLASSES.index("no_object")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            batch: dict with keys:
                - stroke_points: (B, S, P, 6)
                - stroke_masks: (B, S, P)
                - page_mask: (B, S)
                - stroke_centroids: (B, S, 2)
                - stroke_temporal_order: (B, S)
                - group_classes: (B, Q) — GT group classes (for routing during training)
                - group_masks: (B, Q, S) — GT stroke masks per group
                - text_targets: (B, Q, T_text) — GT text token IDs
                - math_targets: (B, Q, T_math) — GT math token IDs
                - text_lengths: (B, Q) — GT text sequence lengths
                - math_lengths: (B, Q) — GT math sequence lengths

        Returns dict with:
            - class_logits: (B, Q, C)
            - mask_logits: (B, Q, S)
            - text_logits: (B, Q, T_text, V_text) — only for text groups
            - math_logits: (B, Q, T_math, V_math) — only for math groups
            - rel_logits: (B, Q, Q, R)
            - aux_outputs: list of intermediate group decoder outputs
        """
        B = batch["stroke_points"].size(0)
        device = batch["stroke_points"].device

        # 1. Encode individual strokes
        stroke_embeddings = self.stroke_encoder.encode_page_strokes(
            stroke_points=batch["stroke_points"],
            stroke_masks=batch["stroke_masks"],
            page_mask=batch["page_mask"],
        )  # (B, S, 128)

        # 2. Contextualize across page
        contextualized = self.page_transformer(
            stroke_embeddings=stroke_embeddings,
            stroke_centroids=batch["stroke_centroids"],
            stroke_temporal_order=batch["stroke_temporal_order"],
            page_mask=batch["page_mask"],
        )  # (B, S, 256)

        # 3. Predict groups
        group_output = self.group_decoder(
            stroke_embeddings=contextualized,
            page_mask=batch["page_mask"],
        )

        # 4. Run recognition decoders on appropriate groups
        # During training, we use GT group assignments for routing
        # (recognition decoders are trained on GT-assigned strokes, not predicted ones)
        text_logits = self._decode_text_groups(
            contextualized, batch, device
        )

        math_logits = self._decode_math_groups(
            contextualized, batch, device
        )

        # 5. Predict relationships
        # Create mask for active groups (non-no_object) based on GT
        group_active_mask = (batch["group_classes"] != self._no_object).float()
        rel_logits = self.relationship_head(
            group_embeddings=group_output["query_embeddings"],
            group_mask=group_active_mask,
        )

        return {
            "class_logits": group_output["class_logits"],
            "mask_logits": group_output["mask_logits"],
            "query_embeddings": group_output["query_embeddings"],
            "text_logits": text_logits,
            "math_logits": math_logits,
            "rel_logits": rel_logits,
            "aux_outputs": group_output["aux_outputs"],
            "contextualized": contextualized,
        }

    def _decode_text_groups(
        self,
        contextualized: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Run text decoder on GT text groups.

        For each group labeled as "text", gather its stroke embeddings
        and decode to text.
        """
        B, Q = batch["group_classes"].shape
        S = contextualized.size(1)
        T = batch["text_targets"].size(2)
        V = self.cfg.text_decoder.vocab_size

        # Initialize output
        text_logits = torch.zeros(B, Q, T, V, device=device)

        for b in range(B):
            for q in range(Q):
                if batch["group_classes"][b, q] != self._text_class:
                    continue
                if batch["text_lengths"][b, q] == 0:
                    continue

                # Gather stroke embeddings for this group using GT mask
                stroke_mask = batch["group_masks"][b, q]  # (S,)
                active_indices = (stroke_mask > 0.5).nonzero(as_tuple=True)[0]

                if len(active_indices) == 0:
                    continue

                group_strokes = contextualized[b, active_indices].unsqueeze(0)  # (1, n, D)
                group_stroke_mask = torch.ones(1, len(active_indices), device=device)

                # Teacher forcing: use GT targets (shifted right for input)
                target = batch["text_targets"][b, q].unsqueeze(0)  # (1, T)

                logits = self.text_decoder(
                    stroke_embeddings=group_strokes,
                    stroke_mask=group_stroke_mask,
                    target_ids=target,
                )  # (1, T, V)

                text_logits[b, q] = logits.squeeze(0)

        return text_logits

    def _decode_math_groups(
        self,
        contextualized: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Run math decoder on GT math groups."""
        B, Q = batch["group_classes"].shape
        T = batch["math_targets"].size(2)
        V = self.cfg.math_decoder.vocab_size

        math_logits = torch.zeros(B, Q, T, V, device=device)

        for b in range(B):
            for q in range(Q):
                if batch["group_classes"][b, q] != self._math_class:
                    continue
                if batch["math_lengths"][b, q] == 0:
                    continue

                stroke_mask = batch["group_masks"][b, q]
                active_indices = (stroke_mask > 0.5).nonzero(as_tuple=True)[0]

                if len(active_indices) == 0:
                    continue

                group_strokes = contextualized[b, active_indices].unsqueeze(0)
                group_stroke_mask = torch.ones(1, len(active_indices), device=device)

                target = batch["math_targets"][b, q].unsqueeze(0)

                logits = self.math_decoder(
                    stroke_embeddings=group_strokes,
                    stroke_mask=group_stroke_mask,
                    target_ids=target,
                )

                math_logits[b, q] = logits.squeeze(0)

        return math_logits

    @torch.no_grad()
    def inference(self, batch: Dict[str, torch.Tensor]) -> Dict:
        """Full inference pipeline — no ground truth needed.

        Returns structured page understanding.
        """
        self.eval()
        B = batch["stroke_points"].size(0)
        device = batch["stroke_points"].device

        # 1. Backbone
        stroke_embeddings = self.stroke_encoder.encode_page_strokes(
            stroke_points=batch["stroke_points"],
            stroke_masks=batch["stroke_masks"],
            page_mask=batch["page_mask"],
        )

        contextualized = self.page_transformer(
            stroke_embeddings=stroke_embeddings,
            stroke_centroids=batch["stroke_centroids"],
            stroke_temporal_order=batch["stroke_temporal_order"],
            page_mask=batch["page_mask"],
        )

        # 2. Group prediction
        group_output = self.group_decoder(
            stroke_embeddings=contextualized,
            page_mask=batch["page_mask"],
        )

        # 3. Extract predicted groups
        class_probs = group_output["class_logits"].softmax(dim=-1)  # (B, Q, C)
        pred_classes = class_probs.argmax(dim=-1)  # (B, Q)
        mask_probs = group_output["mask_logits"].sigmoid()  # (B, Q, S)

        results = []
        for b in range(B):
            page_result = {"groups": [], "relationships": []}

            for q in range(self.cfg.group_decoder.num_queries):
                pred_class = pred_classes[b, q].item()
                if pred_class == self._no_object:
                    continue

                confidence = class_probs[b, q, pred_class].item()
                if confidence < 0.3:
                    continue

                # Get assigned strokes
                stroke_assignment = mask_probs[b, q]  # (S,)
                assigned = (stroke_assignment > 0.5).nonzero(as_tuple=True)[0]

                group_info = {
                    "group_id": q,
                    "type": GROUP_CLASSES[pred_class],
                    "confidence": confidence,
                    "stroke_indices": assigned.cpu().tolist(),
                    "content": None,
                }

                # Decode content for text/math groups
                if pred_class == self._text_class and len(assigned) > 0:
                    group_strokes = contextualized[b, assigned].unsqueeze(0)
                    group_mask = torch.ones(1, len(assigned), device=device)
                    generated = self.text_decoder.generate(group_strokes, group_mask)
                    group_info["content"] = generated[0].cpu().tolist()

                elif pred_class == self._math_class and len(assigned) > 0:
                    group_strokes = contextualized[b, assigned].unsqueeze(0)
                    group_mask = torch.ones(1, len(assigned), device=device)
                    generated = self.math_decoder.beam_search(group_strokes, group_mask)
                    group_info["content"] = generated[0].cpu().tolist()

                page_result["groups"].append(group_info)

            # 4. Predict relationships
            active_mask = (pred_classes[b] != self._no_object).float().unsqueeze(0)
            rel_logits = self.relationship_head(
                group_embeddings=group_output["query_embeddings"][b:b+1],
                group_mask=active_mask,
            )
            rel_preds = rel_logits[0].argmax(dim=-1)  # (Q, Q)

            for i, gi in enumerate(page_result["groups"]):
                for j, gj in enumerate(page_result["groups"]):
                    if i == j:
                        continue
                    qi, qj = gi["group_id"], gj["group_id"]
                    rel_type = rel_preds[qi, qj].item()
                    if rel_type > 0:  # not "none"
                        from config import RELATIONSHIP_TYPES
                        page_result["relationships"].append({
                            "source": qi,
                            "target": qj,
                            "type": RELATIONSHIP_TYPES[rel_type],
                        })

            results.append(page_result)

        return results

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters per component."""
        components = {
            "stroke_encoder": self.stroke_encoder,
            "page_transformer": self.page_transformer,
            "group_decoder": self.group_decoder,
            "text_decoder": self.text_decoder,
            "math_decoder": self.math_decoder,
            "relationship_head": self.relationship_head,
        }
        counts = {}
        total = 0
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters())
            counts[name] = n
            total += n
        counts["total"] = total
        return counts
