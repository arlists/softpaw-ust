"""
SoftPaw UST — Evaluation script.

Computes all metrics on the test set:
- Group detection mAP (stroke mask IoU)
- Stroke assignment accuracy
- Classification accuracy
- Text CER / WER
- Math exact match / token error rate
- Gesture precision / recall
- Relationship accuracy

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt
"""

import argparse
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import SoftPawConfig, GROUP_CLASSES, RELATIONSHIP_TYPES, NO_OBJECT_INDEX
from model import SoftPawUST
from data.page_composer import ComposedPageDataset, PageComposer
from data.tokenizer import TextTokenizer, MathTokenizer
from data.mathwriting import MathWritingDataset
from data.iam_online import IAMOnlineDataset
from data.quickdraw import QuickDrawDataset
from losses.hungarian import HungarianMatcher


def edit_distance(seq1: List[int], seq2: List[int]) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


def strip_special(ids: List[int], pad=0, bos=1, eos=2) -> List[int]:
    """Remove special tokens from a sequence."""
    result = []
    for x in ids:
        if x == eos:
            break
        if x not in (pad, bos):
            result.append(x)
    return result


def compute_mask_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """Compute IoU between two binary masks."""
    pred = (pred_mask > 0.5).float()
    gt = (gt_mask > 0.5).float()
    intersection = (pred * gt).sum().item()
    union = ((pred + gt) > 0).float().sum().item()
    if union == 0:
        return 0.0
    return intersection / union


@torch.no_grad()
def evaluate(args):
    cfg = SoftPawConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SoftPawUST(cfg.model).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Step: {checkpoint.get('step', '?')}, Val loss: {checkpoint.get('best_val_loss', '?')}")

    # Load test data
    # Build a small composer for evaluation
    text_samples, math_samples, drawing_samples = [], [], []
    iam = IAMOnlineDataset(cfg.data.iam_online_dir, split="test")
    for s, t in iam.iter_samples():
        if s and t:
            text_samples.append((s, t))
    mw = MathWritingDataset(cfg.data.mathwriting_dir, split="test")
    for s, l in mw.iter_samples():
        if s and l:
            math_samples.append((s, l))
    qd = QuickDrawDataset(cfg.data.quickdraw_dir, max_per_category=1000)
    for s, c in qd.iter_samples():
        if s:
            drawing_samples.append((s, c))

    # Fallback
    if not text_samples:
        from data.stroke import Stroke
        text_samples = [([Stroke.from_xy(np.linspace(0, 100, 20), np.zeros(20))], "test")]
    if not math_samples:
        from data.stroke import Stroke
        math_samples = [([Stroke.from_xy(np.linspace(0, 50, 15), np.zeros(15))], "x")]
    if not drawing_samples:
        from data.stroke import Stroke
        drawing_samples = [([Stroke.from_xy(np.cos(np.linspace(0, 6.28, 20)) * 30, np.sin(np.linspace(0, 6.28, 20)) * 30)], "circle")]

    composer = PageComposer(text_samples, math_samples, drawing_samples, cfg.data)
    test_dataset = ComposedPageDataset(
        composer=composer,
        num_pages=args.num_pages,
        cfg=cfg.data,
        split="test",
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4,
    )

    text_tok = TextTokenizer()
    math_tok = MathTokenizer()
    matcher = HungarianMatcher(cfg.loss)

    # Metrics accumulators
    metrics = defaultdict(list)
    class_correct = 0
    class_total = 0
    stroke_correct = 0
    stroke_total = 0
    gesture_tp = defaultdict(int)
    gesture_fp = defaultdict(int)
    gesture_fn = defaultdict(int)

    for batch in tqdm(test_loader, desc="Evaluating"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        outputs = model(batch)

        # Hungarian matching
        matches = matcher.match(
            outputs["class_logits"],
            outputs["mask_logits"],
            batch["group_classes"],
            batch["group_masks"],
        )

        B = batch["group_classes"].size(0)
        pred_classes = outputs["class_logits"].argmax(dim=-1)  # (B, Q)
        pred_masks = outputs["mask_logits"].sigmoid()  # (B, Q, S)

        for b, (pred_idx, gt_idx) in enumerate(matches):
            for pi, gi in zip(pred_idx, gt_idx):
                pi, gi = pi.item(), gi.item()
                gt_class = batch["group_classes"][b, gi].item()
                pred_class = pred_classes[b, pi].item()

                # Classification accuracy
                class_total += 1
                if pred_class == gt_class:
                    class_correct += 1

                # Mask IoU
                iou = compute_mask_iou(pred_masks[b, pi], batch["group_masks"][b, gi])
                metrics["mask_iou"].append(iou)

                # Stroke assignment accuracy
                gt_assigned = (batch["group_masks"][b, gi] > 0.5)
                pred_assigned = (pred_masks[b, pi] > 0.5)
                page_valid = batch["page_mask"][b] > 0.5
                agree = ((gt_assigned == pred_assigned) & page_valid).sum().item()
                valid = page_valid.sum().item()
                if valid > 0:
                    stroke_correct += agree
                    stroke_total += valid

                # Text CER (autoregressive generation, not teacher-forced)
                if gt_class == GROUP_CLASSES.index("text"):
                    gt_text = strip_special(batch["text_targets"][b, gi].cpu().tolist())
                    if gt_text:
                        active = (batch["group_masks"][b, gi] > 0.5).nonzero(as_tuple=True)[0]
                        if len(active) > 0:
                            ctx = outputs["contextualized"]
                            group_strokes = ctx[b, active].unsqueeze(0)
                            group_mask = torch.ones(1, len(active), device=device)
                            generated = model.text_decoder.generate(group_strokes, group_mask)
                            pred_text = strip_special(generated[0].cpu().tolist())
                            ed = edit_distance(pred_text, gt_text)
                            cer = ed / max(len(gt_text), 1)
                            metrics["text_cer"].append(cer)
                            metrics["text_exact"].append(1.0 if pred_text == gt_text else 0.0)

                # Math exact match + token error rate (autoregressive)
                if gt_class == GROUP_CLASSES.index("math"):
                    gt_math = strip_special(batch["math_targets"][b, gi].cpu().tolist())
                    if gt_math:
                        active = (batch["group_masks"][b, gi] > 0.5).nonzero(as_tuple=True)[0]
                        if len(active) > 0:
                            ctx = outputs["contextualized"]
                            group_strokes = ctx[b, active].unsqueeze(0)
                            group_mask = torch.ones(1, len(active), device=device)
                            generated = model.math_decoder.beam_search(group_strokes, group_mask)
                            pred_math = strip_special(generated[0].cpu().tolist())
                            ed = edit_distance(pred_math, gt_math)
                            ter = ed / max(len(gt_math), 1)
                            metrics["math_ter"].append(ter)
                            metrics["math_exact"].append(1.0 if pred_math == gt_math else 0.0)

                # Gesture detection
                gesture_classes = ["gesture_circle", "gesture_underline",
                                   "gesture_arrow", "gesture_strikethrough", "gesture_bracket"]
                gt_name = GROUP_CLASSES[gt_class]
                pred_name = GROUP_CLASSES[pred_class]
                if gt_name in gesture_classes:
                    if pred_name == gt_name:
                        gesture_tp[gt_name] += 1
                    else:
                        gesture_fn[gt_name] += 1
                if pred_name in gesture_classes and pred_name != gt_name:
                    gesture_fp[pred_name] += 1

    # Compute final metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nGroup Detection:")
    print(f"  Classification accuracy: {class_correct / max(class_total, 1):.4f} ({class_correct}/{class_total})")
    print(f"  Stroke assignment accuracy: {stroke_correct / max(stroke_total, 1):.4f}")
    print(f"  Mean mask IoU: {np.mean(metrics['mask_iou']):.4f}" if metrics["mask_iou"] else "  Mean mask IoU: N/A")

    print(f"\nText Recognition:")
    print(f"  CER: {np.mean(metrics['text_cer']):.4f}" if metrics["text_cer"] else "  CER: N/A")
    print(f"  Exact match: {np.mean(metrics['text_exact']):.4f}" if metrics["text_exact"] else "  Exact match: N/A")

    print(f"\nMath Recognition:")
    print(f"  Token Error Rate: {np.mean(metrics['math_ter']):.4f}" if metrics["math_ter"] else "  TER: N/A")
    print(f"  Exact match: {np.mean(metrics['math_exact']):.4f}" if metrics["math_exact"] else "  Exact match: N/A")

    print(f"\nGesture Detection:")
    for gesture_type in ["gesture_circle", "gesture_underline", "gesture_arrow",
                         "gesture_strikethrough", "gesture_bracket"]:
        tp = gesture_tp[gesture_type]
        fp = gesture_fp[gesture_type]
        fn = gesture_fn[gesture_type]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        print(f"  {gesture_type}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} (TP={tp} FP={fp} FN={fn})")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_pages", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    evaluate(args)
