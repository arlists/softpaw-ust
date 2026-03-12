"""
Page Composer — Generates synthetic mixed-content note pages.

Takes samples from MathWriting, IAM Online, QuickDraw, and synthetic gestures,
composes them onto a virtual page with full annotations (groups, relationships).

This is the core data generation pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import pickle
import os

from .stroke import (
    Stroke, Group, Relationship, PageAnnotation,
    normalize_strokes, prepare_page_for_model,
)
from .gestures import GestureGenerator
from .tokenizer import TextTokenizer, MathTokenizer
from config import DataConfig, AugmentationConfig, GROUP_CLASSES, RELATIONSHIP_TYPES
from .augmentation import augment_page


# Layout templates — weighted distribution reflecting real note-taking patterns
LAYOUT_TEMPLATES = {
    "clean_rows": 0.40,          # neat linear notes
    "messy_scattered": 0.15,     # random placement
    "marginal_notes": 0.10,      # main content + side annotations
    "clustered": 0.05,           # grouped content areas
    "mixed_density": 0.15,       # dense in some areas, sparse in others
    "crossed_out": 0.10,         # content with corrections/strikethroughs
    "cramped": 0.05,             # everything packed tight, overlapping
}


@dataclass
class ContentSample:
    """A content sample ready for placement on a page."""
    strokes: List[Stroke]
    content_type: str       # "text", "math", "drawing"
    content_label: str      # text transcription, LaTeX, or drawing category
    raw_bbox: Tuple[float, float, float, float]  # original bbox before placement


def _compute_bbox(strokes: List[Stroke]) -> Tuple[float, float, float, float]:
    """Compute bounding box of a set of strokes."""
    all_x = np.concatenate([s.points[:, 0] for s in strokes])
    all_y = np.concatenate([s.points[:, 1] for s in strokes])
    return float(all_x.min()), float(all_y.min()), float(all_x.max()), float(all_y.max())


def _normalize_sample_to_unit(strokes: List[Stroke]) -> Tuple[List[Stroke], Tuple[float, float]]:
    """Normalize a sample's coordinates so it fits in a unit box starting at (0,0).

    Returns: (normalized_strokes, (width, height)) of the original sample.
    """
    bbox = _compute_bbox(strokes)
    x_min, y_min, x_max, y_max = bbox
    width = max(x_max - x_min, 1e-6)
    height = max(y_max - y_min, 1e-6)

    normalized = []
    for s in strokes:
        pts = s.points.copy()
        pts[:, 0] = (pts[:, 0] - x_min) / width
        pts[:, 1] = (pts[:, 1] - y_min) / height
        normalized.append(Stroke(points=pts, stroke_id=s.stroke_id, group_id=s.group_id))

    return normalized, (width, height)


def _place_sample(
    strokes: List[Stroke],
    position: Tuple[float, float],
    scale: Tuple[float, float],
    rotation: float = 0.0,
) -> List[Stroke]:
    """Place a unit-normalized sample at a position with given scale and rotation."""
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    placed = []

    for s in strokes:
        pts = s.points.copy()
        # Scale
        pts[:, 0] *= scale[0]
        pts[:, 1] *= scale[1]
        # Rotate around sample center
        cx = scale[0] / 2
        cy = scale[1] / 2
        if abs(rotation) > 1e-6:
            x = pts[:, 0] - cx
            y = pts[:, 1] - cy
            pts[:, 0] = x * cos_r - y * sin_r + cx
            pts[:, 1] = x * sin_r + y * cos_r + cy
        # Translate
        pts[:, 0] += position[0]
        pts[:, 1] += position[1]
        placed.append(Stroke(points=pts, stroke_id=s.stroke_id, group_id=s.group_id))

    return placed


def _generate_layout_positions(
    template: str,
    num_groups: int,
    group_sizes: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Generate placement positions for content groups based on layout template.

    All positions and sizes are in normalized [0, 1] coordinates.

    Returns: list of (x, y) positions for each group.
    """
    positions = []
    margin = 0.03

    if template == "clean_rows":
        # Top to bottom, left-aligned with slight variation
        cursor_y = margin + np.random.uniform(0.02, 0.06)
        for i, (w, h) in enumerate(group_sizes):
            x = margin + np.random.uniform(0, 0.05)
            y = cursor_y
            positions.append((x, y))
            cursor_y += h + np.random.uniform(0.02, 0.06)
            if cursor_y > 0.95:
                break

    elif template == "messy_scattered":
        # Random positions with collision avoidance
        placed_boxes = []
        for w, h in group_sizes:
            for _ in range(50):  # max attempts
                x = np.random.uniform(margin, 0.95 - w)
                y = np.random.uniform(margin, 0.95 - h)
                # Check collision with placed boxes
                collides = False
                for px, py, pw, ph in placed_boxes:
                    if (x < px + pw + 0.02 and x + w + 0.02 > px and
                            y < py + ph + 0.02 and y + h + 0.02 > py):
                        collides = True
                        break
                if not collides:
                    positions.append((x, y))
                    placed_boxes.append((x, y, w, h))
                    break
            else:
                # Fallback: place with overlap
                positions.append((
                    np.random.uniform(margin, max(margin, 0.9 - w)),
                    np.random.uniform(margin, max(margin, 0.9 - h)),
                ))

    elif template == "marginal_notes":
        # Main content on left 70%, annotations on right 30%
        cursor_y = margin
        split = 0.65
        main_count = max(1, int(num_groups * 0.7))
        for i, (w, h) in enumerate(group_sizes):
            if i < main_count:
                x = margin + np.random.uniform(0, 0.03)
                y = cursor_y
                cursor_y += h + np.random.uniform(0.02, 0.05)
            else:
                x = split + np.random.uniform(0.02, 0.05)
                y = np.random.uniform(margin, 0.85)
            positions.append((x, y))

    elif template == "clustered":
        # 2-3 clusters of content
        n_clusters = np.random.randint(2, 4)
        cluster_centers = [
            (np.random.uniform(0.15, 0.85), np.random.uniform(0.15, 0.85))
            for _ in range(n_clusters)
        ]
        for i, (w, h) in enumerate(group_sizes):
            cluster = cluster_centers[i % n_clusters]
            x = cluster[0] + np.random.uniform(-0.15, 0.15) - w / 2
            y = cluster[1] + np.random.uniform(-0.1, 0.1) - h / 2
            x = np.clip(x, margin, 0.95 - w)
            y = np.clip(y, margin, 0.95 - h)
            positions.append((x, y))

    elif template == "mixed_density":
        # Dense top half (like lecture notes), sparse bottom (like doodles/review)
        cursor_y = margin
        dense_cutoff = np.random.uniform(0.4, 0.65)
        for i, (w, h) in enumerate(group_sizes):
            if cursor_y < dense_cutoff:
                # Dense: tight spacing, slight indent variation
                x = margin + np.random.uniform(0, 0.03)
                y = cursor_y
                cursor_y += h + np.random.uniform(0.01, 0.03)
            else:
                # Sparse: scattered in remaining space
                x = np.random.uniform(margin, max(margin, 0.9 - w))
                y = np.random.uniform(dense_cutoff + 0.05, max(dense_cutoff + 0.06, 0.95 - h))
            positions.append((x, y))

    elif template == "crossed_out":
        # Same as clean_rows — the strikethroughs come from extra gestures
        cursor_y = margin + np.random.uniform(0.02, 0.06)
        for i, (w, h) in enumerate(group_sizes):
            x = margin + np.random.uniform(0, 0.05)
            y = cursor_y
            positions.append((x, y))
            cursor_y += h + np.random.uniform(0.02, 0.05)
            if cursor_y > 0.95:
                break

    elif template == "cramped":
        # Tight packing with intentional overlaps — messy student notes
        cursor_y = margin
        for i, (w, h) in enumerate(group_sizes):
            # Slight random indent, very tight vertical spacing
            x = margin + np.random.uniform(0, 0.08)
            y = cursor_y
            positions.append((x, y))
            # Allow negative spacing (overlap)
            cursor_y += h + np.random.uniform(-0.01, 0.02)
            if cursor_y > 0.98:
                # Wrap to a new "column"
                cursor_y = margin
                # No column offset — just pile on top

    # Pad if we couldn't place all groups
    while len(positions) < num_groups:
        positions.append((np.random.uniform(0.05, 0.7), np.random.uniform(0.05, 0.7)))

    return positions


class PageComposer:
    """Composes synthetic note pages from individual content samples."""

    def __init__(
        self,
        text_samples: List[Tuple[List[Stroke], str]],
        math_samples: List[Tuple[List[Stroke], str]],
        drawing_samples: List[Tuple[List[Stroke], str]],
        cfg: DataConfig,
    ):
        self.text_samples = text_samples
        self.math_samples = math_samples
        self.drawing_samples = drawing_samples
        self.cfg = cfg
        self.gesture_gen = GestureGenerator()
        self.text_tokenizer = TextTokenizer()
        self.math_tokenizer = MathTokenizer()

    def compose_page(self) -> PageAnnotation:
        """Generate a single synthetic note page with full annotations."""
        # Choose layout template
        template = np.random.choice(
            list(LAYOUT_TEMPLATES.keys()),
            p=list(LAYOUT_TEMPLATES.values()),
        )

        # Decide content composition
        n_text = np.random.randint(self.cfg.min_text_groups, self.cfg.max_text_groups + 1)
        n_math = np.random.randint(self.cfg.min_math_groups, self.cfg.max_math_groups + 1)
        n_drawing = np.random.randint(self.cfg.min_drawing_groups, self.cfg.max_drawing_groups + 1)
        n_gestures = np.random.randint(self.cfg.min_gestures, self.cfg.max_gestures + 1)

        # Crossed-out template: more strikethroughs and corrections
        if template == "crossed_out":
            n_gestures = max(n_gestures, np.random.randint(2, 6))

        # Ensure at least one content group
        if n_text + n_math + n_drawing == 0:
            n_text = 1

        # Sample and normalize content
        content_groups: List[ContentSample] = []

        for _ in range(n_text):
            if self.text_samples:
                strokes, label = self.text_samples[np.random.randint(len(self.text_samples))]
                strokes = [Stroke(points=s.points.copy(), stroke_id=s.stroke_id) for s in strokes]
                norm_strokes, size = _normalize_sample_to_unit(strokes)
                content_groups.append(ContentSample(
                    strokes=norm_strokes,
                    content_type="text",
                    content_label=label,
                    raw_bbox=(0, 0, size[0], size[1]),
                ))

        for _ in range(n_math):
            if self.math_samples:
                strokes, label = self.math_samples[np.random.randint(len(self.math_samples))]
                strokes = [Stroke(points=s.points.copy(), stroke_id=s.stroke_id) for s in strokes]
                norm_strokes, size = _normalize_sample_to_unit(strokes)
                content_groups.append(ContentSample(
                    strokes=norm_strokes,
                    content_type="math",
                    content_label=label,
                    raw_bbox=(0, 0, size[0], size[1]),
                ))

        for _ in range(n_drawing):
            if self.drawing_samples:
                strokes, label = self.drawing_samples[np.random.randint(len(self.drawing_samples))]
                strokes = [Stroke(points=s.points.copy(), stroke_id=s.stroke_id) for s in strokes]
                norm_strokes, size = _normalize_sample_to_unit(strokes)
                content_groups.append(ContentSample(
                    strokes=norm_strokes,
                    content_type="drawing",
                    content_label=label,
                    raw_bbox=(0, 0, size[0], size[1]),
                ))

        if not content_groups:
            # Fallback: single text group
            content_groups.append(ContentSample(
                strokes=[Stroke.from_xy(
                    x=np.linspace(0, 1, 20),
                    y=np.full(20, 0.5) + np.random.normal(0, 0.02, 20),
                )],
                content_type="text",
                content_label="hello",
                raw_bbox=(0, 0, 1, 0.1),
            ))

        # Determine sizes for layout
        group_sizes = []
        for cg in content_groups:
            # Scale content to appropriate size
            if cg.content_type == "text":
                w = np.random.uniform(0.25, 0.65)
                h = np.random.uniform(0.03, 0.08)
            elif cg.content_type == "math":
                w = np.random.uniform(0.15, 0.5)
                h = np.random.uniform(0.04, 0.12)
            elif cg.content_type == "drawing":
                s = np.random.uniform(0.1, 0.25)
                w, h = s, s * np.random.uniform(0.7, 1.3)
            else:
                w, h = 0.2, 0.05
            group_sizes.append((w, h))

        # Generate layout positions
        positions = _generate_layout_positions(template, len(content_groups), group_sizes)

        # Place content and build annotations
        all_strokes: List[Stroke] = []
        groups: List[Group] = []
        global_stroke_id = 0
        global_time = 0.0

        for group_id, (sample, position, size) in enumerate(
            zip(content_groups, positions, group_sizes)
        ):
            # Apply rotation for messy layouts
            rotation = 0.0
            if template == "messy_scattered":
                rotation = np.radians(np.random.uniform(-15, 15))
            elif template in ("clean_rows", "marginal_notes"):
                rotation = np.radians(np.random.uniform(-3, 3))

            placed = _place_sample(sample.strokes, position, size, rotation)

            stroke_ids = []
            for s in placed:
                s.stroke_id = global_stroke_id
                s.group_id = group_id
                # Assign temporal order: add inter-group pause
                s.points[:, 3] += global_time
                if s.num_points > 0:
                    global_time = s.points[-1, 3] + np.random.uniform(0.3, 1.5)
                stroke_ids.append(global_stroke_id)
                global_stroke_id += 1
                all_strokes.append(s)

            # Compute placed bounding box
            if placed:
                bbox = _compute_bbox(placed)
            else:
                bbox = (position[0], position[1], position[0] + size[0], position[1] + size[1])

            groups.append(Group(
                group_id=group_id,
                group_type=sample.content_type,
                stroke_ids=stroke_ids,
                content=sample.content_label,
                bounds=bbox,
            ))

        # Add gesture strokes
        gesture_group_start_id = len(groups)
        content_groups_with_bounds = [
            (g, g.bounds) for g in groups if g.bounds is not None
        ]

        relationships: List[Relationship] = []

        for gesture_idx in range(min(n_gestures, len(content_groups_with_bounds))):
            # Pick a random target content group
            target_group, target_bbox = content_groups_with_bounds[
                np.random.randint(len(content_groups_with_bounds))
            ]

            # Choose gesture type
            gesture_type = np.random.choice(GestureGenerator.GESTURE_TYPES)

            # For arrows, pick a second group
            second_bbox = None
            second_group = None
            if gesture_type == "arrow" and len(content_groups_with_bounds) >= 2:
                candidates = [
                    (g, b) for g, b in content_groups_with_bounds
                    if g.group_id != target_group.group_id
                ]
                if candidates:
                    second_group, second_bbox = candidates[np.random.randint(len(candidates))]

            # Generate gesture stroke
            gesture_stroke = self.gesture_gen.generate(
                gesture_type, target_bbox, second_bbox
            )
            gesture_stroke.stroke_id = global_stroke_id
            gesture_group_id = gesture_group_start_id + gesture_idx
            gesture_stroke.group_id = gesture_group_id
            gesture_stroke.points[:, 3] += global_time
            if gesture_stroke.num_points > 0:
                global_time = gesture_stroke.points[-1, 3] + np.random.uniform(0.2, 0.8)
            global_stroke_id += 1
            all_strokes.append(gesture_stroke)

            # Map gesture type to group class
            gesture_class_map = {
                "circle": "gesture_circle",
                "underline": "gesture_underline",
                "arrow": "gesture_arrow",
                "strikethrough": "gesture_strikethrough",
                "bracket": "gesture_bracket",
            }

            groups.append(Group(
                group_id=gesture_group_id,
                group_type=gesture_class_map[gesture_type],
                stroke_ids=[gesture_stroke.stroke_id],
                content=None,
                bounds=gesture_stroke.bbox if gesture_stroke.num_points > 0 else None,
            ))

            # Add relationship
            rel_type_map = {
                "circle": "selects",
                "underline": "emphasizes",
                "arrow": "connects",
                "strikethrough": "deletes",
                "bracket": "selects",
            }
            relationships.append(Relationship(
                source_group_id=gesture_group_id,
                target_group_id=target_group.group_id,
                relationship_type=rel_type_map[gesture_type],
            ))

            # For arrows, add connection to second group
            if gesture_type == "arrow" and second_group is not None:
                relationships.append(Relationship(
                    source_group_id=target_group.group_id,
                    target_group_id=second_group.group_id,
                    relationship_type="connects",
                ))

        # Add "precedes" relationships between sequential content groups
        content_only = [g for g in groups if g.group_type in ("text", "math")]
        for i in range(len(content_only) - 1):
            relationships.append(Relationship(
                source_group_id=content_only[i].group_id,
                target_group_id=content_only[i + 1].group_id,
                relationship_type="precedes",
            ))

        return PageAnnotation(
            strokes=all_strokes,
            groups=groups,
            relationships=relationships,
        )


class ComposedPageDataset(Dataset):
    """PyTorch Dataset that generates or loads composed pages.

    Can either:
    1. Generate pages on-the-fly (for training with unlimited data)
    2. Load pre-generated pages from cache (for reproducible evaluation)
    """

    def __init__(
        self,
        composer: Optional[PageComposer] = None,
        cache_dir: Optional[str] = None,
        num_pages: int = 100_000,
        cfg: Optional[DataConfig] = None,
        split: str = "train",
        augmentation_cfg: Optional[AugmentationConfig] = None,
    ):
        self.composer = composer
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.num_pages = num_pages
        self.cfg = cfg or DataConfig()
        self.split = split
        self.augmentation_cfg = augmentation_cfg
        self.text_tokenizer = TextTokenizer()
        self.math_tokenizer = MathTokenizer()

        # If cache exists, load from it
        self._cached_pages: Optional[List[PageAnnotation]] = None
        if self.cache_dir and self.cache_dir.exists():
            cache_file = self.cache_dir / f"{split}_pages.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    self._cached_pages = pickle.load(f)
                self.num_pages = len(self._cached_pages)

    def __len__(self) -> int:
        return self.num_pages

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get or generate page
        if self._cached_pages is not None:
            page = self._cached_pages[idx]
        elif self.composer is not None:
            page = self.composer.compose_page()
        else:
            raise RuntimeError("No composer or cached pages available")

        # Apply augmentation during training
        if self.split == "train" and self.augmentation_cfg is not None:
            page = augment_page(page, self.augmentation_cfg)

        # Normalize features to [0, 1]
        page = self._normalize_page(page)

        return self._page_to_tensors(page)

    def _normalize_page(self, page: PageAnnotation) -> PageAnnotation:
        """Normalize all stroke features to [0, 1] for model input.

        x, y are already approximately in [0, 1] from page composition.
        Timestamps, altitude, and azimuth need normalization.
        """
        if not page.strokes:
            return page

        max_duration = 5.0
        t_start = min(s.points[0, 3] for s in page.strokes if s.num_points > 0)

        strokes = []
        for s in page.strokes:
            pts = s.points.copy()
            pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
            pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0)
            pts[:, 2] = np.clip(pts[:, 2], 0.0, 1.0)
            pts[:, 3] = np.clip((pts[:, 3] - t_start) / max_duration, 0.0, 1.0)
            pts[:, 4] = np.clip(pts[:, 4] / (np.pi / 2), 0.0, 1.0)
            pts[:, 5] = np.clip(pts[:, 5] / (2 * np.pi), 0.0, 1.0)
            strokes.append(Stroke(points=pts, stroke_id=s.stroke_id, group_id=s.group_id))

        return PageAnnotation(
            strokes=strokes,
            groups=page.groups,
            relationships=page.relationships,
            page_width=page.page_width,
            page_height=page.page_height,
        )

    def _page_to_tensors(self, page: PageAnnotation) -> Dict[str, Any]:
        """Convert a PageAnnotation to model-ready tensors."""
        max_strokes = self.cfg.stroke.max_strokes_per_page
        max_points = self.cfg.stroke.max_points_per_stroke
        max_groups = 32  # from GroupDecoderConfig.num_queries
        max_text_len = self.cfg.max_text_len
        max_math_len = self.cfg.max_math_len

        # Prepare stroke data
        model_input = prepare_page_for_model(page, max_strokes, max_points)

        # Group targets
        num_groups = min(len(page.groups), max_groups)
        group_classes = np.full(max_groups, GROUP_CLASSES.index("no_object"), dtype=np.int64)
        group_masks = np.zeros((max_groups, max_strokes), dtype=np.float32)
        text_targets = np.zeros((max_groups, max_text_len), dtype=np.int64)
        math_targets = np.zeros((max_groups, max_math_len), dtype=np.int64)
        text_lengths = np.zeros(max_groups, dtype=np.int64)
        math_lengths = np.zeros(max_groups, dtype=np.int64)

        for i, group in enumerate(page.groups[:max_groups]):
            group_classes[i] = GROUP_CLASSES.index(group.group_type)

            # Build stroke mask
            for sid in group.stroke_ids:
                if sid < max_strokes:
                    group_masks[i, sid] = 1.0

            # Encode content for text/math groups
            if group.group_type == "text" and group.content:
                ids = self.text_tokenizer.encode(group.content)
                ids = self.text_tokenizer.pad_sequence(ids, max_text_len)
                text_targets[i] = ids
                text_lengths[i] = min(len(self.text_tokenizer.encode(group.content)), max_text_len)

            elif group.group_type == "math" and group.content:
                ids = self.math_tokenizer.encode(group.content)
                ids = self.math_tokenizer.pad_sequence(ids, max_math_len)
                math_targets[i] = ids
                math_lengths[i] = min(len(self.math_tokenizer.encode(group.content)), max_math_len)

        # Relationship targets
        max_rels = max_groups * max_groups
        rel_matrix = np.zeros((max_groups, max_groups), dtype=np.int64)
        for rel in page.relationships:
            src_idx = None
            tgt_idx = None
            for i, g in enumerate(page.groups[:max_groups]):
                if g.group_id == rel.source_group_id:
                    src_idx = i
                if g.group_id == rel.target_group_id:
                    tgt_idx = i
            if src_idx is not None and tgt_idx is not None:
                rel_matrix[src_idx, tgt_idx] = RELATIONSHIP_TYPES.index(rel.relationship_type)

        return {
            # Stroke inputs
            "stroke_points": torch.from_numpy(model_input["stroke_points"]),
            "stroke_masks": torch.from_numpy(model_input["stroke_masks"]),
            "page_mask": torch.from_numpy(model_input["page_mask"]),
            "stroke_centroids": torch.from_numpy(model_input["stroke_centroids"]),
            "stroke_temporal_order": torch.from_numpy(model_input["stroke_temporal_order"]),
            # Group targets
            "group_classes": torch.from_numpy(group_classes),
            "group_masks": torch.from_numpy(group_masks),
            "num_groups": num_groups,
            # Recognition targets
            "text_targets": torch.from_numpy(text_targets),
            "math_targets": torch.from_numpy(math_targets),
            "text_lengths": torch.from_numpy(text_lengths),
            "math_lengths": torch.from_numpy(math_lengths),
            # Relationship targets
            "rel_matrix": torch.from_numpy(rel_matrix),
        }

    def save_cache(self, pages: List[PageAnnotation]):
        """Save generated pages to cache for reproducible evaluation."""
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{self.split}_pages.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(pages, f)
