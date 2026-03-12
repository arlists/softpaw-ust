"""
Google QuickDraw dataset loader.

50M+ drawings across 345 categories with stroke data.

Format: NDJSON with simplified stroke format:
  { "drawing": [[[x0,x1,...],[y0,y1,...]], ...], "word": "cat", ... }

Download: https://github.com/googlecreativelab/quickdraw-dataset
Use the "simplified" format NDJSON files.
"""

from __future__ import annotations

import os
import json
import glob
import numpy as np
from typing import List, Tuple, Optional, Iterator
from pathlib import Path

from .stroke import Stroke


def parse_quickdraw_drawing(drawing_data: list) -> List[np.ndarray]:
    """Parse a QuickDraw simplified drawing into stroke coordinate arrays.

    QuickDraw simplified format: list of strokes, each stroke is [x_array, y_array]
    or [x_array, y_array, t_array] for the "raw" format.

    Returns: list of (num_points, 2 or 3) arrays.
    """
    strokes = []
    for stroke_data in drawing_data:
        if len(stroke_data) >= 2:
            x = np.array(stroke_data[0], dtype=np.float32)
            y = np.array(stroke_data[1], dtype=np.float32)
            if len(stroke_data) >= 3:
                t = np.array(stroke_data[2], dtype=np.float32)
                arr = np.stack([x, y, t], axis=1)
            else:
                arr = np.stack([x, y], axis=1)
            if arr.shape[0] > 0:
                strokes.append(arr)
    return strokes


def quickdraw_to_strokes(
    drawing_data: list,
    category: str,
    start_id: int = 0,
) -> List[Stroke]:
    """Convert a QuickDraw drawing to a list of Stroke objects.

    Synthesizes PencilKit features not present in QuickDraw:
    pressure, altitude, azimuth, and refines timestamps.
    """
    raw_strokes = parse_quickdraw_drawing(drawing_data)
    strokes = []

    global_time = 0.0

    for i, raw in enumerate(raw_strokes):
        x = raw[:, 0]
        y = raw[:, 1]
        timestamps = raw[:, 2] / 1000.0 if raw.shape[1] >= 3 else None

        # Use default synthetic features (same distribution as all content types)
        stroke = Stroke.from_xy(
            x=x,
            y=y,
            stroke_id=start_id + i,
            timestamps=timestamps,
        )

        if timestamps is None and stroke.num_points > 0:
            stroke.points[:, 3] += global_time
            # Drawings have longer pauses between strokes
            global_time = stroke.points[-1, 3] + np.random.uniform(0.1, 0.5)
        elif stroke.num_points > 0:
            global_time = stroke.points[-1, 3] + np.random.uniform(0.1, 0.5)

        strokes.append(stroke)

    return strokes


class QuickDrawDataset:
    """Dataset over QuickDraw NDJSON files.

    Lazily loads from NDJSON files, sampling a configurable number
    of drawings per category.
    """

    def __init__(
        self,
        root_dir: str,
        categories: Optional[List[str]] = None,
        max_per_category: int = 20_000,
    ):
        """
        Args:
            root_dir: Directory containing .ndjson files (one per category)
            categories: List of category names to load. None = load all.
            max_per_category: Max drawings to load per category.
        """
        self.root_dir = Path(root_dir)
        self.max_per_category = max_per_category

        # Find available NDJSON files
        ndjson_files = sorted(glob.glob(str(self.root_dir / "*.ndjson")))

        self.category_files = {}
        for f in ndjson_files:
            cat_name = Path(f).stem.replace(" ", "_").lower()
            # Full name might be like "full_simplified_airplane.ndjson"
            # or just "airplane.ndjson"
            for part in Path(f).stem.split("_"):
                if part not in ("full", "simplified", "raw"):
                    cat_name = Path(f).stem
                    break
            self.category_files[cat_name] = f

        if categories:
            self.category_files = {
                k: v for k, v in self.category_files.items()
                if k in categories
            }

        self._cache: List[Tuple[list, str]] = []
        self._loaded = False

    def _load_all(self):
        """Load samples from all category files into memory."""
        if self._loaded:
            return

        for category, filepath in self.category_files.items():
            count = 0
            with open(filepath, "r") as f:
                for line in f:
                    if count >= self.max_per_category:
                        break
                    try:
                        data = json.loads(line.strip())
                        if data.get("recognized", True):  # only use recognized drawings
                            self._cache.append((data["drawing"], data.get("word", category)))
                            count += 1
                    except (json.JSONDecodeError, KeyError):
                        continue

        self._loaded = True

    def __len__(self) -> int:
        self._load_all()
        return len(self._cache)

    def __getitem__(self, idx: int) -> Tuple[List[Stroke], str]:
        self._load_all()
        drawing_data, category = self._cache[idx]
        strokes = quickdraw_to_strokes(drawing_data, category)
        return strokes, category

    def iter_samples(self) -> Iterator[Tuple[List[Stroke], str]]:
        """Iterate over all samples."""
        self._load_all()
        for drawing_data, category in self._cache:
            strokes = quickdraw_to_strokes(drawing_data, category)
            if strokes:
                yield strokes, category

    def random_sample(self) -> Tuple[List[Stroke], str]:
        """Get a random drawing sample."""
        self._load_all()
        idx = np.random.randint(len(self._cache))
        return self[idx]
