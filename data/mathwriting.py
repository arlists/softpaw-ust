"""
MathWriting dataset loader.

Google's MathWriting dataset (KDD 2025): 630K human-written + 400K synthetic
math expressions with stroke data and LaTeX labels.

Format: InkML files containing stroke coordinates and LaTeX annotations.
Download: https://github.com/google-research/google-research/tree/master/mathwriting
"""

from __future__ import annotations

import os
import glob
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from lxml import etree

from .stroke import Stroke


def parse_inkml_trace(trace_element) -> np.ndarray:
    """Parse an InkML <trace> element into an array of points.

    InkML traces contain space-separated point groups, where each point
    is comma-separated values (typically x, y or x, y, timestamp).

    Returns: (num_points, 2 or 3) array — x, y[, timestamp]
    """
    text = trace_element.text.strip()
    points = []
    for point_str in text.split(","):
        coords = point_str.strip().split()
        if len(coords) >= 2:
            point = [float(c) for c in coords[:3]]  # x, y, [optional t]
            points.append(point)
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.array(points, dtype=np.float32)
    return arr


def parse_inkml_file(filepath: str) -> Optional[Tuple[List[np.ndarray], str]]:
    """Parse a single InkML file.

    Returns: (list of stroke coordinate arrays, latex_label) or None if parsing fails.
    Each stroke array is (num_points, 2+) with at least x, y columns.
    """
    try:
        tree = etree.parse(filepath)
        root = tree.getroot()

        # Handle namespace
        ns = {"ink": "http://www.w3.org/2003/InkML"}

        # Extract LaTeX annotation
        latex = None
        # Try different annotation formats used in MathWriting
        for annotation in root.findall(".//ink:annotation", ns):
            ann_type = annotation.get("type", "")
            if ann_type in ("truth", "label", "writer"):
                if ann_type != "writer":
                    latex = annotation.text
                    break
        # Also try without namespace
        if latex is None:
            for annotation in root.findall(".//annotation"):
                ann_type = annotation.get("type", "")
                if ann_type in ("truth", "label"):
                    latex = annotation.text
                    break

        if latex is None:
            return None

        # Clean LaTeX: remove $ delimiters if present
        latex = latex.strip()
        if latex.startswith("$") and latex.endswith("$"):
            latex = latex[1:-1].strip()

        # Extract traces (strokes)
        strokes = []
        traces = root.findall(".//ink:trace", ns)
        if not traces:
            traces = root.findall(".//trace")

        for trace in traces:
            points = parse_inkml_trace(trace)
            if points.shape[0] > 0:
                strokes.append(points)

        if not strokes:
            return None

        return strokes, latex

    except Exception:
        return None


def load_mathwriting_sample(filepath: str) -> Optional[Tuple[List[Stroke], str]]:
    """Load a single MathWriting sample as a list of Strokes + LaTeX label.

    Synthesizes missing PencilKit features (pressure, altitude, azimuth).
    """
    result = parse_inkml_file(filepath)
    if result is None:
        return None

    raw_strokes, latex = result
    strokes = []

    # Calculate global time offset from first stroke
    global_time = 0.0

    for i, raw in enumerate(raw_strokes):
        x = raw[:, 0]
        y = raw[:, 1]

        # Extract timestamps if available (3rd column)
        if raw.shape[1] >= 3:
            timestamps = raw[:, 2] / 1000.0  # ms → seconds (common InkML format)
        else:
            timestamps = None

        # Use default synthetic features (same distribution as all content types
        # to prevent the model from using pressure/altitude/azimuth to cheat
        # on content classification)
        stroke = Stroke.from_xy(
            x=x,
            y=y,
            stroke_id=i,
            timestamps=timestamps,
        )

        # If we synthesized timestamps, offset by global time
        if timestamps is None and stroke.num_points > 0:
            stroke.points[:, 3] += global_time
            global_time = stroke.points[-1, 3] + np.random.uniform(0.05, 0.3)
        elif stroke.num_points > 0:
            global_time = stroke.points[-1, 3] + np.random.uniform(0.05, 0.3)

        strokes.append(stroke)

    return strokes, latex


class MathWritingDataset:
    """Iterable dataset over MathWriting InkML files."""

    def __init__(self, root_dir: str, split: str = "train"):
        """
        Args:
            root_dir: Path to MathWriting dataset root
            split: One of "train", "val", "test"
        """
        self.root_dir = Path(root_dir)
        self.split = split

        # Find all InkML files
        split_dir = self.root_dir / split
        if split_dir.exists():
            self.files = sorted(glob.glob(str(split_dir / "**" / "*.inkml"), recursive=True))
        else:
            # Try flat structure
            self.files = sorted(glob.glob(str(self.root_dir / "**" / "*.inkml"), recursive=True))

        if not self.files:
            print(f"Warning: No InkML files found in {self.root_dir} for split '{split}'")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Optional[Tuple[List[Stroke], str]]:
        return load_mathwriting_sample(self.files[idx])

    def iter_samples(self):
        """Iterate over all valid samples, skipping parse failures."""
        for filepath in self.files:
            result = load_mathwriting_sample(filepath)
            if result is not None:
                yield result
