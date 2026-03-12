"""
IAM Online Handwriting Database loader.

~13,000 handwritten text samples from ~500 writers with stroke-level data.

Format: XML files with stroke coordinates and text transcriptions.
Download: https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database
(requires free registration)
"""

from __future__ import annotations

import os
import glob
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from lxml import etree

from .stroke import Stroke


def parse_iam_xml(filepath: str) -> Optional[Tuple[List[np.ndarray], str]]:
    """Parse a single IAM Online XML file.

    IAM Online format:
    - <WhiteboardDescription> contains metadata
    - <StrokeSet> contains <Stroke> elements
    - Each <Stroke> contains <Point x="..." y="..." time="..."/> elements
    - Transcription is in the accompanying .txt file or in the XML metadata

    Returns: (list of stroke arrays, transcription) or None
    """
    try:
        tree = etree.parse(filepath)
        root = tree.getroot()

        # Extract strokes
        raw_strokes = []
        stroke_set = root.find(".//StrokeSet")
        if stroke_set is None:
            return None

        for stroke_elem in stroke_set.findall("Stroke"):
            points = []
            for point in stroke_elem.findall("Point"):
                x = float(point.get("x", 0))
                y = float(point.get("y", 0))
                t = float(point.get("time", 0))
                points.append([x, y, t])
            if points:
                raw_strokes.append(np.array(points, dtype=np.float32))

        if not raw_strokes:
            return None

        # Extract transcription
        transcription = None

        # Try WhiteboardDescription
        wb_desc = root.find(".//WhiteboardDescription")
        if wb_desc is not None:
            text_elem = wb_desc.find(".//Text")
            if text_elem is not None and text_elem.text:
                transcription = text_elem.text.strip()

        # Try Transcription element
        if transcription is None:
            trans_elem = root.find(".//Transcription")
            if trans_elem is not None:
                # Collect text lines
                lines = []
                for text_line in trans_elem.findall(".//TextLine"):
                    line_text = text_line.get("text", "")
                    if line_text:
                        lines.append(line_text)
                if lines:
                    transcription = " ".join(lines)

        # Try to get transcription from companion .txt file
        if transcription is None:
            txt_path = filepath.replace(".xml", ".txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    transcription = f.read().strip()

        if transcription is None:
            transcription = ""

        return raw_strokes, transcription

    except Exception:
        return None


def load_iam_sample(filepath: str) -> Optional[Tuple[List[Stroke], str]]:
    """Load a single IAM Online sample as Strokes + text transcription."""
    result = parse_iam_xml(filepath)
    if result is None:
        return None

    raw_strokes, transcription = result
    strokes = []

    for i, raw in enumerate(raw_strokes):
        x = raw[:, 0]
        y = raw[:, 1]
        timestamps = raw[:, 2] / 1000.0 if raw.shape[1] >= 3 else None  # ms → seconds

        # Use default synthetic features (same distribution as all content types)
        stroke = Stroke.from_xy(
            x=x,
            y=y,
            stroke_id=i,
            timestamps=timestamps,
        )
        strokes.append(stroke)

    return strokes, transcription


class IAMOnlineDataset:
    """Dataset over IAM Online Handwriting XML files."""

    def __init__(self, root_dir: str, split: str = "train"):
        """
        Args:
            root_dir: Path to IAM Online dataset root
            split: One of "train", "val", "test"
        """
        self.root_dir = Path(root_dir)
        self.split = split

        # IAM Online has various directory structures
        # Try common patterns
        search_paths = [
            str(self.root_dir / split / "**" / "*.xml"),
            str(self.root_dir / "lineStrokes" / "**" / "*.xml"),
            str(self.root_dir / "**" / "*.xml"),
        ]

        self.files = []
        for pattern in search_paths:
            self.files = sorted(glob.glob(pattern, recursive=True))
            if self.files:
                break

        # Apply split if we loaded from a flat structure
        if self.files and split != "all":
            n = len(self.files)
            if split == "train":
                self.files = self.files[:int(n * 0.8)]
            elif split == "val":
                self.files = self.files[int(n * 0.8):int(n * 0.9)]
            elif split == "test":
                self.files = self.files[int(n * 0.9):]

        if not self.files:
            print(f"Warning: No XML files found in {self.root_dir} for split '{split}'")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Optional[Tuple[List[Stroke], str]]:
        return load_iam_sample(self.files[idx])

    def iter_samples(self):
        """Iterate over all valid samples."""
        for filepath in self.files:
            result = load_iam_sample(filepath)
            if result is not None:
                yield result
