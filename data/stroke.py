"""
Core stroke data structures and normalization utilities.

A stroke is a sequence of points captured from Apple Pencil / PencilKit.
Each point has 6 features: x, y, pressure, timestamp, altitude, azimuth.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


def _synthesize_pressure(n: int) -> np.ndarray:
    """Generate realistic pressure dynamics matching Apple Pencil behavior.

    Real PencilKit pressure pattern:
    1. First 2-6 points: low placeholder (~0.08) due to Bluetooth latency
    2. Ramp up over ~10 points to working pressure
    3. Sustain with slight quasi-periodic variation
    4. Ramp down over last ~5 points as pen lifts

    Values are pre-normalized (divided by maximumPossibleForce ~4.17),
    so typical writing pressure of 1.0-2.0 maps to ~0.24-0.48 here.
    """
    if n <= 2:
        return np.full(n, 0.34, dtype=np.float32)

    # Base writing pressure (varies per stroke, like real writers)
    base_pressure = np.random.normal(0.34, 0.08)  # centered at typical writing
    base_pressure = np.clip(base_pressure, 0.10, 0.72)

    pressure = np.full(n, base_pressure, dtype=np.float32)

    # Phase 1: Initial placeholder (Bluetooth latency artifact)
    n_placeholder = min(np.random.randint(2, 6), n // 3)
    pressure[:n_placeholder] = np.random.uniform(0.06, 0.10)

    # Phase 2: Ramp up to working pressure
    n_ramp_up = min(np.random.randint(5, 12), (n - n_placeholder) // 3)
    if n_ramp_up > 0:
        ramp = np.linspace(pressure[n_placeholder - 1] if n_placeholder > 0 else 0.08,
                           base_pressure, n_ramp_up)
        pressure[n_placeholder:n_placeholder + n_ramp_up] = ramp

    # Phase 3: Sustain with quasi-periodic variation
    sustain_start = n_placeholder + n_ramp_up
    sustain_end = max(sustain_start, n - min(5, n // 5))
    n_sustain = sustain_end - sustain_start
    if n_sustain > 0:
        # Slight oscillation (correlated with writing rhythm)
        t = np.linspace(0, 1, n_sustain)
        freq = np.random.uniform(2, 6)  # oscillation frequency
        variation = 0.03 * np.sin(2 * np.pi * freq * t)
        noise = np.random.normal(0, 0.015, n_sustain)
        pressure[sustain_start:sustain_end] = base_pressure + variation + noise

    # Phase 4: Ramp down (pen lift)
    n_ramp_down = n - sustain_end
    if n_ramp_down > 0:
        ramp = np.linspace(pressure[sustain_end - 1] if sustain_end > 0 else base_pressure,
                           np.random.uniform(0.05, 0.12), n_ramp_down)
        pressure[sustain_end:] = ramp

    return np.clip(pressure, 0.02, 0.95).astype(np.float32)


@dataclass
class StrokePoint:
    """Single point in a stroke."""
    x: float            # horizontal position (pixels)
    y: float            # vertical position (pixels)
    pressure: float     # tip pressure [0, 1]
    timestamp: float    # seconds from page start
    altitude: float     # pencil tilt angle [0, pi/2] (0 = flat, pi/2 = upright)
    azimuth: float      # pencil rotation angle [0, 2*pi]


@dataclass
class Stroke:
    """A single continuous stroke (pen-down to pen-up)."""
    points: np.ndarray          # (num_points, 6) float32
    stroke_id: int = 0          # unique ID within the page
    group_id: int = -1          # which group this stroke belongs to (-1 = unassigned)

    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    @property
    def centroid(self) -> Tuple[float, float]:
        """Mean x, y position of the stroke."""
        return float(self.points[:, 0].mean()), float(self.points[:, 1].mean())

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding box (x_min, y_min, x_max, y_max)."""
        x_min, y_min = self.points[:, :2].min(axis=0)
        x_max, y_max = self.points[:, :2].max(axis=0)
        return float(x_min), float(y_min), float(x_max), float(y_max)

    @property
    def duration(self) -> float:
        """Duration of stroke in seconds."""
        return float(self.points[-1, 3] - self.points[0, 3])

    @property
    def average_speed(self) -> float:
        """Average speed in units/second."""
        if self.num_points < 2:
            return 0.0
        diffs = np.diff(self.points[:, :2], axis=0)
        distances = np.sqrt((diffs ** 2).sum(axis=1))
        dt = self.points[-1, 3] - self.points[0, 3]
        if dt < 1e-6:
            return 0.0
        return float(distances.sum() / dt)

    @property
    def path_length(self) -> float:
        """Total length of the stroke path."""
        if self.num_points < 2:
            return 0.0
        diffs = np.diff(self.points[:, :2], axis=0)
        return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())

    @property
    def is_closed(self) -> bool:
        """Whether the stroke forms a roughly closed shape."""
        if self.num_points < 10:
            return False
        start = self.points[0, :2]
        end = self.points[-1, :2]
        gap = np.sqrt(((end - start) ** 2).sum())
        return gap < self.path_length * 0.15  # end within 15% of path length from start

    @classmethod
    def from_xy(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        stroke_id: int = 0,
        group_id: int = -1,
        pressure: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        altitude: Optional[np.ndarray] = None,
        azimuth: Optional[np.ndarray] = None,
    ) -> Stroke:
        """Create a Stroke from x, y arrays, synthesizing missing features.

        Synthetic features are calibrated to match real Apple Pencil / PencilKit
        data after normalization. Key reference values:
            - PencilKit force: 0 to ~4.17 (maximumPossibleForce)
            - PencilKit altitude: 0 to pi/2 radians
            - PencilKit azimuth: 0 to 2*pi radians
            - PencilKit sample rate: 240 Hz

        Pressure values here are PRE-NORMALIZED (already divided by 4.17),
        so they're in [0, 1] range matching what the app will produce.
        """
        n = len(x)
        points = np.zeros((n, 6), dtype=np.float32)
        points[:, 0] = x
        points[:, 1] = y

        # --- Pressure / Force ---
        # Real PencilKit: 0 to ~4.17. After normalizing by maximumPossibleForce:
        #   Light writing: 0.07-0.19, Normal: 0.24-0.48, Hard: 0.48-0.84
        # We generate pre-normalized values matching this distribution.
        if pressure is not None:
            points[:, 2] = pressure
        else:
            points[:, 2] = _synthesize_pressure(n)

        # --- Timestamps ---
        # Real PencilKit: 240 Hz (4.17ms between points)
        if timestamps is not None:
            points[:, 3] = timestamps
        else:
            # 240 Hz with slight jitter (real devices show 3-5ms intervals)
            dt_base = 1.0 / 240.0  # 4.17ms
            jitter = np.random.normal(0, 0.0005, n - 1)  # ~0.5ms jitter
            dt = np.concatenate([[0.0], np.maximum(dt_base + jitter, 0.002)])
            points[:, 3] = np.cumsum(dt)

        # --- Altitude ---
        # Real PencilKit: 0 (flat) to pi/2 (upright). Writing: 1.1-1.4 rad
        points[:, 4] = altitude if altitude is not None else np.clip(
            np.random.normal(1.25, 0.12, n),  # centered between 1.1-1.4
            0.3, np.pi / 2  # allow some tilt, cap at physical max
        )

        # --- Azimuth ---
        # Real PencilKit: 0 to 2*pi. Right-handed writing: ~2.4-3.9 rad (centered ~3.1)
        # Left-handed: ~0.8-2.4. We mix 85% right-handed, 15% left-handed.
        if azimuth is not None:
            points[:, 5] = azimuth
        else:
            if np.random.random() < 0.85:
                # Right-handed: pencil points upper-left, azimuth ~3.1 rad
                base_az = np.random.normal(3.1, 0.3)
            else:
                # Left-handed: pencil points upper-right, azimuth ~1.5 rad
                base_az = np.random.normal(1.5, 0.3)
            # Small per-point variation (hand doesn't move much within a stroke)
            points[:, 5] = np.clip(
                base_az + np.random.normal(0, 0.05, n),
                0.0, 2 * np.pi
            )

        return cls(points=points, stroke_id=stroke_id, group_id=group_id)


@dataclass
class Group:
    """A group of strokes with a classification."""
    group_id: int
    group_type: str                 # from GROUP_CLASSES
    stroke_ids: List[int]           # indices into the page's stroke list
    content: Optional[str] = None   # text string for text groups, LaTeX for math groups
    bounds: Optional[Tuple[float, float, float, float]] = None  # x, y, w, h normalized


@dataclass
class Relationship:
    """A relationship between two groups."""
    source_group_id: int
    target_group_id: int
    relationship_type: str          # from RELATIONSHIP_TYPES


@dataclass
class PageAnnotation:
    """Complete annotation for a synthetic note page."""
    strokes: List[Stroke]
    groups: List[Group]
    relationships: List[Relationship]
    page_width: float = 1.0         # normalized
    page_height: float = 1.0        # normalized

    @property
    def num_strokes(self) -> int:
        return len(self.strokes)

    @property
    def num_groups(self) -> int:
        return len(self.groups)


def normalize_strokes(
    strokes: List[Stroke],
    page_width: float = 768.0,
    page_height: float = 1024.0,
    max_duration: float = 5.0,
) -> List[Stroke]:
    """Normalize all stroke features to [0, 1] range.

    - x: divide by page_width (default 768 = iPad 9.7" width in points)
    - y: divide by page_height (default 1024 = iPad 9.7" height in points)
    - pressure: already [0, 1] — synthetic data is pre-normalized,
                real PencilKit data must be divided by maximumPossibleForce (~4.17)
    - timestamp: relative to first stroke start, divide by max_duration, clamp
    - altitude: divide by pi/2 (max physical angle)
    - azimuth: divide by 2*pi (full rotation)
    """
    if not strokes:
        return strokes

    # Find global time start
    t_start = min(s.points[0, 3] for s in strokes if s.num_points > 0)

    normalized = []
    for s in strokes:
        pts = s.points.copy()
        pts[:, 0] /= page_width        # x
        pts[:, 1] /= page_height       # y
        # pressure already [0, 1]
        pts[:, 3] = np.clip((pts[:, 3] - t_start) / max_duration, 0, 1)  # time
        pts[:, 4] /= (np.pi / 2)       # altitude
        pts[:, 5] /= (2 * np.pi)       # azimuth
        # Clamp everything to [0, 1]
        pts = np.clip(pts, 0.0, 1.0)
        normalized.append(Stroke(
            points=pts,
            stroke_id=s.stroke_id,
            group_id=s.group_id,
        ))
    return normalized


def subsample_stroke(points: np.ndarray, max_points: int = 128) -> np.ndarray:
    """Uniformly subsample a stroke to max_points if it exceeds the limit."""
    n = points.shape[0]
    if n <= max_points:
        return points
    indices = np.linspace(0, n - 1, max_points, dtype=int)
    return points[indices]


def pad_stroke(points: np.ndarray, max_points: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Pad a stroke to max_points and return (padded_points, attention_mask).

    attention_mask: 1 for real points, 0 for padding.
    """
    n = points.shape[0]
    if n >= max_points:
        return points[:max_points], np.ones(max_points, dtype=np.float32)

    padded = np.zeros((max_points, points.shape[1]), dtype=np.float32)
    padded[:n] = points
    mask = np.zeros(max_points, dtype=np.float32)
    mask[:n] = 1.0
    return padded, mask


def prepare_stroke_for_model(
    stroke: Stroke,
    max_points: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample, then pad a stroke. Returns (points, mask)."""
    pts = subsample_stroke(stroke.points, max_points)
    return pad_stroke(pts, max_points)


def prepare_page_for_model(
    page: PageAnnotation,
    max_strokes: int = 512,
    max_points: int = 128,
) -> dict:
    """Prepare a full page for model input.

    Returns dict with:
        - stroke_points: (max_strokes, max_points, 6) float32
        - stroke_masks: (max_strokes, max_points) float32 — point-level attention mask
        - page_mask: (max_strokes,) float32 — stroke-level attention mask
        - stroke_centroids: (max_strokes, 2) float32 — centroid x, y per stroke
        - stroke_temporal_order: (max_strokes,) int64 — temporal order index
        - group_assignments: (max_strokes,) int64 — group ID per stroke (-1 for padding)
    """
    n_strokes = min(len(page.strokes), max_strokes)

    stroke_points = np.zeros((max_strokes, max_points, 6), dtype=np.float32)
    stroke_masks = np.zeros((max_strokes, max_points), dtype=np.float32)
    page_mask = np.zeros(max_strokes, dtype=np.float32)
    centroids = np.zeros((max_strokes, 2), dtype=np.float32)
    temporal_order = np.zeros(max_strokes, dtype=np.int64)
    group_assignments = np.full(max_strokes, -1, dtype=np.int64)

    # Sort strokes by temporal order (first point timestamp)
    sorted_strokes = sorted(page.strokes[:max_strokes], key=lambda s: s.points[0, 3])

    for i, stroke in enumerate(sorted_strokes):
        pts, mask = prepare_stroke_for_model(stroke, max_points)
        stroke_points[i] = pts
        stroke_masks[i] = mask
        page_mask[i] = 1.0
        cx, cy = stroke.centroid
        centroids[i] = [cx, cy]
        temporal_order[i] = i
        group_assignments[i] = stroke.group_id

    return {
        "stroke_points": stroke_points,
        "stroke_masks": stroke_masks,
        "page_mask": page_mask,
        "stroke_centroids": centroids,
        "stroke_temporal_order": temporal_order,
        "group_assignments": group_assignments,
        "num_strokes": n_strokes,
    }
