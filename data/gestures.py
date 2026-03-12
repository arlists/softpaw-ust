"""
Synthetic gesture generator.

Generates realistic gesture strokes for training:
- Circle (select/group)
- Underline (emphasize)
- Arrow (connect)
- Strikethrough (delete)
- Bracket (group)

Each gesture is generated as a parametric curve with natural variation
in shape. Pressure, altitude, azimuth, and timestamps are synthesized
through Stroke.from_xy() to match real Apple Pencil / PencilKit data.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .stroke import Stroke


@dataclass
class GestureTarget:
    """Describes what content a gesture targets."""
    target_bbox: Tuple[float, float, float, float]  # x_min, y_min, x_max, y_max (normalized)
    gesture_type: str


def _speed_parameterize(n_points: int, gesture_speed: str = "fast") -> np.ndarray:
    """Generate non-uniform parameter values to simulate variable drawing speed.

    Returns t in [0, 1] where consecutive points are:
    - Closely spaced at start/end (slow pen movement → close spatial samples)
    - Widely spaced in middle (fast pen movement → far spatial samples)

    With 240Hz timestamps (from Stroke.from_xy), this creates realistic
    gesture kinematics: at 240Hz, spatial distance between consecutive
    points encodes pen speed.
    """
    u = np.linspace(0, 1, n_points)
    # Sigmoid-like speed profile: slow start, fast middle, slow end
    speed = 0.5 * (1 + np.tanh(6 * (u - 0.15))) * 0.5 * (1 + np.tanh(-6 * (u - 0.85)))
    speed = np.clip(speed, 0.1, 1.0)

    # Sharpen or soften the speed variation by gesture type
    speed_map = {"fast": 1.3, "medium": 1.0, "slow": 0.7}
    exponent = speed_map.get(gesture_speed, 1.0)
    speed = speed ** exponent

    # Integrate speed to get arc-length-like parameter
    cumulative = np.cumsum(speed)
    return cumulative / cumulative[-1]


def generate_circle_gesture(
    target_bbox: Tuple[float, float, float, float],
    n_points: int = 60,
    noise_scale: float = 0.008,
) -> Stroke:
    """Generate a circle gesture around a target bounding box.

    The circle should be slightly larger than the target bbox and
    roughly close on itself.
    """
    x_min, y_min, x_max, y_max = target_bbox
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    rx = (x_max - x_min) / 2 * np.random.uniform(1.1, 1.4)
    ry = (y_max - y_min) / 2 * np.random.uniform(1.1, 1.4)

    # Use speed-parameterized t for natural slow-fast-slow kinematics
    t_param = _speed_parameterize(n_points, "fast")
    total_angle = 2 * np.pi * np.random.uniform(1.0, 1.15)
    start_angle = np.random.uniform(0, 2 * np.pi)
    t = t_param * total_angle + start_angle

    x = cx + rx * np.cos(t)
    y = cy + ry * np.sin(t)

    # Add natural hand tremor
    x += np.random.normal(0, noise_scale, n_points)
    y += np.random.normal(0, noise_scale, n_points)

    return Stroke.from_xy(x.astype(np.float32), y.astype(np.float32))


def generate_underline_gesture(
    target_bbox: Tuple[float, float, float, float],
    n_points: int = 30,
    noise_scale: float = 0.003,
) -> Stroke:
    """Generate an underline gesture below a target bounding box.

    A roughly horizontal line just below the target content.
    """
    x_min, y_min, x_max, y_max = target_bbox

    # Start and end slightly beyond the text bounds
    margin = (x_max - x_min) * np.random.uniform(0.0, 0.1)
    x_start = x_min - margin
    x_end = x_max + margin

    # Position below the content
    gap = (y_max - y_min) * np.random.uniform(0.05, 0.2)
    y_base = y_max + gap

    t = _speed_parameterize(n_points, "fast")
    x = x_start + (x_end - x_start) * t
    # Slight natural curve (not perfectly straight)
    y = y_base + np.sin(t * np.pi) * np.random.uniform(-0.005, 0.005)

    x += np.random.normal(0, noise_scale, n_points)
    y += np.random.normal(0, noise_scale, n_points)

    return Stroke.from_xy(x.astype(np.float32), y.astype(np.float32))


def generate_arrow_gesture(
    source_bbox: Tuple[float, float, float, float],
    target_bbox: Tuple[float, float, float, float],
    n_points: int = 40,
    noise_scale: float = 0.004,
) -> Stroke:
    """Generate an arrow gesture connecting two bounding boxes.

    Line from edge of source bbox to edge of target bbox, with arrowhead.
    """
    # Find closest edges
    sx = (source_bbox[0] + source_bbox[2]) / 2
    sy = (source_bbox[1] + source_bbox[3]) / 2
    tx = (target_bbox[0] + target_bbox[2]) / 2
    ty = (target_bbox[1] + target_bbox[3]) / 2

    # Start from source edge, end at target edge
    dx = tx - sx
    dy = ty - sy
    length = np.sqrt(dx**2 + dy**2)
    if length < 1e-6:
        length = 1e-6

    # Main shaft (80% of points)
    n_shaft = int(n_points * 0.8)
    n_head = n_points - n_shaft

    t = _speed_parameterize(n_shaft, "fast")
    shaft_x = sx + dx * t
    shaft_y = sy + dy * t
    # Slight curve for natural feel
    perp_x = -dy / length
    perp_y = dx / length
    curve = np.sin(t * np.pi) * np.random.uniform(-0.02, 0.02)
    shaft_x += perp_x * curve
    shaft_y += perp_y * curve

    # Arrowhead
    angle = np.arctan2(dy, dx)
    head_length = min(0.03, length * 0.15)
    head_angle = np.random.uniform(0.35, 0.55)  # ~20-30 degrees

    # Two lines of the arrowhead (V-shape)
    n_half = n_head // 2
    n_other = n_head - n_half
    t1 = np.linspace(0, 1, n_half)
    t2 = np.linspace(1, 0, n_other)
    # First line of V: tip → one side
    h1_x = tx - head_length * np.cos(angle + head_angle) * t1
    h1_y = ty - head_length * np.sin(angle + head_angle) * t1
    # Second line of V: other side → tip
    h2_x = tx - head_length * np.cos(angle - head_angle) * t2
    h2_y = ty - head_length * np.sin(angle - head_angle) * t2
    head_x = np.concatenate([h1_x, h2_x])
    head_y = np.concatenate([h1_y, h2_y])

    x = np.concatenate([shaft_x, head_x])
    y = np.concatenate([shaft_y, head_y])

    x += np.random.normal(0, noise_scale, len(x))
    y += np.random.normal(0, noise_scale, len(y))

    return Stroke.from_xy(x.astype(np.float32), y.astype(np.float32))


def generate_strikethrough_gesture(
    target_bbox: Tuple[float, float, float, float],
    n_points: int = 25,
    noise_scale: float = 0.004,
) -> Stroke:
    """Generate a strikethrough gesture crossing through content.

    A roughly horizontal line through the vertical center of the target.
    """
    x_min, y_min, x_max, y_max = target_bbox
    margin = (x_max - x_min) * np.random.uniform(0.0, 0.15)

    y_center = (y_min + y_max) / 2
    # Slight random vertical offset
    y_center += (y_max - y_min) * np.random.uniform(-0.15, 0.15)

    # Slight angle (not perfectly horizontal)
    angle = np.random.uniform(-0.1, 0.1)  # radians

    t = _speed_parameterize(n_points, "fast")
    x = (x_min - margin) + (x_max - x_min + 2 * margin) * t
    y = y_center + np.sin(angle) * (x - x_min)

    x += np.random.normal(0, noise_scale, n_points)
    y += np.random.normal(0, noise_scale, n_points)

    return Stroke.from_xy(x.astype(np.float32), y.astype(np.float32))


def generate_bracket_gesture(
    target_bbox: Tuple[float, float, float, float],
    side: str = "left",
    n_points: int = 35,
    noise_scale: float = 0.005,
) -> Stroke:
    """Generate a bracket/brace gesture next to content.

    A curly brace or bracket on the left or right side of the target.
    """
    x_min, y_min, x_max, y_max = target_bbox
    height = y_max - y_min
    margin = height * 0.1

    if side == "left":
        base_x = x_min - margin - np.random.uniform(0.01, 0.03)
        indent = -np.random.uniform(0.01, 0.025)
    else:
        base_x = x_max + margin + np.random.uniform(0.01, 0.03)
        indent = np.random.uniform(0.01, 0.025)

    t = _speed_parameterize(n_points, "medium")
    y = y_min - margin * 0.5 + (height + margin) * t

    # Curly brace shape: indent at top and bottom, bump in middle
    x = base_x + indent * np.sin(t * np.pi) * np.sin(t * 2 * np.pi)

    x += np.random.normal(0, noise_scale, n_points)
    y += np.random.normal(0, noise_scale, n_points)

    return Stroke.from_xy(x.astype(np.float32), y.astype(np.float32))


class GestureGenerator:
    """Generates synthetic gesture strokes on demand."""

    GESTURE_TYPES = ["circle", "underline", "arrow", "strikethrough", "bracket"]

    def generate(
        self,
        gesture_type: str,
        target_bbox: Tuple[float, float, float, float],
        second_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Stroke:
        """Generate a gesture stroke of the given type.

        Args:
            gesture_type: One of GESTURE_TYPES
            target_bbox: Bounding box of the target content
            second_bbox: Second bbox for arrow gestures (target of arrow)

        Returns: Stroke object
        """
        if gesture_type == "circle":
            return generate_circle_gesture(target_bbox)
        elif gesture_type == "underline":
            return generate_underline_gesture(target_bbox)
        elif gesture_type == "arrow":
            if second_bbox is None:
                # Generate a random nearby target
                dx = np.random.uniform(0.1, 0.3) * np.random.choice([-1, 1])
                dy = np.random.uniform(0.05, 0.15) * np.random.choice([-1, 1])
                w = target_bbox[2] - target_bbox[0]
                h = target_bbox[3] - target_bbox[1]
                second_bbox = (
                    target_bbox[0] + dx,
                    target_bbox[1] + dy,
                    target_bbox[2] + dx,
                    target_bbox[3] + dy,
                )
            return generate_arrow_gesture(target_bbox, second_bbox)
        elif gesture_type == "strikethrough":
            return generate_strikethrough_gesture(target_bbox)
        elif gesture_type == "bracket":
            side = np.random.choice(["left", "right"])
            return generate_bracket_gesture(target_bbox, side=side)
        else:
            raise ValueError(f"Unknown gesture type: {gesture_type}")

    def random_gesture(
        self,
        target_bbox: Tuple[float, float, float, float],
        second_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[Stroke, str]:
        """Generate a random gesture type.

        Returns: (Stroke, gesture_type)
        """
        gesture_type = np.random.choice(self.GESTURE_TYPES)
        stroke = self.generate(gesture_type, target_bbox, second_bbox)
        return stroke, gesture_type
