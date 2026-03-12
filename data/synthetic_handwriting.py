"""
Synthetic handwriting generator — unlimited labeled text training data.

Generates realistic stroke sequences by:
1. Using human-correct stroke decompositions for each character
   (proper stroke count, direction, and order — not glyph outlines)
2. Scaling and positioning characters with natural variation
3. Adding jitter, slant, spacing, speed variation, cursive connections

Each character is defined as the actual pen strokes a human would make,
NOT font outline contours. This is critical — a font outline of 'A' is a
closed polygon, but a human writes 'A' as 3 strokes (two diagonals + crossbar).
The stroke encoder must learn features from realistic writing dynamics.

No external dependencies beyond numpy.
"""

from __future__ import annotations

import numpy as np
import glob
import os
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from dataclasses import dataclass

from .stroke import Stroke


@dataclass
class HandwritingStyle:
    """Parameters controlling synthetic handwriting variation."""
    slant_angle: float = 0.0        # radians, positive = right lean
    letter_spacing: float = 1.0     # multiplier on default spacing
    word_spacing: float = 1.0       # multiplier on space width
    size_variation: float = 0.05    # random scale per character
    baseline_wobble: float = 0.003  # vertical drift amplitude
    stroke_jitter: float = 0.002   # point-level noise
    speed_variation: float = 0.15   # timestamp variation
    pen_lift_prob: float = 0.1      # probability of lifting pen mid-character
    connect_prob: float = 0.3       # probability of connecting adjacent letters (cursive-like)


"""
Writing stroke definitions for ASCII characters.

Each character is defined as a list of strokes (pen-down segments), where each
stroke is a list of (x, y) control points representing the path a human hand
would trace when writing the character.

Coordinate system (y-down, matching page coordinates):
  x: 0 = left edge, 1 = right edge of character cell
  y: 0 = top of x-height (lowercase) or cap height (uppercase)
  y: 1 = baseline
  y < 0 = extends above (ascenders, dots)
  y > 1 = extends below (descenders)

This is NOT glyph outline data. These are the actual pen strokes —
correct stroke count, direction, and order for how humans write.
"""
_WRITING_STROKES: Dict[str, List[List[Tuple[float, float]]]] = {
    # ===== LOWERCASE =====
    'a': [
        [(0.85, 0.15), (0.6, 0.0), (0.2, 0.15), (0.1, 0.5), (0.2, 0.9), (0.6, 1.0), (0.85, 0.85)],
        [(0.85, 0.0), (0.85, 1.0)],
    ],
    'b': [
        [(0.2, -0.6), (0.2, 1.0)],
        [(0.2, 0.4), (0.5, 0.05), (0.85, 0.3), (0.85, 0.7), (0.5, 1.0), (0.2, 0.7)],
    ],
    'c': [
        [(0.8, 0.15), (0.5, 0.0), (0.15, 0.25), (0.1, 0.5), (0.15, 0.75), (0.5, 1.0), (0.8, 0.85)],
    ],
    'd': [
        [(0.8, 0.15), (0.5, 0.0), (0.15, 0.25), (0.1, 0.5), (0.15, 0.75), (0.5, 1.0), (0.8, 0.85)],
        [(0.8, -0.6), (0.8, 1.0)],
    ],
    'e': [
        [(0.1, 0.5), (0.85, 0.5), (0.8, 0.15), (0.5, 0.0), (0.15, 0.2), (0.1, 0.5),
         (0.15, 0.8), (0.5, 1.0), (0.8, 0.85)],
    ],
    'f': [
        [(0.7, -0.5), (0.5, -0.6), (0.35, -0.4), (0.35, 1.0)],
        [(0.15, 0.0), (0.6, 0.0)],
    ],
    'g': [
        [(0.8, 0.15), (0.5, 0.0), (0.15, 0.25), (0.1, 0.5), (0.15, 0.75), (0.5, 1.0), (0.8, 0.85)],
        [(0.8, 0.0), (0.8, 1.3), (0.6, 1.5), (0.3, 1.4)],
    ],
    'h': [
        [(0.2, -0.6), (0.2, 1.0)],
        [(0.2, 0.2), (0.5, 0.0), (0.8, 0.2), (0.8, 1.0)],
    ],
    'i': [
        [(0.5, 0.0), (0.5, 1.0)],
        [(0.48, -0.22), (0.52, -0.18)],  # dot
    ],
    'j': [
        [(0.6, 0.0), (0.6, 1.3), (0.4, 1.5), (0.2, 1.4)],
        [(0.58, -0.22), (0.62, -0.18)],  # dot
    ],
    'k': [
        [(0.2, -0.6), (0.2, 1.0)],
        [(0.7, 0.0), (0.2, 0.55)],
        [(0.3, 0.45), (0.75, 1.0)],
    ],
    'l': [[(0.5, -0.6), (0.5, 1.0)]],
    'm': [
        [(0.1, 1.0), (0.1, 0.0)],
        [(0.1, 0.1), (0.3, 0.0), (0.5, 0.15), (0.5, 1.0)],
        [(0.5, 0.1), (0.7, 0.0), (0.9, 0.15), (0.9, 1.0)],
    ],
    'n': [
        [(0.2, 1.0), (0.2, 0.0)],
        [(0.2, 0.1), (0.5, 0.0), (0.8, 0.2), (0.8, 1.0)],
    ],
    'o': [
        [(0.5, 0.0), (0.15, 0.2), (0.1, 0.5), (0.15, 0.8), (0.5, 1.0),
         (0.85, 0.8), (0.9, 0.5), (0.85, 0.2), (0.5, 0.0)],
    ],
    'p': [
        [(0.2, 0.0), (0.2, 1.5)],
        [(0.2, 0.15), (0.5, 0.0), (0.85, 0.2), (0.85, 0.7), (0.5, 1.0), (0.2, 0.7)],
    ],
    'q': [
        [(0.8, 0.15), (0.5, 0.0), (0.15, 0.2), (0.1, 0.5), (0.15, 0.75), (0.5, 1.0), (0.8, 0.85)],
        [(0.8, 0.0), (0.8, 1.5)],
    ],
    'r': [
        [(0.2, 1.0), (0.2, 0.0)],
        [(0.2, 0.15), (0.5, 0.0), (0.7, 0.1)],
    ],
    's': [
        [(0.75, 0.12), (0.5, 0.0), (0.2, 0.12), (0.2, 0.35), (0.5, 0.5),
         (0.8, 0.65), (0.8, 0.88), (0.5, 1.0), (0.25, 0.88)],
    ],
    't': [
        [(0.4, -0.3), (0.4, 0.9), (0.55, 1.0)],
        [(0.2, 0.0), (0.65, 0.0)],
    ],
    'u': [
        [(0.2, 0.0), (0.2, 0.7), (0.4, 1.0), (0.65, 1.0), (0.8, 0.8)],
        [(0.8, 0.0), (0.8, 1.0)],
    ],
    'v': [[(0.15, 0.0), (0.5, 1.0)], [(0.5, 1.0), (0.85, 0.0)]],
    'w': [
        [(0.05, 0.0), (0.25, 1.0)], [(0.25, 1.0), (0.45, 0.3)],
        [(0.45, 0.3), (0.65, 1.0)], [(0.65, 1.0), (0.95, 0.0)],
    ],
    'x': [[(0.15, 0.0), (0.85, 1.0)], [(0.85, 0.0), (0.15, 1.0)]],
    'y': [
        [(0.15, 0.0), (0.5, 0.65)],
        [(0.85, 0.0), (0.35, 1.2), (0.2, 1.4), (0.1, 1.3)],
    ],
    'z': [[(0.15, 0.0), (0.85, 0.0)], [(0.85, 0.0), (0.15, 1.0)], [(0.15, 1.0), (0.85, 1.0)]],

    # ===== UPPERCASE =====
    'A': [
        [(0.1, 1.0), (0.5, 0.0)], [(0.5, 0.0), (0.9, 1.0)], [(0.25, 0.6), (0.75, 0.6)],
    ],
    'B': [
        [(0.15, 1.0), (0.15, 0.0)],
        [(0.15, 0.0), (0.65, 0.0), (0.8, 0.12), (0.75, 0.35), (0.5, 0.5), (0.15, 0.5)],
        [(0.15, 0.5), (0.7, 0.5), (0.85, 0.65), (0.8, 0.88), (0.6, 1.0), (0.15, 1.0)],
    ],
    'C': [
        [(0.85, 0.15), (0.6, 0.0), (0.25, 0.1), (0.1, 0.35), (0.1, 0.65),
         (0.25, 0.9), (0.6, 1.0), (0.85, 0.85)],
    ],
    'D': [
        [(0.15, 1.0), (0.15, 0.0)],
        [(0.15, 0.0), (0.55, 0.0), (0.85, 0.2), (0.9, 0.5), (0.85, 0.8), (0.55, 1.0), (0.15, 1.0)],
    ],
    'E': [
        [(0.8, 0.0), (0.15, 0.0), (0.15, 1.0), (0.8, 1.0)],
        [(0.15, 0.5), (0.65, 0.5)],
    ],
    'F': [
        [(0.8, 0.0), (0.15, 0.0), (0.15, 1.0)],
        [(0.15, 0.5), (0.65, 0.5)],
    ],
    'G': [
        [(0.85, 0.15), (0.6, 0.0), (0.25, 0.1), (0.1, 0.35), (0.1, 0.65),
         (0.25, 0.9), (0.6, 1.0), (0.85, 0.85), (0.85, 0.5), (0.55, 0.5)],
    ],
    'H': [
        [(0.15, 0.0), (0.15, 1.0)], [(0.85, 0.0), (0.85, 1.0)], [(0.15, 0.5), (0.85, 0.5)],
    ],
    'I': [
        [(0.3, 0.0), (0.7, 0.0)], [(0.5, 0.0), (0.5, 1.0)], [(0.3, 1.0), (0.7, 1.0)],
    ],
    'J': [
        [(0.3, 0.0), (0.75, 0.0)],
        [(0.6, 0.0), (0.6, 0.8), (0.45, 1.0), (0.25, 0.9)],
    ],
    'K': [
        [(0.15, 0.0), (0.15, 1.0)], [(0.8, 0.0), (0.15, 0.5)], [(0.15, 0.5), (0.8, 1.0)],
    ],
    'L': [[(0.15, 0.0), (0.15, 1.0), (0.8, 1.0)]],
    'M': [
        [(0.1, 1.0), (0.1, 0.0)], [(0.1, 0.0), (0.5, 0.6)],
        [(0.5, 0.6), (0.9, 0.0)], [(0.9, 0.0), (0.9, 1.0)],
    ],
    'N': [
        [(0.15, 1.0), (0.15, 0.0)], [(0.15, 0.0), (0.85, 1.0)], [(0.85, 1.0), (0.85, 0.0)],
    ],
    'O': [
        [(0.5, 0.0), (0.15, 0.15), (0.1, 0.5), (0.15, 0.85), (0.5, 1.0),
         (0.85, 0.85), (0.9, 0.5), (0.85, 0.15), (0.5, 0.0)],
    ],
    'P': [
        [(0.15, 1.0), (0.15, 0.0)],
        [(0.15, 0.0), (0.65, 0.0), (0.85, 0.15), (0.8, 0.4), (0.55, 0.5), (0.15, 0.5)],
    ],
    'Q': [
        [(0.5, 0.0), (0.15, 0.15), (0.1, 0.5), (0.15, 0.85), (0.5, 1.0),
         (0.85, 0.85), (0.9, 0.5), (0.85, 0.15), (0.5, 0.0)],
        [(0.65, 0.8), (0.95, 1.1)],
    ],
    'R': [
        [(0.15, 1.0), (0.15, 0.0)],
        [(0.15, 0.0), (0.65, 0.0), (0.85, 0.15), (0.8, 0.4), (0.55, 0.5), (0.15, 0.5)],
        [(0.5, 0.5), (0.85, 1.0)],
    ],
    'S': [
        [(0.8, 0.12), (0.55, 0.0), (0.2, 0.08), (0.15, 0.25), (0.25, 0.42),
         (0.5, 0.5), (0.75, 0.58), (0.85, 0.75), (0.8, 0.92), (0.45, 1.0), (0.2, 0.88)],
    ],
    'T': [[(0.1, 0.0), (0.9, 0.0)], [(0.5, 0.0), (0.5, 1.0)]],
    'U': [
        [(0.15, 0.0), (0.15, 0.7), (0.3, 0.95), (0.5, 1.0), (0.7, 0.95), (0.85, 0.7), (0.85, 0.0)],
    ],
    'V': [[(0.1, 0.0), (0.5, 1.0)], [(0.5, 1.0), (0.9, 0.0)]],
    'W': [
        [(0.05, 0.0), (0.25, 1.0)], [(0.25, 1.0), (0.45, 0.3)],
        [(0.45, 0.3), (0.65, 1.0)], [(0.65, 1.0), (0.95, 0.0)],
    ],
    'X': [[(0.1, 0.0), (0.9, 1.0)], [(0.9, 0.0), (0.1, 1.0)]],
    'Y': [
        [(0.1, 0.0), (0.5, 0.5)], [(0.9, 0.0), (0.5, 0.5)], [(0.5, 0.5), (0.5, 1.0)],
    ],
    'Z': [[(0.1, 0.0), (0.9, 0.0)], [(0.9, 0.0), (0.1, 1.0)], [(0.1, 1.0), (0.9, 1.0)]],

    # ===== DIGITS =====
    '0': [
        [(0.5, 0.0), (0.15, 0.15), (0.1, 0.5), (0.15, 0.85), (0.5, 1.0),
         (0.85, 0.85), (0.9, 0.5), (0.85, 0.15), (0.5, 0.0)],
    ],
    '1': [[(0.3, 0.15), (0.5, 0.0), (0.5, 1.0)], [(0.3, 1.0), (0.7, 1.0)]],
    '2': [
        [(0.2, 0.15), (0.45, 0.0), (0.75, 0.05), (0.85, 0.2), (0.8, 0.4), (0.15, 1.0), (0.85, 1.0)],
    ],
    '3': [
        [(0.2, 0.1), (0.5, 0.0), (0.8, 0.15), (0.8, 0.35), (0.5, 0.5)],
        [(0.5, 0.5), (0.85, 0.65), (0.85, 0.85), (0.5, 1.0), (0.2, 0.9)],
    ],
    '4': [
        [(0.7, 1.0), (0.7, 0.0)],
        [(0.7, 0.0), (0.1, 0.65), (0.9, 0.65)],
    ],
    '5': [
        [(0.8, 0.0), (0.2, 0.0), (0.15, 0.45), (0.5, 0.4), (0.85, 0.55),
         (0.85, 0.8), (0.5, 1.0), (0.2, 0.9)],
    ],
    '6': [
        [(0.75, 0.1), (0.5, 0.0), (0.2, 0.2), (0.1, 0.5), (0.15, 0.85),
         (0.5, 1.0), (0.85, 0.85), (0.85, 0.6), (0.5, 0.45), (0.2, 0.55)],
    ],
    '7': [[(0.15, 0.0), (0.85, 0.0), (0.4, 1.0)]],
    '8': [
        [(0.5, 0.5), (0.2, 0.35), (0.2, 0.12), (0.5, 0.0), (0.8, 0.12), (0.8, 0.35),
         (0.5, 0.5), (0.15, 0.65), (0.15, 0.88), (0.5, 1.0), (0.85, 0.88),
         (0.85, 0.65), (0.5, 0.5)],
    ],
    '9': [
        [(0.8, 0.45), (0.5, 0.55), (0.15, 0.4), (0.15, 0.15), (0.5, 0.0),
         (0.85, 0.15), (0.9, 0.5), (0.8, 0.85), (0.5, 1.0), (0.25, 0.9)],
    ],

    # ===== PUNCTUATION =====
    '.': [[(0.48, 0.95), (0.52, 1.0)]],
    ',': [[(0.5, 0.9), (0.4, 1.2)]],
    '!': [[(0.5, 0.0), (0.5, 0.7)], [(0.48, 0.92), (0.52, 0.97)]],
    '?': [
        [(0.2, 0.15), (0.5, 0.0), (0.8, 0.15), (0.8, 0.35), (0.5, 0.55), (0.5, 0.7)],
        [(0.48, 0.92), (0.52, 0.97)],
    ],
    ':': [[(0.48, 0.28), (0.52, 0.33)], [(0.48, 0.78), (0.52, 0.83)]],
    ';': [[(0.48, 0.28), (0.52, 0.33)], [(0.5, 0.8), (0.4, 1.1)]],
    '-': [[(0.2, 0.5), (0.8, 0.5)]],
    "'": [[(0.5, 0.0), (0.45, 0.2)]],
    '"': [[(0.35, 0.0), (0.3, 0.2)], [(0.65, 0.0), (0.6, 0.2)]],
    '(': [[(0.65, -0.1), (0.35, 0.2), (0.3, 0.5), (0.35, 0.8), (0.65, 1.1)]],
    ')': [[(0.35, -0.1), (0.65, 0.2), (0.7, 0.5), (0.65, 0.8), (0.35, 1.1)]],
    '/': [[(0.8, 0.0), (0.2, 1.0)]],
    '+': [[(0.5, 0.2), (0.5, 0.8)], [(0.2, 0.5), (0.8, 0.5)]],
    '=': [[(0.2, 0.35), (0.8, 0.35)], [(0.2, 0.65), (0.8, 0.65)]],
    '*': [[(0.5, 0.15), (0.5, 0.65)], [(0.2, 0.25), (0.8, 0.55)], [(0.8, 0.25), (0.2, 0.55)]],
    '#': [[(0.3, 0.1), (0.25, 0.9)], [(0.7, 0.1), (0.65, 0.9)],
          [(0.15, 0.35), (0.85, 0.35)], [(0.15, 0.65), (0.85, 0.65)]],
    '&': [
        [(0.8, 1.0), (0.5, 0.5), (0.55, 0.15), (0.4, 0.0), (0.2, 0.1), (0.2, 0.3),
         (0.5, 0.5), (0.2, 0.85), (0.3, 1.0), (0.6, 1.0), (0.9, 0.7)],
    ],
    '@': [
        [(0.75, 0.55), (0.55, 0.35), (0.35, 0.45), (0.35, 0.65), (0.55, 0.75), (0.75, 0.55),
         (0.8, 0.3), (0.65, 0.05), (0.35, 0.0), (0.1, 0.2), (0.1, 0.8), (0.35, 1.0), (0.7, 0.95)],
    ],
}


def _interpolate_writing_stroke(
    control_points: List[Tuple[float, float]],
    n_points: int = 15,
) -> np.ndarray:
    """Interpolate between control points to produce a smooth stroke path.

    Uses arc-length parameterized linear interpolation. The noise and jitter
    added later by the generator make strokes look natural without needing
    spline interpolation here.

    Args:
        control_points: List of (x, y) waypoints defining the stroke path
        n_points: Number of output points

    Returns:
        (n_points, 2) array of interpolated x, y positions
    """
    pts = np.array(control_points, dtype=np.float64)

    if len(pts) <= 1:
        # Single point (dot) — return two very close points for a valid stroke
        pt = pts[0]
        return np.array([
            [pt[0] - 0.01, pt[1] - 0.01],
            [pt[0] + 0.01, pt[1] + 0.01],
        ], dtype=np.float32)

    # Compute cumulative arc length for parameterization
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_dist = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_dist = cum_dist[-1]

    if total_dist < 1e-8:
        return pts.astype(np.float32)

    cum_dist /= total_dist  # normalize to [0, 1]

    # Scale output points by path complexity
    actual_n = max(n_points, len(pts) * 3)

    # Interpolate
    t_new = np.linspace(0, 1, actual_n)
    x_new = np.interp(t_new, cum_dist, pts[:, 0])
    y_new = np.interp(t_new, cum_dist, pts[:, 1])

    return np.column_stack([x_new, y_new]).astype(np.float32)


def _get_writing_strokes(char: str, n_points: int = 15) -> Optional[List[np.ndarray]]:
    """Get pre-defined writing strokes for a character.

    Returns stroke paths that match how a human actually WRITES the character
    (correct stroke count, direction, order), not font outline contours.

    Args:
        char: Character to get strokes for
        n_points: Points per stroke (more = smoother)

    Returns:
        List of (n_points, 2) arrays, or None if character not defined
    """
    if char not in _WRITING_STROKES:
        return None

    stroke_defs = _WRITING_STROKES[char]
    strokes = []

    for control_points in stroke_defs:
        interpolated = _interpolate_writing_stroke(control_points, n_points)
        if len(interpolated) >= 2:
            strokes.append(interpolated)

    return strokes if strokes else None


class SyntheticHandwritingGenerator:
    """Generates unlimited synthetic handwriting training data.

    Uses pre-defined writing stroke decompositions for all ASCII characters
    (correct stroke count, direction, order) with extensive variation in
    style, spacing, and dynamics. No fonts required for core generation.
    """

    # Common English words for procedural text generation
    _NOUNS = [
        "cell", "energy", "force", "mass", "atom", "molecule", "equation", "function",
        "theorem", "proof", "hypothesis", "experiment", "result", "data", "graph",
        "variable", "constant", "matrix", "vector", "derivative", "integral", "limit",
        "sequence", "series", "set", "group", "field", "space", "point", "line",
        "angle", "circle", "triangle", "square", "area", "volume", "surface",
        "wave", "frequency", "amplitude", "phase", "signal", "noise", "filter",
        "system", "model", "algorithm", "process", "method", "technique", "approach",
        "problem", "solution", "answer", "question", "example", "definition", "concept",
        "theory", "law", "principle", "rule", "pattern", "structure", "property",
        "student", "professor", "teacher", "class", "lecture", "exam", "quiz",
        "chapter", "section", "page", "paragraph", "sentence", "word", "letter",
        "book", "textbook", "notebook", "paper", "article", "journal", "report",
        "project", "assignment", "homework", "lab", "study", "review", "summary",
        "note", "idea", "thought", "observation", "conclusion", "introduction",
        "history", "science", "math", "physics", "chemistry", "biology", "language",
        "computer", "program", "code", "software", "hardware", "network", "database",
        "water", "light", "heat", "sound", "pressure", "temperature", "speed",
        "time", "distance", "weight", "height", "length", "width", "depth",
        "color", "shape", "size", "number", "amount", "ratio", "percentage",
        "market", "price", "cost", "value", "profit", "loss", "budget", "income",
        "population", "sample", "mean", "median", "standard", "deviation", "variance",
        "correlation", "regression", "distribution", "probability", "statistics",
        "diagram", "chart", "table", "figure", "illustration", "sketch", "drawing",
        "reaction", "compound", "element", "mixture", "solution", "acid", "base",
        "electron", "proton", "neutron", "nucleus", "orbit", "bond", "charge",
        "plant", "animal", "organism", "species", "gene", "protein", "enzyme",
        "membrane", "tissue", "organ", "muscle", "nerve", "brain", "blood",
    ]

    _VERBS = [
        "is", "are", "was", "were", "has", "have", "had", "does", "do", "did",
        "equals", "defines", "represents", "shows", "demonstrates", "proves",
        "determines", "calculates", "computes", "measures", "estimates", "predicts",
        "increases", "decreases", "changes", "varies", "depends", "relates",
        "contains", "includes", "requires", "produces", "creates", "generates",
        "converts", "transforms", "applies", "uses", "combines", "separates",
        "moves", "flows", "transfers", "absorbs", "emits", "reflects", "refracts",
        "divides", "multiplies", "adds", "subtracts", "integrates", "differentiates",
        "exists", "occurs", "happens", "begins", "ends", "continues", "repeats",
        "describes", "explains", "illustrates", "summarizes", "analyzes", "evaluates",
        "compare", "contrast", "classify", "identify", "define", "list", "outline",
        "review", "study", "learn", "understand", "remember", "practice", "solve",
        "find", "check", "verify", "test", "examine", "investigate", "explore",
        "read", "write", "draw", "sketch", "label", "highlight", "underline",
        "note", "observe", "record", "measure", "count", "calculate", "estimate",
    ]

    _ADJECTIVES = [
        "the", "a", "an", "this", "that", "each", "every", "all", "both",
        "important", "key", "main", "primary", "secondary", "basic", "fundamental",
        "simple", "complex", "linear", "nonlinear", "positive", "negative",
        "large", "small", "high", "low", "long", "short", "fast", "slow",
        "new", "old", "first", "second", "third", "last", "next", "previous",
        "total", "partial", "complete", "average", "maximum", "minimum",
        "equal", "similar", "different", "opposite", "parallel", "perpendicular",
        "constant", "variable", "independent", "dependent", "relative", "absolute",
        "correct", "wrong", "true", "false", "valid", "invalid", "possible",
        "chemical", "physical", "biological", "mathematical", "statistical",
        "electric", "magnetic", "thermal", "nuclear", "atomic", "molecular",
        "organic", "inorganic", "natural", "synthetic", "experimental", "theoretical",
    ]

    _CONNECTORS = [
        "and", "or", "but", "so", "because", "therefore", "however", "thus",
        "when", "where", "while", "since", "if", "then", "also", "not",
        "with", "without", "between", "among", "through", "across", "along",
        "of", "in", "on", "at", "to", "from", "by", "for", "about", "into",
    ]

    _NOTE_PREFIXES = [
        "Note:", "NB:", "Important:", "Remember:", "TODO:", "FIXME:", "Review:",
        "Key point:", "Definition:", "Example:", "Hint:", "Warning:", "Caution:",
        "See also:", "Compare:", "Recall:", "Summary:", "Question:", "Answer:",
        "Step 1:", "Step 2:", "Step 3:", "Part A:", "Part B:", "Theorem:",
        "Proof:", "Given:", "Find:", "Show that:", "Assume:", "Let", "Suppose",
    ]

    _SENTENCE_TEMPLATES = [
        "{adj} {noun} {verb} {adj} {noun}",
        "{noun} {verb} {conn} {noun} {verb}",
        "{prefix} {adj} {noun} {verb} {noun}",
        "{noun} {conn} {noun} {verb} {adj}",
        "{verb} {adj} {noun} {conn} {noun}",
        "{adj} {noun} {verb} {conn} {adj} {noun} {verb}",
        "{prefix} {verb} {adj} {noun}",
        "{noun} {verb} {noun} {conn} {adj} {noun}",
        "{adj} {noun} {conn} {adj} {noun}",
        "{prefix} {noun} {verb} {adj}",
    ]

    # Fixed phrases for realistic complete sentences
    _FIXED_TEXTS = [
        # Academic
        "The mitochondria is the powerhouse of the cell",
        "Newton's second law states that force equals mass times acceleration",
        "Photosynthesis converts carbon dioxide and water into glucose",
        "Osmosis is the movement of water across a semipermeable membrane",
        "The French Revolution began in 1789",
        "DNA replication occurs during the S phase of the cell cycle",
        "Supply and demand determine market equilibrium price",
        "An object in motion stays in motion unless acted upon by an external force",
        "The area under the curve represents the definite integral",
        "Electrons orbit the nucleus in discrete energy levels",
        "Chemical bonds form when atoms share or transfer electrons",
        "The speed of light is approximately 300000 km per second",
        "Natural selection drives the evolution of species over time",
        "The pH scale measures the acidity or alkalinity of a solution",
        "Entropy always increases in an isolated system",
        "The Pythagorean theorem relates the sides of a right triangle",
        "Kinetic energy equals one half mass times velocity squared",
        "Ohm's law states voltage equals current times resistance",
        "The central limit theorem describes the distribution of sample means",
        "Homeostasis maintains stable internal conditions in organisms",
        "Gravity is the force of attraction between masses",
        "A catalyst lowers the activation energy of a reaction",
        "The double helix structure of DNA was discovered in 1953",
        "Standard deviation measures the spread of data around the mean",
        "Tectonic plates move and interact at plate boundaries",
        "The periodic table organizes elements by atomic number",
        "Momentum is conserved in all closed systems",
        "Diffusion moves particles from high to low concentration",
        "The binomial theorem expands powers of sums",
        "Photons behave as both waves and particles",
        # Notes-style
        "important: review chapter 5 before exam",
        "TODO: finish lab report by Friday",
        "see also: page 42 for diagram",
        "Note: this contradicts the textbook on page 89",
        "remember to ask professor about this derivation",
        "key concept: conservation of energy in closed systems",
        "compare figure 3a with figure 3b for differences",
        "rewrite this section more clearly",
        "check: does this formula work for negative values",
        "come back to this problem later",
        "need more practice with integration by parts",
        "review notes from last Tuesday lecture",
        "read sections 4.1 through 4.5 before next class",
        "this is similar to problem 7 from homework 3",
        "ask TA about grading criteria for the project",
        # Pangrams and character coverage
        "the quick brown fox jumps over the lazy dog",
        "pack my box with five dozen liquor jugs",
        "how vexingly quick daft zebras jump",
        "the five boxing wizards jump quickly",
        "bright vixens jump dozy fowl quack",
        "abcdefghijklmnopqrstuvwxyz",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "0123456789",
        "0 1 2 3 4 5 6 7 8 9 10",
        # Lists and structured
        "step 1 step 2 step 3",
        "Monday Tuesday Wednesday Thursday Friday",
        "January February March April May June",
        "first second third fourth fifth",
        "red orange yellow green blue purple",
        "north south east west",
        # Numbers and dates
        "January 15 2024",
        "page 127",
        "3.14159",
        "100 percent",
        "25 degrees Celsius",
        "chapter 12 section 3",
        "figure 4.2",
        "table 7",
        "equation 15",
        "problem set 8",
    ]

    def __init__(self, font_dir: str = "./fonts/handwriting", corpus_file: Optional[str] = None):
        """
        Args:
            font_dir: Kept for API compatibility. Not required — stroke definitions
                      are built-in. The directory path is used to auto-detect corpus.txt.
            corpus_file: Optional path to a text file with one sentence per line.
                         If provided, these sentences are mixed into the text generation pool.
        """
        self.font_dir = Path(font_dir)
        self.font_paths: List[str] = []  # kept for API compat with train.py
        self._external_corpus: List[str] = []

        # Find font files (for reporting only — not used for stroke generation)
        if self.font_dir.exists():
            for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
                self.font_paths.extend(
                    sorted(glob.glob(str(self.font_dir / ext)))
                )

        print(f"  Writing stroke definitions: {len(_WRITING_STROKES)} characters defined")

        # Load external corpus if provided
        if corpus_file and os.path.exists(corpus_file):
            with open(corpus_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and len(line) < 200:  # skip excessively long lines
                        self._external_corpus.append(line)
            if self._external_corpus:
                print(f"  Loaded {len(self._external_corpus)} sentences from external corpus")
        else:
            # Auto-detect corpus file next to font dir
            auto_corpus = Path(font_dir).parent / "corpus.txt"
            if auto_corpus.exists():
                with open(auto_corpus, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and len(line) < 200:
                            self._external_corpus.append(line)
                if self._external_corpus:
                    print(f"  Auto-loaded {len(self._external_corpus)} sentences from {auto_corpus}")

    def generate(
        self,
        text: str,
        style: Optional[HandwritingStyle] = None,
    ) -> Optional[Tuple[List[Stroke], str]]:
        """Generate synthetic handwriting strokes for a text string.

        Uses pre-defined writing stroke decompositions (correct stroke count,
        direction, order) rather than font outline contours. The font is used
        only for character advance widths (spacing).

        Args:
            text: The text to render as handwriting
            style: HandwritingStyle parameters (random if None)

        Returns:
            (list of Stroke objects, text_label) or None if generation fails
        """
        if style is None:
            style = self._random_style()

        all_strokes: List[Stroke] = []
        cursor_x = 0.0
        cursor_y = 0.0
        stroke_id = 0
        global_time = 0.0

        for i, char in enumerate(text):
            if char == ' ':
                cursor_x += 0.15 * style.word_spacing
                global_time += np.random.uniform(0.25, 0.50)
                continue

            # Get writing strokes (human-correct stroke decomposition)
            char_strokes = _get_writing_strokes(char)
            if char_strokes is None:
                # Character not defined — skip and advance cursor
                cursor_x += 0.08
                continue

            if not char_strokes:
                cursor_x += 0.08
                continue

            # Compute character bounds for normalization
            all_pts = np.concatenate(char_strokes)
            if len(all_pts) == 0:
                cursor_x += 0.08
                continue

            char_min = all_pts.min(axis=0)
            char_max = all_pts.max(axis=0)
            char_width = max(char_max[0] - char_min[0], 1e-6)
            char_height = max(char_max[1] - char_min[1], 1e-6)

            # Normalize character to target height, preserve aspect ratio
            target_height = 0.06 * (1 + np.random.uniform(-style.size_variation, style.size_variation))
            scale = target_height / char_height

            # Apply slant
            slant = style.slant_angle + np.random.uniform(-0.05, 0.05)

            for raw_pts in char_strokes:
                # Normalize to target size
                pts = raw_pts.copy()
                pts[:, 0] = (pts[:, 0] - char_min[0]) * scale
                pts[:, 1] = (pts[:, 1] - char_min[1]) * scale

                # Writing strokes are already in y-down (top-down) convention
                # No Y flip needed (unlike font outline contours)

                # Apply slant
                if abs(slant) > 1e-4:
                    pts[:, 0] += pts[:, 1] * np.tan(slant)

                # Baseline wobble
                baseline_offset = style.baseline_wobble * np.sin(
                    2 * np.pi * (cursor_x + pts[:, 0]) * np.random.uniform(2, 5)
                )
                pts[:, 1] += baseline_offset

                # Position on page
                pts[:, 0] += cursor_x
                pts[:, 1] += cursor_y

                # Add point-level jitter
                pts += np.random.normal(0, style.stroke_jitter, pts.shape)

                # Maybe connect to previous stroke (cursive-like)
                if (all_strokes and np.random.random() < style.connect_prob
                        and all_strokes[-1].num_points > 0):
                    prev_end = all_strokes[-1].points[-1, :2]
                    curr_start = pts[0]
                    n_connect = np.random.randint(3, 6)
                    t = np.linspace(0, 1, n_connect)[1:]  # skip first (duplicate)
                    connect_x = prev_end[0] + (curr_start[0] - prev_end[0]) * t
                    connect_y = prev_end[1] + (curr_start[1] - prev_end[1]) * t
                    connect_y += np.random.normal(0, 0.002, len(t))  # slight arc
                    connect_pts = np.column_stack([connect_x, connect_y])
                    pts = np.vstack([connect_pts, pts])

                # Create stroke with full 6 features
                stroke = Stroke.from_xy(
                    x=pts[:, 0].astype(np.float32),
                    y=pts[:, 1].astype(np.float32),
                    stroke_id=stroke_id,
                )

                # Apply time offset
                if stroke.num_points > 0:
                    speed_scale = 1.0 + np.random.uniform(
                        -style.speed_variation, style.speed_variation
                    )
                    stroke.points[:, 3] *= speed_scale
                    stroke.points[:, 3] += global_time
                    global_time = stroke.points[-1, 3] + np.random.uniform(0.05, 0.15)

                all_strokes.append(stroke)
                stroke_id += 1

            # Advance cursor
            cursor_x += char_width * scale * style.letter_spacing + np.random.uniform(0.005, 0.015)

            # Inter-character pause
            global_time += np.random.uniform(0.09, 0.25)

        if not all_strokes:
            return None

        return all_strokes, text

    def _random_style(self) -> HandwritingStyle:
        """Generate a random handwriting style."""
        return HandwritingStyle(
            slant_angle=np.random.uniform(-0.15, 0.25),
            letter_spacing=np.random.uniform(0.8, 1.3),
            word_spacing=np.random.uniform(0.7, 1.5),
            size_variation=np.random.uniform(0.02, 0.1),
            baseline_wobble=np.random.uniform(0.001, 0.006),
            stroke_jitter=np.random.uniform(0.001, 0.004),
            speed_variation=np.random.uniform(0.05, 0.25),
            pen_lift_prob=np.random.uniform(0.0, 0.2),
            connect_prob=np.random.uniform(0.0, 0.6),
        )

    def _generate_random_text(self) -> str:
        """Generate a random text string with high vocabulary diversity.

        Uses procedural generation from word banks + templates to ensure
        the model sees a huge variety of words and sentence structures,
        not just the same 30 phrases on repeat.
        """
        # If external corpus is available, use it 40% of the time
        if self._external_corpus and np.random.random() < 0.4:
            text = self._external_corpus[np.random.randint(len(self._external_corpus))]
            # Occasionally take a substring
            if len(text) > 10 and np.random.random() < 0.2:
                words = text.split()
                start = np.random.randint(0, max(1, len(words) // 2))
                end = np.random.randint(start + 1, len(words) + 1)
                text = " ".join(words[start:end])
            return text

        r = np.random.random()

        if r < 0.35:
            # Procedurally generated sentence from template
            template = self._SENTENCE_TEMPLATES[np.random.randint(len(self._SENTENCE_TEMPLATES))]
            text = template.format(
                noun=self._NOUNS[np.random.randint(len(self._NOUNS))],
                verb=self._VERBS[np.random.randint(len(self._VERBS))],
                adj=self._ADJECTIVES[np.random.randint(len(self._ADJECTIVES))],
                conn=self._CONNECTORS[np.random.randint(len(self._CONNECTORS))],
                prefix=self._NOTE_PREFIXES[np.random.randint(len(self._NOTE_PREFIXES))],
            )
            return text

        elif r < 0.55:
            # Fixed realistic sentence
            return self._FIXED_TEXTS[np.random.randint(len(self._FIXED_TEXTS))]

        elif r < 0.70:
            # Random word sequence (1-6 words) — forces character-level learning
            n_words = np.random.randint(1, 7)
            pool = self._NOUNS + self._VERBS + self._ADJECTIVES
            words = [pool[np.random.randint(len(pool))] for _ in range(n_words)]
            return " ".join(words)

        elif r < 0.80:
            # Note-style with prefix
            prefix = self._NOTE_PREFIXES[np.random.randint(len(self._NOTE_PREFIXES))]
            n_words = np.random.randint(2, 6)
            pool = self._NOUNS + self._VERBS + self._ADJECTIVES
            words = [pool[np.random.randint(len(pool))] for _ in range(n_words)]
            return f"{prefix} {' '.join(words)}"

        elif r < 0.90:
            # Number-heavy content
            patterns = [
                f"{np.random.randint(1, 100)} + {np.random.randint(1, 100)} = {np.random.randint(1, 200)}",
                f"page {np.random.randint(1, 500)}",
                f"chapter {np.random.randint(1, 30)} section {np.random.randint(1, 15)}",
                f"figure {np.random.randint(1, 20)}.{np.random.randint(1, 10)}",
                f"{np.random.randint(1, 12)}/{np.random.randint(1, 28)}/{np.random.randint(2020, 2027)}",
                f"{np.random.uniform(0, 100):.2f}",
                f"problem {np.random.randint(1, 50)}",
                f"step {np.random.randint(1, 10)}: {self._VERBS[np.random.randint(len(self._VERBS))]} {self._NOUNS[np.random.randint(len(self._NOUNS))]}",
                f"table {np.random.randint(1, 15)}: {self._NOUNS[np.random.randint(len(self._NOUNS))]} {self._NOUNS[np.random.randint(len(self._NOUNS))]}",
            ]
            return patterns[np.random.randint(len(patterns))]

        else:
            # Single word — ensures every word gets individual attention
            pool = self._NOUNS + self._VERBS + self._ADJECTIVES
            return pool[np.random.randint(len(pool))]

    def _get_text(self, texts: Optional[List[str]]) -> str:
        """Get a text string — from provided corpus or procedural generation."""
        if texts is not None:
            # Use provided corpus with occasional substring
            text = texts[np.random.randint(len(texts))]
            if len(text) > 5 and np.random.random() < 0.3:
                start = np.random.randint(0, len(text) // 2)
                end = np.random.randint(start + 3, len(text))
                text = text[start:end].strip()
            return text
        return self._generate_random_text()

    def generate_batch(
        self,
        texts: Optional[List[str]] = None,
        count: int = 1000,
    ) -> List[Tuple[List[Stroke], str]]:
        """Generate a batch of synthetic handwriting samples.

        Args:
            texts: List of texts to render. If None, uses procedural generation
                   with built-in word banks for maximum vocabulary diversity.
            count: Number of samples to generate.

        Returns:
            List of (strokes, text_label) tuples.
        """
        samples = []
        for _ in range(count):
            text = self._get_text(texts)
            result = self.generate(text)
            if result is not None:
                samples.append(result)

        return samples

    def iter_samples(
        self,
        texts: Optional[List[str]] = None,
        count: int = 100_000,
    ):
        """Yield synthetic handwriting samples one at a time.

        Args:
            texts: Text corpus. None = procedural generation for max diversity.
            count: Number of samples to generate.
        """
        for _ in range(count):
            text = self._get_text(texts)
            result = self.generate(text)
            if result is not None:
                yield result
