"""
On-the-fly data augmentation for stroke pages.

Applied during training to increase diversity and prevent overfitting.
All augmentations operate on normalized [0,1] coordinates.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional
from .stroke import Stroke, PageAnnotation, Group, Relationship
from config import AugmentationConfig


def augment_page(
    page: PageAnnotation,
    cfg: AugmentationConfig,
) -> PageAnnotation:
    """Apply a suite of augmentations to a page annotation.

    Returns a new PageAnnotation with augmented strokes.
    Groups, relationships, and content labels are preserved.
    """
    if not cfg.enabled:
        return page

    strokes = [Stroke(points=s.points.copy(), stroke_id=s.stroke_id, group_id=s.group_id)
               for s in page.strokes]

    # 1. Group dropout: randomly remove entire groups
    if cfg.group_dropout_prob > 0:
        strokes, groups, rels = _group_dropout(
            strokes, page.groups, page.relationships, cfg.group_dropout_prob
        )
    else:
        groups = list(page.groups)
        rels = list(page.relationships)

    # 2. Spatial jitter: translate all strokes
    if cfg.spatial_jitter > 0:
        strokes = _spatial_jitter(strokes, cfg.spatial_jitter)

    # 3. Scale variation
    if cfg.scale_range != (1.0, 1.0):
        strokes = _random_scale(strokes, cfg.scale_range)

    # 4. Rotation
    if cfg.rotation_range > 0:
        strokes = _random_rotation(strokes, cfg.rotation_range)

    # 5. Stroke point dropout
    if cfg.stroke_point_dropout > 0:
        strokes = _point_dropout(strokes, cfg.stroke_point_dropout)

    # 6. Pressure noise
    if cfg.pressure_noise_std > 0:
        strokes = _pressure_noise(strokes, cfg.pressure_noise_std)

    # 7. Speed perturbation (timestamp scaling)
    if cfg.speed_scale_range != (1.0, 1.0):
        strokes = _speed_perturbation(strokes, cfg.speed_scale_range)

    # 8. Altitude/azimuth noise
    if cfg.altitude_noise_std > 0 or cfg.azimuth_noise_std > 0:
        strokes = _angle_noise(strokes, cfg.altitude_noise_std, cfg.azimuth_noise_std)

    # 9. Feature channel dropout — force model to not depend on synthetic features
    # Academic datasets have no real pressure/altitude/azimuth, so the model
    # must learn to work without them.
    strokes = _feature_channel_dropout(strokes, cfg.feature_channel_dropout_prob)

    # Clamp all values to [0, 1]
    for s in strokes:
        s.points = np.clip(s.points, 0.0, 1.0)

    return PageAnnotation(
        strokes=strokes,
        groups=groups,
        relationships=rels,
        page_width=page.page_width,
        page_height=page.page_height,
    )


def _group_dropout(
    strokes: List[Stroke],
    groups: List[Group],
    relationships: List[Relationship],
    prob: float,
) -> tuple:
    """Randomly remove entire groups with probability `prob`."""
    # Don't drop all groups
    keep_groups = []
    drop_ids = set()
    for g in groups:
        if np.random.random() < prob and len(groups) - len(drop_ids) > 1:
            drop_ids.add(g.group_id)
        else:
            keep_groups.append(g)

    if not drop_ids:
        return strokes, groups, relationships

    # Remove strokes belonging to dropped groups
    kept_strokes = [s for s in strokes if s.group_id not in drop_ids]

    # Remove relationships involving dropped groups
    kept_rels = [
        r for r in relationships
        if r.source_group_id not in drop_ids and r.target_group_id not in drop_ids
    ]

    return kept_strokes, keep_groups, kept_rels


def _spatial_jitter(strokes: List[Stroke], max_jitter: float) -> List[Stroke]:
    """Random translation of all strokes."""
    dx = np.random.uniform(-max_jitter, max_jitter)
    dy = np.random.uniform(-max_jitter, max_jitter)
    for s in strokes:
        s.points[:, 0] += dx
        s.points[:, 1] += dy
    return strokes


def _random_scale(strokes: List[Stroke], scale_range: tuple) -> List[Stroke]:
    """Random uniform scaling around page center."""
    scale = np.random.uniform(*scale_range)
    center_x, center_y = 0.5, 0.5
    for s in strokes:
        s.points[:, 0] = center_x + (s.points[:, 0] - center_x) * scale
        s.points[:, 1] = center_y + (s.points[:, 1] - center_y) * scale
    return strokes


def _random_rotation(strokes: List[Stroke], max_degrees: float) -> List[Stroke]:
    """Random rotation around page center."""
    angle = np.radians(np.random.uniform(-max_degrees, max_degrees))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    cx, cy = 0.5, 0.5
    for s in strokes:
        x = s.points[:, 0] - cx
        y = s.points[:, 1] - cy
        s.points[:, 0] = x * cos_a - y * sin_a + cx
        s.points[:, 1] = x * sin_a + y * cos_a + cy
    return strokes


def _point_dropout(strokes: List[Stroke], prob: float) -> List[Stroke]:
    """Randomly drop individual points from strokes."""
    result = []
    for s in strokes:
        if s.num_points <= 3:
            result.append(s)
            continue
        mask = np.random.random(s.num_points) > prob
        # Always keep first and last point
        mask[0] = True
        mask[-1] = True
        if mask.sum() >= 3:
            s.points = s.points[mask]
        result.append(s)
    return result


def _pressure_noise(strokes: List[Stroke], std: float) -> List[Stroke]:
    """Add Gaussian noise to pressure values."""
    for s in strokes:
        noise = np.random.normal(0, std, s.num_points)
        s.points[:, 2] = np.clip(s.points[:, 2] + noise, 0.0, 1.0)
    return strokes


def _speed_perturbation(strokes: List[Stroke], scale_range: tuple) -> List[Stroke]:
    """Scale timestamps within each stroke to simulate speed variation."""
    for s in strokes:
        scale = np.random.uniform(*scale_range)
        t_start = s.points[0, 3]
        s.points[:, 3] = t_start + (s.points[:, 3] - t_start) * scale
    return strokes


def _angle_noise(strokes: List[Stroke], alt_std: float, azi_std: float) -> List[Stroke]:
    """Add noise to altitude and azimuth angles."""
    for s in strokes:
        if alt_std > 0:
            s.points[:, 4] += np.random.normal(0, alt_std, s.num_points)
            s.points[:, 4] = np.clip(s.points[:, 4], 0.0, 1.0)
        if azi_std > 0:
            s.points[:, 5] += np.random.normal(0, azi_std, s.num_points)
            s.points[:, 5] = np.clip(s.points[:, 5], 0.0, 1.0)
    return strokes


def _feature_channel_dropout(strokes: List[Stroke], prob: float) -> List[Stroke]:
    """Randomly zero out entire feature channels (pressure, altitude, azimuth).

    Academic datasets have no real pressure/altitude/azimuth — all values are
    synthetic. This dropout forces the model to solve tasks primarily from
    x, y geometry and timestamps, treating the other channels as bonus signals.

    Each channel is dropped independently per page (all strokes on the page
    get the same channel zeroed, simulating a sensor being unavailable).
    """
    if prob <= 0:
        return strokes

    # Decide which channels to drop for this page (consistent across all strokes)
    drop_pressure = np.random.random() < prob
    drop_altitude = np.random.random() < prob
    drop_azimuth = np.random.random() < prob

    if not (drop_pressure or drop_altitude or drop_azimuth):
        return strokes

    for s in strokes:
        if drop_pressure:
            s.points[:, 2] = 0.0   # zero pressure
        if drop_altitude:
            s.points[:, 4] = 0.0   # zero altitude
        if drop_azimuth:
            s.points[:, 5] = 0.0   # zero azimuth

    return strokes
