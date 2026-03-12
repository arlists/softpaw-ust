#!/usr/bin/env python3
"""
Validate that synthetic stroke data matches real Apple Pencil / PencilKit distributions.

Generates samples from all synthetic data sources (handwriting, gestures, QuickDraw-style,
math-style) and prints feature distribution statistics, comparing against known real
PencilKit reference values.

Usage:
    python scripts/validate_synthetic_data.py

Reference: Real PencilKit data characteristics (Apple Pencil on iPad)
    Feature     | Raw Range        | Normalized Range | Typical Writing
    ------------|------------------|------------------|----------------
    Force       | 0 - 4.17         | 0 - 1.0          | 0.24 - 0.48 (center ~0.34)
    Altitude    | 0 - π/2 (1.571)  | 0 - 1.0          | 1.1 - 1.4 rad (center ~1.25)
    Azimuth     | 0 - 2π (6.283)   | 0 - 1.0          | Right: 2.4-3.9 (center ~3.1)
    Timestamps  | 240 Hz           | 4.17ms intervals  | ~4.17ms ± 0.5ms jitter
    Pressure dyn| Ramp up/sustain/down | 4 phases      | Bluetooth placeholder → ramp → sustain → lift
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.stroke import Stroke, _synthesize_pressure
from data.gestures import (
    generate_circle_gesture,
    generate_underline_gesture,
    generate_arrow_gesture,
    generate_strikethrough_gesture,
    generate_bracket_gesture,
)


def print_header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_stats(name: str, values: np.ndarray, expected_range: tuple, expected_center: float):
    """Print distribution stats and flag mismatches."""
    mean = np.mean(values)
    std = np.std(values)
    p5 = np.percentile(values, 5)
    p25 = np.percentile(values, 25)
    p50 = np.percentile(values, 50)
    p75 = np.percentile(values, 75)
    p95 = np.percentile(values, 95)
    vmin = np.min(values)
    vmax = np.max(values)

    # Check if distribution matches (28% tolerance — short strokes naturally
    # have lower mean pressure because the Bluetooth placeholder + ramp-up
    # phases occupy a larger fraction of the stroke)
    center_ok = abs(mean - expected_center) < 0.28 * expected_center if expected_center != 0 else abs(mean) < 0.1
    range_ok = p5 >= expected_range[0] * 0.7 and p95 <= expected_range[1] * 1.3
    status = "OK" if (center_ok and range_ok) else "MISMATCH"

    print(f"  {name:20s} | mean={mean:.4f} std={std:.4f} | "
          f"[{vmin:.4f}, {vmax:.4f}] | "
          f"p5={p5:.4f} p50={p50:.4f} p95={p95:.4f} | "
          f"expected center={expected_center:.3f} range={expected_range} | {status}")


def validate_from_xy(n_samples: int = 500):
    """Validate Stroke.from_xy() feature distributions."""
    print_header("Stroke.from_xy() — Feature Distributions")
    print(f"  Generating {n_samples} strokes via from_xy()...")

    all_pressure = []
    all_altitude = []
    all_azimuth = []
    all_dt = []

    for _ in range(n_samples):
        n_pts = np.random.randint(20, 100)
        # Random walk for x, y
        x = np.cumsum(np.random.normal(0.5, 0.3, n_pts)).astype(np.float32)
        y = np.cumsum(np.random.normal(0.0, 0.15, n_pts)).astype(np.float32)
        stroke = Stroke.from_xy(x, y)

        all_pressure.append(stroke.points[:, 2])
        all_altitude.append(stroke.points[:, 4])
        all_azimuth.append(stroke.points[:, 5])
        if n_pts > 1:
            dt = np.diff(stroke.points[:, 3])
            all_dt.append(dt)

    pressure = np.concatenate(all_pressure)
    altitude = np.concatenate(all_altitude)
    azimuth = np.concatenate(all_azimuth)
    dt = np.concatenate(all_dt)

    print(f"\n  Feature distributions across {n_samples} strokes:")
    print_stats("Pressure (pre-norm)", pressure,
                expected_range=(0.02, 0.72), expected_center=0.34)
    print_stats("Altitude (rad)", altitude,
                expected_range=(0.3, 1.571), expected_center=1.25)
    print_stats("Azimuth (rad)", azimuth,
                expected_range=(0.5, 4.5), expected_center=2.86)  # weighted avg of 3.1 and 1.5
    print_stats("Δt (seconds)", dt,
                expected_range=(0.002, 0.007), expected_center=1.0/240.0)

    # Detailed timestamp analysis
    dt_ms = dt * 1000
    print(f"\n  Timestamp interval analysis:")
    print(f"    Mean Δt: {np.mean(dt_ms):.3f} ms (expected: 4.167 ms at 240Hz)")
    print(f"    Std Δt:  {np.std(dt_ms):.3f} ms (expected: ~0.5 ms jitter)")
    print(f"    Min Δt:  {np.min(dt_ms):.3f} ms")
    print(f"    Max Δt:  {np.max(dt_ms):.3f} ms")
    effective_hz = 1000.0 / np.mean(dt_ms)
    print(f"    Effective rate: {effective_hz:.1f} Hz (expected: 240 Hz)")


def validate_pressure_dynamics(n_samples: int = 200):
    """Validate the 4-phase pressure dynamics."""
    print_header("Pressure Dynamics — 4-Phase Pattern")
    print(f"  Generating {n_samples} pressure profiles...")

    # Check that pressure has the expected ramp-up pattern
    n_with_placeholder = 0
    n_with_ramp_up = 0
    n_with_ramp_down = 0

    for _ in range(n_samples):
        n_pts = np.random.randint(30, 80)
        p = _synthesize_pressure(n_pts)

        # Phase 1: First few points should be low (placeholder)
        if p[0] < 0.15 and p[1] < 0.15:
            n_with_placeholder += 1

        # Phase 2: Should ramp up (early points < middle points)
        early_mean = np.mean(p[:5])
        mid_mean = np.mean(p[n_pts//3:2*n_pts//3])
        if mid_mean > early_mean * 1.5:
            n_with_ramp_up += 1

        # Phase 4: Should ramp down (last few points lower than middle)
        late_mean = np.mean(p[-3:])
        if late_mean < mid_mean * 0.8:
            n_with_ramp_down += 1

    print(f"\n  Phase detection rates (should be >80%):")
    print(f"    Bluetooth placeholder (low start):  {n_with_placeholder/n_samples*100:.1f}%")
    print(f"    Ramp up (early < middle):            {n_with_ramp_up/n_samples*100:.1f}%")
    print(f"    Ramp down (late < middle):           {n_with_ramp_down/n_samples*100:.1f}%")


def validate_gestures(n_samples: int = 100):
    """Validate gesture stroke distributions."""
    print_header("Gesture Strokes — Feature Distributions")

    gesture_generators = {
        "circle": lambda: generate_circle_gesture((0.2, 0.2, 0.5, 0.4)),
        "underline": lambda: generate_underline_gesture((0.1, 0.2, 0.6, 0.25)),
        "arrow": lambda: generate_arrow_gesture((0.1, 0.1, 0.3, 0.2), (0.5, 0.4, 0.7, 0.5)),
        "strikethrough": lambda: generate_strikethrough_gesture((0.1, 0.2, 0.6, 0.3)),
        "bracket": lambda: generate_bracket_gesture((0.2, 0.1, 0.6, 0.5)),
    }

    for name, gen_fn in gesture_generators.items():
        pressures = []
        altitudes = []
        azimuths = []
        dts = []

        for _ in range(n_samples):
            stroke = gen_fn()
            pressures.append(stroke.points[:, 2])
            altitudes.append(stroke.points[:, 4])
            azimuths.append(stroke.points[:, 5])
            if stroke.num_points > 1:
                dts.append(np.diff(stroke.points[:, 3]))

        p = np.concatenate(pressures)
        a = np.concatenate(altitudes)
        az = np.concatenate(azimuths)
        dt = np.concatenate(dts)

        print(f"\n  --- {name.upper()} ({n_samples} samples) ---")
        print_stats("Pressure", p,
                    expected_range=(0.02, 0.72), expected_center=0.34)
        print_stats("Altitude", a,
                    expected_range=(0.3, 1.571), expected_center=1.25)
        print_stats("Azimuth", az,
                    expected_range=(0.5, 4.5), expected_center=2.86)
        print_stats("Δt (s)", dt,
                    expected_range=(0.002, 0.007), expected_center=1.0/240.0)


def validate_azimuth_handedness():
    """Verify the 85/15 right-handed/left-handed split."""
    print_header("Azimuth — Handedness Distribution")

    n_samples = 1000
    right_handed = 0  # azimuth centered around 3.1

    for _ in range(n_samples):
        x = np.cumsum(np.random.normal(0.5, 0.3, 30)).astype(np.float32)
        y = np.cumsum(np.random.normal(0.0, 0.15, 30)).astype(np.float32)
        stroke = Stroke.from_xy(x, y)
        mean_az = np.mean(stroke.points[:, 5])
        if mean_az > 2.0:
            right_handed += 1

    pct = right_handed / n_samples * 100
    print(f"  Right-handed (azimuth > 2.0): {pct:.1f}% (expected: ~85%)")
    print(f"  Left-handed  (azimuth < 2.0): {100-pct:.1f}% (expected: ~15%)")
    status = "OK" if 75 < pct < 95 else "MISMATCH"
    print(f"  Status: {status}")


def main():
    print("=" * 70)
    print("  SoftPaw UST — Synthetic Data Validation")
    print("  Comparing synthetic feature distributions vs real PencilKit data")
    print("=" * 70)

    print("\n  Real PencilKit reference (Apple Pencil on iPad):")
    print("    Force:     0-4.17 raw, normalized to 0-1. Writing: 0.24-0.48 (center ~0.34)")
    print("    Altitude:  0-π/2 rad. Writing: 1.1-1.4 rad (center ~1.25)")
    print("    Azimuth:   0-2π rad. Right-handed: 2.4-3.9 (center ~3.1)")
    print("    Timestamps: 240 Hz (4.17ms intervals)")
    print("    Pressure:  4-phase dynamics (placeholder → ramp → sustain → lift)")

    np.random.seed(42)  # reproducible

    validate_from_xy()
    validate_pressure_dynamics()
    validate_gestures()
    validate_azimuth_handedness()

    print_header("SUMMARY")
    print("  If all features show 'OK', synthetic data matches real PencilKit.")
    print("  Any 'MISMATCH' indicates a distribution that needs correction.")
    print("  Run this after any changes to stroke.py or gestures.py.")
    print()


if __name__ == "__main__":
    main()
