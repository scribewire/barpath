"""Live Preview Window Features.

Extracts scale-invariant features from partial windows that work
for real-time lift classification in the live preview.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def extract_window_features(df: pd.DataFrame) -> Dict[str, float]:
    """Extract features from a partial window for live preview classification.

    All features are normalized to be scale-invariant and work on ANY
    partial window, not just complete lifts.

    Args:
        df: DataFrame with at minimum barbell_y_smooth column

    Returns:
        Dict of feature name -> value
    """
    if len(df) < 10 or "barbell_y_smooth" not in df.columns:
        return {}

    frame_h = (
        float(df["frame_height"].iloc[0]) if "frame_height" in df.columns else 720.0
    )
    if frame_h <= 0:
        frame_h = 720.0

    bar_y = _to_float_array(df["barbell_y_smooth"].values) / frame_h
    n = len(bar_y)

    features: Dict[str, float] = {}

    # 1. Start position - INSTANT discriminator (jerk vs clean/snatch)
    features["start_y_norm"] = float(bar_y[0])

    # 2. End position
    features["end_y_norm"] = float(bar_y[-1])

    # 3. Peak height - PRIMARY discriminator (clean vs snatch)
    peak_idx = int(np.argmin(bar_y))
    features["peak_y_norm"] = float(bar_y[peak_idx])

    # 4. Vertical range
    features["y_range"] = float(bar_y.max() - bar_y.min())

    # 5. Peak-to-start ratio (how much of the start height was covered)
    features["peak_to_start_ratio"] = float(
        (bar_y[0] - bar_y[peak_idx]) / (bar_y[0] + 1e-6)
    )

    # 4-7. Velocity features
    if "vel_y_smooth" in df.columns:
        vel = _to_float_array(df["vel_y_smooth"].values) / frame_h
        features["max_upward_vel"] = float(np.min(vel))
        features["max_downward_vel"] = float(np.max(vel))
        features["vel_range"] = float(np.max(vel) - np.min(vel))
        features["vel_zero_crossings"] = float(np.sum((vel[:-1] * vel[1:]) < 0))
    else:
        dy = np.diff(bar_y)
        features["max_upward_vel"] = float(np.min(dy))
        features["max_downward_vel"] = float(np.max(dy))
        features["vel_range"] = float(np.max(dy) - np.min(dy))
        features["vel_zero_crossings"] = float(np.sum((dy[:-1] * dy[1:]) < 0))

    # 8-9. Dip detection (jerk signature)
    if peak_idx > 2 and peak_idx < n - 2:
        post_peak_max = float(np.max(bar_y[peak_idx:]))
        features["has_dip"] = 1.0 if post_peak_max > bar_y[peak_idx] + 0.02 else 0.0
        features["dip_depth_norm"] = float(max(0.0, post_peak_max - bar_y[peak_idx]))
    else:
        features["has_dip"] = 0.0
        features["dip_depth_norm"] = 0.0

    # 10. Peak position (0=start, 1=end)
    features["peak_position"] = float(peak_idx / n)

    # 11. Path efficiency
    if "barbell_x_smooth" in df.columns:
        frame_w = (
            float(df["frame_width"].iloc[0]) if "frame_width" in df.columns else 1280.0
        )
        bar_x = _to_float_array(df["barbell_x_smooth"].values) / frame_w
        dx = np.diff(bar_x)
        dy = np.diff(bar_y)
        total_dist = float(np.sum(np.sqrt(dx**2 + dy**2)))
        straight_dist = float(
            np.sqrt((bar_x[-1] - bar_x[0]) ** 2 + (bar_y[-1] - bar_y[0]) ** 2)
        )
        features["path_efficiency"] = (
            straight_dist / total_dist if total_dist > 0 else 1.0
        )
    else:
        features["path_efficiency"] = 1.0

    # 12-13. Velocity profile halves
    if "vel_y_smooth" in df.columns:
        vel = _to_float_array(df["vel_y_smooth"].values) / frame_h
        mid = len(vel) // 2
        features["mean_vel_first_half"] = float(np.mean(vel[:mid]))
        features["mean_vel_second_half"] = float(np.mean(vel[mid:]))
    else:
        mid = n // 2
        features["mean_vel_first_half"] = float(np.mean(np.diff(bar_y[:mid])))
        features["mean_vel_second_half"] = float(np.mean(np.diff(bar_y[mid:])))

    # 14. Window duration in frames
    features["window_duration_frames"] = float(n)

    # 15. Post-peak descent
    if peak_idx < n - 1:
        features["post_peak_descent"] = float(bar_y[-1] - bar_y[peak_idx])
    else:
        features["post_peak_descent"] = 0.0

    # 16. S-curve score
    if "barbell_x_smooth" in df.columns and n > 20:
        frame_w = (
            float(df["frame_width"].iloc[0]) if "frame_width" in df.columns else 1280.0
        )
        bar_x = _to_float_array(df["barbell_x_smooth"].values) / frame_w
        x_range = bar_x.max() - bar_x.min()
        if x_range > 0:
            x_norm = (bar_x - bar_x.min()) / x_range
            features["s_curve_score"] = float(np.sum(np.abs(np.diff(x_norm))))
        else:
            features["s_curve_score"] = 0.0
    else:
        features["s_curve_score"] = 0.0

    # 17. Acceleration peak
    if "accel_y_smooth" in df.columns:
        accel = _to_float_array(df["accel_y_smooth"].values) / frame_h
        features["max_accel"] = float(np.max(np.abs(accel)))
    else:
        features["max_accel"] = 0.0

    # 18. Bar vs eye level (snatch goes well above eyes, clean does not)
    eye_y = None
    if "left_eye_y" in df.columns and "right_eye_y" in df.columns:
        left_eye = df["left_eye_y"].dropna()
        right_eye = df["right_eye_y"].dropna()
        if len(left_eye) > 0 and len(right_eye) > 0:
            eye_y = float((left_eye.values.mean() + right_eye.values.mean()) / 2) / frame_h
    elif "left_eye_y" in df.columns:
        eye_vals = df["left_eye_y"].dropna()
        if len(eye_vals) > 0:
            eye_y = float(eye_vals.values.mean()) / frame_h

    if eye_y is not None and eye_y > 0:
        features["bar_above_eye"] = 1.0 if bar_y[peak_idx] < eye_y - 0.03 else 0.0
        features["max_above_eye"] = float(max(0.0, eye_y - bar_y[peak_idx]))
    else:
        features["bar_above_eye"] = 0.0
        features["max_above_eye"] = 0.0

    # 19. Shoulder proximity (clean spends time near shoulder, snatch doesn't)
    if "left_shoulder_y" in df.columns and "right_shoulder_y" in df.columns:
        left_sh = df["left_shoulder_y"].dropna()
        right_sh = df["right_shoulder_y"].dropna()
        if len(left_sh) > 0 and len(right_sh) > 0:
            shoulder_y = float((left_sh.mean() + right_sh.mean()) / 2) / frame_h
            near_shoulder = np.abs(bar_y - shoulder_y) < 0.05
            features["pct_near_shoulder"] = float(np.mean(near_shoulder))
        else:
            features["pct_near_shoulder"] = 0.0
    else:
        features["pct_near_shoulder"] = 0.0

    # 20. Jerk early indicator: bar starts going down within first 30%
    if n > 10:
        early_end = max(3, n // 3)
        early_dy = np.diff(bar_y[:early_end])
        features["early_downward"] = 1.0 if np.any(early_dy > 0.01) else 0.0
    else:
        features["early_downward"] = 0.0

    # 21. Final bar height category
    features["final_high"] = 1.0 if bar_y[-1] < 0.2 else 0.0
    features["final_mid"] = 1.0 if 0.2 <= bar_y[-1] < 0.4 else 0.0
    features["final_low"] = 1.0 if bar_y[-1] >= 0.4 else 0.0

    return features


def _to_float_array(values) -> NDArray[np.float64]:
    """Convert values to float64 array."""
    return np.asarray(values, dtype=np.float64)


if __name__ == "__main__":
    # Quick test
    from pathlib import Path

    from .live_training_data import extract_live_windows_from_csv

    # Test on a snatch CSV
    csv_path = Path("outputs/male/snatch/botev_1_snatch/final_analysis.csv")
    windows = extract_live_windows_from_csv(csv_path, max_window_frames=60)

    if windows:
        print(f"Found {len(windows)} windows")
        features = extract_window_features(windows[5])
        print(f"\nSample features (window length={len(windows[5])}):")
        for k, v in sorted(features.items()):
            print(f"  {k}: {v:.4f}")
