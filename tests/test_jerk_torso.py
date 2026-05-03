"""Debug jerk torso length calculation."""

import sys

sys.path.insert(0, "barpath")

import numpy as np
import pandas as pd
from pipeline.lift_detection_features import _safe_savgol, _to_float_array_1d

df = pd.read_csv("outputs/male/jerk/botev_15_jerk/final_analysis.csv")
if "frame" in df.columns:
    df = df.set_index("frame")

fh = float(df["frame_height"].iloc[0])
print(f"frame_height: {fh}")

# Check if shoulder/hip columns exist
for col in ["left_shoulder_y", "right_shoulder_y", "left_hip_y", "right_hip_y"]:
    if col in df.columns:
        vals = df[col].dropna().values
        print(
            f"{col}: len={len(vals)}, min={vals.min() if len(vals) > 0 else 'N/A':.4f}, max={vals.max() if len(vals) > 0 else 'N/A':.4f}"
        )
    else:
        print(f"{col}: MISSING")

if all(
    c in df.columns
    for c in ["left_shoulder_y", "right_shoulder_y", "left_hip_y", "right_hip_y"]
):
    shoulder_y = (
        (df["left_shoulder_y"].astype(float) + df["right_shoulder_y"].astype(float)) / 2
    ).dropna()
    hip_data_y = (
        (df["left_hip_y"].astype(float) + df["right_hip_y"].astype(float)) / 2
    ).dropna()
    print(
        f"\nshoulder_y: len={len(shoulder_y)}, range=[{shoulder_y.min():.4f}, {shoulder_y.max():.4f}]"
    )
    print(
        f"hip_y: len={len(hip_data_y)}, range=[{hip_data_y.min():.4f}, {hip_data_y.max():.4f}]"
    )
    if len(shoulder_y) > 0 and len(hip_data_y) > 0:
        ml = min(len(shoulder_y), len(hip_data_y))
        torso_dist = abs(
            np.asarray(shoulder_y[:ml], dtype=float)
            - np.asarray(hip_data_y[:ml], dtype=float)
        )
        torso_length_px = float(np.median(torso_dist)) * fh
        min_dip = torso_length_px * 0.30
        print(f"\ntorso_dist: median={float(np.median(torso_dist)):.4f}")
        print(f"torso_length_px: {torso_length_px:.1f}px")
        print(f"min_dip_depth_px: {min_dip:.1f}px")

# Check actual dip displacement
y = df["barbell_y_smooth"].interpolate().bfill().ffill().values.astype(float)
n = len(y)
y_smooth = _safe_savgol(
    _to_float_array_1d(y), max_win=min(21, n if n % 2 == 1 else n - 1)
)
ka = (
    df[["left_knee_angle", "right_knee_angle"]]
    .mean(axis=1)
    .interpolate()
    .bfill()
    .ffill()
)
ka_arr = ka.values.astype(np.float64)
ka_vel = np.gradient(ka_arr) * 30
ka_vel_sm = _safe_savgol(ka_vel, max_win=min(7, n // 2 * 2 + 1), polyorder=3)

knee_bending = ka_vel_sm < -20.0
if np.any(knee_bending):
    idx = int(np.argmax(knee_bending))
    subsequent = ka_vel_sm[idx:]
    stop = subsequent >= -20.0
    if np.any(stop):
        dip_end = idx + int(np.argmax(stop))
    else:
        dip_end = n - 1
    y_start = float(y_smooth[idx])
    y_lowest = float(np.max(y_smooth[idx : dip_end + 1]))
    displacement = y_lowest - y_start
    print(f"\nFirst dip: idx={idx}, dip_end={dip_end}")
    print(
        f"y_start={y_start:.1f}, y_lowest={y_lowest:.1f}, displacement={displacement:.1f}px"
    )
