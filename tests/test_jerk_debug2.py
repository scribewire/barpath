"""Debug jerk phase detection - deeper."""

import sys

sys.path.insert(0, "barpath")

import numpy as np
import pandas as pd
from pipeline.lift_detection_features import (
    _safe_savgol,
    _to_float_array_1d,
    _detect_jerk_phases_simple,
)

df = pd.read_csv("outputs/male/jerk/botev_15_jerk/final_analysis.csv")
if "frame" in df.columns:
    df = df.set_index("frame")

y = df["barbell_y_smooth"].interpolate().bfill().ffill().values.astype(float)
n = len(y)
y_smooth = _safe_savgol(
    _to_float_array_1d(y), max_win=min(21, n if n % 2 == 1 else n - 1)
)
fps = 30.0
vel = np.gradient(y_smooth, 1.0 / fps)
vel_smooth = _safe_savgol(vel, max_win=min(15, n if n % 2 == 1 else n - 1))
fh = float(df["frame_height"].iloc[0])

# Knee angles
ka = df[["left_knee_angle", "right_knee_angle"]].mean(axis=1)
ka = ka.interpolate().bfill().ffill()
ka_arr = ka.values.astype(np.float64)

# Run the function directly
phases = _detect_jerk_phases_simple(y_smooth, vel_smooth, ka_arr, fh, fps)
print(
    f"Jerk phases: 0={int(np.sum(phases == 0))} 1={int(np.sum(phases == 1))} 2={int(np.sum(phases == 2))}"
)

# Manual step-through of the detection
ka_vel = np.gradient(ka_arr) * fps
ka_vel_sm = _safe_savgol(ka_vel, max_win=min(7, n // 2 * 2 + 1), polyorder=3)
knee_bending = ka_vel_sm < -20.0
print(f"\nKnee bending frames: {int(np.sum(knee_bending))}")

if np.any(knee_bending):
    bending_indices = np.where(knee_bending)[0]
    for idx in bending_indices[:5]:  # Check first 5
        subsequent = ka_vel_sm[idx:]
        stop = subsequent >= -20.0
        if np.any(stop):
            dip_end = idx + int(np.argmax(stop))
        else:
            dip_end = n - 1
        if dip_end - idx < 3:
            print(f"  idx={idx}: dip_end={dip_end}, too short (delta={dip_end - idx})")
            continue
        y_start = float(y_smooth[idx])
        y_lowest = float(np.max(y_smooth[idx : dip_end + 1]))
        displacement = y_lowest - y_start
        min_dip = fh * 0.09
        print(
            f"  idx={idx}: dip_end={dip_end}, disp={displacement:.1f}px, min_dip={min_dip:.1f}px, ok={displacement >= min_dip}"
        )
        if displacement >= min_dip:
            print(f"  -> DIP DETECTED at frame {idx}")
            break
