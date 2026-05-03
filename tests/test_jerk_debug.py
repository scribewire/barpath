"""Debug jerk phase detection."""

import sys

sys.path.insert(0, "barpath")

import numpy as np
import pandas as pd
from pipeline.lift_detection_features import _safe_savgol, _to_float_array_1d

df = pd.read_csv("outputs/male/jerk/botev_15_jerk/final_analysis.csv")
if "frame" in df.columns:
    df = df.set_index("frame")

y = df["barbell_y_smooth"].interpolate().bfill().ffill().values.astype(float)
n = len(y)
y_smooth = _safe_savgol(
    _to_float_array_1d(y), max_win=min(21, n if n % 2 == 1 else n - 1)
)
fh = float(df["frame_height"].iloc[0])
y_start_norm = float(y_smooth[0]) / fh
print(
    f"Jerk: y_start={y_smooth[0]:.1f}, frame_h={fh:.0f}, y_start_norm={y_start_norm:.3f}, is_jerk={y_start_norm < 0.5}"
)

vel = np.gradient(y_smooth, 1 / 30)
vel_sm = _safe_savgol(vel, max_win=min(15, n if n % 2 == 1 else n - 1))
print(f"vel_sm: min={vel_sm.min():.1f}, max={vel_sm.max():.1f}")

has_knees = "left_knee_angle" in df.columns and "right_knee_angle" in df.columns
print(f"Has knee angles: {has_knees}")
if has_knees:
    ka = df[["left_knee_angle", "right_knee_angle"]].mean(axis=1)
    ka = ka.interpolate().bfill().ffill()
    ka_arr = ka.values.astype(np.float64)
    ka_vel = np.gradient(ka_arr) * 30
    ka_vel_sm = _safe_savgol(ka_vel, max_win=min(7, n // 2 * 2 + 1), polyorder=3)
    print(f"knee_vel_sm: min={ka_vel_sm.min():.1f}, max={ka_vel_sm.max():.1f}")
    knee_bending = ka_vel_sm < -20.0
    print(f"knee_bending frames: {int(np.sum(knee_bending))}")
    if np.any(knee_bending):
        idx = int(np.argmax(knee_bending))
        print(f"First bending at frame {idx}, knee_vel={ka_vel_sm[idx]:.1f}")

# Check if moving_down detection works (fallback without knee angles)
moving_down = vel_sm > 20.0
print(f"bar moving down (vel > 20): {int(np.sum(moving_down))} frames")
if np.any(moving_down):
    idx = int(np.argmax(moving_down))
    print(f"First moving down at frame {idx}, vel={vel_sm[idx]:.1f}")
