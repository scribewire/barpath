"""Debug the new phase detection in detail."""

import sys

sys.path.insert(0, "barpath")

import numpy as np
import pandas as pd
from pipeline.lift_detection_features import (
    _safe_savgol,
    _to_float_array_1d,
    add_phases_to_dataframe,
)

# Test clean
df = pd.read_csv("outputs/male/clean/botev_10_clean/final_analysis.csv")
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

y_start = float(y_smooth[0])
is_jerk = y_start > 0.4
print(f"Clean: y_start={y_start:.3f}, is_jerk={is_jerk}")

# Hip data
hip_raw = df["hip_y_avg"].values.astype(float)
hip_filled = pd.Series(hip_raw).replace(0, np.nan).interpolate().bfill().ffill()
hip_y = hip_filled.values.astype(np.float64)

# Step by step
vel_max = float(np.nanmax(np.abs(vel_smooth)))
vel_threshold = max(10.0, vel_max * 0.05)
bar_moving_up = vel_smooth < -vel_threshold
print(f"  vel_max={vel_max:.1f}, vel_threshold={vel_threshold:.1f}")
print(f"  bar_moving_up: {np.sum(bar_moving_up)} frames")
if np.any(bar_moving_up):
    pull_start = int(np.argmax(bar_moving_up))
    print(f"  pull_start: {pull_start}")

    hip_after = hip_y[pull_start:]
    hw = min(9, len(hip_after) if len(hip_after) % 2 == 1 else len(hip_after) - 1)
    hip_sm = _safe_savgol(_to_float_array_1d(hip_after), max_win=hw, polyorder=3)
    hip_vel = np.gradient(hip_sm)

    hip_std = float(np.std(hip_sm))
    hip_drop_threshold = hip_std * 0.1 if hip_std > 0 else 0.5
    hips_dropping = hip_vel > hip_drop_threshold
    print(f"  hip_std={hip_std:.3f}, hip_drop_threshold={hip_drop_threshold:.3f}")
    print(f"  hips_dropping: {np.sum(hips_dropping)} frames")
    if np.any(hips_dropping):
        pull_under_start = pull_start + int(np.argmax(hips_dropping))
        print(f"  pull_under_start: {pull_under_start}")

        hip_after_pu = hip_y[pull_under_start:]
        hw2 = min(
            9,
            len(hip_after_pu) if len(hip_after_pu) % 2 == 1 else len(hip_after_pu) - 1,
        )
        hip_sm2 = _safe_savgol(
            _to_float_array_1d(hip_after_pu), max_win=hw2, polyorder=3
        )
        hip_vel2 = np.gradient(hip_sm2)
        hips_stopped = hip_vel2 <= hip_drop_threshold * 0.5
        print(f"  hips_stopped: {np.sum(hips_stopped)} frames")
        if np.any(hips_stopped):
            recovery_start = pull_under_start + int(np.argmax(hips_stopped))
            print(f"  recovery_start: {recovery_start}")
        else:
            print("  NO recovery_start found")
    else:
        print("  NO pull_under_start found")
else:
    print("  NO bar_moving_up found")

# Now run the actual function
del df["bar_phase"]
df = add_phases_to_dataframe(df, fps=30.0)
phases = df["bar_phase"].values.astype(int)
print(
    f"\nResult: 0={np.sum(phases == 0)} 1={np.sum(phases == 1)} 2={np.sum(phases == 2)}"
)

# Compare with old
old_df = pd.read_csv("outputs/male/clean/botev_10_clean/final_analysis.csv")
old_phases = old_df["bar_phase"].values.astype(int)
print(
    f"Old:     0={np.sum(old_phases == 0)} 1={np.sum(old_phases == 1)} 2={np.sum(old_phases == 2)}"
)
