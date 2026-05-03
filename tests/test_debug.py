"""Debug velocity and phase boundaries."""

import numpy as np
import pandas as pd

df = pd.read_csv("outputs/male/clean/botev_10_clean/final_analysis.csv")
vel = df["vel_y_smooth"].values.astype(float)
y = df["barbell_y_smooth"].values.astype(float)

# Find peak velocity (most negative = fastest upward)
peak_vel_idx = int(np.argmin(vel))
# Find peak height (lowest y = highest bar)
peak_height_idx = int(np.argmin(y))

print(
    f"Peak vel (most negative): idx={peak_vel_idx}, vel={vel[peak_vel_idx]:.1f}, y={y[peak_vel_idx]:.1f}"
)
print(
    f"Peak height (lowest y):  idx={peak_height_idx}, vel={vel[peak_height_idx]:.1f}, y={y[peak_height_idx]:.1f}"
)
print(f"Total frames: {len(vel)}")

# Check the old phases
phases = df["bar_phase"].values.astype(int)
p1_start = int(np.argmax(phases > 0)) if np.any(phases > 0) else -1
p2_start = int(np.argmax(phases > 1)) if np.any(phases > 1) else -1
print(f"Phase 0->1 transition at frame: {p1_start}")
print(f"Phase 1->2 transition at frame: {p2_start}")

# What does vel > 0.05*max(abs(vel)) give?
thresh = max(10.0, float(np.abs(vel).max()) * 0.05)
bar_up = vel > thresh
print(f"\nvel > {thresh:.1f}: {int(np.sum(bar_up))} frames")
first_up = int(np.argmax(bar_up)) if np.any(bar_up) else -1
print(f"First frame with vel > thresh: {first_up}")

# What about vel < -thresh (negative = upward)?
bar_up2 = vel < -thresh
print(f"\nvel < -{thresh:.1f}: {int(np.sum(bar_up2))} frames")
first_up2 = int(np.argmax(bar_up2)) if np.any(bar_up2) else -1
print(f"First frame with vel < -thresh: {first_up2}")

# Check hip_y_avg
hip = df["hip_y_avg"].values.astype(float)
print(f"\nhip_y_avg: min={hip.min():.1f}, max={hip.max():.1f}")
hip_sm = pd.Series(hip).rolling(9, center=True, min_periods=1).mean()
hip_vel = np.gradient(hip_sm.to_numpy())
print(f"hip_vel: min={hip_vel.min():.3f}, max={hip_vel.max():.3f}")
print(f"hip_vel > 0 (hips dropping): {int(np.sum(hip_vel > 0.01))} frames")
first_drop = int(np.argmax(hip_vel > 0.01)) if np.any(hip_vel > 0.01) else -1
print(f"First hip drop frame: {first_drop}")
