"""Compare old vs new jerk phases frame by frame."""

import numpy as np
import pandas as pd

df = pd.read_csv("outputs/male/jerk/botev_15_jerk/final_analysis.csv")
phases = df["bar_phase"].values.astype(int)
y = df["barbell_y_smooth"].values.astype(float)

print("Frame-by-frame phase comparison (first 60 frames):")
print(f"{'frame':>5s} {'phase':>5s} {'y':>8s}")
for i in range(min(60, len(phases))):
    marker = ""
    if i > 0 and phases[i] != phases[i - 1]:
        marker = f" <- phase {phases[i - 1]}->{phases[i]}"
    print(f"{i:5d} {phases[i]:5d} {y[i]:8.1f}{marker}")

print(
    f"\nPhase 0 (dip) starts at frame: {int(np.argmax(phases == 0)) if np.any(phases == 0) else 'N/A'}"
)
print(
    f"Phase 1 starts at frame: {int(np.argmax(phases == 1)) if np.any(phases == 1) else 'N/A'}"
)
print(
    f"Phase 2 starts at frame: {int(np.argmax(phases == 2)) if np.any(phases == 2) else 'N/A'}"
)
