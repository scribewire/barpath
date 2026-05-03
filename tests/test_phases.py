"""Test that add_phases_to_dataframe matches the normal pipeline."""

import sys

sys.path.insert(0, "barpath")

import numpy as np
import pandas as pd
from pipeline.lift_detection_features import add_phases_to_dataframe

# Test with a clean CSV (should use hip-based detection)
df = pd.read_csv("outputs/male/clean/botev_10_clean/final_analysis.csv")
if "frame" in df.columns:
    df = df.set_index("frame")

# Before: phases from the CSV (assigned by normal pipeline)
old_phases = df["bar_phase"].values.astype(int) if "bar_phase" in df.columns else None

# Remove bar_phase and re-detect
if "bar_phase" in df.columns:
    del df["bar_phase"]

df = add_phases_to_dataframe(df, fps=30.0)
new_phases = df["bar_phase"].values.astype(int)

if old_phases is not None:
    match = np.sum(old_phases == new_phases) / len(old_phases)
    print(f"Clean (botev_10): phase match with normal pipeline = {match:.1%}")
    print(
        f"  Old: 0={np.sum(old_phases == 0)} 1={np.sum(old_phases == 1)} 2={np.sum(old_phases == 2)}"
    )
    print(
        f"  New: 0={np.sum(new_phases == 0)} 1={np.sum(new_phases == 1)} 2={np.sum(new_phases == 2)}"
    )
else:
    print(
        f"Clean (botev_10): 0={np.sum(new_phases == 0)} 1={np.sum(new_phases == 1)} 2={np.sum(new_phases == 2)}"
    )

print()

# Test with a jerk CSV
df2 = pd.read_csv("outputs/male/jerk/botev_15_jerk/final_analysis.csv")
if "frame" in df2.columns:
    df2 = df2.set_index("frame")

old_phases2 = (
    df2["bar_phase"].values.astype(int) if "bar_phase" in df2.columns else None
)
if "bar_phase" in df2.columns:
    del df2["bar_phase"]

df2 = add_phases_to_dataframe(df2, fps=30.0)
new_phases2 = df2["bar_phase"].values.astype(int)

if old_phases2 is not None:
    match2 = np.sum(old_phases2 == new_phases2) / len(old_phases2)
    print(f"Jerk (botev_15): phase match with normal pipeline = {match2:.1%}")
    print(
        f"  Old: 0={np.sum(old_phases2 == 0)} 1={np.sum(old_phases2 == 1)} 2={np.sum(old_phases2 == 2)}"
    )
    print(
        f"  New: 0={np.sum(new_phases2 == 0)} 1={np.sum(new_phases2 == 1)} 2={np.sum(new_phases2 == 2)}"
    )
else:
    print(
        f"Jerk (botev_15): 0={np.sum(new_phases2 == 0)} 1={np.sum(new_phases2 == 1)} 2={np.sum(new_phases2 == 2)}"
    )

print()

# Test with a snatch CSV
df3 = pd.read_csv("outputs/male/snatch/botev_1_snatch/final_analysis.csv")
if "frame" in df3.columns:
    df3 = df3.set_index("frame")

old_phases3 = (
    df3["bar_phase"].values.astype(int) if "bar_phase" in df3.columns else None
)
if "bar_phase" in df3.columns:
    del df3["bar_phase"]

df3 = add_phases_to_dataframe(df3, fps=30.0)
new_phases3 = df3["bar_phase"].values.astype(int)

if old_phases3 is not None:
    match3 = np.sum(old_phases3 == new_phases3) / len(old_phases3)
    print(f"Snatch (botev_1): phase match with normal pipeline = {match3:.1%}")
    print(
        f"  Old: 0={np.sum(old_phases3 == 0)} 1={np.sum(old_phases3 == 1)} 2={np.sum(old_phases3 == 2)}"
    )
    print(
        f"  New: 0={np.sum(new_phases3 == 0)} 1={np.sum(new_phases3 == 1)} 2={np.sum(new_phases3 == 2)}"
    )
else:
    print(
        f"Snatch (botev_1): 0={np.sum(new_phases3 == 0)} 1={np.sum(new_phases3 == 1)} 2={np.sum(new_phases3 == 2)}"
    )
