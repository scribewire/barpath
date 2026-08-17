"""Test that add_phases_to_dataframe matches the normal pipeline phase detection.

These tests require the dataset CSVs under outputs/dataset/male/ (gitignored).
They are skipped automatically when the data is not present.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "barpath"))
sys.path.insert(0, str(Path(__file__).parent.parent / "barpath" / "pipeline"))

from pipeline.lift_detection_features import add_phases_to_dataframe

DATA_BASE = Path("outputs/dataset/male")

PHASE_SAMPLES = [
    ("clean", "botev_10_clean", 0.90),
    ("snatch", "botev_1_snatch", 0.90),
    # jerk re-detection uses a simplified dip/drive detector that is looser
    ("jerk", "botev_15_jerk", 0.70),
]


@pytest.mark.parametrize("category,lift_name,min_match", PHASE_SAMPLES)
def test_phase_detection_matches_pipeline(category: str, lift_name: str, min_match: float) -> None:
    """Re-detected phases should agree with the phases saved by the pipeline."""
    csv_path = DATA_BASE / category / lift_name / "final_analysis.csv"
    if not csv_path.exists():
        pytest.skip(f"dataset CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "frame" in df.columns:
        df = df.set_index("frame")

    old_phases = df["bar_phase"].values.astype(int) if "bar_phase" in df.columns else None
    if "bar_phase" in df.columns:
        del df["bar_phase"]

    df = add_phases_to_dataframe(df, fps=30.0)
    new_phases = df["bar_phase"].values.astype(int)

    assert old_phases is not None, "pipeline CSV is missing bar_phase"
    match = np.sum(old_phases == new_phases) / len(old_phases)
    assert match >= min_match, (
        f"{category} ({lift_name}): phase match {match:.1%} < {min_match:.0%} "
        f"old={np.sum(old_phases == 0)}/0 {np.sum(old_phases == 1)}/1 {np.sum(old_phases == 2)}/2 "
        f"new={np.sum(new_phases == 0)}/0 {np.sum(new_phases == 1)}/1 {np.sum(new_phases == 2)}/2"
    )
