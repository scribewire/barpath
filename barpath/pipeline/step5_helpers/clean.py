from __future__ import annotations

from typing import List

import pandas as pd

from .classics_phase_detection import ClassicsPhases


def compute_bar_phase_from_clean_phases(
    df: pd.DataFrame, phases: ClassicsPhases, *, phase_col: str = "bar_phase"
) -> pd.Series:
    """
    Build a per-frame phase label series from clean phase boundaries.

    Output matches the pipeline's convention of integer "phase IDs" that increment
    at each boundary:
      - Phase 0: [t0, t1)
      - Phase 1: [t1, t2)
      - Phase 2: [t2, t3)
      - Phase 3: [t3, t4]  (inclusive end)

    Frames before t0 (if present) are assigned phase 0 by default.
    """
    if df.empty:
        return pd.Series(dtype="int64", name=phase_col)

    idx = df.index
    phase = pd.Series(0, index=idx, dtype="int64", name=phase_col)

    t0 = int(phases["t0"])
    t1 = int(phases["t1"])
    t2 = int(phases["t2"])
    t3 = int(phases["t3"])
    t4 = int(phases["t4"])

    # Assign phases by index slices. `.loc` is inclusive on both ends for label-based indices,
    # so we use half-open ranges by subtracting 1 where appropriate when indices are integers.
    # Because frame indices may not be contiguous, we apply masks instead of slicing by position.
    phase.loc[idx >= t1] = 1
    phase.loc[idx >= t2] = 2
    phase.loc[idx >= t3] = 3

    # If t4 is earlier than some boundary (shouldn't happen, but guard), clamp.
    phase.loc[idx > t4] = phase.loc[idx <= t4].iloc[-1] if (idx <= t4).any() else 0

    # If there are frames before t0 (rare after Step 2 truncation), keep them as 0.
    # Optionally you could set them to -1, but that would break downstream assumptions.
    _ = t0  # kept for clarity

    return phase


def check_clean_faults(df: pd.DataFrame, phases: ClassicsPhases) -> List[str]:
    """
    Clean-specific critique checks (ported from `5_critique_lift.py`).

    Returns a list of critique strings suitable for report output.
    """
    critiques: List[str] = []

    t0, t1, t2, t3, t4 = (
        phases["t0"],
        phases["t1"],
        phases["t2"],
        phases["t3"],
        phases["t4"],
    )

    try:
        frame_height = float(df["frame_height"].iloc[0])
    except Exception:
        frame_height = 0.0

    def get_phase_df(start: int, end: int) -> pd.DataFrame:
        return df.loc[start:end]

    # --- First Pull (t0 -> t1) ---
    p1 = get_phase_df(t0, t1)
    if not p1.empty:
        # Check 1: Positive acceleration
        if "accel_y_smooth" in p1.columns:
            if float(p1["accel_y_smooth"].mean()) < -200:
                critiques.append("First Pull: You are slowing down in the first pull")

        # Check 2: Vertical path
        if "barbell_x_stable" in p1.columns:
            x_start = float(p1["barbell_x_stable"].iloc[0])
            max_dev = float((p1["barbell_x_stable"] - x_start).abs().max())
            if frame_height > 0 and max_dev > (frame_height * 0.08):
                critiques.append(
                    "First Pull: The bar is being kicked out/pulled back too far"
                )

    # --- Second Pull (t1 -> t2) ---
    p2 = get_phase_df(t1, t2)
    if not p2.empty:
        # Check 1: Positive, increasing acceleration
        if "accel_y_smooth" in p2.columns:
            mid = len(p2) // 2
            if mid > 0:
                first_half_mean = float(p2["accel_y_smooth"].iloc[:mid].mean())
                second_half_mean = float(p2["accel_y_smooth"].iloc[mid:].mean())
                if (
                    second_half_mean < first_half_mean
                    or float(p2["accel_y_smooth"].mean()) <= 0
                ):
                    critiques.append("Second Pull: You are hitching in the second pull")

        # Check 2: Flat feet at power position (end of phase)
        if "left_ankle_y" in p2.columns and "right_ankle_y" in p2.columns:
            ankle_mean_series: pd.Series = p2[["left_ankle_y", "right_ankle_y"]].mean(
                axis=1
            )  # type: ignore
            start_y = float(ankle_mean_series.iloc[0])
            end_y = float(ankle_mean_series.iloc[-1])
            if frame_height > 0 and (start_y - end_y) > (frame_height * 0.05):
                critiques.append("Second Pull: You are jumping too soon")

        # Check 3: Straight arms (early bend)
        if "left_elbow_angle" in p2.columns and "right_elbow_angle" in p2.columns:
            mid = len(p2) // 2
            if mid > 0:
                min_elbow_first_half = float(
                    p2[["left_elbow_angle", "right_elbow_angle"]].iloc[:mid].min().min()
                )
                if min_elbow_first_half < 140:
                    critiques.append("Second Pull: You are bending your arms too early")

    # --- Third Pull (t2 -> t3) ---
    p3 = get_phase_df(t2, t3)
    if not p3.empty and "hip_y_avg" in df.columns:
        hip_peak_y = float(df.loc[t2, "hip_y_avg"])
        threshold_drop = frame_height * 0.08 if frame_height > 0 else 0.0

        drop_mask = p3["hip_y_avg"] > (hip_peak_y + threshold_drop)
        if bool(drop_mask.any()):
            drop_frame: int = int(drop_mask.idxmax())  # type: ignore
            time_taken = float(df.loc[drop_frame, "time_s"]) - float(
                df.loc[t2, "time_s"]
            )
            if time_taken > 0.6:
                critiques.append("Third Pull: You are getting stuck in the transition")
        else:
            duration = float(df.loc[t3, "time_s"]) - float(df.loc[t2, "time_s"])
            if duration > 0.9:
                critiques.append("Third Pull: You are getting stuck in the transition")

    # --- Recovery (t3 -> t4) ---
    p4 = get_phase_df(t3, t4)
    if not p4.empty and "vel_y_smooth" in p4.columns:
        severe_downward = int((p4["vel_y_smooth"] < -100).sum())
        moderate_downward = int((p4["vel_y_smooth"] < -50).sum())
        if severe_downward > 5 or moderate_downward > int(len(p4) * 0.2):
            critiques.append("Recovery: Your recovery is too tiring")

    return critiques
