from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TypedDict, cast

import pandas as pd


class ClassicsPhases(TypedDict):
    """
    Frame-index keyed phase boundaries for a clean.

    Conventions (Y=0 is top of frame):
    - t0: start of lift initiation (bar begins moving up)
    - t1: end of first pull (bar passes knees)
    - t2: end of second pull / turnover initiation (hip reaches highest/most-extended)
    - t3: bottom of catch (hip reaches lowest after turnover)
    - t4: peak bar height (minimum bar Y)
    """

    t0: int
    t1: int
    t2: int
    t3: int
    t4: int


@dataclass(frozen=True)
class PhaseDetectionParams:
    """
    Parameters for clean phase detection.

    These mirror the heuristic used in Step 5 previously.
    """

    start_search_limit: int = 30
    start_threshold_frac: float = 0.02


def _require_columns(df: pd.DataFrame, required: List[str]) -> Optional[str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        return ", ".join(missing)
    return None


def identify_classics_phases(
    df: pd.DataFrame, params: PhaseDetectionParams = PhaseDetectionParams()
) -> Optional[ClassicsPhases]:
    """
    Identify clean phases using the heuristics that were previously embedded in `5_critique_lift.py`.

    Returns a dict of phase frame indices, or None if phases cannot be identified.

    Notes:
    - Assumes `df` is indexed by frame number (as in the pipeline output CSV when "frame" is set as index).
    - Uses pixel-space vertical coordinates; smaller Y means higher in the frame.
    """
    required = [
        "barbell_y_stable",
        "hip_y_avg",
        "left_knee_y",
        "right_knee_y",
        "frame_height",
        "time_s",
    ]
    missing = _require_columns(df, required)
    if missing:
        print(f"Error: Missing columns {missing}")
        return None

    if df.empty:
        return None

    try:
        frame_height = float(df["frame_height"].iloc[0])
    except Exception:
        return None

    start_search_limit = int(min(params.start_search_limit, len(df)))
    baseline_y = float(df["barbell_y_stable"].iloc[:start_search_limit].mean())
    threshold_px = frame_height * float(params.start_threshold_frac)

    mask_started = df["barbell_y_stable"] < (baseline_y - threshold_px)
    if not bool(mask_started.any()):
        return None

    t0_raw = mask_started.idxmax()
    t0_frame: int = int(t0_raw)

    df_post_t0 = df.loc[t0_raw:]
    if df_post_t0.empty:
        return None

    left_knee_px = df_post_t0["left_knee_y"] * frame_height
    right_knee_px = df_post_t0["right_knee_y"] * frame_height
    knee_y_lowest_px = pd.concat([left_knee_px, right_knee_px], axis=1).min(axis=1)
    mask_at_knees = df_post_t0["barbell_y_stable"] <= knee_y_lowest_px
    if not bool(mask_at_knees.any()):
        return None
    t1_raw = mask_at_knees.idxmax()
    t1_frame: int = int(t1_raw)

    df_post_t1 = df.loc[t1_raw:]
    if df_post_t1.empty:
        return None

    bar_peak_raw = df_post_t1["barbell_y_stable"].idxmin()
    bar_peak_frame: int = int(bar_peak_raw)

    if bar_peak_frame >= t1_frame:
        search_window = df.loc[t1_raw:bar_peak_raw]
    else:
        search_window = df_post_t1.iloc[:10]

    if search_window.empty or "hip_y_avg" not in search_window.columns:
        return None
    t2_raw = search_window["hip_y_avg"].idxmin()
    t2_frame: int = int(t2_raw)

    df_post_t2 = df.loc[t2_raw:]
    if df_post_t2.empty:
        return None
    t3_raw = df_post_t2["hip_y_avg"].idxmax()
    t3_frame: int = int(t3_raw)

    t4_raw = df["barbell_y_stable"].idxmin()
    t4_frame = int(cast(int, t4_raw))

    return ClassicsPhases(
        t0=t0_frame, t1=t1_frame, t2=t2_frame, t3=t3_frame, t4=t4_frame
    )
