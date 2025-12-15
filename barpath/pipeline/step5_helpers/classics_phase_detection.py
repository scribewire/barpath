from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TypedDict

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

    # For t0 detection: how many initial frames to estimate baseline bar height
    start_search_limit: int = 30
    # For t0 detection: threshold (as fraction of frame height) bar must move upward to count as "started"
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

    # Frame height is stored per-row but should be constant.
    try:
        frame_height = float(df["frame_height"].iloc[0])
    except Exception:
        return None

    # 1) Identify Start of Lift (t0): bar rises above baseline by a small threshold.
    start_search_limit = int(min(params.start_search_limit, len(df)))
    baseline_y = float(df["barbell_y_stable"].iloc[:start_search_limit].mean())
    threshold_px = frame_height * float(params.start_threshold_frac)

    # Bar moving "up" means Y decreases.
    mask_started = df["barbell_y_stable"] < (baseline_y - threshold_px)
    if not bool(mask_started.any()):
        return None

    # idxmax on a boolean Series gives the first index with True
    t0_frame: int = int(mask_started.idxmax())  # type: ignore

    # 2) End of First Pull (t1): bar at/above knees (bar_y <= knee_y_avg_px).
    df_post_t0 = df.loc[t0_frame:]
    if df_post_t0.empty:
        return None

    # Knee y columns are normalized; convert to pixels (same as original logic).
    knee_y_avg_px = (
        (df_post_t0["left_knee_y"] + df_post_t0["right_knee_y"]) / 2.0 * frame_height
    )
    mask_at_knees = df_post_t0["barbell_y_stable"] <= knee_y_avg_px
    if not bool(mask_at_knees.any()):
        return None
    t1_frame: int = int(mask_at_knees.idxmax())  # type: ignore

    # 3) End of Second Pull (t2): "Hip turnover" proxy:
    # within [t1, bar_peak], find frame where hips are highest (min hip_y_avg).
    df_post_t1 = df.loc[t1_frame:]
    if df_post_t1.empty:
        return None

    bar_peak_frame: int = int(df_post_t1["barbell_y_stable"].idxmin())  # type: ignore

    # Guard ordering: if peak happens before t1 for any reason, fall back to a small window.
    if bar_peak_frame >= t1_frame:
        search_window = df.loc[t1_frame:bar_peak_frame]
    else:
        search_window = df_post_t1.iloc[:10]

    if search_window.empty or "hip_y_avg" not in search_window.columns:
        return None
    t2_frame: int = int(search_window["hip_y_avg"].idxmin())  # type: ignore

    # 4) End of Third Pull (t3): bottom of catch (hips lowest after turnover => max hip_y_avg).
    df_post_t2 = df.loc[t2_frame:]
    if df_post_t2.empty:
        return None
    t3_frame: int = int(df_post_t2["hip_y_avg"].idxmax())  # type: ignore

    # 5) End of Recovery / peak bar height (t4): global bar peak (min bar_y).
    t4_frame: int = int(df["barbell_y_stable"].idxmin())  # type: ignore

    return ClassicsPhases(
        t0=t0_frame, t1=t1_frame, t2=t2_frame, t3=t3_frame, t4=t4_frame
    )
