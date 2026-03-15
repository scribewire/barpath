"""
Kinematics calculations for Step 2: Data Analysis.

Handles barbell position processing, velocity/acceleration calculations,
and phase detection.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from analysis_utils import safe_savgol_smooth
from config import (
    PHASE_HIP_DROP_STD_FACTOR,
    PHASE_HIP_SMOOTH_WINDOW,
    PHASE_VEL_THRESHOLD_FACTOR,
    SAVGOL_POLY_ORDER,
    SAVGOL_POSITION_WINDOW,
    SAVGOL_VELOCITY_WINDOW,
    TRUNCATION_BUFFER_SECONDS,
)


def calculate_stabilized_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate camera-shake-stabilized barbell positions.

    Adds columns:
        - total_shake_x, total_shake_y: cumulative camera shake
        - barbell_x_stable, barbell_y_stable: shake-compensated positions

    Args:
        df: DataFrame with shake_dx, shake_dy, and barbell_center columns

    Returns:
        DataFrame with stabilized position columns
    """
    df["total_shake_x"] = df["shake_dx"].cumsum()
    df["total_shake_y"] = df["shake_dy"].cumsum()

    if "barbell_center" in df.columns:
        df["barbell_x_raw"] = df["barbell_center"].apply(
            lambda x: x[0] if isinstance(x, (list, tuple)) else np.nan
        )
        df["barbell_y_raw"] = df["barbell_center"].apply(
            lambda x: x[1] if isinstance(x, (list, tuple)) else np.nan
        )
    else:
        print(
            "Warning: 'barbell_center' column not found. No barbell data will be processed."
        )
        df["barbell_x_raw"] = np.nan
        df["barbell_y_raw"] = np.nan

    df["barbell_x_stable"] = df["barbell_x_raw"] - df["total_shake_x"]
    df["barbell_y_stable"] = df["barbell_y_raw"] - df["total_shake_y"]

    return df


def smooth_barbell_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Savitzky-Golay smoothing to stabilized barbell positions.

    Adds columns: barbell_x_smooth, barbell_y_smooth

    Args:
        df: DataFrame with barbell_x_stable, barbell_y_stable columns

    Returns:
        DataFrame with smoothed position columns
    """
    x_filled = df["barbell_x_stable"].interpolate(method="linear").bfill().ffill()
    y_filled = df["barbell_y_stable"].interpolate(method="linear").bfill().ffill()

    n = len(x_filled)
    window = min(SAVGOL_POSITION_WINDOW, n // 2 * 2 + 1)

    if n >= 5 and n >= window:
        print(f"Applying barbell position smoothing with window {window}...")
        df["barbell_x_smooth"] = safe_savgol_smooth(
            pd.Series(x_filled), window=window, poly=SAVGOL_POLY_ORDER
        )
        df["barbell_y_smooth"] = safe_savgol_smooth(
            pd.Series(y_filled), window=window, poly=SAVGOL_POLY_ORDER
        )
    else:
        print(
            "Warning: Not enough data to smooth barbell position. Using unsmoothed values."
        )
        df["barbell_x_smooth"] = df["barbell_x_stable"]
        df["barbell_y_smooth"] = df["barbell_y_stable"]

    return df


def truncate_at_knee_pass(
    df: pd.DataFrame, fps: float, frame_height: int
) -> pd.DataFrame:
    """
    Truncate data to start 1 second before bar passes knee height.

    Args:
        df: DataFrame with barbell_y_smooth and knee_y_avg columns
        fps: Video frames per second
        frame_height: Video frame height

    Returns:
        Truncated DataFrame
    """
    if "knee_y_avg" not in df.columns:
        df["knee_y_avg"] = (
            df[["left_knee_y", "right_knee_y"]].mean(axis=1) * frame_height
        )

    if not bool(df["barbell_y_smooth"].notna().any()) or not bool(
        df["knee_y_avg"].notna().any()
    ):
        print("Warning: Cannot determine knee pass frame. Keeping all data at start.")
        return df

    bar_above_knee = df["barbell_y_smooth"] < df["knee_y_avg"]
    frames_above_knee = df[bar_above_knee].index.values

    if len(frames_above_knee) > 0:
        knee_pass_frame = int(frames_above_knee[0])
        frames_before = int(fps * TRUNCATION_BUFFER_SECONDS)
        first_frame = int(df.index.values[0])
        start_frame = max(first_frame, knee_pass_frame - frames_before)

        print(
            f"Bar passes knee at frame {knee_pass_frame}. "
            f"Keeping data from frame {start_frame} onwards ({TRUNCATION_BUFFER_SECONDS}s before knee pass)."
        )
        return df.loc[start_frame:].copy()
    else:
        print("Warning: Bar never detected above knee. Keeping all data at start.")
        return df


def truncate_at_peak_height(df: pd.DataFrame) -> pd.DataFrame:
    """
    Truncate data at peak barbell height (minimum Y value).

    Args:
        df: DataFrame with barbell_y_smooth column

    Returns:
        Truncated DataFrame
    """
    if not bool(df["barbell_y_smooth"].notna().any()):
        print("Warning: No barbell Y data found. Cannot truncate at peak height.")
        return df

    peak_height_idx = df["barbell_y_smooth"].idxmin()
    print(
        f"Peak height detected at frame {peak_height_idx}. Truncating data after this point."
    )
    return df.loc[:peak_height_idx].copy()


def calculate_time_and_kinematics(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    Calculate time, velocity, acceleration, and specific power.

    Adds columns:
        - time_s: Time in seconds from first frame
        - dt: Time delta between frames
        - vel_y_px_s: Raw vertical velocity
        - vel_y_smooth: Smoothed vertical velocity
        - accel_y_smooth: Smoothed vertical acceleration
        - specific_power_y_smooth: Smoothed specific power

    Args:
        df: DataFrame with barbell_y_smooth column
        fps: Video frames per second

    Returns:
        DataFrame with kinematic columns
    """
    if df.index.is_monotonic_increasing:
        df["time_s"] = (df.index - df.index[0]) / fps
    else:
        print("Warning: Frame indices are not monotonic. Using sequential time.")
        df["time_s"] = np.arange(len(df)) / fps

    df["dt"] = df["time_s"].diff().fillna(1 / fps)

    df["vel_y_px_s"] = (df["barbell_y_smooth"].diff() / df["dt"]) * -1

    vel_filled = df["vel_y_px_s"].interpolate(method="linear").fillna(0)
    n = len(vel_filled)
    vel_window = min(SAVGOL_VELOCITY_WINDOW, n // 2 * 2 + 1)

    if n >= 5:
        print(f"Applying Savitzky-Golay velocity smoothing with window {vel_window}...")
        df["vel_y_smooth"] = safe_savgol_smooth(
            pd.Series(vel_filled), window=vel_window, poly=SAVGOL_POLY_ORDER
        )
    else:
        print("Warning: Not enough data to smooth velocity.")
        df["vel_y_smooth"] = vel_filled

    df["accel_y_smooth"] = df["vel_y_smooth"].diff() / df["dt"]
    df["specific_power_y_smooth"] = df["accel_y_smooth"] * df["vel_y_smooth"]

    return df


def detect_three_phases(df: pd.DataFrame, fps: float) -> pd.Series:
    """
    Assign one of three kinematic bar phases to every frame.

    Phase labels (integer):
      0 - Pull: barbell moving upward, hips have not yet dropped
      1 - Pull-under: hips are actively descending under the bar
      2 - Recovery: from when hips stop descending until peak height

    Detection is purely kinematic using smoothed signals.

    Args:
        df: DataFrame with vel_y_smooth and hip_y_avg columns
        fps: Video frames per second

    Returns:
        pd.Series of phase integers indexed by frame
    """
    phase = pd.Series(0, index=df.index, dtype="int64", name="bar_phase")

    if "vel_y_smooth" not in df.columns or "hip_y_avg" not in df.columns:
        return phase

    vel = df["vel_y_smooth"].fillna(0)
    hip_y = df["hip_y_avg"]

    vel_threshold = max(10.0, float(vel.abs().max()) * PHASE_VEL_THRESHOLD_FACTOR)
    bar_moving_up = vel > vel_threshold

    if not bar_moving_up.any():
        return phase

    pull_start_idx = int(bar_moving_up.idxmax())

    hip_after_pull = hip_y.loc[pull_start_idx:]
    hip_smooth = safe_savgol_smooth(
        hip_after_pull, window=PHASE_HIP_SMOOTH_WINDOW, poly=3
    )
    hip_vel = hip_smooth.diff().fillna(0)

    hip_drop_threshold = (
        float(hip_smooth.std()) * PHASE_HIP_DROP_STD_FACTOR
        if float(hip_smooth.std()) > 0
        else 0.5
    )
    hips_dropping = hip_vel > hip_drop_threshold

    pull_under_start_idx: Optional[int] = None
    if hips_dropping.any():
        pull_under_start_idx = int(hips_dropping.idxmax())

    recovery_start_idx: Optional[int] = None
    if pull_under_start_idx is not None:
        hip_after_pu = hip_y.loc[pull_under_start_idx:]
        hip_smooth_pu = safe_savgol_smooth(
            hip_after_pu, window=PHASE_HIP_SMOOTH_WINDOW, poly=3
        )
        hip_vel_pu = hip_smooth_pu.diff().fillna(0)
        hips_stopped = hip_vel_pu <= hip_drop_threshold * 0.5

        if hips_stopped.any():
            recovery_start_idx = int(hips_stopped.idxmax())

    if pull_under_start_idx is not None:
        phase.loc[df.index >= pull_under_start_idx] = 1
    if recovery_start_idx is not None:
        phase.loc[df.index >= recovery_start_idx] = 2

    return phase


def assign_phases_from_classics(df: pd.DataFrame, phases: Any) -> pd.DataFrame:
    """
    Assign bar phases from classics phase detection results.

    Maps the t0-t4 boundaries to the 3-phase system:
        Pull: t0 -> t2 (entire upward drive)
        Pull-under: t2 -> t3 (hips dropping under bar)
        Recovery: t3 -> t4 (stand-up)

    Args:
        df: DataFrame to modify
        phases: Dict with t0, t1, t2, t3, t4 frame indices

    Returns:
        DataFrame with bar_phase and phase_change columns
    """
    idx = df.index
    bar_phase = pd.Series(0, index=idx, dtype="int64", name="bar_phase")

    t2 = int(phases["t2"])
    t3 = int(phases["t3"])

    bar_phase.loc[idx >= t2] = 1
    bar_phase.loc[idx >= t3] = 2

    df["bar_phase"] = bar_phase
    df["phase_change"] = df["bar_phase"].diff().fillna(0).ne(0)

    print(
        f"Classics phase mapping -> Pull:[t0->t2], Pull-under:[t2->t3], Recovery:[t3->t4] "
        f"(t2={t2}, t3={t3})"
    )

    return df


def assign_phases_kinematic(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    Assign bar phases using kinematic detection (fallback).

    Args:
        df: DataFrame to modify
        fps: Video frames per second

    Returns:
        DataFrame with bar_phase and phase_change columns
    """
    df["bar_phase"] = detect_three_phases(df, fps)
    df["phase_change"] = df["bar_phase"].diff().fillna(0).ne(0)
    return df
