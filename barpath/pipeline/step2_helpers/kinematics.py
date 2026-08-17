"""
Kinematics calculations for Step 2: Data Analysis.

Handles barbell position processing, velocity/acceleration calculations,
and phase detection for snatch, clean, and jerk.
"""

from typing import Any, cast

import numpy as np
import pandas as pd
from analysis_utils import safe_savgol_smooth
from config import (
    JERK_DIP_VELOCITY_THRESHOLD,
    JERK_DRIVE_VELOCITY_THRESHOLD,
    JERK_MIN_KNEE_BEND_ANGLE,
    PHASE_HIP_DROP_STD_FACTOR,
    PHASE_HIP_SMOOTH_WINDOW,
    PHASE_VEL_THRESHOLD_FACTOR,
    SAVGOL_POLY_ORDER,
    SAVGOL_POSITION_WINDOW,
    SAVGOL_VELOCITY_WINDOW,
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
        print("Warning: 'barbell_center' column not found. No barbell data will be processed.")
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
    # Replace Inf values with NaN before interpolation
    x_clean = df["barbell_x_stable"].replace([np.inf, -np.inf], np.nan)
    y_clean = df["barbell_y_stable"].replace([np.inf, -np.inf], np.nan)

    # Interpolate and fill
    x_filled = x_clean.interpolate(method="linear").bfill().ffill()
    y_filled = y_clean.interpolate(method="linear").bfill().ffill()

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
        print("Warning: Not enough data to smooth barbell position. Using unsmoothed values.")
        df["barbell_x_smooth"] = x_filled
        df["barbell_y_smooth"] = y_filled

    return df


def truncate_at_knee_pass(df: pd.DataFrame, fps: float, frame_height: int) -> pd.DataFrame:
    """
    Truncate data to only include frames after the bar passes the knees.

    Finds where bar_y drops below knee_y and keeps 1 second of data before that point,
    plus all data until peak bar height.

    Args:
        df: DataFrame with barbell and knee positions
        fps: Frames per second for timing calculations
        frame_height: Frame height in pixels (for threshold scaling)

    Returns:
        Truncated DataFrame
    """
    if len(df) < 10:
        return df

    if "barbell_y_smooth" not in df.columns or "left_knee_y" not in df.columns:
        return df

    bar_y_px = df["barbell_y_smooth"] * frame_height
    knee_y_px = (
        pd.concat(
            [
                df["left_knee_y"].fillna(0),
                df["right_knee_y"].fillna(0),
            ],
            axis=1,
        ).min(axis=1)
        * frame_height
    )

    bar_at_or_below_knees = bar_y_px <= knee_y_px + 10

    if bar_at_or_below_knees.any():
        knee_pass_idx = bar_at_or_below_knees.idxmax()
        print(f"  Bar passes knees at frame {knee_pass_idx}")

        one_sec_buffer = int(fps)
        start_idx = max(0, knee_pass_idx - one_sec_buffer)

        bar_y_smooth = df["barbell_y_smooth"]
        peak_idx = bar_y_smooth.idxmin()

        print(f"  Peak height at frame {peak_idx}")

        df = df.loc[start_idx:peak_idx]
        print(
            f"  Truncated data: {start_idx} -> {peak_idx} ({len(df)} frames, {len(df) / fps:.1f}s)"
        )

    return df


class InsufficientDataError(Exception):
    """Raised when there's not enough valid data to analyze."""


def truncate_at_peak_height(df: pd.DataFrame) -> pd.DataFrame:
    """
    Truncate data to only include frames up to peak bar height.

    Args:
        df: DataFrame with barbell_y_smooth column

    Returns:
        DataFrame truncated at peak height

    Raises:
        InsufficientDataError: If barbell_y_smooth is all NA or has too few valid values
    """
    if len(df) < 10:
        raise InsufficientDataError(f"Too few frames ({len(df)}) to analyze")

    if "barbell_y_smooth" not in df.columns:
        raise InsufficientDataError("Missing barbell_y_smooth column")

    # Check for all NA values
    bar_y = df["barbell_y_smooth"]
    valid_count = bar_y.notna().sum()

    if valid_count == 0:
        raise InsufficientDataError(
            "All barbell position values are NA - no tracking data available"
        )

    # Check for too few valid values (less than 20% of total)
    if valid_count < len(df) * 0.2:
        raise InsufficientDataError(
            f"Too few valid barbell positions ({valid_count}/{len(df)} frames) - "
            f"insufficient tracking data"
        )

    peak_idx = bar_y.idxmin()
    df = df.loc[:peak_idx]
    print(f"  Truncated at peak height (frame {peak_idx}, {len(df)} frames)")

    return df


def calculate_time_and_kinematics(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    Calculate time and kinematic derivatives (velocity, acceleration).

    Adds columns:
        - time_s: time from lift start in seconds
        - dt: time step between frames
        - vel_y_smooth: smoothed vertical velocity (px/s)
        - accel_y_smooth: smoothed vertical acceleration (px/s²)
        - specific_power_y_smooth: power (accel * velocity)

    Args:
        df: DataFrame with frame-indexed barbell positions
        fps: Frames per second

    Returns:
        DataFrame with kinematic columns added
    """
    df["time_s"] = (df.index - df.index[0]) / fps
    df["dt"] = df["time_s"].diff().fillna(1.0 / fps)

    bar_y_filled = df["barbell_y_smooth"].interpolate(method="linear").bfill().ffill()
    vel_raw = bar_y_filled.diff() / df["dt"]
    vel_filled = vel_raw.interpolate(method="linear").bfill().ffill()

    n = len(vel_filled)
    window = min(SAVGOL_VELOCITY_WINDOW, n // 2 * 2 + 1)

    if n >= 5 and n >= window:
        print(f"  Smoothing velocity with window {window}...")
        df["vel_y_smooth"] = safe_savgol_smooth(vel_filled, window=window, poly=SAVGOL_POLY_ORDER)
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

    vel_max_val = vel.abs().max()
    if isinstance(vel_max_val, (int, float, np.number)):
        vel_max_scalar: float = float(vel_max_val)
    else:
        vel_max_scalar = float(vel_max_val.item())
    vel_threshold = max(10.0, vel_max_scalar * PHASE_VEL_THRESHOLD_FACTOR)
    bar_moving_up = vel > vel_threshold

    if not bar_moving_up.any():
        return phase

    pull_start_idx = int(bar_moving_up.idxmax())

    hip_after_pull = hip_y.loc[pull_start_idx:]
    hip_smooth = safe_savgol_smooth(hip_after_pull, window=PHASE_HIP_SMOOTH_WINDOW, poly=3)
    hip_vel = hip_smooth.diff().fillna(0)

    hip_std_val = hip_smooth.std()
    if isinstance(hip_std_val, (int, float, np.number)):
        hip_std_float: float = float(hip_std_val)
    else:
        hip_std_float = float(hip_std_val.item())
    hip_drop_threshold = hip_std_float * PHASE_HIP_DROP_STD_FACTOR if hip_std_float > 0 else 0.5
    hips_dropping = hip_vel > hip_drop_threshold

    pull_under_start_idx: int | None = None
    if hips_dropping.any():
        pull_under_start_idx = int(hips_dropping.idxmax())

    recovery_start_idx: int | None = None
    if pull_under_start_idx is not None:
        hip_after_pu = hip_y.loc[pull_under_start_idx:]
        hip_smooth_pu = safe_savgol_smooth(hip_after_pu, window=PHASE_HIP_SMOOTH_WINDOW, poly=3)
        hip_vel_pu = hip_smooth_pu.diff().fillna(0)
        hips_stopped = hip_vel_pu <= hip_drop_threshold * 0.5

        if hips_stopped.any():
            recovery_start_idx = int(hips_stopped.idxmax())

    if pull_under_start_idx is not None:
        phase.loc[df.index >= pull_under_start_idx] = 1
    if recovery_start_idx is not None:
        phase.loc[df.index >= recovery_start_idx] = 2

    return phase


def detect_jerk_phases(df: pd.DataFrame, fps: float) -> pd.Series:
    """
    Assign one of three jerk phases to every frame.

    Phase labels (integer):
      0 - Dip: knees bending, bar moving down slightly with the dip
      1 - Drive: knees extending, bar moving upward rapidly
      2 - Recovery: bar decelerating, stabilization phase

    Phase Definitions:
      - Dip begins: when knee angles start decreasing (knees bending) significantly
      - Dip ends/Drive begins: when knee angles start increasing (knees extending)
      - Drive ends/Recovery begins: when bar velocity drops below 25% of peak drive velocity
      - Recovery ends: when bar reaches maximum height (minimum y)

    Args:
        df: DataFrame with barbell_y_smooth, vel_y_smooth, left_knee_angle, right_knee_angle columns
        fps: Video frames per second

    Returns:
        pd.Series of phase integers indexed by frame
    """
    phase = pd.Series(0, index=df.index, dtype="int64", name="bar_phase")

    if "barbell_y_smooth" not in df.columns or "vel_y_smooth" not in df.columns:
        print("Warning: Missing required columns for jerk phase detection")
        return phase

    bar_y = df["barbell_y_smooth"].interpolate(method="linear").bfill().ffill()
    vel = df["vel_y_smooth"].fillna(0)

    if len(bar_y) < 10:
        print("Warning: Not enough data for jerk phase detection")
        return phase

    has_knee_angles = "left_knee_angle" in df.columns and "right_knee_angle" in df.columns

    # Compute torso (shoulder-to-hip) distance for minimum dip depth
    frame_height = float(df["frame_height"].iloc[0]) if "frame_height" in df.columns else 1080.0
    torso_length_px = frame_height * 0.3  # default fallback
    if all(
        c in df.columns
        for c in ["left_shoulder_y", "right_shoulder_y", "left_hip_y", "right_hip_y"]
    ):
        shoulder_y = ((df["left_shoulder_y"] + df["right_shoulder_y"]) / 2).dropna()
        hip_y = ((df["left_hip_y"] + df["right_hip_y"]) / 2).dropna()
        if len(shoulder_y) > 0 and len(hip_y) > 0:
            min_len = min(len(shoulder_y), len(hip_y))
            torso_dist = abs(
                np.asarray(shoulder_y[:min_len], dtype=float)
                - np.asarray(hip_y[:min_len], dtype=float)
            )
            torso_length_px = float(np.median(torso_dist)) * frame_height

    min_dip_depth_px = torso_length_px * 0.30  # dip must be >= 30% of torso length
    knee_angles = None
    knee_vel_smooth = None

    if has_knee_angles:
        knee_angles = df[["left_knee_angle", "right_knee_angle"]].mean(axis=1)
        knee_angles = pd.Series(knee_angles, index=df.index, dtype="float64")
        knee_angles = knee_angles.interpolate(method="linear").bfill().ffill()
        knee_vel = knee_angles.diff().fillna(0) * fps
        knee_vel_series = pd.Series(knee_vel, index=df.index, dtype="float64").astype("float64")
        knee_vel_smooth = safe_savgol_smooth(
            cast(pd.Series, knee_vel_series),
            window=min(7, len(knee_vel_series) // 2 * 2 + 1),
            poly=3,
        )
    else:
        print("Warning: No knee angle data, falling back to barbell-only detection")

    vel_series = pd.Series(vel, index=df.index, dtype="float64")
    vel_smooth = safe_savgol_smooth(vel_series, window=min(7, len(vel_series) // 2 * 2 + 1), poly=3)

    dip_start_idx: int | None = None
    drive_start_idx: int | None = None

    if has_knee_angles and knee_vel_smooth is not None and knee_angles is not None:
        knee_angle_decreasing = knee_vel_smooth < -JERK_DIP_VELOCITY_THRESHOLD

        if not bool(knee_angle_decreasing.any()):
            print("Warning: Could not detect dip phase - no knee bending found")
            return phase

        down_indices = knee_angle_decreasing[knee_angle_decreasing].index

        for idx in down_indices:
            subsequent_indices = knee_angles.loc[idx:].index
            if len(subsequent_indices) < 5:
                continue

            subsequent_knee_vel = knee_vel_smooth.loc[idx:]
            stop_bending = subsequent_knee_vel >= -JERK_DIP_VELOCITY_THRESHOLD

            if bool(stop_bending.any()):
                dip_end_idx = int(stop_bending.idxmax())
            else:
                dip_end_idx = int(subsequent_indices[-1])

            knee_at_start = float(knee_angles.loc[idx])
            knee_at_lowest = float(knee_angles.loc[idx:dip_end_idx].min())
            angle_change = knee_at_start - knee_at_lowest

            if angle_change >= JERK_MIN_KNEE_BEND_ANGLE:
                # Also check that the dip translates to meaningful vertical displacement
                y_at_dip_start = float(bar_y.loc[idx])
                y_at_dip_lowest = float(bar_y.loc[idx:dip_end_idx].max())
                dip_displacement = y_at_dip_lowest - y_at_dip_start
                if dip_displacement < min_dip_depth_px:
                    continue

                dip_start_idx = int(idx)
                print(f"  Jerk dip detected (knee bend): frame {dip_start_idx} -> {dip_end_idx}")
                print(f"    Knee angle change: {angle_change:.1f} degrees")
                print(
                    f"    Dip displacement: {dip_displacement:.1f}px "
                    f"({dip_displacement / torso_length_px * 100:.0f}% of torso)"
                )
                break

        if dip_start_idx is not None:
            after_dip_knee_vel = knee_vel_smooth.loc[dip_start_idx:]
            knee_extending = after_dip_knee_vel > JERK_DRIVE_VELOCITY_THRESHOLD

            if bool(knee_extending.any()):
                drive_start_idx = int(knee_extending.idxmax())
                print(f"  Jerk drive starts at frame {drive_start_idx} (knees extending)")
    else:
        moving_down = vel_smooth > JERK_DIP_VELOCITY_THRESHOLD

        if not bool(moving_down.any()):
            print("Warning: Could not detect dip phase - no downward bar movement found")
            return phase

        down_indices = moving_down[moving_down].index

        for idx in down_indices:
            subsequent_indices = bar_y.loc[idx:].index
            if len(subsequent_indices) < 5:
                continue

            subsequent_vel = vel_smooth.loc[idx:]
            stop_moving_down = subsequent_vel <= JERK_DIP_VELOCITY_THRESHOLD

            if bool(stop_moving_down.any()):
                dip_end_idx = int(stop_moving_down.idxmax())
            else:
                dip_end_idx = int(subsequent_indices[-1])

            y_at_start = float(bar_y.loc[idx])
            y_at_lowest = float(bar_y.loc[idx:dip_end_idx].max())
            displacement = y_at_lowest - y_at_start

            if displacement >= min_dip_depth_px:
                dip_start_idx = int(idx)
                print(f"  Jerk dip detected (barbell): frame {dip_start_idx} -> {dip_end_idx}")
                print(
                    f"    Displacement: {displacement:.1f}px "
                    f"({displacement / torso_length_px * 100:.0f}% of torso)"
                )
                break

        if dip_start_idx is not None:
            after_dip = vel_smooth.loc[dip_start_idx:]
            moving_up = after_dip < -JERK_DRIVE_VELOCITY_THRESHOLD

            if bool(moving_up.any()):
                drive_start_idx = int(moving_up.idxmax())
                print(f"  Jerk drive starts at frame {drive_start_idx}")

    if dip_start_idx is None:
        print("Warning: No dip detected")
        return phase

    recovery_start_idx: int | None = None
    if drive_start_idx is not None:
        drive_velocities = vel_smooth.loc[drive_start_idx:]
        peak_drive_vel = drive_velocities.min()
        _peak_drive_vel_float = (
            float(peak_drive_vel.item())
            if isinstance(peak_drive_vel, np.generic)
            else float(peak_drive_vel)
        )

        peak_drive_idx = int(drive_velocities.idxmin())

        after_peak_velocities = vel_smooth.loc[peak_drive_idx:]
        velocity_crossed_zero = after_peak_velocities >= 0

        if bool(velocity_crossed_zero.any()):
            recovery_start_idx = int(velocity_crossed_zero.idxmax())
            print(f"  Jerk recovery starts at frame {recovery_start_idx}")

    if drive_start_idx is not None:
        phase.loc[df.index >= drive_start_idx] = 1
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


def assign_phases_kinematic(
    df: pd.DataFrame, fps: float, lift_type: str = "snatch"
) -> pd.DataFrame:
    """
    Assign bar phases using kinematic detection (fallback).

    Args:
        df: DataFrame to modify
        fps: Video frames per second
        lift_type: Type of lift ("snatch", "clean", or "jerk")

    Returns:
        DataFrame with bar_phase and phase_change columns
    """
    if lift_type.lower() == "jerk":
        df["bar_phase"] = detect_jerk_phases(df, fps)
    else:
        df["bar_phase"] = detect_three_phases(df, fps)
    df["phase_change"] = df["bar_phase"].diff().fillna(0).ne(0)
    return df
