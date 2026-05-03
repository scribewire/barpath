"""
Unified Feature Extraction for Lift Detection Model.

Consolidates feature extraction from the training script
(outputs/plans/lift_detection_training.py) and the pipeline
(step4_helpers/feature_extraction.py) into a single module
that produces exactly the 37 features the trained model expects.

Model path: barpath/models/lift_detection/lift_detection_model.pkl
Classes: ["clean", "clean_jerk", "jerk", "snatch"]
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import signal

# Biomechanical constants
VERTICAL_MOTION_BALANCE_RATIO = 0.3


# ============================================================================
# Utility helpers
# ============================================================================


def _to_float_array_1d(values: Any) -> NDArray[np.float64]:
    """Force input into a 1D float64 ndarray."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return cast(NDArray[np.float64], arr)


def _safe_interp_numeric(series: pd.Series) -> NDArray[np.float64]:
    """Numeric coercion + interpolation with robust typing."""
    numeric = pd.to_numeric(series, errors="coerce")
    if not isinstance(numeric, pd.Series):
        numeric = pd.Series(numeric, index=series.index, dtype="float64")
    numeric = numeric.interpolate(method="linear", limit_direction="both")
    return _to_float_array_1d(numeric.to_numpy(dtype=np.float64))


def _safe_savgol(
    y: NDArray[np.float64], max_win: int = 21, polyorder: int = 3
) -> NDArray[np.float64]:
    """Savitzky-Golay smoothing with safe window handling."""
    n = int(len(y))
    if n < 5:
        return y.copy()
    win = min(max_win, n if n % 2 == 1 else n - 1)
    if win < 5:
        return y.copy()
    po = min(polyorder, win - 1)
    filtered = signal.savgol_filter(y, window_length=win, polyorder=po)
    return _to_float_array_1d(filtered)


def _safe_float(value: object) -> float:
    """Safely convert a value to float, replacing NaN/inf with 0.0."""
    if isinstance(value, (int, float, np.number)):
        result = float(value)
    elif hasattr(value, "iloc"):
        result = float(value.iloc[0]) if len(cast(Any, value)) > 0 else 0.0
    else:
        result = 0.0
    return 0.0 if not np.isfinite(result) else result


# ============================================================================
# Trajectory extraction
# ============================================================================


def extract_trajectory(df: pd.DataFrame) -> NDArray[np.float64]:
    """Extract multi-channel kinematic trajectory as Nx3 float64 array.

    Channels: [y_position_norm, velocity_norm, acceleration_norm]
    """
    if "barbell_y_smooth" not in df.columns:
        return np.empty((0, 3), dtype=np.float64)

    y = _safe_interp_numeric(cast(pd.Series, df["barbell_y_smooth"]))
    y_range = float(np.nanmax(y) - np.nanmin(y))
    if y_range > 0:
        y = (y - np.nanmin(y)) / y_range
    else:
        y = np.zeros_like(y)

    vel = np.zeros_like(y)
    if "vel_y_smooth" in df.columns:
        vel = _safe_interp_numeric(cast(pd.Series, df["vel_y_smooth"]))
        vel_max = float(np.nanmax(np.abs(vel)))
        if vel_max > 0:
            vel = vel / vel_max

    accel = np.zeros_like(y)
    if "accel_y_smooth" in df.columns:
        accel = _safe_interp_numeric(cast(pd.Series, df["accel_y_smooth"]))
        accel_max = float(np.nanmax(np.abs(accel)))
        if accel_max > 0:
            accel = accel / accel_max

    mask = ~(np.isnan(y) | np.isnan(vel) | np.isnan(accel))
    if not bool(np.any(mask)):
        return np.empty((0, 3), dtype=np.float64)

    traj = np.column_stack((y[mask], vel[mask], accel[mask])).astype(
        np.float64, copy=False
    )
    return cast(NDArray[np.float64], traj)


def extract_joint_data(df: pd.DataFrame) -> Dict[str, NDArray[np.float64]]:
    """Extract joint position data for biomechanical analysis.

    Returns dict of joint_name -> Nx2 array (x_norm, y_norm) where
    coordinates are normalized by frame_height.
    """
    fh = float(df["frame_height"].iloc[0]) if "frame_height" in df.columns else 1.0
    if fh == 0.0:
        fh = 1.0

    joints: Dict[str, NDArray[np.float64]] = {}
    joint_cols = {
        "left_ankle": ["left_ankle_x", "left_ankle_y"],
        "right_ankle": ["right_ankle_x", "right_ankle_y"],
        "left_knee": ["left_knee_x", "left_knee_y"],
        "right_knee": ["right_knee_x", "right_knee_y"],
        "left_hip": ["left_hip_x", "left_hip_y"],
        "right_hip": ["right_hip_x", "right_hip_y"],
        "left_shoulder": ["left_shoulder_x", "left_shoulder_y"],
        "right_shoulder": ["right_shoulder_x", "right_shoulder_y"],
        "left_elbow": ["left_elbow_x", "left_elbow_y"],
        "right_elbow": ["right_elbow_x", "right_elbow_y"],
        "left_wrist": ["left_wrist_x", "left_wrist_y"],
        "right_wrist": ["right_wrist_x", "right_wrist_y"],
    }

    for joint_name, cols in joint_cols.items():
        if all(c in df.columns for c in cols):
            x = _safe_interp_numeric(cast(pd.Series, df[cols[0]])) / fh
            y = _safe_interp_numeric(cast(pd.Series, df[cols[1]])) / fh
            joints[joint_name] = np.column_stack((x, y)).astype(np.float64)

    return joints


# ============================================================================
# Trajectory shape features (from training script)
# ============================================================================


def compute_trajectory_shape_features(
    trajectory: NDArray[np.float64],
) -> Dict[str, float]:
    """Compute basic trajectory shape features from kinematic channels."""
    from scipy.stats import kurtosis, skew

    features: Dict[str, float] = {}
    if len(trajectory) < 10:
        return {f"shape_{k}": 0.0 for k in range(20)}

    y = _to_float_array_1d(trajectory[:, 0])

    y_range = float(np.max(y) - np.nanmin(y))
    features["shape_y_range"] = y_range
    features["shape_y_start"] = float(y[0])
    features["shape_y_end"] = float(y[-1])
    features["shape_y_min"] = float(np.min(y))
    features["shape_y_max"] = float(np.max(y))

    features["shape_y_start_to_min"] = (
        features["shape_y_min"] - features["shape_y_start"]
    )
    features["shape_y_min_to_end"] = features["shape_y_end"] - features["shape_y_min"]

    features["shape_y_skewness"] = float(skew(y)) if len(y) > 3 else 0.0
    features["shape_y_kurtosis"] = float(kurtosis(y)) if len(y) > 4 else 0.0

    dy = _to_float_array_1d(np.diff(y))
    distances = _to_float_array_1d(np.abs(dy))
    total_distance = float(np.sum(distances))
    straight_distance = float(abs(y[-1] - y[0]))

    features["shape_total_distance"] = total_distance
    features["shape_straight_distance"] = straight_distance
    features["shape_path_efficiency"] = (
        (straight_distance / total_distance) if total_distance > 0.0 else 0.0
    )
    features["shape_path_tortuosity"] = (
        (total_distance / straight_distance) if straight_distance > 0.0 else 0.0
    )

    n = len(y)
    first_half = _to_float_array_1d(y[: n // 2])
    second_half = _to_float_array_1d(y[n // 2 :])

    features["shape_y_mean_first_half"] = (
        float(np.mean(first_half)) if len(first_half) > 0 else 0.0
    )
    features["shape_y_mean_second_half"] = (
        float(np.mean(second_half)) if len(second_half) > 0 else 0.0
    )
    features["shape_y_half_diff"] = (
        features["shape_y_mean_second_half"] - features["shape_y_mean_first_half"]
    )

    y_min_idx = int(np.argmin(y))
    features["shape_y_min_position"] = float(y_min_idx) / float(len(y))

    return features


# ============================================================================
# Velocity features (from training script)
# ============================================================================


def compute_velocity_features(trajectory: NDArray[np.float64]) -> Dict[str, float]:
    """Compute velocity and acceleration features from kinematic channels."""
    features: Dict[str, float] = {}
    if len(trajectory) < 10:
        return {f"vel_{k}": 0.0 for k in range(10)}

    vel = _to_float_array_1d(trajectory[:, 1])

    features["vel_mean"] = float(np.mean(vel))
    features["vel_std"] = float(np.std(vel)) if len(vel) > 1 else 0.0
    features["vel_max"] = float(np.max(vel))
    features["vel_min"] = float(np.min(vel))
    features["vel_range"] = features["vel_max"] - features["vel_min"]

    accel = _to_float_array_1d(trajectory[:, 2])
    features["vel_accel_mean"] = float(np.mean(accel)) if len(accel) > 0 else 0.0
    features["vel_accel_max"] = float(np.max(accel)) if len(accel) > 0 else 0.0
    features["vel_accel_min"] = float(np.min(accel)) if len(accel) > 0 else 0.0

    n = len(vel)
    q1 = _to_float_array_1d(vel[: n // 4] if n > 4 else vel)
    q2 = _to_float_array_1d(vel[n // 4 : n // 2] if n > 4 else vel)
    q3 = _to_float_array_1d(vel[n // 2 : 3 * n // 4] if n > 4 else vel)
    q4 = _to_float_array_1d(vel[3 * n // 4 :] if n > 4 else vel)

    features["vel_mean_q1"] = float(np.mean(q1)) if len(q1) > 0 else 0.0
    features["vel_mean_q2"] = float(np.mean(q2)) if len(q2) > 0 else 0.0
    features["vel_mean_q3"] = float(np.mean(q3)) if len(q3) > 0 else 0.0
    features["vel_mean_q4"] = float(np.mean(q4)) if len(q4) > 0 else 0.0

    return features


# ============================================================================
# Phase features (from training script)
# ============================================================================


def compute_phase_features(trajectory: NDArray[np.float64]) -> Dict[str, float]:
    """Compute phase-based features from trajectory peaks and valleys."""
    features: Dict[str, float] = {}
    if len(trajectory) < 20:
        return {f"phase_{k}": 0.0 for k in range(15)}

    y = _to_float_array_1d(trajectory[:, 0])
    y_smooth = _safe_savgol(y, max_win=21, polyorder=3)

    peaks, _ = signal.find_peaks(-y_smooth, prominence=0.01, distance=10)
    valleys, _ = signal.find_peaks(y_smooth, prominence=0.01, distance=10)

    features["phase_n_peaks"] = float(len(peaks))
    features["phase_n_valleys"] = float(len(valleys))

    if len(peaks) > 0:
        features["phase_first_peak_pos"] = float(peaks[0]) / float(len(y))
        features["phase_last_peak_pos"] = float(peaks[-1]) / float(len(y))
        features["phase_peak_span"] = (
            features["phase_last_peak_pos"] - features["phase_first_peak_pos"]
        )
        if len(peaks) > 1:
            features["phase_peak_gap"] = float(peaks[1] - peaks[0]) / float(len(y))
            features["phase_two_distinct_peaks"] = (
                1.0 if features["phase_peak_gap"] > 0.2 else 0.0
            )
        else:
            features["phase_peak_gap"] = 0.0
            features["phase_two_distinct_peaks"] = 0.0
    else:
        features["phase_first_peak_pos"] = 0.0
        features["phase_last_peak_pos"] = 0.0
        features["phase_peak_span"] = 0.0
        features["phase_peak_gap"] = 0.0
        features["phase_two_distinct_peaks"] = 0.0

    if len(valleys) > 0:
        features["phase_first_valley_pos"] = float(valleys[0]) / float(len(y))
        features["phase_valley_depth"] = float(y_smooth[valleys[0]]) - float(
            np.min(y_smooth)
        )
    else:
        features["phase_first_valley_pos"] = 0.0
        features["phase_valley_depth"] = 0.0

    y_min_idx = int(np.argmin(y_smooth))
    y_max_idx = int(np.argmax(y_smooth))
    features["phase_y_min_pos"] = float(y_min_idx) / float(len(y))
    features["phase_y_max_pos"] = float(y_max_idx) / float(len(y))
    features["phase_early_peak"] = 1.0 if y_min_idx < (len(y_smooth) // 2) else 0.0

    mid = len(y_smooth) // 2
    first_half = (
        _to_float_array_1d(y_smooth[:mid])
        if mid > 0
        else np.empty((0,), dtype=np.float64)
    )
    second_half = (
        _to_float_array_1d(y_smooth[mid:])
        if mid > 0
        else np.empty((0,), dtype=np.float64)
    )

    first_half_var = float(np.var(first_half)) if len(first_half) > 0 else 0.0
    second_half_var = float(np.var(second_half)) if len(second_half) > 0 else 0.0

    features["phase_first_half_var"] = first_half_var
    features["phase_second_half_var"] = second_half_var
    features["phase_var_ratio"] = (
        (second_half_var / first_half_var) if first_half_var > 0.0 else 0.0
    )

    return features


# ============================================================================
# S-curve detection (from training script)
# ============================================================================


def detect_s_curve_pattern(trajectory: NDArray[np.float64]) -> Dict[str, float]:
    """Detect S-curve trajectory pattern from vertical position channel."""
    if len(trajectory) < 10:
        return {
            "s_curve_score": 0.0,
            "s_curve_detected": 0.0,
            "upward_motion": 0.0,
            "downward_motion": 0.0,
            "n_upward_peaks": 0.0,
            "n_downward_peaks": 0.0,
        }

    y = _to_float_array_1d(trajectory[:, 0])
    dy = _to_float_array_1d(np.gradient(y))

    upward_peaks, _ = signal.find_peaks(-dy, height=0.001)
    upward_motion = float(np.sum(-dy[dy < 0]))

    downward_peaks, _ = signal.find_peaks(dy, height=0.001)
    downward_motion = float(abs(np.sum(dy[dy > 0])))

    total_motion = upward_motion + downward_motion
    if total_motion > 0:
        s_curve_score = min(upward_motion, downward_motion) / total_motion
    else:
        s_curve_score = 0.0

    return {
        "s_curve_score": s_curve_score,
        "s_curve_detected": 1.0
        if s_curve_score > VERTICAL_MOTION_BALANCE_RATIO
        else 0.0,
        "upward_motion": upward_motion,
        "downward_motion": downward_motion,
        "n_upward_peaks": float(len(upward_peaks)),
        "n_downward_peaks": float(len(downward_peaks)),
    }


# ============================================================================
# Clean & jerk pattern detection (from training script)
# ============================================================================


def detect_clean_jerk_pattern(trajectory: NDArray[np.float64]) -> Dict[str, float]:
    """Detect clean & jerk specific two-phase pattern from kinematic trajectory."""
    features: Dict[str, float] = {}
    if len(trajectory) < 30:
        return {f"cj_{k}": 0.0 for k in range(10)}

    y = _to_float_array_1d(trajectory[:, 0])
    y_smooth = _safe_savgol(y, max_win=21, polyorder=3)

    peaks, _ = signal.find_peaks(-y_smooth, prominence=0.02, distance=15)
    features["cj_n_major_peaks"] = float(len(peaks))

    if len(peaks) >= 2:
        first_peak = int(peaks[0])
        second_peak = int(peaks[1])

        features["cj_two_phase_detected"] = 1.0
        features["cj_phase_gap"] = float(second_peak - first_peak) / float(len(y))

        between_peaks = _to_float_array_1d(y_smooth[first_peak:second_peak])
        features["cj_intermediate_y_mean"] = (
            float(np.mean(between_peaks)) if len(between_peaks) > 0 else 0.0
        )
        features["cj_intermediate_y_max"] = (
            float(np.max(between_peaks)) if len(between_peaks) > 0 else 0.0
        )

        dy = _to_float_array_1d(np.diff(y_smooth))
        between_dy = _to_float_array_1d(
            dy[first_peak:second_peak] if second_peak <= len(dy) else dy[first_peak:]
        )
        if len(between_dy) > 0:
            near_zero = float(np.sum(np.abs(between_dy) < 0.001)) / float(
                len(between_dy)
            )
            features["cj_pause_ratio"] = near_zero
        else:
            features["cj_pause_ratio"] = 0.0
    else:
        features["cj_two_phase_detected"] = 0.0
        features["cj_phase_gap"] = 0.0
        features["cj_intermediate_y_mean"] = 0.0
        features["cj_intermediate_y_max"] = 0.0
        features["cj_pause_ratio"] = 0.0

    n = len(y_smooth)
    third = n // 3
    a = (
        _to_float_array_1d(y_smooth[:third])
        if third > 0
        else np.empty((0,), dtype=np.float64)
    )
    b = (
        _to_float_array_1d(y_smooth[third : 2 * third])
        if (2 * third) > third
        else np.empty((0,), dtype=np.float64)
    )
    c = (
        _to_float_array_1d(y_smooth[2 * third :])
        if n > (2 * third)
        else np.empty((0,), dtype=np.float64)
    )

    features["cj_first_third_var"] = float(np.var(a)) if len(a) > 0 else 0.0
    features["cj_second_third_var"] = float(np.var(b)) if len(b) > 0 else 0.0
    features["cj_third_third_var"] = float(np.var(c)) if len(c) > 0 else 0.0

    return features


def detect_clean_jerk_split_point(df: pd.DataFrame) -> Optional[int]:
    """Detect the frame index where the jerk begins in a clean+jerk lift.

    Uses a global-maximum + backward-valley search heuristic on the smoothed
    barbell vertical trajectory.  In image coordinates (Y=0 at top, Y
    increases downward), a physical dip is a local maximum in ``barbell_y``.

    Algorithm:
        1. global_max_idx = argmin(barbell_y_smooth) -> jerk lockout.
        2. Search backwards for the jerk dip bottom (local maximum in y).
        3. Search backwards from the dip for the last stable shoulder position.
        4. Validation gates confirm the split is plausible.

    Args:
        df: DataFrame with ``barbell_y_smooth`` and optionally ``vel_y_smooth``.

    Returns:
        Frame index where the jerk begins, or ``None`` if no split detected.
    """
    if "barbell_y_smooth" not in df.columns or len(df) < 80:
        return None

    y = _safe_interp_numeric(cast(pd.Series, df["barbell_y_smooth"]))
    if "vel_y_smooth" in df.columns:
        vel = _safe_interp_numeric(cast(pd.Series, df["vel_y_smooth"]))
    else:
        vel_raw = np.gradient(y)
        vel = _safe_savgol(vel_raw)

    # 1. Global maximum height = argmin(barbell_y_smooth) in image coords
    global_max_idx = int(np.argmin(y))

    # Basic sanity checks on global max position
    if global_max_idx < 20 or global_max_idx > len(y) - 10:
        return None

    # 3. Find jerk dip bottom in [0 : global_max_idx]
    pre_max = y[: global_max_idx + 1]
    if len(pre_max) < 20:
        return None

    y_smooth_pre = _safe_savgol(pre_max)
    y_range = float(np.nanmax(y) - np.nanmin(y))
    prom = max(0.02 * y_range, 0.01) if y_range > 0 else 0.01

    # Local maxima in y = physical dips
    peaks, _ = signal.find_peaks(y_smooth_pre, prominence=prom, distance=10)
    if len(peaks) == 0:
        return None

    # Select peak closest to global_max_idx (the dip just before lockout)
    dip_bottom_idx = int(peaks[np.argmin(np.abs(peaks - global_max_idx))])

    # The dip must be reasonably before global max
    if global_max_idx - dip_bottom_idx < 5:
        return None

    # 4. Find split point: last stable shoulder height before dip.
    #    The shoulder is where the bar pauses between clean recovery and
    #    the jerk dip.  We find it as the point of minimum |velocity|
    #    between the clean peak (deepest physical high point before dip)
    #    and the dip bottom.
    split_frame: Optional[int] = None

    # Find the clean peak: deepest local minimum before the dip
    valleys, _ = signal.find_peaks(-y_smooth_pre, prominence=prom * 0.3, distance=5)
    valid_valleys = valleys[valleys < dip_bottom_idx]
    clean_peak_idx: Optional[int] = None
    if len(valid_valleys) > 0:
        # Select the deepest valley (smallest y) before the dip
        clean_peak_idx = int(valid_valleys[np.argmin(y_smooth_pre[valid_valleys])])

    if clean_peak_idx is None:
        # Fallback: use a small offset from the start
        clean_peak_idx = 10

    # Search between clean peak and dip for the stable shoulder (min |vel|)
    search_start = min(clean_peak_idx + 5, dip_bottom_idx - 3)
    if search_start < dip_bottom_idx:
        vel_abs = np.abs(vel[search_start:dip_bottom_idx])
        if len(vel_abs) > 0:
            local_min_vel_idx = search_start + int(np.argmin(vel_abs))
            split_frame = int(local_min_vel_idx)

    if split_frame is None:
        search_end = max(0, dip_bottom_idx - 30)
        for i in range(dip_bottom_idx, search_end, -1):
            if abs(vel[i]) < 20.0:
                split_frame = int(i)
                break

    if split_frame is None:
        return None

    # 5. Validation gates
    # Dip must have reasonable duration
    if dip_bottom_idx - split_frame < 3:
        return None

    # Drive must have reasonable duration
    if global_max_idx - dip_bottom_idx < 3:
        return None

    # The bar between split and dip should be at shoulder height (lower y
    # than the first third of the lift, which is near the floor).
    first_third_mean = float(np.mean(y[: len(y) // 3]))
    mid_mean = float(np.mean(y[split_frame:dip_bottom_idx]))
    if mid_mean >= first_third_mean:
        return None

    return split_frame


# ============================================================================
# Pipeline-specific features (from step4_helpers/feature_extraction.py)
# ============================================================================


def _extract_dip_depth_norm(df: pd.DataFrame) -> float:
    """Compute dip_depth_norm: normalized dip depth for jerk detection.

    Requires: bar_phase, barbell_y_smooth, frame_height columns.
    Computed from Phase 0 (Dip) of the trajectory.
    """
    if "bar_phase" not in df.columns:
        return 0.0
    if "barbell_y_smooth" not in df.columns:
        return 0.0

    phases = df["bar_phase"].dropna().astype(int)
    bar_y = df["barbell_y_smooth"].dropna()
    min_len = min(len(bar_y), len(phases))
    bar_y_arr = np.asarray(bar_y.iloc[:min_len], dtype=float)
    phase_arr = np.asarray(phases.iloc[:min_len], dtype=int)

    dip_mask = phase_arr == 0
    dip_y = bar_y_arr[dip_mask]

    if len(dip_y) > 1:
        dip_lowest = float(np.min(dip_y))
        dip_highest = float(np.max(dip_y))
        dip_depth = dip_highest - dip_lowest
        frame_h = (
            float(df["frame_height"].iloc[0]) if "frame_height" in df.columns else 1.0
        )
        return float(dip_depth / frame_h) if frame_h > 0 else 0.0
    return 0.0


def _extract_elbow_angle_variance(df: pd.DataFrame) -> float:
    """Compute elbow_angle_variance: average variance of left and right elbow angles.

    Requires: left_elbow_angle, right_elbow_angle columns.
    """
    if "left_elbow_angle" not in df.columns or "right_elbow_angle" not in df.columns:
        return 0.0

    var_left_result = df["left_elbow_angle"].dropna().var()
    var_right_result = df["right_elbow_angle"].dropna().var()
    var_left = _safe_float(var_left_result)
    var_right = _safe_float(var_right_result)
    return (var_left + var_right) / 2


def _extract_min_elbow_angle_early(df: pd.DataFrame) -> float:
    """Compute min_elbow_angle_early: minimum elbow angle in first 60% of lift.

    Requires: left_elbow_angle, right_elbow_angle columns.
    """
    if "left_elbow_angle" not in df.columns or "right_elbow_angle" not in df.columns:
        return 180.0

    elbow_angles = df[["left_elbow_angle", "right_elbow_angle"]].dropna()
    if len(elbow_angles) == 0:
        return 180.0

    n = len(df)
    early_frames = elbow_angles.iloc[: int(n * 0.6)]
    if len(early_frames) > 0:
        return float(early_frames.min().min())
    return 180.0


def _extract_min_knee_angle_catch(df: pd.DataFrame) -> float:
    """Compute min_knee_angle_catch: minimum knee angle in last 30% of lift.

    Requires: left_knee_angle, right_knee_angle columns.
    """
    if "left_knee_angle" not in df.columns or "right_knee_angle" not in df.columns:
        return 180.0

    knee_angles = df[["left_knee_angle", "right_knee_angle"]].dropna()
    if len(knee_angles) == 0:
        return 180.0

    n = len(df)
    late_frames = knee_angles.iloc[int(n * 0.7) :]
    if len(late_frames) > 0:
        return float(late_frames.min().min())
    return 180.0


def _extract_max_vel_y(df: pd.DataFrame) -> float:
    """Compute max_vel_y from raw velocity column."""
    if "vel_y_smooth" not in df.columns:
        return 0.0
    vel = df["vel_y_smooth"].dropna()
    if len(vel) == 0:
        return 0.0
    return float(np.asarray(vel, dtype=float).max())


def _extract_mean_vel_y_first_half(df: pd.DataFrame) -> float:
    """Compute mean_vel_y_first_half from raw velocity column."""
    if "vel_y_smooth" not in df.columns:
        return 0.0
    vel = df["vel_y_smooth"].dropna()
    if len(vel) == 0:
        return 0.0
    vel_arr = np.asarray(vel, dtype=float)
    n = len(vel_arr)
    first_half = vel_arr[: n // 2]
    return float(np.mean(first_half)) if len(first_half) > 0 else 0.0


def _extract_pull_duration_frac(df: pd.DataFrame) -> float:
    """Compute pull_duration_frac: Phase 0 duration as fraction of total."""
    if "bar_phase" not in df.columns or "time_s" not in df.columns:
        return 0.0

    phases = df["bar_phase"].dropna()
    times = df["time_s"].dropna()
    if len(phases) == 0 or len(times) == 0:
        return 0.0

    times_arr = np.asarray(times, dtype=float)
    phases_arr = np.asarray(phases, dtype=float)
    total_time = float(times_arr[-1] - times_arr[0])

    if total_time <= 0:
        return 0.0

    phase_0_mask = phases_arr == 0
    phase_0_times = times_arr[phase_0_mask]
    phase_0_time = (
        float(np.sum(np.diff(phase_0_times))) if len(phase_0_times) > 0 else 0.0
    )
    return phase_0_time / total_time


def _extract_turnover_duration_frac(df: pd.DataFrame) -> float:
    """Compute turnover_duration_frac: Phase 1 duration as fraction of total."""
    if "bar_phase" not in df.columns or "time_s" not in df.columns:
        return 0.0

    phases = df["bar_phase"].dropna()
    times = df["time_s"].dropna()
    if len(phases) == 0 or len(times) == 0:
        return 0.0

    times_arr = np.asarray(times, dtype=float)
    phases_arr = np.asarray(phases, dtype=float)
    total_time = float(times_arr[-1] - times_arr[0])

    if total_time <= 0:
        return 0.0

    phase_1_mask = phases_arr == 1
    phase_1_times = times_arr[phase_1_mask]
    phase_1_time = (
        float(np.sum(np.diff(phase_1_times))) if len(phase_1_times) > 0 else 0.0
    )
    return phase_1_time / total_time


# ============================================================================
# Unified feature extraction
# ============================================================================

# Feature names in the exact order the model expects (sorted alphabetically).
_MODEL_FEATURE_NAMES: List[str] = [
    "cj_n_major_peaks",
    "cj_phase_gap",
    "cj_two_phase_detected",
    "dip_depth_norm",
    "elbow_angle_variance",
    "max_vel_y",
    "mean_vel_y_first_half",
    "min_elbow_angle_early",
    "min_knee_angle_catch",
    "phase_early_peak",
    "phase_first_peak_pos",
    "phase_last_peak_pos",
    "phase_n_peaks",
    "phase_n_valleys",
    "phase_peak_gap",
    "phase_two_distinct_peaks",
    "phase_y_min_pos",
    "pull_duration_frac",
    "s_curve_detected",
    "s_curve_score",
    "shape_path_efficiency",
    "shape_path_tortuosity",
    "shape_y_end",
    "shape_y_kurtosis",
    "shape_y_max",
    "shape_y_min",
    "shape_y_min_position",
    "shape_y_range",
    "shape_y_skewness",
    "shape_y_start",
    "trajectory_length",
    "turnover_duration_frac",
    "vel_max",
    "vel_mean",
    "vel_min",
    "vel_range",
    "vel_std",
]


def get_model_feature_names() -> List[str]:
    """Return the list of feature names the model expects, in order."""
    return list(_MODEL_FEATURE_NAMES)


def extract_model_features(df: pd.DataFrame) -> Dict[str, float]:
    """Extract exactly the 37 features the lift detection model expects.

    Combines trajectory-based features (from training script) with
    joint-angle and pipeline-specific features to produce the complete
    feature vector in the model's expected sorted order.

    Args:
        df: DataFrame with kinematic data. Must contain at minimum:
            - barbell_y_smooth, vel_y_smooth, accel_y_smooth (trajectory)
            - bar_phase (phase labels 0/1/2)
            - time_s (timestamps)
            - frame_height (frame dimensions)
            - Joint columns for angle computation (optional, default 0.0):
              left_elbow_angle, right_elbow_angle,
              left_knee_angle, right_knee_angle

    Returns:
        Dict of 37 feature name -> value pairs.
    """
    features: Dict[str, float] = {}

    # Extract trajectory (Nx3: [y_norm, vel_norm, accel_norm])
    trajectory = extract_trajectory(df)
    if len(trajectory) < 10:
        # Return all zeros if trajectory too short
        return {name: 0.0 for name in _MODEL_FEATURE_NAMES}

    # --- Trajectory shape features (model needs 6 of them) ---
    shape_feats = compute_trajectory_shape_features(trajectory)
    for name in [
        "shape_path_efficiency",
        "shape_path_tortuosity",
        "shape_y_end",
        "shape_y_kurtosis",
        "shape_y_max",
        "shape_y_min",
        "shape_y_min_position",
        "shape_y_range",
        "shape_y_skewness",
        "shape_y_start",
    ]:
        features[name] = shape_feats.get(name, 0.0)

    # --- Velocity features (model needs 5) ---
    vel_feats = compute_velocity_features(trajectory)
    for name in ["vel_max", "vel_mean", "vel_min", "vel_range", "vel_std"]:
        features[name] = vel_feats.get(name, 0.0)

    # --- Phase features (model needs 7) ---
    phase_feats = compute_phase_features(trajectory)
    for name in [
        "phase_early_peak",
        "phase_first_peak_pos",
        "phase_last_peak_pos",
        "phase_n_peaks",
        "phase_n_valleys",
        "phase_peak_gap",
        "phase_two_distinct_peaks",
        "phase_y_min_pos",
    ]:
        features[name] = phase_feats.get(name, 0.0)

    # --- S-curve features (model needs 2) ---
    s_curve_feats = detect_s_curve_pattern(trajectory)
    features["s_curve_score"] = s_curve_feats.get("s_curve_score", 0.0)
    features["s_curve_detected"] = s_curve_feats.get("s_curve_detected", 0.0)

    # --- Clean & jerk pattern features (model needs 3) ---
    cj_feats = detect_clean_jerk_pattern(trajectory)
    features["cj_n_major_peaks"] = cj_feats.get("cj_n_major_peaks", 0.0)
    features["cj_phase_gap"] = cj_feats.get("cj_phase_gap", 0.0)
    features["cj_two_phase_detected"] = cj_feats.get("cj_two_phase_detected", 0.0)

    # --- Pipeline-specific features (model needs 5) ---
    features["dip_depth_norm"] = _extract_dip_depth_norm(df)
    features["elbow_angle_variance"] = _extract_elbow_angle_variance(df)
    features["max_vel_y"] = _extract_max_vel_y(df)
    features["mean_vel_y_first_half"] = _extract_mean_vel_y_first_half(df)
    features["min_elbow_angle_early"] = _extract_min_elbow_angle_early(df)
    features["min_knee_angle_catch"] = _extract_min_knee_angle_catch(df)
    features["pull_duration_frac"] = _extract_pull_duration_frac(df)
    features["turnover_duration_frac"] = _extract_turnover_duration_frac(df)

    # --- Direct value ---
    features["trajectory_length"] = float(len(trajectory))

    return features


def extract_model_features_as_array(
    df: pd.DataFrame, feature_names: Optional[List[str]] = None
) -> NDArray[np.float64]:
    """Extract features as a 1D numpy array in model-expected order.

    Args:
        df: DataFrame with kinematic data
        feature_names: Ordered list of feature names. If None, uses the
                       model's default sorted order.

    Returns:
        1D float64 array of features ready for scaling + prediction.
    """
    if feature_names is None:
        feature_names = _MODEL_FEATURE_NAMES

    feat_dict = extract_model_features(df)
    arr = np.array(
        [float(feat_dict.get(name, 0.0)) for name in feature_names],
        dtype=np.float64,
    )
    return cast(NDArray[np.float64], arr)


# ============================================================================
# Joint angle computation (Fix Issue 5)
# ============================================================================


def compute_joint_angle(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    """Compute angle at point b given three (x, y) points.

    Uses cosine rule. Returns angle in degrees [0, 180].
    """
    ba_x = a[0] - b[0]
    ba_y = a[1] - b[1]
    bc_x = c[0] - b[0]
    bc_y = c[1] - b[1]

    mag_ba = (ba_x**2 + ba_y**2) ** 0.5
    mag_bc = (bc_x**2 + bc_y**2) ** 0.5

    if mag_ba < 1e-8 or mag_bc < 1e-8:
        return 180.0

    cos_angle = (ba_x * bc_x + ba_y * bc_y) / (mag_ba * mag_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))

    return float(math.degrees(math.acos(cos_angle)))


def compute_joint_angles_from_landmarks(
    landmarks: Dict[int, tuple[float, float, float, float]],
    frame_width: int,
    frame_height: int,
) -> Dict[str, float]:
    """Compute joint angles from MediaPipe pose landmarks.

    Args:
        landmarks: Dict of {landmark_index: (x_norm, y_norm, z, visibility)}
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels

    Returns:
        Dict with left_elbow_angle, right_elbow_angle,
        left_knee_angle, right_knee_angle in degrees.
    """
    angles: Dict[str, float] = {}

    # MediaPipe landmark indices:
    # 11=left_shoulder, 12=right_shoulder
    # 13=left_elbow, 14=right_elbow
    # 15=left_wrist, 16=right_wrist
    # 23=left_hip, 24=right_hip
    # 25=left_knee, 26=right_knee
    # 27=left_ankle, 28=right_ankle

    def _get_px(idx: int) -> Optional[tuple[float, float]]:
        """Get pixel coordinates for a landmark index."""
        if idx not in landmarks:
            return None
        x_norm, y_norm, _z, vis = landmarks[idx]
        if vis < 0.1:
            return None
        return (x_norm * frame_width, y_norm * frame_height)

    # Left elbow angle: shoulder(11) -> elbow(13) -> wrist(15)
    shoulder = _get_px(11)
    elbow = _get_px(13)
    wrist = _get_px(15)
    if shoulder and elbow and wrist:
        angles["left_elbow_angle"] = compute_joint_angle(shoulder, elbow, wrist)
    else:
        angles["left_elbow_angle"] = 180.0

    # Right elbow angle: shoulder(12) -> elbow(14) -> wrist(16)
    shoulder = _get_px(12)
    elbow = _get_px(14)
    wrist = _get_px(16)
    if shoulder and elbow and wrist:
        angles["right_elbow_angle"] = compute_joint_angle(shoulder, elbow, wrist)
    else:
        angles["right_elbow_angle"] = 180.0

    # Left knee angle: hip(23) -> knee(25) -> ankle(27)
    hip = _get_px(23)
    knee = _get_px(25)
    ankle = _get_px(27)
    if hip and knee and ankle:
        angles["left_knee_angle"] = compute_joint_angle(hip, knee, ankle)
    else:
        angles["left_knee_angle"] = 180.0

    # Right knee angle: hip(24) -> knee(26) -> ankle(28)
    hip = _get_px(24)
    knee = _get_px(26)
    ankle = _get_px(28)
    if hip and knee and ankle:
        angles["right_knee_angle"] = compute_joint_angle(hip, knee, ankle)
    else:
        angles["right_knee_angle"] = 180.0

    return angles


def compute_joint_positions_from_landmarks(
    landmarks: Dict[int, tuple[float, float, float, float]],
) -> Dict[str, float]:
    """Extract normalized joint positions from MediaPipe landmarks.

    Returns dict with keys like left_knee_x, left_knee_y, etc.
    Values are normalized [0, 1] coordinates.
    """
    # MediaPipe index to joint name mapping
    joint_map = {
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist",
        23: "left_hip",
        24: "right_hip",
        25: "left_knee",
        26: "right_knee",
        27: "left_ankle",
        28: "right_ankle",
    }

    positions: Dict[str, float] = {}
    for idx, name in joint_map.items():
        if idx in landmarks:
            x_norm, y_norm, _z, _vis = landmarks[idx]
            positions[f"{name}_x"] = float(x_norm)
            positions[f"{name}_y"] = float(y_norm)
        else:
            positions[f"{name}_x"] = 0.0
            positions[f"{name}_y"] = 0.0

    return positions


# ============================================================================
# Phase detection for live data (Fix Issue 4)
# ============================================================================


def detect_phases_from_velocity(
    vel_smooth: NDArray[np.float64],
    y_smooth: Optional[NDArray[np.float64]] = None,
    fps: float = 30.0,
) -> NDArray[np.int64]:
    """Detect 3 phases from smoothed velocity signal.

    Phase 0 (Pull): bar accelerating upward (velocity becoming more negative)
    Phase 1 (Pull-under): bar decelerating and turning over (peak velocity to peak height)
    Phase 2 (Recovery): bar descending from peak (velocity positive or near zero)

    In image coordinates (Y=0 at top):
    - Negative velocity = bar moving UP (ascending)
    - Positive velocity = bar moving DOWN (descending)

    Args:
        vel_smooth: Smoothed velocity array (image coordinates)
        y_smooth: Optional smoothed y-position array for finding peak height.
                  If None, computed by cumulative integration of velocity.
        fps: Frames per second

    Returns:
        Array of phase labels (0, 1, or 2) same length as vel_smooth.
    """
    n = len(vel_smooth)
    if n < 10:
        return np.zeros(n, dtype=np.int64)

    # Smooth the velocity for stable boundary detection
    vel_for_phases = _safe_savgol(
        vel_smooth, max_win=min(21, n if n % 2 == 1 else n - 1)
    )

    # Compute y-position if not provided (integrate velocity)
    if y_smooth is None:
        y_smooth = np.cumsum(vel_for_phases)

    # Key transition points:
    # 1. Peak velocity: most negative = fastest upward movement (end of pull)
    peak_vel_idx = int(np.argmin(vel_for_phases))

    # 2. Peak height: minimum y-value = highest bar position (end of pull-under)
    peak_height_idx = int(np.argmin(y_smooth))

    # Ensure peak_height comes after peak_velocity (physically required)
    if peak_height_idx <= peak_vel_idx:
        # Search for minimum y after peak velocity
        search_start = peak_vel_idx + 1
        if search_start < n:
            peak_height_idx = search_start + int(np.argmin(y_smooth[search_start:]))
        else:
            peak_height_idx = peak_vel_idx + 1
            if peak_height_idx >= n:
                peak_height_idx = n - 1

    # Ensure minimum phase 1 length (at least 2 frames)
    if peak_height_idx - peak_vel_idx < 2:
        peak_height_idx = min(peak_vel_idx + 2, n - 1)

    # Assign phases
    phases = np.zeros(n, dtype=np.int64)
    phases[:peak_vel_idx] = 0  # Pull (acceleration phase)
    phases[peak_vel_idx:peak_height_idx] = 1  # Pull-under (deceleration/turnover)
    phases[peak_height_idx:] = 2  # Recovery (descent)

    return phases


def detect_phases_from_trajectory(
    y_smooth: NDArray[np.float64],
    fps: float = 30.0,
) -> NDArray[np.int64]:
    """Detect 3 phases from smoothed barbell y-position trajectory.

    Uses position-based heuristics:
    - Phase 0 (Pull): bar moving upward (y decreasing in image coords)
    - Phase 1 (Pull-under): bar near peak, rapid transition
    - Phase 2 (Recovery): bar stable or descending from peak

    Args:
        y_smooth: Smoothed y-position (image coordinates, lower = higher)
        fps: Frames per second

    Returns:
        Array of phase labels (0, 1, or 2) same length as y_smooth.
    """
    n = len(y_smooth)
    if n < 10:
        return np.zeros(n, dtype=np.int64)

    vel = np.gradient(y_smooth)
    return detect_phases_from_velocity(vel, y_smooth=y_smooth, fps=fps)


def _detect_three_phases_hip(
    vel_smooth: NDArray[np.float64],
    hip_y: NDArray[np.float64],
    fps: float,
) -> NDArray[np.int64]:
    """Detect pull / pull-under / recovery using hip-y velocity.

    Mirrors step2_helpers/kinematics.detect_three_phases exactly.
    In image coordinates (Y=0 at top):
      - Negative bar velocity = bar moving UP (ascending)
      - Positive hip velocity = hips moving DOWN (athlete squatting under)

    Phase 0: Pull — bar moving up, hips not yet dropping
    Phase 1: Pull-under — hips actively descending under the bar
    Phase 2: Recovery — hips stopped descending
    """
    n = len(vel_smooth)
    if n < 10:
        return np.zeros(n, dtype=np.int64)

    vel_max = float(np.nanmax(np.abs(vel_smooth)))
    vel_threshold = max(10.0, vel_max * 0.05)  # PHASE_VEL_THRESHOLD_FACTOR
    # In image coords: negative velocity = bar moving UP
    bar_moving_up = vel_smooth < -vel_threshold

    if not np.any(bar_moving_up):
        return np.zeros(n, dtype=np.int64)

    pull_start = int(np.argmax(bar_moving_up))

    # Hip velocity after pull starts
    hip_after = hip_y[pull_start:]
    if len(hip_after) < 5:
        return np.zeros(n, dtype=np.int64)

    hip_smooth = _safe_savgol(_to_float_array_1d(hip_after), max_win=9, polyorder=3)
    hip_vel = np.gradient(hip_smooth)

    hip_std = float(np.std(hip_smooth))
    hip_drop_threshold = (
        hip_std * 0.1 if hip_std > 0 else 0.5
    )  # PHASE_HIP_DROP_STD_FACTOR
    hips_dropping = hip_vel > hip_drop_threshold

    pull_under_start: Optional[int] = None
    if np.any(hips_dropping):
        pull_under_start = pull_start + int(np.argmax(hips_dropping))

    recovery_start: Optional[int] = None
    if pull_under_start is not None:
        hip_after_pu = hip_y[pull_under_start:]
        if len(hip_after_pu) >= 5:
            hip_smooth_pu = _safe_savgol(
                _to_float_array_1d(hip_after_pu), max_win=9, polyorder=3
            )
            hip_vel_pu = np.gradient(hip_smooth_pu)
            hips_stopped = hip_vel_pu <= hip_drop_threshold * 0.5
            if np.any(hips_stopped):
                recovery_start = pull_under_start + int(np.argmax(hips_stopped))

    phases = np.zeros(n, dtype=np.int64)
    if pull_under_start is not None:
        phases[pull_under_start:] = 1
    if recovery_start is not None:
        phases[recovery_start:] = 2
    return phases


def _detect_jerk_phases_simple(
    bar_y: NDArray[np.float64],
    vel_smooth: NDArray[np.float64],
    knee_angles: Optional[NDArray[np.float64]],
    min_dip_depth_px: float,
    fps: float,
) -> NDArray[np.int64]:
    """Simplified jerk phase detection for live preview.

    Mirrors step2_helpers/kinematics.detect_jerk_phases but without
    print statements and DataFrame overhead.

    Phase 0: Dip — knees bending, bar moving down
    Phase 1: Drive — knees extending, bar moving up
    Phase 2: Recovery — bar decelerating
    """
    n = len(bar_y)
    if n < 10:
        return np.zeros(n, dtype=np.int64)

    vel_sm = _safe_savgol(_to_float_array_1d(vel_smooth), max_win=7, polyorder=3)
    knee_vel_sm: Optional[NDArray[np.float64]] = None

    if knee_angles is not None and len(knee_angles) == n:
        ka = pd.Series(knee_angles, dtype="float64").interpolate().bfill().ffill()
        ka_vel = ka.diff().fillna(0) * fps
        knee_vel_sm = _safe_savgol(
            ka_vel.values.astype(np.float64),
            max_win=min(7, n // 2 * 2 + 1),
            polyorder=3,
        )

    dip_start: Optional[int] = None
    drive_start: Optional[int] = None

    if knee_vel_sm is not None:
        # Use knee angle velocity: negative = knees bending
        knee_bending = knee_vel_sm < -20.0  # JERK_DIP_VELOCITY_THRESHOLD
        if np.any(knee_bending):
            bending_indices = np.where(knee_bending)[0]
            for idx in bending_indices:
                subsequent = knee_vel_sm[idx:]
                stop = subsequent >= -20.0
                if np.any(stop):
                    dip_end = idx + int(np.argmax(stop))
                else:
                    dip_end = n - 1
                if dip_end - idx < 3:
                    continue
                # Check minimum dip depth
                y_start = float(bar_y[idx])
                y_lowest = float(np.max(bar_y[idx : dip_end + 1]))
                if (y_lowest - y_start) >= min_dip_depth_px:
                    dip_start = int(idx)
                    # Find drive start: knee vel > 0 after dip
                    after_dip = knee_vel_sm[dip_start:]
                    extending = after_dip > 20.0  # JERK_DRIVE_VELOCITY_THRESHOLD
                    if np.any(extending):
                        drive_start = dip_start + int(np.argmax(extending))
                    break
    else:
        # Fallback: use bar velocity for dip detection
        moving_down = vel_sm > 20.0
        if np.any(moving_down):
            down_indices = np.where(moving_down)[0]
            for idx in down_indices:
                subsequent_vel = vel_sm[idx:]
                stop_down = subsequent_vel <= 20.0
                if np.any(stop_down):
                    dip_end = idx + int(np.argmax(stop_down))
                else:
                    dip_end = n - 1
                if dip_end - idx < 3:
                    continue
                y_start = float(bar_y[idx])
                y_lowest = float(np.max(bar_y[idx : dip_end + 1]))
                if (y_lowest - y_start) >= min_dip_depth_px:
                    dip_start = int(idx)
                    after_dip = vel_sm[dip_start:]
                    moving_up = after_dip < -20.0
                    if np.any(moving_up):
                        drive_start = dip_start + int(np.argmax(moving_up))
                    break

    if dip_start is None:
        return np.zeros(n, dtype=np.int64)

    # Recovery: velocity drops below 25% of peak drive velocity
    recovery_start: Optional[int] = None
    if drive_start is not None:
        drive_vels = vel_sm[drive_start:]
        peak_vel = float(np.min(drive_vels))
        recovery_thresh = peak_vel * 0.25
        past_drive = vel_sm[drive_start:]
        slowing = past_drive > recovery_thresh
        if np.any(slowing):
            recovery_start = drive_start + int(np.argmax(slowing))

    phases = np.zeros(n, dtype=np.int64)
    if drive_start is not None:
        phases[dip_start:drive_start] = 0  # Dip
        phases[drive_start:] = 1  # Drive
    else:
        phases[dip_start:] = 0  # All dip, no clear drive
    if recovery_start is not None:
        phases[recovery_start:] = 2  # Recovery
    return phases


def add_phases_to_dataframe(
    df: pd.DataFrame,
    fps: float = 30.0,
) -> pd.DataFrame:
    """Add bar_phase column to DataFrame using the same detection as the
    normal post-process pipeline.

    Mirrors step2_helpers/kinematics.assign_phases_kinematic:
      - For clean/snatch: uses hip_y_avg velocity to detect pull-under
      - For jerk (bar starts high): uses knee angles / bar velocity for dip-drive

    Modifies df in place and returns it.
    """
    if "barbell_y_smooth" not in df.columns:
        df["bar_phase"] = 0
        return df

    y = df["barbell_y_smooth"].interpolate().bfill().ffill().values.astype(float)
    n = len(y)
    if n < 10:
        df["bar_phase"] = 0
        return df

    y_smooth = _safe_savgol(
        _to_float_array_1d(y), max_win=min(21, n if n % 2 == 1 else n - 1)
    )
    vel = np.gradient(y_smooth, 1.0 / fps if fps > 0 else 1.0 / 30.0)
    vel_smooth = _safe_savgol(vel, max_win=min(15, n if n % 2 == 1 else n - 1))

    # Detect if this is a jerk: bar starts at a high position.
    # Normalize by frame_height to get [0,1] coords; < 0.5 means bar
    # is in the upper half of the frame (shoulders/chest), not floor.
    frame_h = float(df["frame_height"].iloc[0]) if "frame_height" in df.columns else 1.0
    if frame_h <= 0:
        frame_h = 1.0
    y_start_norm = float(y_smooth[0])
    is_jerk = y_start_norm < 0.5

    if is_jerk:
        # Jerk phase detection
        knee_angles_arr: Optional[NDArray[np.float64]] = None
        if "left_knee_angle" in df.columns and "right_knee_angle" in df.columns:
            ka = df[["left_knee_angle", "right_knee_angle"]].mean(axis=1)
            ka = ka.interpolate().bfill().ffill()
            knee_angles_arr = ka.values.astype(np.float64)

        # Compute torso length for minimum dip depth threshold
        torso_length_px = frame_h * 0.3  # fallback
        if all(
            c in df.columns
            for c in [
                "left_shoulder_y",
                "right_shoulder_y",
                "left_hip_y",
                "right_hip_y",
            ]
        ):
            shoulder_y = (
                (
                    df["left_shoulder_y"].astype(float)
                    + df["right_shoulder_y"].astype(float)
                )
                / 2
            ).dropna()
            hip_data_y = (
                (df["left_hip_y"].astype(float) + df["right_hip_y"].astype(float)) / 2
            ).dropna()
            if len(shoulder_y) > 0 and len(hip_data_y) > 0:
                ml = min(len(shoulder_y), len(hip_data_y))
                torso_dist = abs(
                    np.asarray(shoulder_y[:ml], dtype=float)
                    - np.asarray(hip_data_y[:ml], dtype=float)
                )
                torso_length_px = float(np.median(torso_dist)) * frame_h
        min_dip = torso_length_px * 0.30

        phases = _detect_jerk_phases_simple(
            y_smooth, vel_smooth, knee_angles_arr, min_dip, fps
        )
    else:
        # Clean / snatch phase detection — use hip_y_avg when available
        hip_y = None
        if "hip_y_avg" in df.columns:
            hip_raw = df["hip_y_avg"].values.astype(float)
            # hip_y_avg of 0 means no data; check for valid values
            if np.any(hip_raw > 0):
                hip_filled = (
                    pd.Series(hip_raw).replace(0, np.nan).interpolate().bfill().ffill()
                )
                hip_y = hip_filled.values.astype(np.float64)

        if hip_y is not None and len(hip_y) == n and np.any(hip_y > 0):
            phases = _detect_three_phases_hip(vel_smooth, hip_y, fps)
        else:
            # Fallback: velocity-only (same as old behavior)
            peak_vel_idx = int(np.argmin(vel_smooth))
            peak_height_idx = int(np.argmin(y_smooth))
            if peak_height_idx <= peak_vel_idx:
                search_start = peak_vel_idx + 1
                if search_start < n:
                    peak_height_idx = search_start + int(
                        np.argmin(y_smooth[search_start:])
                    )
                else:
                    peak_height_idx = min(peak_vel_idx + 1, n - 1)
            if peak_height_idx - peak_vel_idx < 2:
                peak_height_idx = min(peak_vel_idx + 2, n - 1)
            phases = np.zeros(n, dtype=np.int64)
            phases[:peak_vel_idx] = 0
            phases[peak_vel_idx:peak_height_idx] = 1
            phases[peak_height_idx:] = 2

    df["bar_phase"] = phases
    return df


def build_lift_dataframe(
    barbell_y: List[float],
    barbell_x: List[float],
    timestamps_ms: List[float],
    landmarks_list: List[Dict[int, tuple[float, float, float, float]]],
    frame_width: int,
    frame_height: int,
    fps: float = 30.0,
) -> pd.DataFrame:
    """Build a DataFrame from live lift buffer data.

    Converts raw live preview data into the format expected by
    extract_model_features().

    Args:
        barbell_y: List of barbell y-center values (pixels)
        barbell_x: List of barbell x-center values (pixels)
        timestamps_ms: List of timestamps in milliseconds
        landmarks_list: List of landmark dicts per frame
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        fps: Frames per second

    Returns:
        DataFrame with all columns needed for feature extraction.
    """
    n = len(barbell_y)
    if n < 10:
        return pd.DataFrame()

    # Keep pixel coordinates to match main pipeline format
    y_arr = _to_float_array_1d(barbell_y)
    x_arr = _to_float_array_1d(barbell_x)

    y_smooth = _safe_savgol(y_arr, max_win=min(21, n if n % 2 == 1 else n - 1))
    x_smooth = _safe_savgol(x_arr, max_win=min(21, n if n % 2 == 1 else n - 1))

    # Compute velocity (normalized units per second)
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
    vel_y = np.gradient(y_smooth, dt)
    vel_y_smooth = _safe_savgol(vel_y, max_win=min(15, n if n % 2 == 1 else n - 1))

    # Compute acceleration
    accel_y = np.gradient(vel_y_smooth, dt)
    accel_y_smooth = _safe_savgol(accel_y, max_win=min(15, n if n % 2 == 1 else n - 1))

    # Time array
    time_s = [(ts - timestamps_ms[0]) / 1000.0 for ts in timestamps_ms]

    # Build DataFrame
    df_data: Dict[str, Any] = {
        "frame": list(range(n)),
        "time_s": time_s,
        "barbell_y_smooth": y_smooth.tolist(),
        "barbell_x_smooth": x_smooth.tolist(),
        "vel_y_smooth": vel_y_smooth.tolist(),
        "accel_y_smooth": accel_y_smooth.tolist(),
        "frame_height": [float(frame_height)] * n,
        "frame_width": [float(frame_width)] * n,
    }

    # Compute joint data per frame
    joint_angle_keys = [
        "left_elbow_angle",
        "right_elbow_angle",
        "left_knee_angle",
        "right_knee_angle",
    ]
    joint_pos_keys = [
        "left_shoulder_x",
        "left_shoulder_y",
        "right_shoulder_x",
        "right_shoulder_y",
        "left_elbow_x",
        "left_elbow_y",
        "right_elbow_x",
        "right_elbow_y",
        "left_wrist_x",
        "left_wrist_y",
        "right_wrist_x",
        "right_wrist_y",
        "left_hip_x",
        "left_hip_y",
        "right_hip_x",
        "right_hip_y",
        "left_knee_x",
        "left_knee_y",
        "right_knee_x",
        "right_knee_y",
        "left_ankle_x",
        "left_ankle_y",
        "right_ankle_x",
        "right_ankle_y",
    ]

    for key in joint_angle_keys + joint_pos_keys:
        df_data[key] = []

    hip_y_avgs: List[float] = []

    for i in range(n):
        lm = landmarks_list[i] if i < len(landmarks_list) else {}

        # Joint angles
        angles = compute_joint_angles_from_landmarks(lm, frame_width, frame_height)
        for key in joint_angle_keys:
            df_data[key].append(angles.get(key, 180.0))

        # Joint positions
        positions = compute_joint_positions_from_landmarks(lm)
        for key in joint_pos_keys:
            df_data[key].append(positions.get(key, 0.0))

        # Hip y average
        left_hip_y = positions.get("left_hip_y", 0.0)
        right_hip_y = positions.get("right_hip_y", 0.0)
        if left_hip_y > 0 and right_hip_y > 0:
            hip_y_avgs.append((left_hip_y + right_hip_y) / 2)
        elif left_hip_y > 0:
            hip_y_avgs.append(left_hip_y)
        elif right_hip_y > 0:
            hip_y_avgs.append(right_hip_y)
        else:
            hip_y_avgs.append(0.0)

    df_data["hip_y_avg"] = hip_y_avgs

    df = pd.DataFrame(df_data)
    df = df.set_index("frame")

    # Detect phases
    df = add_phases_to_dataframe(df, fps)

    return df


# ============================================================================
# Model loading and prediction
# ============================================================================

try:
    import pickle
except ImportError:
    pickle = None  # type: ignore[assignment]


def load_lift_detection_model(
    model_path: str,
) -> Optional[Dict[str, Any]]:
    """Load the lift detection model from a pickle file.

    Args:
        model_path: Path to lift_detection_model.pkl

    Returns:
        Dict with 'classifier', 'scaler', 'feature_names' keys,
        or None if loading fails.
    """
    if pickle is None:
        return None

    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        return cast(Dict[str, Any], model_data)
    except Exception:
        return None


def predict_lift_type(
    df: pd.DataFrame,
    model_data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Predict lift type from kinematic DataFrame using loaded model.

    Args:
        df: DataFrame with kinematic data (same format as final_analysis.csv)
        model_data: Dict from load_lift_detection_model()

    Returns:
        Dict with:
            - predicted_class: str ("clean", "jerk", "snatch", or "clean_jerk")
            - probabilities: dict of class -> probability
            - confidence: float (max probability)
            - is_clean_jerk: bool (True when model says clean_jerk or heuristic triggers)
        or None if prediction fails.
    """
    try:
        classifier = model_data["classifier"]
        scaler = model_data["scaler"]
        feature_names = cast(List[str], model_data["feature_names"])

        feat_dict = extract_model_features(df)
        X = np.array(
            [[float(feat_dict.get(name, 0.0)) for name in feature_names]],
            dtype=np.float64,
        )
        X_scaled = scaler.transform(X)

        prediction = str(classifier.predict(X_scaled)[0])
        probas = np.asarray(classifier.predict_proba(X_scaled)[0], dtype=np.float64)
        classes = [str(c) for c in classifier.classes_]
        class_probs = dict(zip(classes, probas.tolist()))

        confidence = float(max(probas))

        # Clean & jerk heuristic: only triggers when the model predicts
        # "jerk" or "clean" AND the trajectory has a convincing two-phase
        # structure (large gap between peaks, high plateau, stationary bar).
        # With a 4-class model the "clean_jerk" class handles most cases;
        # this heuristic is a fallback for edge cases.
        # is_clean_jerk is True when the model directly predicts "clean_jerk"
        # OR when the fallback heuristic detects a two-phase pattern.
        is_clean_jerk = prediction == "clean_jerk"

        if not is_clean_jerk and prediction in ("jerk", "clean") and confidence > 0.5:
            split_point = detect_clean_jerk_split_point(df)
            traj_len = feat_dict.get("trajectory_length", 0.0)
            if split_point is not None and traj_len > 80:
                is_clean_jerk = True

        return {
            "predicted_class": prediction,
            "probabilities": class_probs,
            "confidence": confidence,
            "is_clean_jerk": is_clean_jerk,
        }
    except Exception:
        return None
