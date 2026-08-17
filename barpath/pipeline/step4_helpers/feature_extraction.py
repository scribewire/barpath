"""
Feature extraction for ML-based technique analysis.

Two extraction paths, both reading from final_analysis.csv:
  - extract_trajectory(): Multi-channel kinematic time-series (used for trajectory-shape lift classification)
  - extract_technique_features(): ~26 scalar features for fault discrimination

ALL features are camera-angle-invariant — derived from kinematic time-series
(velocity, acceleration, joint angles, phase timing) rather than absolute
pixel positions. NO horizontal position features are used, as camera angle
compensation has been removed from the pipeline.

PHASE VALIDATION: If all 3 phases for the respective lift are not found in a
given lift CSV file, that file must be discarded and not used for training.

DIRECTION NORMALIZATION: The athlete may face left or right in the frame.
All left/right joint comparisons use abs() or symmetric operations to be
direction-invariant. Vertical-only features (*_y values, velocities,
accelerations) are inherently direction-invariant.

BAR OSCILLATION: At the end of a clean or beginning of a jerk, the bar may
oscillate sinusoidally (2-4 Hz, amplitude < 5% of total displacement) due to
bar elasticity under heavy load. This is NORMAL and must NOT be counted as
recovery_bounce or unstable_recovery. The velocity threshold in recovery_bounce
counting (~20% of peak recovery velocity) filters out bar oscillation while
catching genuine squat bounces.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

PHASE_IDS_COMMON: set[int] = {0, 1, 2}


def validate_phases(df: pd.DataFrame) -> bool:
    """Check that bar_phase contains all 3 phases (0, 1, 2).

    Discard any CSV that doesn't have all 3 phases — incomplete
    data produces unreliable features for training.
    """
    if "bar_phase" not in df.columns:
        return False
    phases = set(df["bar_phase"].dropna().astype(int).unique())
    return PHASE_IDS_COMMON.issubset(phases)


def extract_trajectory(df: pd.DataFrame) -> np.ndarray:
    """Extract multi-channel kinematic trajectory (lift-type classifier input).

    Input: final_analysis.csv loaded as DataFrame (frame-indexed).
    Returns: np.ndarray of shape (N_frames, 3) with channels:
      - channel 0: barbell_y_smooth (vertical bar position, normalized)
      - channel 1: vel_y_smooth (vertical velocity, normalized)
      - channel 2: accel_y_smooth (vertical acceleration, normalized)

    This multi-channel approach captures the lift's kinematic signature
    (shape + speed profile) rather than relying on absolute pixel positions
    that vary with camera angle.
    """
    y_arr = df["barbell_y_smooth"].interpolate().to_numpy()
    y_range = float(np.nanmax(y_arr) - np.nanmin(y_arr))
    if y_range > 0:
        y = ((y_arr - np.nanmin(y_arr)) / y_range).astype(float)
    else:
        y = np.zeros_like(y_arr, dtype=float)

    vel = np.zeros_like(y, dtype=float)
    if "vel_y_smooth" in df.columns:
        vel_raw = df["vel_y_smooth"].interpolate().to_numpy().astype(float)
        vel_max = float(np.nanmax(np.abs(vel_raw)))
        if vel_max > 0:
            vel = vel_raw / vel_max

    accel = np.zeros_like(y, dtype=float)
    if "accel_y_smooth" in df.columns:
        accel_raw = df["accel_y_smooth"].interpolate().values.astype(float)
        accel_max = float(np.nanmax(np.abs(accel_raw)))
        if accel_max > 0:
            accel = accel_raw / accel_max

    mask = ~(np.isnan(y) | np.isnan(vel) | np.isnan(accel))
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float64)

    return np.column_stack([y[mask], vel[mask], accel[mask]]).astype(np.float64)


def extract_technique_features(df: pd.DataFrame, lift_type: str = "clean") -> dict[str, float]:
    """Extract all scalar features for Technique Analysis (Random Forest or rule-based).

    Input: final_analysis.csv loaded as DataFrame (frame-indexed).
    lift_type: "snatch", "clean", or "jerk" — determines which features to compute.
    Returns: ~26 scalar features as flat dict, all camera-angle-invariant.

    PHASE VALIDATION: Caller must verify that bar_phase contains all 3 phases
    (0, 1, 2) before calling this function for training data.
    """
    features: dict[str, float] = {}
    features.update(_extract_velocity_power_scalars(df))
    features.update(_extract_joint_angle_scalars(df))
    features.update(_extract_body_position_scalars(df))
    features.update(_extract_phase_timing_scalars(df))
    features.update(_extract_time_series_profile_features(df))
    if lift_type in ("clean", "snatch"):
        features.update(_extract_recovery_bounce_features(df))
    if lift_type == "jerk":
        features.update(_extract_jerk_specific_features(df))
    return features


def _extract_velocity_power_scalars(df: pd.DataFrame) -> dict[str, float]:
    """Extract velocity and power-related features (all camera-angle-invariant)."""
    features: dict[str, float] = {}

    if "vel_y_smooth" in df.columns:
        vel = df["vel_y_smooth"].dropna()
        if len(vel) > 0:
            vel_arr = np.asarray(vel, dtype=float)
            features["max_vel_y"] = float(vel_arr.max())
            n = len(vel)
            first_half = vel.iloc[: n // 2]
            if len(first_half) > 0:
                features["mean_vel_y_first_half"] = float(
                    np.asarray(first_half, dtype=float).mean()
                )
            else:
                features["mean_vel_y_first_half"] = 0.0

            q1 = np.asarray(vel.iloc[: n // 4], dtype=float)
            q4 = np.asarray(vel.iloc[3 * n // 4 :], dtype=float)
            features["vel_range_q1_q4"] = float(
                q4.mean() - q1.mean() if len(q1) > 0 and len(q4) > 0 else 0.0
            )
        else:
            features["max_vel_y"] = 0.0
            features["mean_vel_y_first_half"] = 0.0
            features["vel_range_q1_q4"] = 0.0
    else:
        features["max_vel_y"] = 0.0
        features["mean_vel_y_first_half"] = 0.0
        features["vel_range_q1_q4"] = 0.0

    if "accel_y_smooth" in df.columns:
        accel = df["accel_y_smooth"].dropna()
        if len(accel) > 0:
            accel_arr = np.asarray(accel, dtype=float)
            features["peak_accel_y"] = float(accel_arr.max())
            features["min_accel_y"] = float(accel_arr.min())
            features["accel_positive_frac"] = float(np.sum(accel_arr > 0) / len(accel_arr))
        else:
            features["peak_accel_y"] = 0.0
            features["min_accel_y"] = 0.0
            features["accel_positive_frac"] = 0.0
    else:
        features["peak_accel_y"] = 0.0
        features["min_accel_y"] = 0.0
        features["accel_positive_frac"] = 0.0

    return features


def _extract_joint_angle_scalars(df: pd.DataFrame) -> dict[str, float]:
    """Extract joint angle-related features (all camera-angle-invariant)."""
    features: dict[str, float] = {}
    n = len(df)

    if "left_elbow_angle" in df.columns and "right_elbow_angle" in df.columns:
        elbow_angles = df[["left_elbow_angle", "right_elbow_angle"]].dropna()
        if len(elbow_angles) > 0:
            early_frames = elbow_angles.iloc[: int(n * 0.6)]
            if len(early_frames) > 0:
                features["min_elbow_angle_early"] = float(early_frames.min().min())
            else:
                features["min_elbow_angle_early"] = 180.0

            var_left_result = df["left_elbow_angle"].dropna().var()
            var_right_result = df["right_elbow_angle"].dropna().var()
            var_left = _safe_float(var_left_result)
            var_right = _safe_float(var_right_result)
            features["elbow_angle_variance"] = (var_left + var_right) / 2

            if "bar_phase" in df.columns:
                pull_mask = df["bar_phase"] == 0
                pull_elbows = elbow_angles.loc[pull_mask]
                if len(pull_elbows) > 0:
                    features["elbow_range_pull"] = float(
                        pull_elbows.max().max() - pull_elbows.min().min()
                    )
                else:
                    features["elbow_range_pull"] = 0.0
            else:
                features["elbow_range_pull"] = float(
                    elbow_angles.max().max() - elbow_angles.min().min()
                )
        else:
            features["min_elbow_angle_early"] = 180.0
            features["elbow_angle_variance"] = 0.0
            features["elbow_range_pull"] = 0.0
    else:
        features["min_elbow_angle_early"] = 180.0
        features["elbow_angle_variance"] = 0.0
        features["elbow_range_pull"] = 0.0

    if "left_knee_angle" in df.columns and "right_knee_angle" in df.columns:
        knee_angles = df[["left_knee_angle", "right_knee_angle"]].dropna()
        if len(knee_angles) > 0:
            late_frames = knee_angles.iloc[int(n * 0.7) :]
            if len(late_frames) > 0:
                features["min_knee_angle_catch"] = float(late_frames.min().min())
            else:
                features["min_knee_angle_catch"] = 180.0

            if "bar_phase" in df.columns:
                pull_mask = df["bar_phase"] == 0
                pull_knees = knee_angles.loc[pull_mask]
                if len(pull_knees) > 0:
                    left_var = float(pull_knees.iloc[:, 0].var())
                    right_var = float(pull_knees.iloc[:, 1].var())
                    features["knee_angle_variance_pull"] = (left_var + right_var) / 2
                else:
                    features["knee_angle_variance_pull"] = 0.0
            else:
                left_var = float(knee_angles.iloc[:, 0].var())
                right_var = float(knee_angles.iloc[:, 1].var())
                features["knee_angle_variance_pull"] = (left_var + right_var) / 2
        else:
            features["min_knee_angle_catch"] = 180.0
            features["knee_angle_variance_pull"] = 0.0
    else:
        features["min_knee_angle_catch"] = 180.0
        features["knee_angle_variance_pull"] = 0.0

    if "left_knee_x" in df.columns and "right_knee_x" in df.columns:
        knee_width = (df["left_knee_x"] - df["right_knee_x"]).abs()
        early_frames = knee_width.iloc[: int(n * 0.3)]
        if len(early_frames) > 1:
            features["knee_width_change_early"] = float(
                early_frames.iloc[-1] - early_frames.iloc[0]
            )
        else:
            features["knee_width_change_early"] = 0.0
    else:
        features["knee_width_change_early"] = 0.0

    return features


def _extract_body_position_scalars(df: pd.DataFrame) -> dict[str, float]:
    """Extract body position-related features (all camera-angle-invariant).

    Direction normalization: All horizontal left/right differences use abs()
    so they are direction-invariant. Vertical features are inherently
    direction-invariant.
    """
    features: dict[str, float] = {}
    n = len(df)

    if "hip_y_avg" in df.columns and "barbell_y_smooth" in df.columns:
        hip_y = df["hip_y_avg"].dropna()
        bar_y = df["barbell_y_smooth"].dropna()

        if len(hip_y) > 0 and len(bar_y) > 0:
            early_frames = min(int(n * 0.3), len(hip_y), len(bar_y))
            if early_frames > 1:
                hip_y_arr = np.asarray(hip_y, dtype=float)
                bar_y_arr = np.asarray(bar_y, dtype=float)
                hip_change = float(hip_y_arr[early_frames - 1] - hip_y_arr[0])
                bar_change = float(bar_y_arr[early_frames - 1] - bar_y_arr[0])
                features["hip_rise_vs_bar_rise_early"] = (
                    hip_change / bar_change if abs(bar_change) > 1e-6 else 0.0
                )
            else:
                features["hip_rise_vs_bar_rise_early"] = 0.0

            hip_y_arr = np.asarray(hip_y, dtype=float)
            fh = float(df["frame_height"].iloc[0]) if "frame_height" in df.columns else 1.0
            features["hip_height_at_catch_norm"] = float(hip_y_arr.min()) / fh if fh > 0 else 0.0
        else:
            features["hip_rise_vs_bar_rise_early"] = 0.0
            features["hip_height_at_catch_norm"] = 0.0
    else:
        features["hip_rise_vs_bar_rise_early"] = 0.0
        features["hip_height_at_catch_norm"] = 0.0

    if (
        "left_shoulder_y" in df.columns
        and "right_shoulder_y" in df.columns
        and "left_hip_y" in df.columns
        and "right_hip_y" in df.columns
    ):
        shoulder_y = ((df["left_shoulder_y"] + df["right_shoulder_y"]) / 2).dropna()
        hip_y = ((df["left_hip_y"] + df["right_hip_y"]) / 2).dropna()
        if len(shoulder_y) > 0 and len(hip_y) > 0:
            min_len = min(len(shoulder_y), len(hip_y))
            separation = np.abs(
                np.asarray(shoulder_y[:min_len], dtype=float)
                - np.asarray(hip_y[:min_len], dtype=float)
            )
            if "bar_phase" in df.columns:
                pull_mask = df["bar_phase"] == 0
                pull_sep = separation[pull_mask[:min_len]]
                features["shoulder_hip_separation_pull"] = float(
                    pull_sep.max() if len(pull_sep) > 0 else separation.max()
                )
            else:
                features["shoulder_hip_separation_pull"] = float(separation.max())
        else:
            features["shoulder_hip_separation_pull"] = 0.0
    else:
        features["shoulder_hip_separation_pull"] = 0.0

    if "left_ankle_y" in df.columns and "right_ankle_y" in df.columns:
        ankle_y = ((df["left_ankle_y"] + df["right_ankle_y"]) / 2).dropna()
        if len(ankle_y) > 0:
            late_phase_0_start = int(n * 0.6)
            if late_phase_0_start < len(ankle_y):
                ankle_y_arr = np.asarray(ankle_y, dtype=float)
                late_ankle = ankle_y_arr[late_phase_0_start:]
                if len(late_ankle) > 1:
                    features["ankle_rise_late_pull"] = float(late_ankle[-1] - late_ankle[0])
                else:
                    features["ankle_rise_late_pull"] = 0.0
            else:
                features["ankle_rise_late_pull"] = 0.0
        else:
            features["ankle_rise_late_pull"] = 0.0
    else:
        features["ankle_rise_late_pull"] = 0.0

    return features


def _extract_phase_timing_scalars(df: pd.DataFrame) -> dict[str, float]:
    """Extract phase timing-related features."""
    features: dict[str, float] = {}

    if "bar_phase" in df.columns and "time_s" in df.columns:
        phases = df["bar_phase"].dropna()
        times = df["time_s"].dropna()

        if len(phases) > 0 and len(times) > 0:
            times_arr = np.asarray(times, dtype=float)
            phases_arr = np.asarray(phases, dtype=float)
            total_time = float(times_arr[-1] - times_arr[0])

            if total_time > 0:
                phase_0_mask = phases_arr == 0
                phase_1_mask = phases_arr == 1
                phase_2_mask = phases_arr == 2

                phase_0_times = times_arr[phase_0_mask]
                phase_0_time = (
                    float(np.sum(np.diff(phase_0_times))) if len(phase_0_times) > 0 else 0.0
                )
                phase_1_times = times_arr[phase_1_mask]
                phase_1_time = (
                    float(np.sum(np.diff(phase_1_times))) if len(phase_1_times) > 0 else 0.0
                )
                phase_2_times = times_arr[phase_2_mask]
                phase_2_time = (
                    float(np.sum(np.diff(phase_2_times))) if len(phase_2_times) > 0 else 0.0
                )

                features["pull_duration_frac"] = phase_0_time / total_time
                features["turnover_duration_frac"] = phase_1_time / total_time
                features["pull_to_recovery_ratio"] = (
                    phase_0_time / phase_2_time if phase_2_time > 0 else 0.0
                )
            else:
                features["pull_duration_frac"] = 0.0
                features["turnover_duration_frac"] = 0.0
                features["pull_to_recovery_ratio"] = 0.0
        else:
            features["pull_duration_frac"] = 0.0
            features["turnover_duration_frac"] = 0.0
            features["pull_to_recovery_ratio"] = 0.0
    else:
        features["pull_duration_frac"] = 0.0
        features["turnover_duration_frac"] = 0.0
        features["pull_to_recovery_ratio"] = 0.0

    if "vel_y_smooth" in df.columns:
        vel = df["vel_y_smooth"].dropna()
        if len(vel) > 0:
            vel_arr = np.asarray(vel, dtype=float)
            peak_idx = int(np.argmax(vel_arr))
            features["peak_vel_phase_frac"] = float(peak_idx / len(vel_arr))
        else:
            features["peak_vel_phase_frac"] = 0.0
    else:
        features["peak_vel_phase_frac"] = 0.0

    return features


def _extract_time_series_profile_features(df: pd.DataFrame) -> dict[str, float]:
    """Extract time-series profile features (all camera-angle-invariant).

    These features capture the shape of the velocity and acceleration
    distributions rather than absolute spatial measurements.
    """
    features: dict[str, float] = {}

    if "vel_y_smooth" in df.columns:
        vel = df["vel_y_smooth"].dropna()
        if len(vel) > 3:
            vel_arr = np.asarray(vel, dtype=float)
            _skew = cast(float, pd.Series(vel_arr).skew())
            features["vel_profile_skewness"] = float(_skew if np.std(vel_arr) > 1e-6 else 0.0)
        else:
            features["vel_profile_skewness"] = 0.0
    else:
        features["vel_profile_skewness"] = 0.0

    if "accel_y_smooth" in df.columns:
        accel = df["accel_y_smooth"].dropna()
        if len(accel) > 5:
            accel_arr = np.asarray(accel, dtype=float)
            if "bar_phase" in df.columns:
                pull_mask = df["bar_phase"] == 0
                pull_accel = accel_arr[pull_mask[: len(accel_arr)]]
            else:
                pull_accel = accel_arr[: len(accel_arr) // 2]

            if len(pull_accel) > 5:
                from scipy.signal import find_peaks

                peaks, _ = find_peaks(pull_accel, height=np.std(pull_accel) * 0.5)
                features["accel_peaks_count"] = float(len(peaks))
            else:
                features["accel_peaks_count"] = 0.0
        else:
            features["accel_peaks_count"] = 0.0
    else:
        features["accel_peaks_count"] = 0.0

    return features


def _extract_recovery_bounce_features(df: pd.DataFrame) -> dict[str, float]:
    """Extract recovery bounce features for clean/snatch.

    Counts the number of significant downward velocity reversals during the
    Recovery phase (Phase 2). Multiple bounces during recovery from the squat
    indicate the athlete may be near their squat strength limit.

    IMPORTANT: Bar oscillation (small sinusoidal vibrations at 2-4 Hz that
    occur at the end of a clean / beginning of a jerk due to bar elasticity)
    must NOT be counted as a bounce. The velocity threshold is set to ~20%
    of the peak recovery velocity to filter out oscillation while catching
    real bounces.
    """
    features: dict[str, float] = {}
    features["recovery_bounce_count"] = 0.0

    if "vel_y_smooth" not in df.columns or "bar_phase" not in df.columns:
        return features

    vel = df["vel_y_smooth"].dropna()
    phases = df["bar_phase"].dropna()

    if len(vel) == 0 or len(phases) == 0:
        return features

    min_len = min(len(vel), len(phases))
    vel_arr = np.asarray(vel.iloc[:min_len], dtype=float)
    phase_arr = np.asarray(phases.iloc[:min_len], dtype=int)

    recovery_mask = phase_arr == 2
    recovery_vel = vel_arr[recovery_mask]

    if len(recovery_vel) < 5:
        return features

    peak_recovery_vel = float(np.max(np.abs(recovery_vel)))
    min_bounce_velocity = peak_recovery_vel * 0.20

    min_bounce_velocity = max(min_bounce_velocity, 15.0)

    direction_changes = 0
    for i in range(1, len(recovery_vel)):
        prev_up = recovery_vel[i - 1] < -min_bounce_velocity
        curr_down = recovery_vel[i] > min_bounce_velocity
        if prev_up and curr_down:
            direction_changes += 1

    features["recovery_bounce_count"] = float(direction_changes)

    return features


def _extract_jerk_specific_features(df: pd.DataFrame) -> dict[str, float]:
    """Extract jerk-specific features from Dip and Drive phases.

    Jerk phases: 0=Dip, 1=Drive, 2=Recovery

    Only computed when lift_type == "jerk". Camera-angle-invariant.
    """
    features: dict[str, float] = {}

    if "bar_phase" not in df.columns:
        features["dip_depth_norm"] = 0.0
        features["drive_peak_vel"] = 0.0
        features["dip_pause_detected"] = 0.0
        return features

    phases = df["bar_phase"].dropna().astype(int)

    if "barbell_y_smooth" in df.columns and "frame_height" in df.columns:
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
            frame_h = float(df["frame_height"].iloc[0]) if "frame_height" in df.columns else 1.0
            features["dip_depth_norm"] = float(dip_depth / frame_h) if frame_h > 0 else 0.0
        else:
            features["dip_depth_norm"] = 0.0
    else:
        features["dip_depth_norm"] = 0.0

    if "vel_y_smooth" in df.columns:
        vel = df["vel_y_smooth"].dropna()
        min_len = min(len(vel), len(phases))
        vel_arr = np.asarray(vel.iloc[:min_len], dtype=float)
        phase_arr = np.asarray(phases.iloc[:min_len], dtype=int)

        drive_mask = phase_arr == 1
        drive_vel = vel_arr[drive_mask]

        if len(drive_vel) > 0:
            features["drive_peak_vel"] = float(np.min(drive_vel))
        else:
            features["drive_peak_vel"] = 0.0
    else:
        features["drive_peak_vel"] = 0.0

    has_pause = False
    if "vel_y_smooth" in df.columns:
        vel = df["vel_y_smooth"].dropna()
        min_len = min(len(vel), len(phases))
        vel_arr = np.asarray(vel.iloc[:min_len], dtype=float)
        phase_arr = np.asarray(phases.iloc[:min_len], dtype=int)

        dip_mask_short = phase_arr == 0
        dip_vel = vel_arr[dip_mask_short]

        if len(dip_vel) > 5:
            sign_changes = np.diff(np.sign(dip_vel))
            for i, sc in enumerate(sign_changes):
                if sc > 0 and i > 2:
                    has_pause = True
                    break

    features["dip_pause_detected"] = 1.0 if has_pause else 0.0

    return features


def _safe_float(value: object) -> float:
    """Safely convert a value to float, replacing NaN/inf with 0.0."""
    if isinstance(value, (int, float, np.number)):
        result = float(value)
    elif hasattr(value, "iloc"):
        result = float(value.iloc[0]) if len(value) > 0 else 0.0  # type: ignore[index]
    else:
        result = 0.0
    return 0.0 if not np.isfinite(result) else result
