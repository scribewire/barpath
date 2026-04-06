"""
Feature extraction for ML-based technique analysis.

Two extraction paths, both reading from final_analysis.csv:
  - extract_trajectory(): Normalized (x, y) bar path for DTW comparison
  - extract_smart_features(): ~18 scalar features for fault discrimination

Phase IDs (from bar_phase column):
  - 0: Pull (early + late portions)
  - 1: Pull-under
  - 2: Recovery
"""

from typing import Dict

import numpy as np
import pandas as pd


def extract_trajectory(df: pd.DataFrame) -> np.ndarray:
    """
    Extract normalized barbell trajectory for Fast Analysis (DTW).

    Input: final_analysis.csv loaded as DataFrame (frame-indexed).
    Returns: np.ndarray of shape (N_frames, 2), normalized by frame_height.
    """
    fh = float(df["frame_height"].iloc[0])
    x = np.asarray(df["barbell_x_smooth"].interpolate().values, dtype=float) / fh
    y = np.asarray(df["barbell_y_smooth"].interpolate().values, dtype=float) / fh
    return np.column_stack([x, y])


def extract_smart_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract scalar features for Smart Analysis (Random Forest).

    Input: final_analysis.csv loaded as DataFrame (frame-indexed).
    Returns: ~18 scalar features as flat dict.
    """
    fh = float(df["frame_height"].iloc[0])
    features = {}
    features.update(_extract_velocity_power_scalars(df, fh))
    features.update(_extract_joint_angle_scalars(df, fh))
    features.update(_extract_barbell_path_scalars(df, fh))
    features.update(_extract_body_position_scalars(df, fh))
    features.update(_extract_phase_timing_scalars(df))
    features.update(_extract_additional_scalars(df, fh))
    return features


def _extract_velocity_power_scalars(df: pd.DataFrame, fh: float) -> Dict[str, float]:
    """Extract velocity and power-related features."""
    features = {}

    if "vel_y_smooth" in df.columns:
        vel = df["vel_y_smooth"].dropna()
        if len(vel) > 0:
            vel_arr = np.asarray(vel, dtype=float)
            features["max_vel_y"] = float(vel_arr.max())
            n = len(vel)
            first_half = vel.iloc[: n // 2]
            if len(first_half) > 0:
                first_half_arr = np.asarray(first_half, dtype=float)
                features["mean_vel_y_first_half"] = float(first_half_arr.mean())
            else:
                features["mean_vel_y_first_half"] = 0.0
        else:
            features["max_vel_y"] = 0.0
            features["mean_vel_y_first_half"] = 0.0
    else:
        features["max_vel_y"] = 0.0
        features["mean_vel_y_first_half"] = 0.0

    if "accel_y_smooth" in df.columns:
        accel = df["accel_y_smooth"].dropna()
        if len(accel) > 0:
            accel_arr = np.asarray(accel, dtype=float)
            features["peak_accel_y"] = float(accel_arr.max())
            features["min_accel_y"] = float(accel_arr.min())
        else:
            features["peak_accel_y"] = 0.0
            features["min_accel_y"] = 0.0
    else:
        features["peak_accel_y"] = 0.0
        features["min_accel_y"] = 0.0

    return features


def _extract_joint_angle_scalars(df: pd.DataFrame, fh: float) -> Dict[str, float]:
    """Extract joint angle-related features."""
    features = {}
    n = len(df)

    if "left_elbow_angle" in df.columns and "right_elbow_angle" in df.columns:
        elbow_angles = df[["left_elbow_angle", "right_elbow_angle"]].dropna()
        if len(elbow_angles) > 0:
            early_frames = elbow_angles.iloc[: int(n * 0.6)]
            if len(early_frames) > 0:
                min_val = float(early_frames.min().min())
                features["min_elbow_angle_early"] = min_val
            else:
                features["min_elbow_angle_early"] = 180.0

            var_left_result = df["left_elbow_angle"].dropna().var()
            var_right_result = df["right_elbow_angle"].dropna().var()
            # Convert to float, handling both scalar and Series returns
            if isinstance(var_left_result, (int, float, np.number)):
                var_left = float(var_left_result)
            else:
                var_left = (
                    float(var_left_result.iloc[0]) if len(var_left_result) > 0 else 0.0
                )
            if isinstance(var_right_result, (int, float, np.number)):
                var_right = float(var_right_result)
            else:
                var_right = (
                    float(var_right_result.iloc[0])
                    if len(var_right_result) > 0
                    else 0.0
                )
            # Replace NaN/inf with 0.0
            var_left = 0.0 if not np.isfinite(var_left) else var_left
            var_right = 0.0 if not np.isfinite(var_right) else var_right
            features["elbow_angle_variance"] = (var_left + var_right) / 2
        else:
            features["min_elbow_angle_early"] = 180.0
            features["elbow_angle_variance"] = 0.0
    else:
        features["min_elbow_angle_early"] = 180.0
        features["elbow_angle_variance"] = 0.0

    if "left_knee_angle" in df.columns and "right_knee_angle" in df.columns:
        knee_angles = df[["left_knee_angle", "right_knee_angle"]].dropna()
        if len(knee_angles) > 0:
            late_frames = knee_angles.iloc[int(n * 0.7) :]
            if len(late_frames) > 0:
                features["min_knee_angle_catch"] = float(late_frames.min().min())
            else:
                features["min_knee_angle_catch"] = 180.0
        else:
            features["min_knee_angle_catch"] = 180.0
    else:
        features["min_knee_angle_catch"] = 180.0

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


def _extract_barbell_path_scalars(df: pd.DataFrame, fh: float) -> Dict[str, float]:
    """Extract barbell path-related features."""
    features = {}

    if "barbell_x_smooth" in df.columns:
        x = df["barbell_x_smooth"].dropna()
        if len(x) > 0:
            x_arr = np.asarray(x, dtype=float)
            x_start = float(x_arr[0])
            max_dev = float(np.abs(x_arr - x_start).max())
            features["max_horiz_deviation"] = max_dev / fh if fh > 0 else 0.0

            n = len(x_arr)
            first_third = x_arr[: n // 3]
            if len(first_third) > 0:
                features["horiz_deviation_first_third"] = (
                    float(np.abs(first_third - x_start).max()) / fh if fh > 0 else 0.0
                )
            else:
                features["horiz_deviation_first_third"] = 0.0
        else:
            features["max_horiz_deviation"] = 0.0
            features["horiz_deviation_first_third"] = 0.0
    else:
        features["max_horiz_deviation"] = 0.0
        features["horiz_deviation_first_third"] = 0.0

    return features


def _extract_body_position_scalars(df: pd.DataFrame, fh: float) -> Dict[str, float]:
    """Extract body position-related features."""
    features = {}
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
                if abs(bar_change) > 1e-6:
                    features["hip_rise_vs_bar_rise_early"] = hip_change / bar_change
                else:
                    features["hip_rise_vs_bar_rise_early"] = 0.0
            else:
                features["hip_rise_vs_bar_rise_early"] = 0.0

            hip_min_arr = np.asarray(hip_y, dtype=float)
            hip_min = float(hip_min_arr.min())
            features["hip_height_at_catch_norm"] = hip_min / fh if fh > 0 else 0.0
        else:
            features["hip_rise_vs_bar_rise_early"] = 0.0
            features["hip_height_at_catch_norm"] = 0.0
    else:
        features["hip_rise_vs_bar_rise_early"] = 0.0
        features["hip_height_at_catch_norm"] = 0.0

    if "hip_x_avg" in df.columns:
        hip_x = df["hip_x_avg"].dropna()
        if len(hip_x) > 0:
            hip_x_arr = np.asarray(hip_x, dtype=float)
            start_idx = int(n * 0.8)
            if start_idx < len(hip_x_arr) - 1:
                features["horizontal_displacement_recovery"] = float(
                    abs(hip_x_arr[-1] - hip_x_arr[start_idx])
                )
            else:
                features["horizontal_displacement_recovery"] = 0.0
        else:
            features["horizontal_displacement_recovery"] = 0.0
    else:
        features["horizontal_displacement_recovery"] = 0.0

    return features


def _extract_phase_timing_scalars(df: pd.DataFrame) -> Dict[str, float]:
    """Extract phase timing-related features."""
    features = {}

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
                    float(np.sum(np.diff(phase_0_times)))
                    if len(phase_0_times) > 0
                    else 0.0
                )
                phase_1_times = times_arr[phase_1_mask]
                phase_1_time = (
                    float(np.sum(np.diff(phase_1_times)))
                    if len(phase_1_times) > 0
                    else 0.0
                )
                phase_2_times = times_arr[phase_2_mask]
                phase_2_time = (
                    float(np.sum(np.diff(phase_2_times)))
                    if len(phase_2_times) > 0
                    else 0.0
                )

                features["pull_duration_frac"] = phase_0_time / total_time
                features["turnover_duration_frac"] = phase_1_time / total_time

                if phase_2_time > 0:
                    features["pull_to_recovery_ratio"] = phase_0_time / phase_2_time
                else:
                    features["pull_to_recovery_ratio"] = 0.0
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

    return features


def _extract_additional_scalars(df: pd.DataFrame, fh: float) -> Dict[str, float]:
    """Extract additional discriminative features."""
    features = {}
    n = len(df)

    if "left_ankle_y" in df.columns and "right_ankle_y" in df.columns:
        ankle_y = (df["left_ankle_y"] + df["right_ankle_y"]) / 2
        ankle_y = ankle_y.dropna()

        if len(ankle_y) > 0 and "bar_phase" in df.columns:
            late_phase_0_start = int(n * 0.6)

            if late_phase_0_start < len(ankle_y):
                ankle_y_arr = np.asarray(ankle_y, dtype=float)
                late_ankle = ankle_y_arr[late_phase_0_start:]
                if len(late_ankle) > 1:
                    features["ankle_rise_late_pull"] = float(
                        late_ankle[-1] - late_ankle[0]
                    )
                else:
                    features["ankle_rise_late_pull"] = 0.0
            else:
                features["ankle_rise_late_pull"] = 0.0
        else:
            features["ankle_rise_late_pull"] = 0.0
    else:
        features["ankle_rise_late_pull"] = 0.0

    if (
        "left_shoulder_x" in df.columns
        and "right_shoulder_x" in df.columns
        and "barbell_x_smooth" in df.columns
    ):
        shoulder_x = (df["left_shoulder_x"] + df["right_shoulder_x"]) / 2
        bar_x = df["barbell_x_smooth"]

        if len(shoulder_x) > 0 and len(bar_x) > 0:
            shoulder_x_arr = np.asarray(shoulder_x, dtype=float)
            bar_x_arr = np.asarray(bar_x, dtype=float)
            first_shoulder = float(shoulder_x_arr[0])
            first_bar = float(bar_x_arr[0])
            features["shoulder_over_bar_sign"] = float(
                np.sign(first_shoulder - first_bar)
            )
        else:
            features["shoulder_over_bar_sign"] = 0.0
    else:
        features["shoulder_over_bar_sign"] = 0.0

    return features
