"""
Landmark processing utilities for Step 2: Data Analysis.

Handles unpacking of landmark data, calculating joint angles, and
deriving body position metrics.
"""

from typing import List

import numpy as np
import pandas as pd
from config import LANDMARKS_TO_TRACK


def unpack_landmarks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unpack the 'landmarks' dict column into per-joint x/y/z/vis columns.

    Args:
        df: DataFrame with a 'landmarks' column containing dicts

    Returns:
        DataFrame with additional columns for each tracked landmark
    """
    for name in LANDMARKS_TO_TRACK:
        df[name] = df["landmarks"].apply(
            lambda x, _n=name: x.get(_n) if isinstance(x, dict) else None
        )
        df[f"{name}_x"] = df[name].apply(
            lambda x: x[0] if (x is not None and len(x) >= 4 and x[3] > 0.1) else np.nan
        )
        df[f"{name}_y"] = df[name].apply(
            lambda x: x[1] if (x is not None and len(x) >= 4 and x[3] > 0.1) else np.nan
        )
        df[f"{name}_z"] = df[name].apply(
            lambda x: x[2] if (x is not None and len(x) >= 4 and x[3] > 0.1) else np.nan
        )
        df[f"{name}_vis"] = df[name].apply(
            lambda x: x[3] if (x is not None and len(x) >= 4) else np.nan
        )

    return df


def get_pixel_pos(
    row: pd.Series, name: str, frame_width: int, frame_height: int
) -> np.ndarray:
    """
    Get pixel-space position for a landmark.

    Args:
        row: DataFrame row
        name: Landmark name (without _x/_y suffix)
        frame_width: Video frame width in pixels
        frame_height: Video frame height in pixels

    Returns:
        np.ndarray of [x, y] in pixels, or [nan, nan] if unavailable
    """
    x_norm = row.get(f"{name}_x")
    y_norm = row.get(f"{name}_y")

    if x_norm is None or y_norm is None:
        return np.array([np.nan, np.nan])

    try:
        x_float = float(x_norm)
        y_float = float(y_norm)
        if np.isnan(x_float) or np.isnan(y_float):
            return np.array([np.nan, np.nan])
        return np.array([x_float * frame_width, y_float * frame_height])
    except (TypeError, ValueError):
        return np.array([np.nan, np.nan])


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate the angle at p2 formed by points p1-p2-p3.

    Args:
        p1, p2, p3: 2D points as numpy arrays

    Returns:
        Angle in degrees, or NaN if any point is invalid
    """
    if np.any(np.isnan(p1)) or np.any(np.isnan(p2)) or np.any(np.isnan(p3)):
        return np.nan

    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def calculate_joint_angles(
    df: pd.DataFrame, frame_width: int, frame_height: int
) -> pd.DataFrame:
    """
    Calculate all joint angles from landmark positions.

    Adds columns for knee and elbow angles (left and right).

    Args:
        df: DataFrame with landmark x/y columns
        frame_width: Video frame width
        frame_height: Video frame height

    Returns:
        DataFrame with additional angle columns
    """

    def get_pos(row, name):
        return get_pixel_pos(row, name, frame_width, frame_height)

    df["left_knee_angle"] = df.apply(
        lambda row: calculate_angle(
            get_pos(row, "left_hip"),
            get_pos(row, "left_knee"),
            get_pos(row, "left_ankle"),
        ),
        axis=1,
    )

    df["right_knee_angle"] = df.apply(
        lambda row: calculate_angle(
            get_pos(row, "right_hip"),
            get_pos(row, "right_knee"),
            get_pos(row, "right_ankle"),
        ),
        axis=1,
    )

    df["left_elbow_angle"] = df.apply(
        lambda row: calculate_angle(
            get_pos(row, "left_shoulder"),
            get_pos(row, "left_elbow"),
            get_pos(row, "left_wrist"),
        ),
        axis=1,
    )

    df["right_elbow_angle"] = df.apply(
        lambda row: calculate_angle(
            get_pos(row, "right_shoulder"),
            get_pos(row, "right_elbow"),
            get_pos(row, "right_wrist"),
        ),
        axis=1,
    )

    return df


def calculate_lifter_angle(landmarks: dict) -> float:
    """
    Calculate the lifter's body angle from landmarks.

    This measures the forward lean angle based on shoulder-hip alignment.

    Args:
        landmarks: Dict of landmark name -> [x, y, z, visibility]

    Returns:
        Angle in degrees, or NaN if landmarks unavailable
    """
    if not isinstance(landmarks, dict):
        return np.nan

    required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    for name in required:
        if name not in landmarks:
            return np.nan
        lm = landmarks[name]
        if not isinstance(lm, (list, tuple)) or len(lm) < 3:
            return np.nan

    try:
        ls = landmarks["left_shoulder"]
        rs = landmarks["right_shoulder"]
        lh = landmarks["left_hip"]
        rh = landmarks["right_hip"]

        shoulder_mid_x = (ls[0] + rs[0]) / 2
        shoulder_mid_y = (ls[1] + rs[1]) / 2
        hip_mid_x = (lh[0] + rh[0]) / 2
        hip_mid_y = (lh[1] + rh[1]) / 2

        dx = shoulder_mid_x - hip_mid_x
        dy = shoulder_mid_y - hip_mid_y

        angle = np.degrees(np.arctan2(dx, -dy))
        return float(angle)
    except Exception:
        return np.nan


def calculate_hip_y_average(df: pd.DataFrame, frame_height: int) -> pd.DataFrame:
    """
    Calculate average hip Y position in pixel space.

    Used for phase detection (hips dropping = pull-under).

    Args:
        df: DataFrame with left_hip_y and right_hip_y columns
        frame_height: Video frame height

    Returns:
        DataFrame with 'hip_y_avg' column added
    """
    df["hip_y_avg"] = df[["left_hip_y", "right_hip_y"]].mean(axis=1) * frame_height
    return df


def calculate_knee_y_average(df: pd.DataFrame, frame_height: int) -> pd.DataFrame:
    """
    Calculate average knee Y position in pixel space.

    Used for truncation (bar passing knee).

    Args:
        df: DataFrame with left_knee_y and right_knee_y columns
        frame_height: Video frame height

    Returns:
        DataFrame with 'knee_y_avg' column added
    """
    df["knee_y_avg"] = df[["left_knee_y", "right_knee_y"]].mean(axis=1) * frame_height
    return df


def drop_intermediate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop intermediate columns that are not needed in the final CSV.

    Keeps smoothed/stabilized data, drops raw tuples and intermediate values.

    Args:
        df: DataFrame with all columns

    Returns:
        DataFrame with intermediate columns removed
    """
    cols_to_drop: List[str] = []

    cols_to_drop.append("landmarks")
    cols_to_drop.extend(["shake_dx", "shake_dy"])

    for name in LANDMARKS_TO_TRACK:
        cols_to_drop.append(name)

    if "barbell_center" in df.columns:
        cols_to_drop.append("barbell_center")
    if "barbell_box" in df.columns:
        cols_to_drop.append("barbell_box")
    cols_to_drop.extend(["barbell_x_raw", "barbell_y_raw"])

    world_cols = [c for c in df.columns if "world" in c and "corrected" not in c]
    cols_to_drop.extend(world_cols)

    if "vel_y_px_s" in df.columns:
        cols_to_drop.append("vel_y_px_s")

    cols_to_drop = list(dict.fromkeys(cols_to_drop))
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    return df.drop(columns=cols_to_drop)
