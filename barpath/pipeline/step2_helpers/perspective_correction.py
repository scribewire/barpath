"""
Perspective correction helpers for Step 2: Data Analysis.

This module contains functions for calculating perspective-corrected lateral bar
displacement using MediaPipe world landmarks and camera angle estimation.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def unpack_world_landmarks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract world landmark coordinates from the world_landmarks column.

    Unpacks the nested dictionary of world landmarks into separate columns
    for easier processing (x, y, z coordinates for each body part).

    Args:
        df (pd.DataFrame): DataFrame with 'world_landmarks' column

    Returns:
        pd.DataFrame: DataFrame with new columns for each landmark coordinate
    """
    # Unpack world landmarks
    for name in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]:
        df[f"{name}_world"] = df["world_landmarks"].apply(
            lambda x: x.get(name) if isinstance(x, dict) else None
        )
        df[f"{name}_world_x"] = df[f"{name}_world"].apply(
            lambda x: x[0] if (x is not None and len(x) >= 4 and x[3] > 0.1) else np.nan
        )
        df[f"{name}_world_y"] = df[f"{name}_world"].apply(
            lambda x: x[1] if (x is not None and len(x) >= 4 and x[3] > 0.1) else np.nan
        )
        df[f"{name}_world_z"] = df[f"{name}_world"].apply(
            lambda x: x[2] if (x is not None and len(x) >= 4 and x[3] > 0.1) else np.nan
        )

    return df


def calculate_reference_camera_angle(
    df: pd.DataFrame, first_idx
) -> Tuple[Optional[float], float]:
    """
    Calculate the reference camera yaw angle from the first frame.

    Uses shoulder positions to determine the camera angle relative to the lifter.
    The yaw angle represents how much the camera is rotated away from a
    perpendicular view (90°) of the lifter.

    Args:
        df (pd.DataFrame): DataFrame with unpacked world landmark coordinates
        first_idx: Index of the first frame to use as reference

    Returns:
        Tuple[Optional[float], float]: (camera_yaw_deg, correction_factor)
            - camera_yaw_deg: Angle in degrees (None if calculation fails)
            - correction_factor: Scaling factor (1.0 / cos(yaw_rad))
    """
    reference_camera_yaw_deg = None
    lateral_correction_factor = 1.0

    try:
        # Get shoulder positions from first frame
        l_sh_x = df.loc[first_idx, "left_shoulder_world_x"]
        l_sh_y = df.loc[first_idx, "left_shoulder_world_y"]
        l_sh_z = df.loc[first_idx, "left_shoulder_world_z"]
        r_sh_x = df.loc[first_idx, "right_shoulder_world_x"]
        r_sh_y = df.loc[first_idx, "right_shoulder_world_y"]
        r_sh_z = df.loc[first_idx, "right_shoulder_world_z"]

        if not any(pd.isna([l_sh_x, l_sh_y, l_sh_z, r_sh_x, r_sh_y, r_sh_z])):
            left_shoulder_world = np.array([l_sh_x, l_sh_y, l_sh_z])
            right_shoulder_world = np.array([r_sh_x, r_sh_y, r_sh_z])

            # Calculate lateral axis (left->right shoulder vector)
            lateral_vector = right_shoulder_world - left_shoulder_world
            shoulder_width_m = np.linalg.norm(lateral_vector)

            if shoulder_width_m >= 0.1:  # Realistic shoulder width
                lateral_axis = lateral_vector / shoulder_width_m

                # Calculate camera yaw from reference frame
                # Project lateral axis onto horizontal plane (XZ in world coords)
                lateral_xz = np.array([lateral_axis[0], lateral_axis[2]])
                lateral_xz_norm = np.linalg.norm(lateral_xz)

                if lateral_xz_norm > 0.01:
                    # Angle from camera X-axis (shoulder line relative to camera)
                    # When lifter faces camera: shoulders perpendicular to view → 90°
                    # When lifter is perpendicular to camera: shoulders parallel to view → 0°
                    camera_yaw_rad = np.arctan2(
                        abs(lateral_axis[0]), abs(lateral_axis[2])
                    )
                    reference_camera_yaw_deg = np.degrees(camera_yaw_rad)

                    # Calculate correction factor: 1 / cos(yaw)
                    # This expands the compressed horizontal displacement
                    lateral_correction_factor = 1.0 / np.cos(camera_yaw_rad)

    except Exception as e:
        print(
            f"  Warning: Could not calculate reference camera angle from first frame: {e}"
        )

    return reference_camera_yaw_deg, lateral_correction_factor


def apply_perspective_correction(
    df: pd.DataFrame,
    reference_camera_yaw_deg: Optional[float],
    lateral_correction_factor: float,
    first_idx,
) -> pd.DataFrame:
    """
    Apply the perspective correction factor to all frames.

    Uses the reference camera angle from the first frame to scale the observed
    horizontal displacement, accounting for how the camera angle compresses
    the apparent lateral movement.

    Args:
        df (pd.DataFrame): DataFrame with barbell_x_smooth and other data
        reference_camera_yaw_deg (Optional[float]): Reference camera angle in degrees
        lateral_correction_factor (float): Scaling factor (1/cos(yaw_rad))
        first_idx: Index of the first frame

    Returns:
        pd.DataFrame: DataFrame with corrected horizontal positions
    """
    # Initialize correction columns
    df["barbell_x_corrected_px"] = np.nan
    df["camera_yaw_deg"] = np.nan
    df["lateral_correction_factor"] = np.nan

    # Get the baseline horizontal position (first frame)
    if "barbell_x_smooth" not in df.columns:
        print("  Warning: No barbell_x_smooth data. Skipping perspective correction.")
        return df

    first_smooth_x = (
        df["barbell_x_smooth"].dropna().iloc[0]
        if len(df["barbell_x_smooth"].dropna()) > 0
        else None
    )

    if first_smooth_x is None:
        print("  Warning: No barbell_x_smooth data. Skipping perspective correction.")
        return df

    # Apply correction to each frame
    for idx in df.index:
        if pd.notna(df.loc[idx, "barbell_x_smooth"]):
            # Store the reference angle and correction factor for all frames
            df.loc[idx, "camera_yaw_deg"] = reference_camera_yaw_deg
            df.loc[idx, "lateral_correction_factor"] = lateral_correction_factor

            # Calculate the observed horizontal displacement from baseline
            observed_displacement = df.loc[idx, "barbell_x_smooth"] - first_smooth_x

            # Apply the correction factor to scale up the displacement
            corrected_displacement = observed_displacement * lateral_correction_factor

            # Calculate the corrected X position
            df.loc[idx, "barbell_x_corrected_px"] = (
                first_smooth_x + corrected_displacement
            )

    return df


def calculate_perspective_correction(
    df: pd.DataFrame, frame_width: int, frame_height: int
) -> pd.DataFrame:
    """
    Calculate perspective-corrected lateral bar displacement using world landmarks.

    This function uses a single reference camera angle from the first frame (start of lift)
    to calculate a correction factor that scales the observed horizontal displacement.

    The correction process:
    1. Extracts world landmarks (3D coordinates in meters)
    2. Defines the lifter's lateral axis from shoulders at the FIRST frame
    3. Calculates camera yaw angle (shoulder line relative to camera view)
       - 0° = lifter perpendicular to camera (side view, no correction needed)
       - 90° = lifter facing camera (shoulders perpendicular to view, max correction)
    4. Applies a scaling factor to observed horizontal displacement based on angle
    5. Correction factor = 1 / cos(yaw_angle) to expand compressed lateral movement

    Args:
        df (pd.DataFrame): DataFrame with landmarks and barbell data
        frame_width (int): Video frame width in pixels
        frame_height (int): Video frame height in pixels

    Returns:
        pd.DataFrame: DataFrame with additional columns:
            - barbell_x_corrected_px: Corrected horizontal position in pixels
            - camera_yaw_deg: Estimated camera yaw angle (from reference frame)
            - lateral_correction_factor: Scaling factor applied to horizontal displacement
    """
    # Check if world landmarks exist
    if "world_landmarks" not in df.columns:
        print(
            "Warning: No world_landmarks column found. Skipping perspective correction."
        )
        return df

    # Unpack world landmarks into separate columns
    df = unpack_world_landmarks(df)

    # Get the first frame index
    first_idx = df.index[0]

    # --- STEP 1: Calculate reference camera angle from the FIRST frame ---
    reference_camera_yaw_deg, lateral_correction_factor = (
        calculate_reference_camera_angle(df, first_idx)
    )

    if reference_camera_yaw_deg is None:
        print(
            "  Warning: Could not establish reference camera angle. Skipping perspective correction."
        )
        return df

    print(
        f"  Reference camera yaw angle: {reference_camera_yaw_deg:.1f}° (from frame {first_idx})"
    )
    print(f"  Lateral correction factor: {lateral_correction_factor:.3f}x")

    # --- STEP 2: Apply correction factor to all frames ---
    df = apply_perspective_correction(
        df, reference_camera_yaw_deg, lateral_correction_factor, first_idx
    )

    return df
