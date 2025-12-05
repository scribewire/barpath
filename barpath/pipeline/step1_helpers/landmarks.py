"""
Landmark collection helpers for Step 1: Data Collection.

This module contains functions for extracting and processing MediaPipe pose landmarks
(both normalized and world coordinates) for video analysis and perspective correction.
"""

from typing import Optional, Tuple

import mediapipe as mp
import numpy as np


def get_landmark_enums(landmark_names):
    """
    Convert landmark string names to MediaPipe PoseLandmark enum objects.

    Args:
        landmark_names (set or list): Set of landmark names (e.g., 'left_shoulder')

    Returns:
        dict: Mapping of landmark name to MediaPipe PoseLandmark enum
    """
    landmark_enums = {
        name: mp.solutions.pose.PoseLandmark[name.upper()]  # type: ignore[attr-defined]
        for name in landmark_names
    }
    return landmark_enums


def extract_landmarks(pose_landmarks, landmark_enums: dict) -> Optional[dict]:
    """
    Extract normalized pose landmarks from MediaPipe results.

    Normalized landmarks are in [0, 1] range relative to image dimensions
    and include (x, y, z, visibility) for each landmark.

    Args:
        pose_landmarks: MediaPipe pose_landmarks object
        landmark_enums (dict): Mapping of landmark name to enum

    Returns:
        dict: Dictionary mapping landmark name to (x, y, z, visibility) tuple
    """
    landmarks_data = {}

    for name, enum in landmark_enums.items():
        lm = pose_landmarks.landmark[enum]
        landmarks_data[name] = (lm.x, lm.y, lm.z, lm.visibility)

    return landmarks_data


def extract_world_landmarks(
    pose_world_landmarks, landmark_enums: dict
) -> Optional[dict]:
    """
    Extract world pose landmarks from MediaPipe results.

    World landmarks are in meters relative to the hip center and are not
    affected by image dimensions. They provide 3D spatial information.

    Args:
        pose_world_landmarks: MediaPipe pose_world_landmarks object
        landmark_enums (dict): Mapping of landmark name to enum

    Returns:
        dict: Dictionary mapping landmark name to (x, y, z, visibility) tuple
    """
    world_landmarks_data = {}

    for name, enum in landmark_enums.items():
        wlm = pose_world_landmarks.landmark[enum]
        world_landmarks_data[name] = (wlm.x, wlm.y, wlm.z, wlm.visibility)

    return world_landmarks_data


def process_pose_results(
    results_pose, landmark_enums: dict
) -> Tuple[Optional[dict], Optional[dict], Optional[np.ndarray]]:
    """
    Process MediaPipe pose results and extract both landmark types.

    Handles cases where pose detection succeeds or fails gracefully.

    Args:
        results_pose: MediaPipe Pose results object
        landmark_enums (dict): Mapping of landmark name to enum

    Returns:
        tuple: (landmarks_data, world_landmarks_data, segmentation_mask)
            - landmarks_data (dict or None): Normalized landmarks
            - world_landmarks_data (dict or None): World landmarks in meters
            - segmentation_mask (np.ndarray or None): Binary person mask
    """
    landmarks_data = None
    world_landmarks_data = None
    segmentation_mask = None

    if results_pose and results_pose.pose_landmarks:
        # Extract normalized landmarks
        landmarks_data = extract_landmarks(results_pose.pose_landmarks, landmark_enums)

        # Extract world landmarks if available
        if results_pose.pose_world_landmarks:
            world_landmarks_data = extract_world_landmarks(
                results_pose.pose_world_landmarks, landmark_enums
            )

        # Extract segmentation mask if available
        if results_pose.segmentation_mask is not None:
            # Create a binary mask (1 for person, 0 for background)
            segmentation_mask = (results_pose.segmentation_mask > 0.5).astype(np.uint8)

    return landmarks_data, world_landmarks_data, segmentation_mask


def get_ankle_positions(
    pose_landmarks, mp_pose_solution, frame_width: int, frame_height: int
) -> Optional[np.ndarray]:
    """
    Extract ankle positions from pose landmarks.

    Used for initial barbell detection by finding the position near the lifter's feet.

    Args:
        pose_landmarks: MediaPipe pose_landmarks object
        mp_pose_solution: MediaPipe pose solution module
        frame_width (int): Video frame width in pixels
        frame_height (int): Video frame height in pixels

    Returns:
        np.ndarray: Average ankle position [x, y] in pixels, or None if not available
    """
    l_ankle = pose_landmarks.landmark[mp_pose_solution.PoseLandmark.LEFT_ANKLE]
    r_ankle = pose_landmarks.landmark[mp_pose_solution.PoseLandmark.RIGHT_ANKLE]

    l_visible = l_ankle.visibility > 0.3
    r_visible = r_ankle.visibility > 0.3

    l_pos = (
        np.array([l_ankle.x * frame_width, l_ankle.y * frame_height])
        if l_visible
        else None
    )
    r_pos = (
        np.array([r_ankle.x * frame_width, r_ankle.y * frame_height])
        if r_visible
        else None
    )

    # Return average position if both visible, otherwise return available one
    if l_visible and r_visible and l_pos is not None and r_pos is not None:
        return (l_pos + r_pos) / 2
    elif l_visible and l_pos is not None:
        return l_pos
    elif r_visible and r_pos is not None:
        return r_pos

    return None
