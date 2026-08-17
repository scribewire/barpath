"""
Landmark collection helpers for Step 1: Data Collection.

This module contains functions for extracting and processing MediaPipe pose landmarks
(both normalized and world coordinates) for video analysis and perspective correction.

Updated for mediapipe 0.10.x Tasks API.
"""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

POSE_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

POSE_LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

POSE_LANDMARK_INDEX = {name: i for i, name in enumerate(POSE_LANDMARK_NAMES)}


def get_pose_landmarker_model_path() -> Path:
    """
    Get the path to the pose landmarker model, downloading if necessary.

    Returns:
        Path to the pose landmarker .task file
    """
    models_dir = Path(__file__).parent.parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "pose_landmarker_heavy.task"

    if not model_path.exists():
        print(f"Downloading pose landmarker model to {model_path}...")
        urlretrieve(POSE_LANDMARKER_MODEL_URL, str(model_path))
        print("Download complete.")

    return model_path


def get_landmark_enums(landmark_names):
    """
    Convert landmark string names to landmark indices.

    Args:
        landmark_names (set or list): Set of landmark names (e.g., 'left_shoulder')

    Returns:
        dict: Mapping of landmark name to index
    """
    landmark_enums = {}
    for name in landmark_names:
        if name in POSE_LANDMARK_INDEX:
            landmark_enums[name] = POSE_LANDMARK_INDEX[name]
        else:
            raise ValueError(f"Unknown landmark name: {name}")
    return landmark_enums


def extract_landmarks(pose_landmarks, landmark_enums: dict) -> dict | None:
    """
    Extract normalized pose landmarks from MediaPipe results.

    Normalized landmarks are in [0, 1] range relative to image dimensions
    and include (x, y, z, visibility) for each landmark.

    Args:
        pose_landmarks: MediaPipe pose_landmarks protobuf message
        landmark_enums (dict): Mapping of landmark name to index

    Returns:
        dict: Dictionary mapping landmark name to (x, y, z, visibility) tuple
    """
    landmarks_data = {}

    for name, idx in landmark_enums.items():
        lm = pose_landmarks[idx]
        landmarks_data[name] = (
            lm.x,
            lm.y,
            lm.z,
            lm.visibility if hasattr(lm, "visibility") else 1.0,
        )

    return landmarks_data


def extract_world_landmarks(pose_world_landmarks, landmark_enums: dict) -> dict | None:
    """
    Extract world pose landmarks from MediaPipe results.

    World landmarks are in meters relative to the hip center and are not
    affected by image dimensions. They provide 3D spatial information.

    Args:
        pose_world_landmarks: MediaPipe pose_world_landmarks protobuf message
        landmark_enums (dict): Mapping of landmark name to index

    Returns:
        dict: Dictionary mapping landmark name to (x, y, z, visibility) tuple
    """
    world_landmarks_data = {}

    for name, idx in landmark_enums.items():
        wlm = pose_world_landmarks[idx]
        world_landmarks_data[name] = (
            wlm.x,
            wlm.y,
            wlm.z,
            wlm.visibility if hasattr(wlm, "visibility") else 1.0,
        )

    return world_landmarks_data


def process_pose_results(
    results_pose, landmark_enums: dict
) -> tuple[dict | None, dict | None, np.ndarray | None]:
    """
    Process MediaPipe pose results and extract both landmark types.

    Handles cases where pose detection succeeds or fails gracefully.

    Args:
        results_pose: MediaPipe PoseLandmarkerResult object
        landmark_enums (dict): Mapping of landmark name to index

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
        pose_landmarks_list = results_pose.pose_landmarks
        if len(pose_landmarks_list) > 0:
            pose_landmarks = pose_landmarks_list[0]
            landmarks_data = extract_landmarks(pose_landmarks, landmark_enums)

    if results_pose and results_pose.pose_world_landmarks:
        world_landmarks_list = results_pose.pose_world_landmarks
        if len(world_landmarks_list) > 0:
            world_landmarks = world_landmarks_list[0]
            world_landmarks_data = extract_world_landmarks(world_landmarks, landmark_enums)

    if results_pose and results_pose.segmentation_masks:
        seg_masks = results_pose.segmentation_masks
        if len(seg_masks) > 0:
            mask = seg_masks[0]
            mask_array = mask.numpy_view()
            segmentation_mask = (mask_array > 0.5).astype(np.uint8)

    return landmarks_data, world_landmarks_data, segmentation_mask


def get_ankle_positions(pose_landmarks, frame_width: int, frame_height: int) -> np.ndarray | None:
    """
    Extract ankle positions from pose landmarks.

    Used for initial barbell detection by finding the position near the lifter's feet.

    Args:
        pose_landmarks: MediaPipe pose_landmarks protobuf message
        frame_width (int): Video frame width in pixels
        frame_height (int): Video frame height in pixels

    Returns:
        np.ndarray: Average ankle position [x, y] in pixels, or None if not available
    """
    left_ankle_idx = POSE_LANDMARK_INDEX["left_ankle"]
    right_ankle_idx = POSE_LANDMARK_INDEX["right_ankle"]

    l_ankle = pose_landmarks[left_ankle_idx]
    r_ankle = pose_landmarks[right_ankle_idx]

    l_visible = l_ankle.visibility > 0.3
    r_visible = r_ankle.visibility > 0.3

    l_pos = np.array([l_ankle.x * frame_width, l_ankle.y * frame_height]) if l_visible else None
    r_pos = np.array([r_ankle.x * frame_width, r_ankle.y * frame_height]) if r_visible else None

    if l_visible and r_visible and l_pos is not None and r_pos is not None:
        return (l_pos + r_pos) / 2
    elif l_visible and l_pos is not None:
        return l_pos
    elif r_visible and r_pos is not None:
        return r_pos

    return None
