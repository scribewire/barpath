"""
Step 1 Helper modules for data collection.

This package contains helper modules for the data collection pipeline:
- stabilization: Camera motion estimation using optical flow
- landmarks: Pose landmark extraction and processing
"""

from .landmarks import (
    extract_landmarks,
    extract_world_landmarks,
    get_ankle_positions,
    get_landmark_enums,
    get_pose_landmarker_model_path,
    process_pose_results,
)
from .stabilization import (
    StabilizationParams,
    create_background_mask,
    detect_features,
    estimate_motion,
    track_features,
    update_features,
)

__all__ = [
    # Stabilization
    "StabilizationParams",
    "create_background_mask",
    "detect_features",
    "estimate_motion",
    "extract_landmarks",
    "extract_world_landmarks",
    "get_ankle_positions",
    # Landmarks
    "get_landmark_enums",
    "get_pose_landmarker_model_path",
    "process_pose_results",
    "track_features",
    "update_features",
]
