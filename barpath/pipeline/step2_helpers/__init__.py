"""
Step 2 Helper modules for data analysis.

This package contains helper modules for the analysis pipeline:
- perspective_correction: Perspective correction using world landmarks
- landmark_processing: Landmark unpacking and angle calculations
- kinematics: Barbell position, velocity, and phase detection
"""

from .kinematics import (
    assign_phases_from_classics,
    assign_phases_kinematic,
    calculate_stabilized_position,
    calculate_time_and_kinematics,
    detect_three_phases,
    smooth_barbell_position,
    truncate_at_knee_pass,
    truncate_at_peak_height,
)
from .landmark_processing import (
    calculate_hip_y_average,
    calculate_joint_angles,
    calculate_knee_y_average,
    calculate_lifter_angle,
    drop_intermediate_columns,
    get_pixel_pos,
    unpack_landmarks,
)
from .perspective_correction import (
    apply_perspective_correction,
    calculate_perspective_correction,
    calculate_reference_camera_angle,
    unpack_world_landmarks,
)

__all__ = [
    "unpack_landmarks",
    "get_pixel_pos",
    "calculate_joint_angles",
    "calculate_lifter_angle",
    "calculate_hip_y_average",
    "calculate_knee_y_average",
    "drop_intermediate_columns",
    "calculate_stabilized_position",
    "smooth_barbell_position",
    "truncate_at_knee_pass",
    "truncate_at_peak_height",
    "calculate_time_and_kinematics",
    "detect_three_phases",
    "assign_phases_from_classics",
    "assign_phases_kinematic",
    "unpack_world_landmarks",
    "calculate_reference_camera_angle",
    "apply_perspective_correction",
    "calculate_perspective_correction",
]
