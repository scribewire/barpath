"""
Step 2 Helper modules for data analysis.

This package contains helper modules for the analysis pipeline:
- landmark_processing: Landmark unpacking and angle calculations
- kinematics: Barbell position, velocity, and phase detection
"""

from .kinematics import (
    assign_phases_from_classics,
    assign_phases_kinematic,
    calculate_stabilized_position,
    calculate_time_and_kinematics,
    smooth_barbell_position,
    truncate_at_knee_pass,
    truncate_at_peak_height,
)
from .landmark_processing import (
    calculate_hip_y_average,
    calculate_joint_angles,
    calculate_lifter_angle,
    detect_facing_direction,
    drop_intermediate_columns,
    unpack_landmarks,
)

__all__ = [
    "unpack_landmarks",
    "calculate_joint_angles",
    "calculate_lifter_angle",
    "detect_facing_direction",
    "calculate_hip_y_average",
    "drop_intermediate_columns",
    "calculate_stabilized_position",
    "smooth_barbell_position",
    "truncate_at_knee_pass",
    "truncate_at_peak_height",
    "calculate_time_and_kinematics",
    "assign_phases_from_classics",
    "assign_phases_kinematic",
]
