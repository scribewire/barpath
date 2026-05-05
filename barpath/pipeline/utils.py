"""
Shared utility functions for the barpath pipeline.

This module contains helper functions used across multiple pipeline steps
to avoid code duplication and maintain consistency.
"""

import ast

try:
    import cv2
except ImportError:
    cv2 = None
try:
    import pandas as pd
except ImportError:
    pd = None


# ============================================================================
# VIDEO DRAWING UTILITIES
# ============================================================================


def draw_legend(image, colors):
    """
    Draws a color legend on the image.

    Args:
        image (np.ndarray): The frame/image to draw on
        colors (dict): Dictionary of color names to BGR tuples

    Returns:
        int: Y offset after legend (for text placement)
    """
    if cv2 is None:
        return 0

    y_offset = 30
    for i, (name, color) in enumerate(colors.items()):
        cv2.rectangle(
            image, (15, 10 + i * y_offset), (35, 30 + i * y_offset), color, -1
        )
        x, y = 45, 25 + i * y_offset
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            cv2.putText(
                image, name, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 2, cv2.LINE_AA,
            )
        cv2.putText(
            image,
            name,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return 15 + len(colors) * y_offset


def get_connection_color(lm1_name, lm2_name, color_scheme):
    """
    Determines the color for a skeleton connection based on body part.

    Args:
        lm1_name (str): First landmark name
        lm2_name (str): Second landmark name
        color_scheme (dict): Color scheme dictionary

    Returns:
        tuple: BGR color tuple
    """
    # Check for torso connections
    if ("shoulder" in lm1_name or "hip" in lm1_name) and (
        "shoulder" in lm2_name or "hip" in lm2_name
    ):
        return color_scheme.get("Torso", (255, 255, 0))

    # Check for left side
    if "left" in lm1_name and "left" in lm2_name:
        if any(part in lm1_name for part in ["shoulder", "elbow", "wrist"]):
            return color_scheme.get("Left Arm", (0, 165, 255))
        if any(part in lm1_name for part in ["hip", "knee", "ankle"]):
            return color_scheme.get("Left Leg", (255, 0, 128))

    # Check for right side
    if "right" in lm1_name and "right" in lm2_name:
        if any(part in lm1_name for part in ["shoulder", "elbow", "wrist"]):
            return color_scheme.get("Right Arm", (0, 255, 255))
        if any(part in lm1_name for part in ["hip", "knee", "ankle"]):
            return color_scheme.get("Right Leg", (0, 255, 0))

    return (255, 255, 255)


def parse_landmarks_from_string(landmarks_str):
    """
    Safely parses the landmark dictionary string from the CSV.

    Args:
        landmarks_str (str): String representation of landmarks dict

    Returns:
        dict: Parsed landmarks dictionary, or empty dict if parsing fails
    """
    try:
        if pd is not None and pd.isna(landmarks_str):
            return {}
        if landmarks_str == "{}" or landmarks_str == "":
            return {}
        return ast.literal_eval(landmarks_str)
    except Exception:
        return {}


def parse_barbell_box(box_str):
    """
    Parses the barbell box string from CSV.

    Args:
        box_str (str): String representation of box coordinates

    Returns:
        tuple: (x1, y1, x2, y2) as integers, or None if parsing fails
    """
    try:
        if pd is not None and pd.isna(box_str):
            return None
        if box_str == "" or box_str is None:
            return None
        values = [float(v.strip()) for v in str(box_str).split(",")]
        if len(values) == 4:
            return tuple(map(int, values))
    except Exception:
        pass
    return None


# ============================================================================
# CONSTANTS
# ============================================================================

# MediaPipe landmark names (used by 1_collect_data.py)
# Canonical source of truth is LANDMARKS_TO_TRACK in config.py;
# this set is kept for convenience and must stay in sync.
LANDMARK_NAMES = {
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_eye",
    "right_eye",
}

# Color scheme for visualization
COLOR_SCHEME = {
    "Torso": (255, 255, 0),  # Cyan
    "Left Arm": (0, 165, 255),  # Orange
    "Right Arm": (0, 255, 255),  # Yellow
    "Left Leg": (255, 0, 128),  # Purple
    "Right Leg": (0, 255, 0),  # Green
    "Barbell Box": (255, 0, 255),  # Magenta
    "Barbell Path": (0, 0, 255),  # Red
}
