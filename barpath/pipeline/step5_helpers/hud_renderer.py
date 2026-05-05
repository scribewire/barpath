"""HUD rendering functions extracted from 5_render_video.py.

This module contains:
- draw_skeleton_overlay: Pose skeleton drawing with body-part coloring
- draw_bar_path_trail: Phase-colored barbell path trail
- draw_hud_overlay: Orchestrator function for per-frame HUD compositing
"""

import cv2
import numpy as np

from barpath.pipeline.config import (
    LANDMARK_RADIUS,
    PHASE_COLORS_BGR,
    SKELETON_LINE_THICKNESS,
)
from barpath.pipeline.utils import (
    COLOR_SCHEME,
    draw_legend,
    get_connection_color,
    parse_landmarks_from_string,
)

# Phase color schemes for different lift types (moved from 5_render_video.py)
PHASE_COLOR_SCHEMES = {
    "snatch": {
        0: (0, 0, 255),  # Red - Pull
        1: (0, 165, 255),  # Orange - Pull-under
        2: (0, 255, 0),  # Green - Recovery
    },
    "clean": {
        0: (0, 0, 255),  # Red - Pull
        1: (0, 165, 255),  # Orange - Pull-under
        2: (0, 255, 0),  # Green - Recovery
    },
    "jerk": {
        0: (255, 0, 0),  # Blue - Dip
        1: (255, 255, 0),  # Yellow - Drive
        2: (0, 255, 0),  # Green - Recovery
    },
    "clean_jerk": {
        0: (0, 0, 255),  # Red - Clean Pull
        1: (0, 165, 255),  # Orange - Clean Pull-under
        2: (0, 255, 0),  # Green - Clean Recovery
        3: (255, 0, 0),  # Blue - Jerk Dip
        4: (255, 255, 0),  # Yellow - Jerk Drive
        5: (0, 255, 255),  # Cyan - Jerk Recovery
    },
}

# Phase names for legend (moved from 5_render_video.py)
PHASE_NAMES = {
    "snatch": {0: "Pull", 1: "Pull-under", 2: "Recovery"},
    "clean": {0: "Pull", 1: "Pull-under", 2: "Recovery"},
    "jerk": {0: "Dip", 1: "Drive", 2: "Recovery"},
    "clean_jerk": {
        0: "Pull",
        1: "Pull-under",
        2: "Recovery",
        3: "Dip",
        4: "Drive",
        5: "Recovery",
    },
}

# Legend colors (moved from 5_render_video.py)
LEGEND_COLORS = {
    "Barbell Box": COLOR_SCHEME["Barbell Box"],
    "Pull": PHASE_COLORS_BGR[0],
    "Pull-under": PHASE_COLORS_BGR[1],
    "Recovery": PHASE_COLORS_BGR[2],
    "Dip": (255, 0, 0),
    "Drive": (255, 255, 0),
}

# Skeleton connections (moved from 5_render_video.py)
SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def draw_skeleton_overlay(frame, landmarks_str, frame_width, frame_height, legend_colors):
    """Draw pose skeleton overlay on the frame.

    Args:
        frame: OpenCV frame/image to draw on (modified in place)
        landmarks_str: String representation of landmarks dict from CSV
        frame_width: Width of the frame in pixels
        frame_height: Height of the frame in pixels
        legend_colors: Color scheme dict for body part coloring

    Returns:
        last_head_pos: tuple (x, y) of head position or None
    """
    landmarks = parse_landmarks_from_string(landmarks_str)
    if not landmarks:
        return None

    landmark_pixels = {}
    for name, (x, y, z, vis) in landmarks.items():
        if vis > 0.1:
            px = int(x * frame_width)
            py = int(y * frame_height)
            landmark_pixels[name] = (px, py)

    for lm1_name, lm2_name in SKELETON_CONNECTIONS:
        if lm1_name in landmark_pixels and lm2_name in landmark_pixels:
            p1 = landmark_pixels[lm1_name]
            p2 = landmark_pixels[lm2_name]
            color = get_connection_color(lm1_name, lm2_name, legend_colors)
            cv2.line(frame, p1, p2, color, SKELETON_LINE_THICKNESS)

    for name, (px, py) in landmark_pixels.items():
        cv2.circle(frame, (px, py), LANDMARK_RADIUS, (255, 255, 255), -1)

    # Determine head position for velocity text placement
    last_head_pos = None
    for key in ["left_eye", "right_eye", "nose"]:
        if key in landmark_pixels:
            last_head_pos = landmark_pixels[key]
            break
    if last_head_pos is None:
        for key in ["left_shoulder", "right_shoulder"]:
            if key in landmark_pixels:
                last_head_pos = landmark_pixels[key]
                break
    if last_head_pos is None and len(landmark_pixels) > 0:
        last_head_pos = list(landmark_pixels.values())[0]

    return last_head_pos


def draw_bar_path_trail(frame, path_points, path_phases, max_path_index,
                        current_shake_x, current_shake_y, lift_type):
    """Draw phase-colored barbell path trail on the frame.

    Args:
        frame: OpenCV frame/image to draw on (modified in place)
        path_points: Nx2 array of (x, y) barbell positions
        path_phases: N array of phase indices
        max_path_index: Number of path points to draw (up to current frame)
        current_shake_x: X shake offset for current frame
        current_shake_y: Y shake offset for current frame
        lift_type: Type of lift for phase color mapping

    Returns:
        None (frame modified in place)
    """
    if max_path_index < 2:
        return

    points_to_draw = path_points[:max_path_index].copy()
    phases_to_draw = path_phases[:max_path_index]

    points_to_draw[:, 0] += current_shake_x
    points_to_draw[:, 1] += current_shake_y
    points_to_draw = points_to_draw.astype(np.int32)

    phase_scheme = PHASE_COLOR_SCHEMES.get(lift_type, PHASE_COLOR_SCHEMES["snatch"])

    for i in range(len(points_to_draw) - 1):
        p1 = (int(points_to_draw[i, 0]), int(points_to_draw[i, 1]))
        p2 = (int(points_to_draw[i + 1, 0]), int(points_to_draw[i + 1, 1]))
        phase_index = int(phases_to_draw[i])
        color = phase_scheme.get(phase_index, (255, 255, 255))
        cv2.line(frame, p1, p2, color, 3)


def draw_hud_overlay(frame, df, df_row, frame_width, frame_height, lift_type, hud_config,
                     path_points, path_phases, max_path_index, shake_x, shake_y,
                     landmarks_str, legend_colors, analysis_result=None, baselines=None):
    """Orchestrator function for per-frame HUD compositing.

    Calls each HUD element function based on hud_config toggles.

    Args:
        frame: OpenCV frame/image to draw on (modified in place)
        df: Full DataFrame with kinematic data
        df_row: Current frame's row from DataFrame
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        lift_type: Type of lift for phase color mapping
        hud_config: HUDConfig dataclass with element toggles
        path_points: Nx2 array of barbell positions
        path_phases: N array of phase indices
        max_path_index: Number of path points to draw
        shake_x: X shake offset
        shake_y: Y shake offset
        landmarks_str: String representation of landmarks dict
        legend_colors: Color scheme dict for body part coloring
        analysis_result: Optional dict from Step 4 critique (for error markers)
        baselines: Optional baseline data dict (for knee angle coloring)

    Returns:
        last_head_pos: tuple (x, y) or None
    """
    # Draw bar path trail (always drawn)
    draw_bar_path_trail(frame, path_points, path_phases, max_path_index,
                        shake_x, shake_y, lift_type)

    # Draw skeleton overlay
    last_head_pos = None
    if hud_config.show_skeleton:
        last_head_pos = draw_skeleton_overlay(frame, landmarks_str, frame_width,
                                               frame_height, legend_colors)

    # Draw velocity sparkline
    if hud_config.show_sparkline:
        from .sparkline import draw_velocity_sparkline
        frame = draw_velocity_sparkline(frame, df, frame_width, frame_height, lift_type)

    # Draw power zone band
    if hud_config.show_power_zones:
        from .power_band import draw_power_zone_band
        frame = draw_power_zone_band(frame, df, frame_width, frame_height)

    # Draw knee angles
    if hud_config.show_angles:
        from .joint_angles import draw_knee_angles
        bar_phase = df_row.get('bar_phase', None) if df_row is not None else None
        frame = draw_knee_angles(frame, df_row, frame_width, frame_height,
                                  bar_phase, baselines)

    # Draw error markers
    if hud_config.show_error_markers and analysis_result:
        from .error_markers import draw_error_markers
        draw_error_markers(frame, analysis_result, df, path_points, path_phases,
                           max_path_index, shake_x, shake_y, lift_type)

    return frame, last_head_pos
