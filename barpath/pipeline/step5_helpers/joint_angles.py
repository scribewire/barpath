"""Knee angle display with baseline color coding for HUD overlay.

Displays left and right knee angles at bottom-center of frame,
color-coded against phase-specific baseline thresholds.
"""

import cv2
import numpy as np

from barpath.pipeline.config import (
    ANGLE_BORDERLINE_MARGIN,
    ANGLE_FALLBACK_MAX,
    ANGLE_FALLBACK_MIN,
    ANGLE_FONT_SCALE,
    ANGLE_FONT_THICKNESS,
    ANGLE_GREEN_BGR,
    ANGLE_RED_BGR,
    ANGLE_TEXT_POSITION_Y_RATIO,
    ANGLE_YELLOW_BGR,
)


def _get_angle_color(angle, p25, p75):
    """Determine color for knee angle based on baseline thresholds.

    Args:
        angle: Current knee angle in degrees
        p25: 25th percentile baseline threshold
        p75: 75th percentile baseline threshold

    Returns:
        BGR color tuple: green (within range), yellow (borderline), red (outside)
    """
    if p25 <= angle <= p75:
        return ANGLE_GREEN_BGR
    elif p25 * (1 - ANGLE_BORDERLINE_MARGIN) <= angle <= p75 * (1 + ANGLE_BORDERLINE_MARGIN):
        return ANGLE_YELLOW_BGR
    else:
        return ANGLE_RED_BGR


def draw_knee_angles(frame, df_row, frame_width, frame_height, bar_phase, baselines=None):
    """Draw knee angle display at bottom-center of frame.

    Args:
        frame: OpenCV frame/image to draw on (modified in place)
        df_row: Current frame's row from DataFrame (must have left_knee_angle, right_knee_angle)
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        bar_phase: Current bar phase index (for phase-specific baselines)
        baselines: Optional dict of baseline data (keyed by lift_type_gender)

    Returns:
        frame (modified in place)
    """
    if df_row is None:
        return frame

    # Extract knee angles
    left_angle = df_row.get('left_knee_angle', None)
    right_angle = df_row.get('right_knee_angle', None)

    # Determine thresholds
    if baselines is not None:
        # TODO: Load phase-specific baselines from pro_baseline_report.json
        # For now, use fallback thresholds
        p25 = ANGLE_FALLBACK_MIN
        p75 = ANGLE_FALLBACK_MAX
    else:
        p25 = ANGLE_FALLBACK_MIN
        p75 = ANGLE_FALLBACK_MAX

    # Position: bottom-center
    y = int(frame_height * ANGLE_TEXT_POSITION_Y_RATIO)

    # Draw left knee angle
    if left_angle is not None and not (isinstance(left_angle, float) and np.isnan(left_angle)):
        left_text = f"L Knee: {int(left_angle)}\u00b0"
        left_color = _get_angle_color(float(left_angle), p25, p75)
        # Calculate position for left text (left half of center)
        left_text_size = cv2.getTextSize(left_text, cv2.FONT_HERSHEY_SIMPLEX,
                                          ANGLE_FONT_SCALE, ANGLE_FONT_THICKNESS)[0]
        left_x = (frame_width - left_text_size[0]) // 2 - 10
        cv2.putText(frame, left_text, (left_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    ANGLE_FONT_SCALE, left_color, ANGLE_FONT_THICKNESS, cv2.LINE_AA)

    # Draw right knee angle
    if right_angle is not None and not (isinstance(right_angle, float) and np.isnan(right_angle)):
        right_text = f"R Knee: {int(right_angle)}\u00b0"
        right_color = _get_angle_color(float(right_angle), p25, p75)
        # Calculate position for right text (right half of center)
        right_text_size = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX,
                                           ANGLE_FONT_SCALE, ANGLE_FONT_THICKNESS)[0]
        right_x = (frame_width + right_text_size[0]) // 2 + 10 - right_text_size[0]
        cv2.putText(frame, right_text, (right_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    ANGLE_FONT_SCALE, right_color, ANGLE_FONT_THICKNESS, cv2.LINE_AA)

    return frame
