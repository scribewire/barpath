"""Velocity sparkline rendering for HUD overlay.

Draws a phase-colored sparkline in the top-right corner of the frame,
showing barbell vertical velocity over time. Includes a playhead
indicating the current video frame position.
"""

import cv2
import numpy as np

from barpath.pipeline.config import (
    SPARKLINE_AXIS_COLOR_BGR,
    SPARKLINE_HEIGHT_RATIO,
    SPARKLINE_LINE_THICKNESS,
    SPARKLINE_MARGIN_X,
    SPARKLINE_MARGIN_Y,
    SPARKLINE_WIDTH_RATIO,
)

from .hud_renderer import PHASE_COLOR_SCHEMES


def draw_velocity_sparkline(frame, df, frame_width, frame_height, lift_type,
                            current_frame=None):
    """Draw phase-colored velocity sparkline in top-right corner.

    Pre-computes full curve from frame 1 — drawn identically every frame,
    not building up incrementally. Playhead shows current frame position.

    Args:
        frame: OpenCV frame/image to draw on (modified in place)
        df: DataFrame with kinematic data (must have 'vel_y_smooth' and 'bar_phase')
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        lift_type: Type of lift for phase color mapping
        current_frame: Current video frame index for playhead (optional)

    Returns:
        tuple: (frame, sparkline_box) where sparkline_box is (x, y, w, h)
    """
    # Check for required columns
    if 'vel_y_smooth' not in df.columns or 'bar_phase' not in df.columns:
        return frame, None

    vel_series = df['vel_y_smooth'].dropna().values
    phases = df['bar_phase'].dropna().values

    if len(vel_series) == 0:
        return frame, None

    # Calculate bounding box with proportional sizing and minimum clamp
    sparkline_w = max(100, int(frame_width * SPARKLINE_WIDTH_RATIO))
    sparkline_h = max(50, int(frame_height * SPARKLINE_HEIGHT_RATIO))
    x = frame_width - SPARKLINE_MARGIN_X - sparkline_w
    y = SPARKLINE_MARGIN_Y

    # Normalize velocity to sparkline box
    vel_min = vel_series.min()
    vel_max = vel_series.max()
    x_scale = sparkline_w / len(vel_series)
    y_scale = sparkline_h / (vel_max - vel_min + 1e-6)

    # Compute point coordinates
    points = []
    for i in range(len(vel_series)):
        px = x + int(i * x_scale)
        py = y + sparkline_h - int((vel_series[i] - vel_min) * y_scale)
        points.append((px, py))

    # Draw phase-colored segments
    phase_scheme = PHASE_COLOR_SCHEMES.get(lift_type, PHASE_COLOR_SCHEMES["snatch"])
    for i in range(len(points) - 1):
        phase_idx = int(phases[min(i, len(phases) - 1)])
        color = phase_scheme.get(phase_idx, (255, 255, 255))
        cv2.line(frame, points[i], points[i + 1], color, SPARKLINE_LINE_THICKNESS)

    # Draw axis lines (subtle)
    cv2.line(frame, (x, y + sparkline_h), (x + sparkline_w, y + sparkline_h),
             SPARKLINE_AXIS_COLOR_BGR, 1)
    cv2.line(frame, (x, y), (x, y + sparkline_h), SPARKLINE_AXIS_COLOR_BGR, 1)

    # Draw "Velocity" label with outline
    from .hud_renderer import draw_text_with_outline
    draw_text_with_outline(frame, "Velocity", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 255))

    # Draw playhead if current_frame is provided
    if current_frame is not None and len(df) > 0:
        df_index = df.index
        first_frame = int(df_index.min())
        last_frame = int(df_index.max())
        if last_frame > first_frame:
            # Map current_frame to position in the sparkline
            t = (current_frame - first_frame) / (last_frame - first_frame)
            t = max(0.0, min(1.0, t))
            playhead_x = x + int(t * sparkline_w)
            cv2.line(frame, (playhead_x, y), (playhead_x, y + sparkline_h),
                     (255, 255, 255), 2)

    return frame, (x, y, sparkline_w, sparkline_h)
