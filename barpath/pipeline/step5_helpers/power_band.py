"""Power zone band rendering for HUD overlay.

Draws a horizontal intensity gradient band below the sparkline,
showing specific power over time.
"""

import cv2
import numpy as np

from barpath.pipeline.config import (
    POWER_BAND_GAP,
    POWER_BAND_HEIGHT,
    SPARKLINE_HEIGHT_RATIO,
    SPARKLINE_MARGIN_X,
    SPARKLINE_MARGIN_Y,
    SPARKLINE_WIDTH_RATIO,
)


def draw_power_zone_band(frame, df, frame_width, frame_height,
                         sparkline_box=None, current_frame=None):
    """Draw power zone intensity band below sparkline.

    Uses same time axis as sparkline for aligned visualization.
    Pre-computed full band — drawn once, not building up incrementally.
    "Power" label placed between sparkline and band.

    Args:
        frame: OpenCV frame/image to draw on (modified in place)
        df: DataFrame with kinematic data (must have 'specific_power_y_smooth')
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        sparkline_box: (x, y, w, h) of sparkline for positioning (optional)
        current_frame: Current video frame index for playhead (optional)

    Returns:
        frame (modified in place)
    """
    # Check for required column
    if 'specific_power_y_smooth' not in df.columns:
        return frame

    power_series = df['specific_power_y_smooth'].dropna().values
    if len(power_series) == 0:
        return frame

    # Calculate position — directly below sparkline, same X and width
    sparkline_w = max(100, int(frame_width * SPARKLINE_WIDTH_RATIO))
    if sparkline_box is not None:
        sl_x, sl_y, sl_w, sl_h = sparkline_box
        band_x = sl_x
        band_w = sl_w
    else:
        band_x = frame_width - SPARKLINE_MARGIN_X - sparkline_w
        band_y_base = SPARKLINE_MARGIN_Y + max(50, int(frame_height * SPARKLINE_HEIGHT_RATIO))
        band_w = sparkline_w

    # "Power" label goes right below sparkline
    label_gap = 2
    label_text_size = cv2.getTextSize("Power", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_h = label_text_size[1] + label_gap * 2

    if sparkline_box is not None:
        band_y = sl_y + sl_h + label_h + POWER_BAND_GAP
    else:
        band_y = band_y_base + label_h + POWER_BAND_GAP

    label_y = band_y - POWER_BAND_GAP - label_gap

    band_h = POWER_BAND_HEIGHT

    # Normalize power to intensity
    power_max = power_series.max()
    if power_max <= 0:
        power_max = 1.0

    # Draw per-column rectangles with intensity-based color
    px_per_col = max(1, band_w / len(power_series))
    for i in range(len(power_series)):
        intensity = power_series[i] / power_max
        # Single-hue warm orange gradient
        B = int(50 + (1 - intensity) * 200)
        G = int(120 + (1 - intensity) * 135)
        R = int(200 + (1 - intensity) * 55)
        x_start = band_x + int(i * px_per_col)
        cv2.rectangle(frame,
                      (x_start, band_y),
                      (x_start + max(1, int(px_per_col)), band_y + band_h),
                      (B, G, R), -1)

    # Draw "Power" label underneath sparkline with outline
    from .hud_renderer import draw_text_with_outline
    draw_text_with_outline(frame, "Power", (band_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 255))

    # Draw playhead if current_frame is provided
    if current_frame is not None and len(df) > 0:
        df_index = df.index
        first_frame = int(df_index.min())
        last_frame = int(df_index.max())
        if last_frame > first_frame:
            t = (current_frame - first_frame) / (last_frame - first_frame)
            t = max(0.0, min(1.0, t))
            playhead_x = band_x + int(t * band_w)
            cv2.line(frame, (playhead_x, band_y), (playhead_x, band_y + band_h),
                     (255, 255, 255), 2)

    return frame
