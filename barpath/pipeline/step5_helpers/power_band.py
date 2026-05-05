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


def draw_power_zone_band(frame, df, frame_width, frame_height):
    """Draw power zone intensity band below sparkline.

    Uses same time axis as sparkline for aligned visualization.
    Pre-computed full band — drawn once, not building up incrementally.

    Args:
        frame: OpenCV frame/image to draw on (modified in place)
        df: DataFrame with kinematic data (must have 'specific_power_y_smooth')
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels

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
    band_x = frame_width - SPARKLINE_MARGIN_X - sparkline_w
    band_y = SPARKLINE_MARGIN_Y + max(50, int(frame_height * SPARKLINE_HEIGHT_RATIO)) + POWER_BAND_GAP
    band_w = sparkline_w
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

    # Draw "Power" label
    cv2.putText(frame, "Power", (band_x, band_y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (80, 80, 80), 1, cv2.LINE_AA)

    return frame
