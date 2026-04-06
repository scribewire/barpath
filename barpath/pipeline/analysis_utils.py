"""
Shared analysis utilities for the barpath pipeline.

This module contains common functions used across multiple pipeline steps,
consolidated here to avoid code duplication.
"""

import typing
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def calculate_sg_window(
    data_length: int, default_window: int, poly_order: int = 3
) -> int:
    """
    Calculate an appropriate Savitzky-Golay window size.

    The window must be:
    - Odd
    - >= poly_order + 2
    - <= data_length

    Args:
        data_length: Number of data points
        default_window: Preferred window size
        poly_order: Polynomial order for the filter

    Returns:
        Valid window size
    """
    min_window = poly_order + 2
    if min_window % 2 == 0:
        min_window += 1

    max_window = data_length if data_length % 2 == 1 else data_length - 1

    window = min(default_window, max_window)
    window = max(window, min_window)

    if window % 2 == 0:
        window -= 1

    return window


def safe_savgol_smooth(series: pd.Series, window: int = 11, poly: int = 3) -> pd.Series:
    """
    Apply Savitzky-Golay smoothing with automatic window adjustment.

    Handles NaN and Inf values by interpolation and automatically clamps the window
    to be valid for the data length.

    Args:
        series: Input data series
        window: Desired window size
        poly: Polynomial order

    Returns:
        Smoothed series with same index as input
    """
    # Replace Inf values with NaN first
    series_clean = series.replace([np.inf, -np.inf], np.nan)

    # Interpolate to fill NaN values
    filled = series_clean.interpolate(method="linear").bfill().ffill()

    # Check if we still have NaN values (e.g., all values were NaN)
    if filled.isna().all():
        return series_clean

    # Replace any remaining NaN with 0 (edge case for all-NaN series)
    filled = filled.fillna(0)

    n = len(filled)

    w = calculate_sg_window(n, window, poly)

    if n < w or w < poly + 1:
        return filled

    try:
        smoothed = savgol_filter(filled.values, w, poly)
        return pd.Series(smoothed, index=series.index)
    except ValueError as e:
        print(
            f"Warning: Savitzky-Golay smoothing failed: {e}. Returning unsmoothed data."
        )
        return filled


def calculate_max_specific_power(
    df: pd.DataFrame, phases: typing.Any, t1_key: str = "t1", t3_key: str = "t3"
) -> Optional[dict]:
    """
    Calculate maximum specific power between two phase boundaries.

    This is used to find peak power output during the pull-under phase
    of Olympic lifts.

    Args:
        df: DataFrame with kinematic data including 'specific_power_y_smooth'
        phases: Dict with phase boundary frame indices (e.g., {'t1': 100, 't3': 150})
        t1_key: Key for start frame in phases dict
        t3_key: Key for end frame in phases dict

    Returns:
        Dict with 'max_power_px' and optionally 'max_power_real' (W/kg), or None
    """
    if phases is None or t1_key not in phases or t3_key not in phases:
        return None

    try:
        t1 = int(phases[t1_key])
        t3 = int(phases[t3_key])

        if "specific_power_y_smooth" not in df.columns:
            return None

        power_segment = df.loc[t1:t3, "specific_power_y_smooth"]

        if power_segment.empty:
            return None

        max_power_px = float(power_segment.abs().max())

        if np.isnan(max_power_px):
            return None

        result: dict[str, Optional[float]] = {"max_power_px": max_power_px}

        if "px_to_m_conversion" in df.columns:
            px_to_m = df["px_to_m_conversion"].dropna()
            if len(px_to_m) > 0:
                px_to_m_val = float(px_to_m.iloc[0])
                if not np.isnan(px_to_m_val) and px_to_m_val > 0:
                    max_power_real = max_power_px * (px_to_m_val**2)
                    result["max_power_real"] = max_power_real

        return result
    except Exception as e:
        print(f"Warning: Could not calculate max specific power: {e}")
        return None


def calculate_pixel_to_meter_conversion(
    df: pd.DataFrame, endcap_width_m: float = 0.05
) -> Optional[float]:
    """
    Calculate pixel-to-meter conversion factor based on barbell endcap width.

    Args:
        df: DataFrame with barbell_box data
        endcap_width_m: Real-world width of barbell endcap in metres

    Returns:
        Pixels-to-metres factor, or None if cannot be calculated
    """
    if "barbell_box" not in df.columns:
        return None

    try:
        widths = []
        for box in df["barbell_box"]:
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                x1, y1, x2, y2 = box[:4]
                width_px = abs(float(x2) - float(x1))
                if width_px > 0:
                    widths.append(width_px)

        if not widths:
            return None

        median_width_px = float(np.median(widths))
        px_to_m = endcap_width_m / median_width_px

        print(f"Endcap detection: median width = {median_width_px:.1f} px")
        print(f"Pixel-to-meter conversion: 1 px = {px_to_m * 1000:.3f} mm")
        return px_to_m
    except Exception as e:
        print(f"Warning: Could not calculate pixel-to-meter conversion: {e}")
        return None
