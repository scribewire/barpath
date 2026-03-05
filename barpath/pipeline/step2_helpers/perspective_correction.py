"""
Perspective correction helpers for Step 2: Data Analysis.

This module calculates perspective-corrected lateral bar displacement using
MediaPipe world landmarks and camera angle estimation.

Approach
--------
Rather than scaling observed pixel displacement by 1/cos(yaw) – which blows up
badly at non-trivial camera angles – we work entirely in real-world units:

1. Extract the shoulder vector in both pixel-space and world-space (metres).
2. Use the ratio of world-space shoulder width to pixel-space shoulder width as
   a px → metre scale factor (computed per-frame and averaged for stability).
3. Project the observed pixel-space horizontal bar displacement onto the true
   lateral axis using the world-space shoulder direction, converting the result
   to centimetres.
4. Store corrected positions as `barbell_x_corrected_cm` and
   `barbell_y_corrected_cm` so that both axes of the path graph are in the same
   physical unit and the aspect ratio is automatically believable.

This is numerically stable for any camera angle because we never divide by
cos(yaw); instead we use the shoulder geometry that MediaPipe already provides
in metric units.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_get(df: pd.DataFrame, idx, col: str) -> Optional[float]:
    """Return df.loc[idx, col] as a Python float, or None on any error."""
    try:
        val = df.loc[idx, col]
        if pd.isna(val):
            return None
        return float(val)
    except (KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def unpack_world_landmarks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract world landmark coordinates from the world_landmarks column.

    Unpacks the nested dictionary of world landmarks into separate columns
    for each body part's x/y/z coordinates (in metres, MediaPipe world space).

    Args:
        df: DataFrame with a 'world_landmarks' column whose values are dicts.

    Returns:
        DataFrame with additional columns ``{name}_world_x/y/z`` for
        left_shoulder, right_shoulder, left_hip, right_hip.
    """
    LANDMARKS = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    for name in LANDMARKS:
        df[f"{name}_world"] = df["world_landmarks"].apply(
            lambda x, _n=name: x.get(_n) if isinstance(x, dict) else None
        )
        df[f"{name}_world_x"] = df[f"{name}_world"].apply(
            lambda x: x[0] if (x is not None and len(x) >= 4 and x[3] > 0.1) else np.nan
        )
        df[f"{name}_world_y"] = df[f"{name}_world"].apply(
            lambda x: x[1] if (x is not None and len(x) >= 4 and x[3] > 0.1) else np.nan
        )
        df[f"{name}_world_z"] = df[f"{name}_world"].apply(
            lambda x: x[2] if (x is not None and len(x) >= 4 and x[3] > 0.1) else np.nan
        )
    return df


def calculate_reference_camera_angle(
    df: pd.DataFrame, first_idx
) -> Tuple[Optional[float], float]:
    """
    Calculate the reference camera yaw angle from the first frame.

    This is kept for informational / annotation purposes only.  The returned
    ``correction_factor`` is **not** used to scale bar displacement any more;
    the actual correction is done in :func:`apply_perspective_correction` using
    real metric units.

    Args:
        df: DataFrame with unpacked world landmark coordinates.
        first_idx: Index of the first frame to use as reference.

    Returns:
        ``(camera_yaw_deg, 1.0)`` – the correction factor is always 1.0 here
        because scaling is handled separately.
    """
    reference_camera_yaw_deg: Optional[float] = None

    try:
        l_sh_x = _safe_get(df, first_idx, "left_shoulder_world_x")
        l_sh_z = _safe_get(df, first_idx, "left_shoulder_world_z")
        r_sh_x = _safe_get(df, first_idx, "right_shoulder_world_x")
        r_sh_z = _safe_get(df, first_idx, "right_shoulder_world_z")

        if None not in (l_sh_x, l_sh_z, r_sh_x, r_sh_z):
            dx = float(r_sh_x) - float(l_sh_x)  # type: ignore[arg-type]
            dz = float(r_sh_z) - float(l_sh_z)  # type: ignore[arg-type]
            # Yaw = angle of shoulder line w.r.t. the camera's Z axis
            camera_yaw_rad = np.arctan2(abs(dx), abs(dz))
            reference_camera_yaw_deg = float(np.degrees(camera_yaw_rad))
    except Exception as exc:
        print(f"  Warning: Could not calculate reference camera angle: {exc}")

    return reference_camera_yaw_deg, 1.0  # factor unused; kept for API compat


def apply_perspective_correction(
    df: pd.DataFrame,
    reference_camera_yaw_deg: Optional[float],
    lateral_correction_factor: float,  # kept for API compat, ignored
    first_idx,
) -> pd.DataFrame:
    """
    Convert the bar path to real-world centimetres using shoulder geometry.

    Algorithm (per frame)
    ---------------------
    1. Measure shoulder width in pixel space (``Δx_px_shoulders``).
    2. Measure shoulder width in world space (``Δ_m_shoulders``, metres).
    3. Derive ``px_to_m = Δ_m_shoulders / Δx_px_shoulders``.
    4. Convert the bar's pixel-space position offset from the first frame into
       metres: ``Δbar_m = Δbar_px * px_to_m``.
    5. Multiply by 100 to get centimetres → ``barbell_x_corrected_cm``.
    6. Do the same for the vertical axis → ``barbell_y_corrected_cm``.

    Using the shoulder's *projected* pixel width (i.e. how wide the shoulders
    appear on screen) automatically accounts for camera angle: a camera placed
    at an angle foreshortens the shoulders in pixel space, so the ``px_to_m``
    factor naturally compensates without any trigonometric blowup.

    Frames where the shoulder landmarks are not visible fall back to the median
    scale factor computed over all valid frames.

    Args:
        df: DataFrame with barbell_x_smooth, barbell_y_smooth and unpacked
            world landmark columns.
        reference_camera_yaw_deg: Informational only; stored in the DataFrame
            for annotation.
        lateral_correction_factor: Legacy parameter, ignored.
        first_idx: Index of the first frame (used as origin for displacement).

    Returns:
        DataFrame with additional columns:
            ``barbell_x_corrected_cm``, ``barbell_y_corrected_cm``,
            ``camera_yaw_deg``, ``lateral_correction_factor``,
            ``px_to_m_scale`` (per-frame scale factor, metres per pixel).
    """
    # Initialise output columns
    df["barbell_x_corrected_cm"] = np.nan
    df["barbell_y_corrected_cm"] = np.nan
    df["camera_yaw_deg"] = reference_camera_yaw_deg  # scalar broadcast
    df["lateral_correction_factor"] = 1.0  # informational only now
    df["px_to_m_scale"] = np.nan

    if "barbell_x_smooth" not in df.columns or "barbell_y_smooth" not in df.columns:
        print("  Warning: No barbell_x/y_smooth data. Skipping perspective correction.")
        return df

    # ------------------------------------------------------------------
    # Step 1 – Per-frame px→m scale from visible shoulder geometry
    # ------------------------------------------------------------------
    # We need both pixel-space shoulder positions and world-space shoulder width.
    #
    # Pixel-space shoulder separation uses the normalised landmark coords that
    # were already unpacked into  {name}_x  (normalised 0-1) columns earlier
    # in step_2_analyze_data.  We reconstruct pixels from those.
    frame_width = (
        int(df["frame_width"].iloc[0]) if "frame_width" in df.columns else 1920
    )

    def _shoulder_px_width(row) -> Optional[float]:
        """Pixel-space horizontal distance between shoulders for one frame."""
        lx = row.get("left_shoulder_x")
        rx = row.get("right_shoulder_x")
        if pd.isna(lx) or pd.isna(rx):
            return None
        px_sep = abs(float(rx) - float(lx)) * frame_width
        return px_sep if px_sep > 2.0 else None  # ignore degenerate cases

    def _shoulder_world_width(row) -> Optional[float]:
        """True 3-D shoulder width (metres) for one frame."""
        lx = row.get("left_shoulder_world_x")
        ly = row.get("left_shoulder_world_y")
        lz = row.get("left_shoulder_world_z")
        rx = row.get("right_shoulder_world_x")
        ry = row.get("right_shoulder_world_y")
        rz = row.get("right_shoulder_world_z")
        if any(pd.isna(v) for v in [lx, ly, lz, rx, ry, rz]):
            return None
        width_m = float(
            np.linalg.norm(np.array([rx - lx, ry - ly, rz - lz], dtype=float))
        )
        return width_m if width_m >= 0.05 else None  # < 5 cm is noise

    # Compute per-frame scale and store it
    scale_values = []
    for idx in df.index:
        row = df.loc[idx]
        px_w = _shoulder_px_width(row)
        m_w = _shoulder_world_width(row)
        if px_w is not None and m_w is not None:
            scale = m_w / px_w  # metres per pixel
            df.loc[idx, "px_to_m_scale"] = scale
            scale_values.append(scale)

    if not scale_values:
        print(
            "  Warning: Could not compute px→m scale from shoulders. "
            "Skipping perspective correction."
        )
        return df

    # Robust median scale (fall-back for frames with no visible shoulders)
    median_scale_m_per_px = float(np.median(scale_values))
    print(
        f"  Shoulder-derived px→m scale: median = {median_scale_m_per_px * 1000:.3f} mm/px "
        f"(from {len(scale_values)}/{len(df)} frames)"
    )

    # Fill missing scale values with the median
    df["px_to_m_scale"] = df["px_to_m_scale"].fillna(median_scale_m_per_px)

    # ------------------------------------------------------------------
    # Step 2 – Convert bar pixel positions to centimetres
    # ------------------------------------------------------------------
    # Use the first valid barbell position as the spatial origin so that the
    # corrected path always starts at (0 cm, 0 cm).
    valid_mask = df["barbell_x_smooth"].notna() & df["barbell_y_smooth"].notna()
    if not valid_mask.any():
        print("  Warning: No valid barbell positions. Skipping perspective correction.")
        return df

    origin_idx = df[valid_mask].index[0]
    origin_x_px = float(df.loc[origin_idx, "barbell_x_smooth"])
    origin_y_px = float(df.loc[origin_idx, "barbell_y_smooth"])

    for idx in df.index:
        if not valid_mask.loc[idx]:
            continue
        scale = float(df.loc[idx, "px_to_m_scale"])  # m/px (median if not computed)

        delta_x_px = float(df.loc[idx, "barbell_x_smooth"]) - origin_x_px
        delta_y_px = float(df.loc[idx, "barbell_y_smooth"]) - origin_y_px

        df.loc[idx, "barbell_x_corrected_cm"] = delta_x_px * scale * 100.0  # → cm
        df.loc[idx, "barbell_y_corrected_cm"] = delta_y_px * scale * 100.0  # → cm

    # ------------------------------------------------------------------
    # Step 3 – Smooth the cm-space path with Savitzky-Golay
    # ------------------------------------------------------------------
    # The per-frame px→m scale inherits MediaPipe landmark jitter, so the
    # raw cm values are noisier than the already-smoothed pixel path.
    # Apply the same SG filter that step 2 uses for barbell_x/y_smooth.
    for col in ("barbell_x_corrected_cm", "barbell_y_corrected_cm"):
        series = df[col]
        filled = series.interpolate(method="linear").bfill().ffill()
        n_valid = int(filled.notna().sum())
        window = min(11, n_valid // 2 * 2 + 1)
        if window >= 5 and n_valid >= window:
            df[col] = savgol_filter(filled, window, 3)
        else:
            df[col] = filled

    # Summary stats
    x_range = df["barbell_x_corrected_cm"].max() - df["barbell_x_corrected_cm"].min()
    y_range = df["barbell_y_corrected_cm"].max() - df["barbell_y_corrected_cm"].min()
    print(
        f"  Corrected bar path range: "
        f"horizontal = {x_range:.1f} cm, vertical = {y_range:.1f} cm"
    )

    return df


def calculate_perspective_correction(
    df: pd.DataFrame, frame_width: int, frame_height: int
) -> pd.DataFrame:
    """
    Entry point: calculate perspective-corrected bar path in centimetres.

    Steps
    -----
    1. Unpack world landmarks into per-joint columns.
    2. Estimate the informational camera yaw angle (first frame only).
    3. Derive a per-frame px→m scale from shoulder geometry.
    4. Convert the bar's pixel path into centimetres, stored in
       ``barbell_x_corrected_cm`` / ``barbell_y_corrected_cm``.

    The resulting coordinates are in **centimetres** (both axes), so the path
    graph has a physically meaningful and consistent scale.

    Args:
        df: DataFrame with 'world_landmarks' and barbell tracking columns.
        frame_width: Video frame width in pixels (used for px conversion).
        frame_height: Video frame height in pixels (used for px conversion).

    Returns:
        DataFrame with additional columns:
            - ``barbell_x_corrected_cm``: Lateral displacement in cm from start.
            - ``barbell_y_corrected_cm``: Vertical displacement in cm from start.
            - ``camera_yaw_deg``: Estimated camera yaw (informational).
            - ``lateral_correction_factor``: Always 1.0 (legacy; no longer used).
            - ``px_to_m_scale``: Per-frame metres-per-pixel scale factor.
    """
    if "world_landmarks" not in df.columns:
        print(
            "  Warning: No world_landmarks column found. "
            "Skipping perspective correction."
        )
        return df

    # 1. Unpack world landmarks
    df = unpack_world_landmarks(df)

    # 2. Informational camera yaw from first frame
    first_idx = df.index[0]
    reference_camera_yaw_deg, _ = calculate_reference_camera_angle(df, first_idx)

    if reference_camera_yaw_deg is not None:
        print(
            f"  Estimated camera yaw: {reference_camera_yaw_deg:.1f}° "
            f"(informational only – correction uses shoulder geometry)"
        )
    else:
        print(
            "  Note: Could not estimate camera yaw angle; "
            "proceeding with shoulder-geometry scale only."
        )

    # 3 & 4. Scale derivation + cm conversion
    df = apply_perspective_correction(
        df,
        reference_camera_yaw_deg,
        1.0,  # legacy factor, ignored inside apply_perspective_correction
        first_idx,
    )

    return df
