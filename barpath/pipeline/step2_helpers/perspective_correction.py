"""
Perspective correction helpers for Step 2: Data Analysis.

This module converts the bar path from pixel space to real-world centimetres
using MediaPipe shoulder geometry as a reference ruler.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from config import (
    BARBELL_ENDCAP_WIDTH_M,
    PERSPECTIVE_IQR_MULTIPLIER,
    PERSPECTIVE_MIN_VALID_FRAMES,
    PERSPECTIVE_ORIGIN_SEARCH_FRAMES,
    PERSPECTIVE_PATH_SG_POLY,
    PERSPECTIVE_PATH_SG_WINDOW,
    PERSPECTIVE_SCALE_SG_POLY,
    PERSPECTIVE_SCALE_SG_WINDOW,
    PERSPECTIVE_SIDE_ANGLE_THRESHOLD_DEG,
    PERSPECTIVE_YAW_SG_POLY,
    PERSPECTIVE_YAW_SG_WINDOW,
)
from scipy.signal import savgol_filter


def _make_odd(n: int) -> int:
    """Return *n* if odd, else *n - 1* (SG window must be odd and >= poly+2)."""
    return n if n % 2 == 1 else n - 1


def _safe_get(df: pd.DataFrame, idx, col: str) -> Optional[float]:
    """Return ``df.loc[idx, col]`` as a Python float, or ``None`` on any error."""
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
    Unpack the ``world_landmarks`` dict column into per-joint
    ``{name}_world_x/y/z`` columns for shoulders and hips.

    Args:
        df: DataFrame with a ``world_landmarks`` column whose values are
            dicts mapping joint names to ``[x, y, z, visibility]`` lists.

    Returns:
        DataFrame with additional columns for ``left_shoulder``,
        ``right_shoulder``, ``left_hip``, and ``right_hip``.
        All four joints are needed: shoulders for the angled-view scale,
        hips for the side-on vertical-reference scale.
    """
    LANDMARKS = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    for name in LANDMARKS:
        col = df["world_landmarks"].apply(
            lambda x, _n=name: x.get(_n) if isinstance(x, dict) else None
        )
        df[f"{name}_world_x"] = col.apply(
            lambda x: x[0] if (x is not None and len(x) >= 4 and x[3] > 0.1) else np.nan
        )
        df[f"{name}_world_y"] = col.apply(
            lambda x: x[1] if (x is not None and len(x) >= 4 and x[3] > 0.1) else np.nan
        )
        df[f"{name}_world_z"] = col.apply(
            lambda x: x[2] if (x is not None and len(x) >= 4 and x[3] > 0.1) else np.nan
        )
    return df


def calculate_reference_camera_angle(
    df: pd.DataFrame, first_idx
) -> Tuple[Optional[float], pd.Series]:
    """
    Estimate the camera yaw angle from shoulder world landmarks across all frames.
    Returns both the reference yaw (from first valid frame) and a per-frame yaw series.

    Yaw is the angle between the shoulder line and the camera's depth (Z)
    axis, computed from the world-space X and Z offsets of the two shoulders.

    Args:
        df: DataFrame with unpacked world landmark columns.
        first_idx: Index of the first (reference) frame.

    Returns:
        ``(reference_camera_yaw_deg, yaw_series)`` – reference yaw from the first
        valid frame, and a smoothed per-frame yaw series for continuous tracking.
    """
    yaw_series = pd.Series(np.nan, index=df.index, dtype=float)

    for idx in df.index:
        try:
            l_sh_x = _safe_get(df, idx, "left_shoulder_world_x")
            l_sh_z = _safe_get(df, idx, "left_shoulder_world_z")
            r_sh_x = _safe_get(df, idx, "right_shoulder_world_x")
            r_sh_z = _safe_get(df, idx, "right_shoulder_world_z")

            if (
                l_sh_x is not None
                and l_sh_z is not None
                and r_sh_x is not None
                and r_sh_z is not None
            ):
                dx = float(r_sh_x) - float(l_sh_x)
                dz = float(r_sh_z) - float(l_sh_z)
                if abs(dz) > 1e-6:
                    camera_yaw_rad = np.arctan2(abs(dx), abs(dz))
                    yaw_series.loc[idx] = float(np.degrees(camera_yaw_rad))
        except Exception:
            pass

    yaw_series = _smooth_yaw_series(yaw_series)

    reference_camera_yaw_deg: Optional[float] = None
    valid_yaw = yaw_series.dropna()
    if len(valid_yaw) > 0:
        reference_camera_yaw_deg = float(valid_yaw.iloc[0])

    return reference_camera_yaw_deg, yaw_series


def _smooth_yaw_series(yaw_series: pd.Series) -> pd.Series:
    """
    Smooth the per-frame yaw series using Savitzky-Golay filtering.

    Args:
        yaw_series: Per-frame camera yaw in degrees.

    Returns:
        Smoothed yaw series with the same index.
    """
    valid = yaw_series.dropna()
    if len(valid) < 5:
        return yaw_series

    interpolated = yaw_series.interpolate(method="linear").bfill().ffill()
    n_valid = int(interpolated.notna().sum())

    win = _make_odd(min(PERSPECTIVE_YAW_SG_WINDOW, n_valid))
    win = max(win, PERSPECTIVE_YAW_SG_POLY + 2)

    if n_valid >= win and win > PERSPECTIVE_YAW_SG_POLY:
        smoothed = savgol_filter(
            interpolated.values.astype(float), win, PERSPECTIVE_YAW_SG_POLY
        )
        yaw_series = pd.Series(smoothed, index=yaw_series.index)

    return yaw_series


def _robust_smooth_scale(raw_scale: pd.Series) -> pd.Series:
    """
    Clean and smooth a per-frame px->m scale series.

    Steps
    -----
    1. IQR outlier rejection: frames outside the Tukey fence
       ``[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`` are set to NaN.
    2. Linear interpolation across the resulting gaps; bfill/ffill at
       boundaries so the series has no NaN after this step.
    3. Savitzky-Golay smoothing with a wide window to suppress MediaPipe's
       high-frequency landmark jitter.

    Parameters
    ----------
    raw_scale:
        Per-frame scale values in metres-per-pixel.  NaN where the scale
        could not be computed (e.g. shoulders not visible).

    Returns
    -------
    pd.Series
        Cleaned, interpolated, and smoothed scale with the same index as
        the input.
    """
    valid = raw_scale.dropna()
    if len(valid) == 0:
        return raw_scale

    # Step 1 -- IQR outlier rejection
    q1 = float(valid.quantile(0.25))
    q3 = float(valid.quantile(0.75))
    iqr = q3 - q1

    if iqr > 0:
        lo = q1 - PERSPECTIVE_IQR_MULTIPLIER * iqr
        hi = q3 + PERSPECTIVE_IQR_MULTIPLIER * iqr
        outlier_mask = raw_scale.notna() & ((raw_scale < lo) | (raw_scale > hi))
        n_out = int(outlier_mask.sum())
        if n_out > 0:
            print(
                f"  Scale outlier rejection: removed {n_out} frames "
                f"(fence [{lo * 1000:.3f}, {hi * 1000:.3f}] mm/px)"
            )
        cleaned = raw_scale.copy()
        cleaned[outlier_mask] = np.nan
    else:
        # All valid values identical -- no outliers possible.
        cleaned = raw_scale.copy()

    # Step 2 -- Interpolate across gaps
    interpolated = cleaned.interpolate(method="linear").bfill().ffill()

    n_valid = int(interpolated.notna().sum())
    if n_valid < 5:
        # Too few points to smooth meaningfully.
        return interpolated

    # Step 3 -- Savitzky-Golay smoothing
    win = _make_odd(min(PERSPECTIVE_SCALE_SG_WINDOW, n_valid))
    win = max(win, PERSPECTIVE_SCALE_SG_POLY + 2)
    if win % 2 == 0:
        win -= 1

    if n_valid >= win and win > PERSPECTIVE_SCALE_SG_POLY:
        smoothed = savgol_filter(
            interpolated.values.astype(float), win, PERSPECTIVE_SCALE_SG_POLY
        )
        # Clamp to a physically plausible range so SG ringing at boundaries
        # cannot produce negative or absurdly large scale values.
        med = float(np.median(valid.to_numpy(dtype=float)))
        smoothed = np.clip(smoothed, med * 0.3, med * 3.0)
        return pd.Series(smoothed, index=raw_scale.index)

    return interpolated


def _hip_shoulder_vertical_scale(raw_scale: pd.Series) -> pd.Series:
    """
    Identical pipeline to :func:`_robust_smooth_scale` – just re-uses it.

    Kept as a named alias so call-sites are self-documenting.
    """
    return _robust_smooth_scale(raw_scale)


def _calculate_barbell_endcap_scale(
    df: pd.DataFrame, frame_width: int
) -> Tuple[pd.Series, int]:
    """
    Calculate px->m scale using the barbell endcap width.

    Path C: Uses the known real-world width of barbell endcaps (50mm)
    to compute scale from the detected barbell bounding box width.

    Args:
        df: DataFrame with barbell_box column
        frame_width: Video frame width in pixels

    Returns:
        Tuple of (raw_scale_series, valid_frame_count)
    """
    raw_scale_series = pd.Series(np.nan, index=df.index, dtype=float)
    valid_count = 0

    for idx in df.index:
        box = df.loc[idx, "barbell_box"]
        if box is None or (isinstance(box, float) and pd.isna(box)):
            continue
        try:
            if isinstance(box, str):
                values = [float(v.strip()) for v in box.split(",")]
                if len(values) != 4:
                    continue
                x1, _, x2, _ = values
            elif isinstance(box, (list, tuple)):
                if len(box) != 4:
                    continue
                x1, _, x2, _ = box[0], box[1], box[2], box[3]
            else:
                continue

            box_width_px = abs(x2 - x1)
            if box_width_px > 20:
                scale = BARBELL_ENDCAP_WIDTH_M / box_width_px
                raw_scale_series.loc[idx] = scale
                valid_count += 1
        except Exception:
            continue

    return raw_scale_series, valid_count


def _find_robust_origin(
    df: pd.DataFrame, valid_mask: pd.Series
) -> Tuple[Optional[int], float, float]:
    """
    Find a robust origin frame for barbell position reference.

    Searches through the first N frames to find one with reliable
    barbell tracking (not just first valid frame).

    Args:
        df: DataFrame with barbell_x_smooth, barbell_y_smooth columns
        valid_mask: Boolean mask of valid barbell positions

    Returns:
        Tuple of (origin_frame_index, origin_x_px, origin_y_px) or
        (None, nan, nan) if no suitable origin found
    """
    valid_indices = df[valid_mask].index.tolist()

    if len(valid_indices) == 0:
        return None, np.nan, np.nan

    search_frames = min(PERSPECTIVE_ORIGIN_SEARCH_FRAMES, len(valid_indices))

    for i in range(search_frames):
        idx = valid_indices[i]
        x = df.loc[idx, "barbell_x_smooth"]
        y = df.loc[idx, "barbell_y_smooth"]

        if pd.notna(x) and pd.notna(y):
            return int(idx), float(x), float(y)

    first_valid = valid_indices[0]
    return (
        int(first_valid),
        float(df.loc[first_valid, "barbell_x_smooth"]),
        float(df.loc[first_valid, "barbell_y_smooth"]),
    )


# ---------------------------------------------------------------------------
# Core correction
# ---------------------------------------------------------------------------


def apply_perspective_correction(
    df: pd.DataFrame,
    reference_camera_yaw_deg: Optional[float],
    yaw_series: pd.Series,
    lateral_correction_factor: float,  # kept for API compat, ignored
    first_idx,
    is_side_on: bool = False,
) -> pd.DataFrame:
    """
    Convert the bar path from pixel space to real-world centimetres.

    Algorithm
    ---------
    Two scale-derivation paths are tried in order:

    **Path A – shoulder-width scale** (used for angled-view shots):

    1. For each frame compute:

       - ``pixel_shoulder_width``  = horizontal pixel distance between the
         two shoulder landmarks (normalised coords * frame_width).
       - ``world_shoulder_width``  = Euclidean 3-D distance between the two
         shoulder world landmarks (metres, from MediaPipe).
       - ``raw_scale``  = world_shoulder_width / pixel_shoulder_width  (m/px).

    **Path B – hip-to-shoulder vertical scale** (used for side-on shots, or
    as a fallback when Path A yields too few valid frames):

    1. For each frame compute the *vertical* pixel distance between the
       shoulder midpoint and the hip midpoint, and compare it to the
       world-space vertical distance between the same points.  Vertical
       distances are not foreshortened by a horizontal-axis yaw rotation,
       so this is reliable from any camera angle.

    Whichever path runs, the resulting raw scale series is:

    2. Cleaned with :func:`_robust_smooth_scale` (IQR rejection +
       interpolation + SG smoothing) and stored in ``px_to_m_scale``.

    3. Reduced to a **single stable scalar**:

       - Preferred: median of the first half of the Pull phase (phase 0),
         where the athlete is upright and landmarks are most reliable.
       - Fallback: median of the entire smoothed scale series.

       This single scalar is used for the whole lift.  Using a per-frame
       rising scale on cumulative pixel displacement would inflate the path
       during pull-under when the pixel shoulder width shrinks due to
       shoulder rotation and partial occlusion.

    4. Convert bar pixel displacements to centimetres::

           barbell_x_corrected_cm = (barbell_x_smooth - origin_x_px) * scalar * 100
           barbell_y_corrected_cm = (barbell_y_smooth - origin_y_px) * scalar * 100

       The origin is the first valid barbell position.  Both axes use the
       same scalar because yaw is a rotation about the vertical axis and
       does not foreshorten the vertical axis.

    5. Apply a final wide Savitzky-Golay pass on the cm-space coordinates
       to absorb any residual jitter.

    Args:
        df: DataFrame with ``barbell_x_smooth``, ``barbell_y_smooth``, and
            unpacked landmark / world-landmark columns.
        reference_camera_yaw_deg: Informational; stored in the output.
        lateral_correction_factor: Legacy parameter -- ignored.
        first_idx: Index of the first frame (used as displacement origin).
        is_side_on: When True, Path B (hip-shoulder vertical) is used
            instead of Path A (shoulder width).

    Returns:
        DataFrame with additional columns:
            ``barbell_x_corrected_cm``, ``barbell_y_corrected_cm``,
            ``camera_yaw_deg``, ``lateral_correction_factor``,
            ``px_to_m_scale``, ``scale_method``.
    """
    # Initialise output columns.
    df["barbell_x_corrected_cm"] = np.nan
    df["barbell_y_corrected_cm"] = np.nan
    df["camera_yaw_deg"] = reference_camera_yaw_deg
    df["lateral_correction_factor"] = 1.0
    df["px_to_m_scale"] = np.nan
    df["scale_method"] = "none"

    if "barbell_x_smooth" not in df.columns or "barbell_y_smooth" not in df.columns:
        print("  Warning: No barbell_x/y_smooth data. Skipping perspective correction.")
        return df

    frame_width = (
        int(df["frame_width"].iloc[0]) if "frame_width" in df.columns else 1920
    )
    frame_height = (
        int(df["frame_height"].iloc[0]) if "frame_height" in df.columns else 1080
    )

    # ------------------------------------------------------------------
    # Step 1 -- Per-frame raw px->m scale
    #
    # Path A: shoulder horizontal width  (reliable for angled-view shots)
    # Path B: hip-to-shoulder vertical distance  (reliable for all angles,
    #         and the only option when the camera is near-perpendicular and
    #         the pixel shoulder width collapses to near-zero)
    # ------------------------------------------------------------------

    def _shoulder_px_width(row) -> Optional[float]:
        """Horizontal pixel distance between shoulders for one frame."""
        lx = row.get("left_shoulder_x")
        rx = row.get("right_shoulder_x")
        if pd.isna(lx) or pd.isna(rx):
            return None
        # Normalised coords (0-1) * frame_width -> pixels.
        px_sep = abs(float(rx) - float(lx)) * frame_width
        return px_sep if px_sep > 2.0 else None  # ignore degenerate near-zero cases

    def _shoulder_world_width(row) -> Optional[float]:
        """True 3-D Euclidean shoulder width in metres for one frame."""
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

    def _hip_shoulder_px_vert(row) -> Optional[float]:
        """
        Vertical pixel distance between the shoulder midpoint and the hip
        midpoint for one frame.  Normalised Y coords * frame_height -> px.
        """
        ls_y = row.get("left_shoulder_y")
        rs_y = row.get("right_shoulder_y")
        lh_y = row.get("left_hip_y")
        rh_y = row.get("right_hip_y")
        if any(pd.isna(v) for v in [ls_y, rs_y, lh_y, rh_y]):
            return None
        sh_mid_y = (float(ls_y) + float(rs_y)) / 2.0 * frame_height
        hip_mid_y = (float(lh_y) + float(rh_y)) / 2.0 * frame_height
        dist = abs(sh_mid_y - hip_mid_y)
        return dist if dist > 5.0 else None  # < 5 px is noise

    def _hip_shoulder_world_vert(row) -> Optional[float]:
        """
        World-space vertical (Y-axis) distance between the shoulder midpoint
        and the hip midpoint in metres.  MediaPipe world Y is vertical and is
        not foreshortened by a horizontal yaw rotation, so this is reliable
        from any camera angle.
        """
        ls_wy = row.get("left_shoulder_world_y")
        rs_wy = row.get("right_shoulder_world_y")
        lh_wy = row.get("left_hip_world_y")
        rh_wy = row.get("right_hip_world_y")
        if any(pd.isna(v) for v in [ls_wy, rs_wy, lh_wy, rh_wy]):
            return None
        sh_mid_wy = (float(ls_wy) + float(rs_wy)) / 2.0
        hip_mid_wy = (float(lh_wy) + float(rh_wy)) / 2.0
        dist = abs(sh_mid_wy - hip_mid_wy)
        return dist if dist >= 0.05 else None  # < 5 cm is noise

    # Build the raw scale series using whichever path is appropriate.
    raw_scale_series = pd.Series(np.nan, index=df.index, dtype=float)
    n_raw = 0
    scale_method_label = "none"

    if not is_side_on:
        # Path A: shoulder horizontal width
        for idx in df.index:
            row = df.loc[idx]
            px_w = _shoulder_px_width(row)
            m_w = _shoulder_world_width(row)
            if px_w is not None and m_w is not None:
                raw_scale_series.loc[idx] = m_w / px_w  # metres per pixel
                n_raw += 1
        scale_method_label = "shoulder_width"

    if is_side_on or n_raw < PERSPECTIVE_MIN_VALID_FRAMES:
        # Path B: hip-to-shoulder vertical distance.
        # Used when explicitly side-on, or when Path A yielded too few frames.
        if n_raw < PERSPECTIVE_MIN_VALID_FRAMES and not is_side_on:
            print(
                f"  Path A (shoulder width) only gave {n_raw} valid frames; "
                "falling back to hip-shoulder vertical scale."
            )
        raw_scale_series_b = pd.Series(np.nan, index=df.index, dtype=float)
        n_raw_b = 0
        for idx in df.index:
            row = df.loc[idx]
            px_v = _hip_shoulder_px_vert(row)
            m_v = _hip_shoulder_world_vert(row)
            if px_v is not None and m_v is not None:
                raw_scale_series_b.loc[idx] = m_v / px_v  # metres per pixel
                n_raw_b += 1
        if n_raw_b > 0:
            raw_scale_series = raw_scale_series_b
            n_raw = n_raw_b
            scale_method_label = "hip_shoulder_vertical"
        elif n_raw == 0:
            print(
                "  Warning: Could not compute px->m scale from shoulder width "
                "or hip-shoulder vertical distance. Trying barbell endcap width..."
            )
            raw_scale_series_c, n_raw_c = _calculate_barbell_endcap_scale(
                df, frame_width
            )
            if n_raw_c >= PERSPECTIVE_MIN_VALID_FRAMES:
                raw_scale_series = raw_scale_series_c
                n_raw = n_raw_c
                scale_method_label = "barbell_endcap"
                print(f"  Using barbell endcap scale: {n_raw_c} valid frames")
            else:
                print(
                    f"  Warning: Could not compute px->m scale. "
                    f"Barbell endcap: {n_raw_c} frames (need {PERSPECTIVE_MIN_VALID_FRAMES}). "
                    "Skipping perspective correction."
                )
                return df

    df["scale_method"] = scale_method_label
    print(f"  Scale method: {scale_method_label} ({n_raw} valid frames)")

    # ------------------------------------------------------------------
    # Step 2 -- Outlier rejection + temporal smoothing of the scale series
    # ------------------------------------------------------------------
    smooth_scale_series = _robust_smooth_scale(raw_scale_series)
    df["px_to_m_scale"] = smooth_scale_series

    # ------------------------------------------------------------------
    # Step 3 -- Derive a single stable scalar
    # ------------------------------------------------------------------
    # The per-frame scale can drift during the pull-under (shoulder occlusion
    # for Path A; minor posture changes for Path B).  Using a single scalar
    # from a stable early window eliminates this artifact while preserving
    # the true path shape -- only the axis units change from pixels to cm.
    stable_scale_scalar: float

    if "bar_phase" in df.columns:
        pull_frames = df[df["bar_phase"] == 0]
        if len(pull_frames) >= 4:
            # First half of Pull: athlete is upright and landmarks are most visible.
            first_half = pull_frames.iloc[: max(4, len(pull_frames) // 2)]
            valid_pull_scale = smooth_scale_series.loc[first_half.index].dropna()
            if len(valid_pull_scale) >= 2:
                stable_scale_scalar = float(valid_pull_scale.median())
                print(
                    f"  Stable px->m scale from first-half Pull phase "
                    f"({len(valid_pull_scale)} frames): "
                    f"{stable_scale_scalar * 1000:.3f} mm/px"
                )
            else:
                stable_scale_scalar = float(smooth_scale_series.median())
                print(
                    f"  Stable px->m scale (full median fallback): "
                    f"{stable_scale_scalar * 1000:.3f} mm/px"
                )
        else:
            stable_scale_scalar = float(smooth_scale_series.median())
            print(
                f"  Stable px->m scale (full median, insufficient Pull frames): "
                f"{stable_scale_scalar * 1000:.3f} mm/px"
            )
    else:
        stable_scale_scalar = float(smooth_scale_series.median())
        print(
            f"  Stable px->m scale (full median, no phase data): "
            f"{stable_scale_scalar * 1000:.3f} mm/px"
        )

    # ------------------------------------------------------------------
    # Step 4 -- Convert bar pixel positions to centimetres
    # ------------------------------------------------------------------
    valid_mask = df["barbell_x_smooth"].notna() & df["barbell_y_smooth"].notna()
    if not valid_mask.any():
        print("  Warning: No valid barbell positions. Skipping perspective correction.")
        return df

    origin_idx, origin_x_px, origin_y_px = _find_robust_origin(df, valid_mask)
    if origin_idx is None:
        print(
            "  Warning: Could not find reliable origin frame. Skipping perspective correction."
        )
        return df

    print(f"  Using robust origin frame {origin_idx} for barbell position reference")

    delta_x_px = df.loc[valid_mask, "barbell_x_smooth"] - origin_x_px
    delta_y_px = df.loc[valid_mask, "barbell_y_smooth"] - origin_y_px

    df.loc[valid_mask, "barbell_x_corrected_cm"] = (
        delta_x_px * stable_scale_scalar * 100.0
    )
    df.loc[valid_mask, "barbell_y_corrected_cm"] = (
        delta_y_px * stable_scale_scalar * 100.0
    )

    # ------------------------------------------------------------------
    # Step 5 -- Final SG smoothing of the cm-space path
    # ------------------------------------------------------------------
    for col in ("barbell_x_corrected_cm", "barbell_y_corrected_cm"):
        series = pd.Series(df[col])
        filled = series.interpolate(method="linear").bfill().ffill()
        n_valid = int(filled.notna().sum())
        win = _make_odd(min(PERSPECTIVE_PATH_SG_WINDOW, n_valid))
        win = max(win, PERSPECTIVE_PATH_SG_POLY + 2)
        if win % 2 == 0:
            win -= 1
        if n_valid >= win and win > PERSPECTIVE_PATH_SG_POLY:
            df[col] = savgol_filter(
                filled.values.astype(float), win, PERSPECTIVE_PATH_SG_POLY
            )
        else:
            df[col] = filled

    x_range = float(df["barbell_x_corrected_cm"].max()) - float(
        df["barbell_x_corrected_cm"].min()
    )
    y_range = float(df["barbell_y_corrected_cm"].max()) - float(
        df["barbell_y_corrected_cm"].min()
    )
    print(
        f"  Corrected bar path range: "
        f"horizontal = {x_range:.1f} cm, vertical = {y_range:.1f} cm"
    )

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def calculate_perspective_correction(
    df: pd.DataFrame, frame_width: int, frame_height: int
) -> pd.DataFrame:
    """
    Entry point: calculate perspective-corrected bar path in centimetres.

    Steps
    -----
    1. Unpack world landmarks into per-joint ``_world_x/y/z`` columns.
    2. Estimate the camera yaw angle from the first frame.
    3. Choose scale method:

       - |yaw| >= 10 deg (angled view): use shoulder horizontal width as
         the px->m ruler (Path A).
       - |yaw| < 10 deg (side-on view): use hip-to-shoulder vertical
         distance as the px->m ruler (Path B).  The shoulder pixel width
         collapses to near-zero from a side-on camera, but the vertical
         distance is not foreshortened by a horizontal yaw and is perfectly
         reliable.

    4. Derive a stable px->m scalar (outlier-rejected, smoothed, then
       median of the early Pull phase).
    5. Convert bar pixel displacement to centimetres with that scalar.

    All lifts produce valid ``barbell_x/y_corrected_cm`` columns so that
    the superimposed comparison graphs share a common unit (cm).

    Args:
        df: DataFrame with ``world_landmarks`` and barbell tracking columns.
        frame_width:  Video frame width  in pixels.
        frame_height: Video frame height in pixels.

    Returns:
        DataFrame with additional columns:

        ``barbell_x_corrected_cm``
            Horizontal displacement in cm from the first frame.
        ``barbell_y_corrected_cm``
            Vertical displacement in cm from the first frame.
        ``camera_yaw_deg``
            Estimated camera yaw in degrees (informational).
        ``lateral_correction_factor``
            Always 1.0 (legacy placeholder, no longer used).
        ``px_to_m_scale``
            Per-frame smoothed metres-per-pixel scale factor.
        ``scale_method``
            String tag: ``"shoulder_width"`` or ``"hip_shoulder_vertical"``.
    """
    if "world_landmarks" not in df.columns:
        print(
            "  Warning: No world_landmarks column found. "
            "Skipping perspective correction."
        )
        return df

    # 1. Unpack world landmarks
    df = unpack_world_landmarks(df)

    # 2. Estimate camera yaw from multiple frames (continuous tracking)
    first_idx = df.index[0]
    reference_camera_yaw_deg, yaw_series = calculate_reference_camera_angle(
        df, first_idx
    )

    if reference_camera_yaw_deg is not None:
        print(
            f"  Estimated camera yaw (reference): {reference_camera_yaw_deg:.1f} degrees"
        )
    else:
        print(
            "  Note: Could not estimate camera yaw angle; "
            "skipping perspective correction."
        )
        df["camera_yaw_deg"] = np.nan
        df["lateral_correction_factor"] = 1.0
        df["px_to_m_scale"] = np.nan
        df["barbell_x_corrected_cm"] = np.nan
        df["barbell_y_corrected_cm"] = np.nan
        df["scale_method"] = "none"
        return df

    df["camera_yaw_deg"] = yaw_series

    # 3. Choose scale method based on camera yaw (use median for stability)
    valid_yaw = yaw_series.dropna()
    median_yaw = (
        float(valid_yaw.median()) if len(valid_yaw) > 0 else reference_camera_yaw_deg
    )
    is_side_on = abs(median_yaw) < PERSPECTIVE_SIDE_ANGLE_THRESHOLD_DEG
    if is_side_on:
        print(
            f"  Camera yaw median {median_yaw:.1f} deg is within the "
            f"+/-{PERSPECTIVE_SIDE_ANGLE_THRESHOLD_DEG} deg side-angle threshold. "
            "Using hip-to-shoulder vertical distance as scale reference "
            "(shoulder pixel width is degenerate from a side-on view)."
        )
    else:
        print(
            f"  Camera yaw median {median_yaw:.1f} deg -- using "
            "shoulder width as scale reference."
        )

    # 4 & 5. Scale derivation + cm conversion
    df = apply_perspective_correction(
        df,
        reference_camera_yaw_deg,
        yaw_series,
        1.0,  # legacy factor, ignored inside apply_perspective_correction
        first_idx,
        is_side_on=is_side_on,
    )

    return df
