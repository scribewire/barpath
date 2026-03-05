import argparse
import gc
import os
import pickle
from typing import Optional, cast

import numpy as np
import pandas as pd
from pandas import Series
from scipy.signal import savgol_filter
from step2_helpers import calculate_perspective_correction
from utils import calculate_angle, calculate_lifter_angle

# ---------------------------------------------------------------------------
# Helper: safe Savitzky-Golay smoother
# ---------------------------------------------------------------------------


def _savgol_smooth(series: pd.Series, window: int = 11, poly: int = 3) -> pd.Series:  # type: ignore[type-arg]
    """
    Apply Savitzky-Golay smoothing to *series* after forward/back-filling NaNs.

    The window is automatically clamped to be odd and ≤ len(series).  Returns
    the original series unchanged when there are too few points to smooth.
    """
    filled = series.interpolate(method="linear").bfill().ffill()
    n = len(filled)
    # Clamp window: must be odd, >= poly+1, <= n
    w = min(window, n if n % 2 == 1 else n - 1)
    w = max(w, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)
    if n < w or w < poly + 1:
        return filled
    return pd.Series(savgol_filter(filled, w, poly), index=series.index)


# ---------------------------------------------------------------------------
# Pixel-to-meter conversion
# ---------------------------------------------------------------------------


def calculate_pixel_to_meter_conversion(df, endcap_width_m: float = 0.05):
    """
    Calculate pixel-to-meter conversion factor based on barbell endcap width.

    Args:
        df: DataFrame with barbell_box data
        endcap_width_m: Real-world width of barbell endcap in metres (default 0.05 m = 50 mm)

    Returns:
        float | None: Pixels-to-metres factor, or None if it cannot be calculated.
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


# ---------------------------------------------------------------------------
# Maximum specific power
# ---------------------------------------------------------------------------


def calculate_max_specific_power(df, phases):
    """
    Calculate maximum specific power between t1 (end of pull) and t3 (end of pull-under).

    Args:
        df: DataFrame with calculated kinematics
        phases: dict with 't1' and 't3' frame indices (ClassicsPhases)

    Returns:
        dict with 'max_power_px' and optionally 'max_power_real' (W/kg), or None.
    """
    if phases is None or "t1" not in phases or "t3" not in phases:
        return None

    try:
        t1 = int(phases["t1"])
        t3 = int(phases["t3"])

        if "specific_power_y_smooth" not in df.columns:
            return None

        power_segment = df.loc[t1:t3, "specific_power_y_smooth"]
        if power_segment.empty:
            return None

        max_power_px = float(power_segment.abs().max())
        if np.isnan(max_power_px):
            return None

        px_to_m = calculate_pixel_to_meter_conversion(df)
        result: dict[str, Optional[float]] = {"max_power_px": max_power_px}

        if px_to_m is not None:
            max_power_real = max_power_px * (px_to_m**2)
            result["max_power_real"] = max_power_real
            print(
                f"Maximum specific power: {max_power_px:.2f} px²/s³ = {max_power_real:.2f} W/kg"
            )
        else:
            result["max_power_real"] = None
            print(
                f"Maximum specific power: {max_power_px:.2f} px²/s³ (real-world conversion unavailable)"
            )

        return result
    except Exception as e:
        print(f"Warning: Could not calculate max specific power: {e}")
        return None


# ---------------------------------------------------------------------------
# New 3-phase detection: pull / pull-under / recovery
# ---------------------------------------------------------------------------


def _detect_three_phases(df: pd.DataFrame, fps: float) -> pd.Series:
    """
    Assign one of three kinematic bar phases to every frame.

    Phase labels (integer):
      0 – Pull         : barbell moving upward AND lifter's hips have not yet
                         begun dropping after the bar started moving.
      1 – Pull-under   : hips are actively descending under the bar.
                         Starts when hips begin to drop (after bar is rising);
                         ends when hips stop descending.
      2 – Recovery     : from when hips stop descending until bar reaches
                         peak height.

    Detection is purely kinematic, using smoothed signals already present on
    *df*:
      - ``vel_y_smooth``  : bar vertical velocity (positive = upward, px/s)
      - ``hip_y_avg``     : average hip Y pixel coordinate (larger = lower in frame)

    All three phases are guaranteed to be present.  If detection fails at any
    step the function falls back gracefully by extending the previous phase.
    """
    phase = pd.Series(0, index=df.index, dtype="int64", name="bar_phase")

    if "vel_y_smooth" not in df.columns or "hip_y_avg" not in df.columns:
        return phase

    vel = df["vel_y_smooth"].fillna(0)
    hip_y = df["hip_y_avg"]

    # ----- Step 1: find when bar first moves upward (start of Pull) ----------
    # Use a small threshold to ignore noise
    vel_threshold = max(10.0, float(vel.abs().max()) * 0.05)
    bar_moving_up = vel > vel_threshold

    if not bool(bar_moving_up.any()):
        # Bar never moves upward – keep everything as phase 0
        return phase

    pull_start_idx = int(bar_moving_up.idxmax())  # type: ignore[arg-type]

    # ----- Step 2: find when hips begin to drop AFTER the bar starts rising --
    # "Hips dropping" means hip_y_avg increasing (Y grows downward).
    # We look for the first sustained increase in hip_y after pull_start.
    hip_after_pull = hip_y.loc[pull_start_idx:]

    # Smooth hip_y so transient noise doesn't trigger a false phase boundary
    hip_smooth = _savgol_smooth(hip_after_pull, window=9, poly=3)

    # Hip velocity (positive = hips moving down in frame = dropping)
    hip_vel = hip_smooth.diff().fillna(0)

    hip_drop_threshold = (
        float(hip_smooth.std()) * 0.1 if float(hip_smooth.std()) > 0 else 0.5
    )
    hips_dropping = hip_vel > hip_drop_threshold

    pull_under_start_idx: int | None = None
    if bool(hips_dropping.any()):
        pull_under_start_idx = int(hips_dropping.idxmax())  # type: ignore[arg-type]

    # ----- Step 3: find when hips stop descending (end of Pull-under) --------
    recovery_start_idx: int | None = None
    if pull_under_start_idx is not None:
        hip_after_pu = hip_y.loc[pull_under_start_idx:]
        hip_smooth_pu = _savgol_smooth(hip_after_pu, window=9, poly=3)
        hip_vel_pu = hip_smooth_pu.diff().fillna(0)

        # Hips stop dropping when hip_vel goes negative or near zero
        hips_stopped = hip_vel_pu <= hip_drop_threshold * 0.5

        if bool(hips_stopped.any()):
            recovery_start_idx = int(hips_stopped.idxmax())  # type: ignore[arg-type]

    # ----- Assign phases ------------------------------------------------------
    # Phase 0 (Pull): pull_start_idx → pull_under_start_idx
    # Phase 1 (Pull-under): pull_under_start_idx → recovery_start_idx
    # Phase 2 (Recovery): recovery_start_idx → end

    if pull_under_start_idx is not None:
        phase.loc[df.index >= pull_under_start_idx] = 1
    if recovery_start_idx is not None:
        phase.loc[df.index >= recovery_start_idx] = 2

    # Frames before pull_start_idx stay at 0 (pre-lift / beginning of Pull)

    return phase


# ---------------------------------------------------------------------------
# Step 2 main function
# ---------------------------------------------------------------------------


def step_2_analyze_data(input_data, output_path):
    print("--- Step 2: Analyzing Data ---")

    metadata = input_data.get("metadata", {})
    df_list = input_data.get("data", [])

    lift_type = str(metadata.get("lift_type", "none")).lower()

    if not df_list:
        print("Error: No data found in pickle file.")
        return

    df = pd.DataFrame(df_list)

    # Free the raw list now that we have a DataFrame
    del df_list
    if "data" in input_data:
        del input_data["data"]
    gc.collect()

    if "frame" not in df.columns:
        print("Error: No 'frame' column in data.")
        return

    df = df.set_index("frame").sort_index()

    frame_gaps = df.index.to_series().diff()
    frame_gaps_numeric = cast(Series, pd.to_numeric(frame_gaps, errors="coerce"))
    if (frame_gaps_numeric.fillna(0) > 1).any():
        print(f"Warning: Detected {(frame_gaps_numeric > 1).sum()} gaps.")

    # --- Metadata ---
    frame_width = metadata.get("frame_width", 1920)
    frame_height = metadata.get("frame_height", 1080)
    fps = metadata.get("fps", 30.0)

    df["frame_width"] = frame_width
    df["frame_height"] = frame_height

    # -----------------------------------------------------------------------
    # Unpack raw landmark data into per-joint columns
    # -----------------------------------------------------------------------
    LANDMARKS_TO_TRACK = {
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
    }

    for name in LANDMARKS_TO_TRACK:
        df[name] = df["landmarks"].apply(
            lambda x, _n=name: x.get(_n) if isinstance(x, dict) else None
        )
        df[f"{name}_x"] = df[name].apply(
            lambda x: x[0] if (x is not None and len(x) >= 4 and x[3] > 0.1) else np.nan
        )
        df[f"{name}_y"] = df[name].apply(
            lambda x: x[1] if (x is not None and len(x) >= 4 and x[3] > 0.1) else np.nan
        )
        df[f"{name}_z"] = df[name].apply(
            lambda x: x[2] if (x is not None and len(x) >= 4 and x[3] > 0.1) else np.nan
        )
        df[f"{name}_vis"] = df[name].apply(
            lambda x: x[3] if (x is not None and len(x) >= 4) else np.nan
        )

    # -----------------------------------------------------------------------
    # Compute pixel-space joint positions (needed for angles and hip_y_avg)
    # -----------------------------------------------------------------------
    def get_pixel_pos(row, name: str) -> np.ndarray:
        x_norm = row.get(f"{name}_x")
        y_norm = row.get(f"{name}_y")
        if pd.isna(x_norm) or pd.isna(y_norm):
            return np.array([np.nan, np.nan])
        return np.array([x_norm * frame_width, y_norm * frame_height])

    # -----------------------------------------------------------------------
    # Calculate joint angles (from smoothed joint positions)
    # -----------------------------------------------------------------------
    df["left_knee_angle"] = df.apply(
        lambda row: calculate_angle(
            get_pixel_pos(row, "left_hip"),
            get_pixel_pos(row, "left_knee"),
            get_pixel_pos(row, "left_ankle"),
        ),
        axis=1,
    )

    df["right_knee_angle"] = df.apply(
        lambda row: calculate_angle(
            get_pixel_pos(row, "right_hip"),
            get_pixel_pos(row, "right_knee"),
            get_pixel_pos(row, "right_ankle"),
        ),
        axis=1,
    )

    df["left_elbow_angle"] = df.apply(
        lambda row: calculate_angle(
            get_pixel_pos(row, "left_shoulder"),
            get_pixel_pos(row, "left_elbow"),
            get_pixel_pos(row, "left_wrist"),
        ),
        axis=1,
    )

    df["right_elbow_angle"] = df.apply(
        lambda row: calculate_angle(
            get_pixel_pos(row, "right_shoulder"),
            get_pixel_pos(row, "right_elbow"),
            get_pixel_pos(row, "right_wrist"),
        ),
        axis=1,
    )

    # -----------------------------------------------------------------------
    # Lifter angle – tracked per-frame, raw MediaPipe output
    # -----------------------------------------------------------------------
    df["lifter_angle"] = df["landmarks"].apply(
        lambda x: calculate_lifter_angle(x) if isinstance(x, dict) else np.nan
    )

    # -----------------------------------------------------------------------
    # Hip average (pixel space) – used for phase detection
    # -----------------------------------------------------------------------
    df["hip_y_avg"] = df[["left_hip_y", "right_hip_y"]].mean(axis=1) * frame_height

    # -----------------------------------------------------------------------
    # Barbell stabilised coordinates
    # -----------------------------------------------------------------------
    df["total_shake_x"] = df["shake_dx"].cumsum()
    df["total_shake_y"] = df["shake_dy"].cumsum()

    if "barbell_center" in df.columns:
        df["barbell_x_raw"] = df["barbell_center"].apply(
            lambda x: x[0] if isinstance(x, (list, tuple)) else np.nan
        )
        df["barbell_y_raw"] = df["barbell_center"].apply(
            lambda x: x[1] if isinstance(x, (list, tuple)) else np.nan
        )
    else:
        print(
            "Warning: 'barbell_center' column not found. No barbell data will be processed."
        )
        df["barbell_x_raw"] = np.nan
        df["barbell_y_raw"] = np.nan

    df["barbell_x_stable"] = df["barbell_x_raw"] - df["total_shake_x"]
    df["barbell_y_stable"] = df["barbell_y_raw"] - df["total_shake_y"]

    # -----------------------------------------------------------------------
    # Smooth barbell position
    # -----------------------------------------------------------------------
    x_filled = df["barbell_x_stable"].interpolate(method="linear").bfill().ffill()
    y_filled = df["barbell_y_stable"].interpolate(method="linear").bfill().ffill()

    pos_window = min(11, len(x_filled) // 2 * 2 + 1)
    if pos_window >= 5 and len(x_filled) >= pos_window:
        print(f"Applying barbell position smoothing with window {pos_window}...")
        df["barbell_x_smooth"] = savgol_filter(x_filled, pos_window, 3)
        df["barbell_y_smooth"] = savgol_filter(y_filled, pos_window, 3)
    else:
        print(
            "Warning: Not enough data to smooth barbell position. Using unsmoothed values."
        )
        df["barbell_x_smooth"] = df["barbell_x_stable"]
        df["barbell_y_smooth"] = df["barbell_y_stable"]

    # -----------------------------------------------------------------------
    # Truncate: discard frames before bar passes knee (keep 1 s before)
    # -----------------------------------------------------------------------
    df["knee_y_avg"] = df[["left_knee_y", "right_knee_y"]].mean(axis=1) * frame_height

    if bool(df["barbell_y_smooth"].notna().any()) and bool(
        df["knee_y_avg"].notna().any()
    ):
        bar_above_knee = df["barbell_y_smooth"] < df["knee_y_avg"]
        frames_above_knee = df[bar_above_knee].index.values

        if len(frames_above_knee) > 0:
            knee_pass_frame = int(frames_above_knee[0])
            frames_before = int(fps)
            first_frame = int(df.index.values[0])
            start_frame = max(first_frame, knee_pass_frame - frames_before)

            print(
                f"Bar passes knee at frame {knee_pass_frame}. "
                f"Keeping data from frame {start_frame} onwards (1s before knee pass)."
            )
            df = df.loc[start_frame:].copy()
        else:
            print("Warning: Bar never detected above knee. Keeping all data at start.")
    else:
        print("Warning: Cannot determine knee pass frame. Keeping all data at start.")

    # -----------------------------------------------------------------------
    # Truncate at peak barbell height
    # -----------------------------------------------------------------------
    if bool(df["barbell_y_smooth"].notna().any()):
        peak_height_idx = df["barbell_y_smooth"].idxmin()
        print(
            f"Peak height detected at frame {peak_height_idx}. Truncating data after this point."
        )
        df = df.loc[:peak_height_idx].copy()
    else:
        print("Warning: No barbell Y data found. Cannot truncate at peak height.")

    # -----------------------------------------------------------------------
    # Kinematics: time, velocity, acceleration, specific power
    # -----------------------------------------------------------------------
    if df.index.is_monotonic_increasing:
        df["time_s"] = (df.index - df.index[0]) / fps
    else:
        print("Warning: Frame indices are not monotonic. Using sequential time.")
        df["time_s"] = np.arange(len(df)) / fps

    df["dt"] = df["time_s"].diff().fillna(1 / fps)

    # Velocity from smoothed barbell position (positive = upward because we invert Y)
    df["vel_y_px_s"] = (df["barbell_y_smooth"].diff() / df["dt"]) * -1

    # Smooth velocity
    vel_filled = df["vel_y_px_s"].interpolate(method="linear").fillna(0)
    vel_window = min(15, len(vel_filled) // 2 * 2 + 1)
    if vel_window >= 5:
        print(f"Applying Savitzky-Golay velocity smoothing with window {vel_window}...")
        df["vel_y_smooth"] = savgol_filter(vel_filled, vel_window, 3)
    else:
        print("Warning: Not enough data to smooth velocity.")
        df["vel_y_smooth"] = vel_filled

    # Smoothed acceleration and specific power
    df["accel_y_smooth"] = df["vel_y_smooth"].diff() / df["dt"]
    df["specific_power_y_smooth"] = df["accel_y_smooth"] * df["vel_y_smooth"]

    # -----------------------------------------------------------------------
    # Bar path phase detection
    # -----------------------------------------------------------------------
    # For clean/snatch lifts we attempt the classics-aware 3-phase mapping
    # (pull → pull-under → recovery).  For all other lift types we fall back
    # to the kinematic 3-phase detector that uses barbell velocity and hip
    # position only.
    phases = None

    if lift_type in ("clean", "snatch"):
        # Lazy import to keep the module self-contained
        from step5_helpers.classics_phase_detection import (
            identify_classics_phases,  # type: ignore
        )

        phases = identify_classics_phases(df)

    if phases is not None:
        # Map classics phase boundaries (t0-t4) to the 3 new phases:
        #   Pull        : t0 → t1   (bar off floor to hip extension / end of second pull)
        #   Pull-under  : t1 → t3   (turnover: hips rise then drop to catch)
        #   Recovery    : t3 → t4   (stand up to peak bar height)
        #
        # t1 = end of first pull (bar at knee)
        # t2 = end of second pull / hip extension peak
        # t3 = bottom of catch / lowest hip position
        # t4 = peak bar height
        #
        # We deliberately merge the old "second pull" and "third pull" into
        # "pull" (t0→t2 is the entire upward drive), then split at t2 for
        # pull-under and t3 for recovery.
        #
        # Revised mapping that matches the new phase definitions:
        #   Pull        : t0 → t2   (entire upward drive, bar moving up, hips rising)
        #   Pull-under  : t2 → t3   (hips dropping under bar)
        #   Recovery    : t3 → t4   (stand-up)
        idx = df.index
        bar_phase = pd.Series(0, index=idx, dtype="int64", name="bar_phase")

        t2 = int(phases["t2"])
        t3 = int(phases["t3"])

        bar_phase.loc[idx >= t2] = 1  # Pull-under begins at hip extension peak
        bar_phase.loc[idx >= t3] = 2  # Recovery begins when hips stop dropping

        df["bar_phase"] = bar_phase
        df["phase_change"] = df["bar_phase"].diff().fillna(0).ne(0)

        print(
            f"Classics phase mapping → Pull:[t0→t2], Pull-under:[t2→t3], Recovery:[t3→t4] "
            f"(t2={t2}, t3={t3})"
        )
    else:
        # Kinematic 3-phase fallback for non-classics lifts or when detection fails
        if lift_type in ("clean", "snatch"):
            print(
                "Warning: Could not identify classics phases. "
                "Falling back to kinematic 3-phase detection."
            )
        df["bar_phase"] = _detect_three_phases(df, fps)
        df["phase_change"] = df["bar_phase"].diff().fillna(0).ne(0)

    # -----------------------------------------------------------------------
    # Perspective correction (requires world landmarks)
    # -----------------------------------------------------------------------
    has_world_landmarks = "world_landmarks" in df.columns and bool(
        df["world_landmarks"].notna().any()
    )

    if has_world_landmarks:
        print("Calculating perspective-corrected bar path...")
        df = calculate_perspective_correction(df, frame_width, frame_height)

        valid_frames = df["barbell_x_corrected_cm"].notna().sum()
        if valid_frames > 10:
            print(
                f"  Perspective correction calculated for {valid_frames}/{len(df)} frames"
            )
            corrected_x_range = (
                df["barbell_x_corrected_cm"].max() - df["barbell_x_corrected_cm"].min()
            )
            corrected_y_range = (
                df["barbell_y_corrected_cm"].max() - df["barbell_y_corrected_cm"].min()
            )
            print(
                f"  Corrected bar path range: "
                f"horizontal = {corrected_x_range:.1f} cm, vertical = {corrected_y_range:.1f} cm"
            )
            avg_yaw = df["camera_yaw_deg"].dropna()
            if len(avg_yaw) > 0:
                avg_yaw_val = float(avg_yaw.iloc[0])
                if not pd.isna(avg_yaw_val):
                    print(f"  Estimated camera yaw: {avg_yaw_val:.1f}°")
        elif valid_frames > 0:
            print(
                f"  Warning: Only {valid_frames} frames with perspective correction (need >10)"
            )
    else:
        print("Skipping perspective correction (no world landmarks available)")

    # -----------------------------------------------------------------------
    # Maximum specific power (classics lifts only)
    # -----------------------------------------------------------------------
    if phases is not None and lift_type in ("clean", "snatch"):
        print("\n--- Maximum Specific Power Analysis ---")
        px_to_m_factor = calculate_pixel_to_meter_conversion(df)
        if px_to_m_factor is not None:
            df["px_to_m_conversion"] = px_to_m_factor

        max_power_result = calculate_max_specific_power(df, phases)
        if max_power_result is not None:
            if max_power_result.get("max_power_real") is not None:
                print(
                    f"Peak power output (pull→pull-under): {max_power_result['max_power_real']:.2f} W/kg"
                )
            else:
                print(
                    f"Peak power output (pull→pull-under): {max_power_result['max_power_px']:.2f} px²/s³ "
                    "(endcap detection failed)"
                )

    # -----------------------------------------------------------------------
    # Preserve landmark string for video rendering
    # -----------------------------------------------------------------------
    df["landmarks_str"] = df["landmarks"].apply(
        lambda x: str(x) if isinstance(x, dict) else "{}"
    )

    def box_to_str(x):
        if isinstance(x, (list, tuple)):
            values = []
            for v in x:
                values.append(v.item() if hasattr(v, "item") else float(v))
            return ",".join(f"{v:.2f}" for v in values)
        return ""

    if "barbell_box" in df.columns:
        df["barbell_box_str"] = df["barbell_box"].apply(box_to_str)
    else:
        df["barbell_box_str"] = ""

    # -----------------------------------------------------------------------
    # Drop raw / intermediate columns — only smoothed data goes to CSV
    # -----------------------------------------------------------------------
    cols_to_drop: list[str] = []

    # Raw landmark dict columns
    cols_to_drop.append("landmarks")

    # Shake components (cumulative totals are kept)
    cols_to_drop.extend(["shake_dx", "shake_dy"])

    # Per-joint tuple column (the raw x/y/z columns are kept in the CSV)
    for name in LANDMARKS_TO_TRACK:
        cols_to_drop.append(name)  # tuple column — x/y/z/vis columns are kept

    # Barbell raw / unstabilised position
    if "barbell_center" in df.columns:
        cols_to_drop.append("barbell_center")
    if "barbell_box" in df.columns:
        cols_to_drop.append("barbell_box")
    cols_to_drop.extend(["barbell_x_raw", "barbell_y_raw"])

    # World landmark intermediate columns (keep only final perspective results)
    world_cols = [c for c in df.columns if "world" in c]
    cols_to_drop.extend(world_cols)

    # Velocity from raw (unsmoothed) position
    if "vel_y_px_s" in df.columns:
        cols_to_drop.append("vel_y_px_s")

    # Remove duplicates and only drop columns that actually exist
    cols_to_drop = list(dict.fromkeys(cols_to_drop))
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    df = df.drop(columns=cols_to_drop)

    df.to_csv(output_path)
    print(f"\nAnalysis complete. Enriched data saved to '{output_path}'")
    print(f"Saved {len(df)} frames with {len(df.columns)} columns")

    barbell_tracked = df["barbell_y_stable"].notna().sum()
    print(
        f"Barbell tracked in {barbell_tracked}/{len(df)} frames "
        f"({100 * barbell_tracked / len(df):.1f}%)"
    )

    # Phase summary
    if "bar_phase" in df.columns:
        phase_names = {0: "Pull", 1: "Pull-under", 2: "Recovery"}
        counts = df["bar_phase"].value_counts().sort_index()
        print("Phase breakdown:")
        for pid, count in counts.items():
            pid_int = int(pid)  # type: ignore[arg-type]  # value_counts keys are ints at runtime
            label = phase_names.get(pid_int, f"Phase {pid_int}")
            print(f"  {label}: {count} frames ({100 * count / len(df):.1f}%)")


# ---------------------------------------------------------------------------
# Main (standalone CLI)
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Analyze raw data and save to CSV."
    )
    parser.add_argument(
        "--input",
        default="raw_data.pkl",
        help="Path to the raw data pickle file from Step 1.",
    )
    parser.add_argument(
        "--output",
        default="final_analysis.csv",
        help="Path to save the final analysis CSV file.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        return

    try:
        with open(args.input, "rb") as f:
            input_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file {args.input}: {e}")
        return

    step_2_analyze_data(input_data, args.output)


if __name__ == "__main__":
    main()
