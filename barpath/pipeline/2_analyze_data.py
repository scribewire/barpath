import argparse
import gc
import os
import pickle
from typing import cast

import numpy as np
import pandas as pd
from pandas import Series
from scipy.signal import savgol_filter
from step2_helpers import calculate_perspective_correction
from utils import calculate_angle, calculate_lifter_angle


# --- Step 2: Data Analysis Function ---
def step_2_analyze_data(input_data, output_path):
    print("--- Step 2: Analyzing Data ---")

    # Unpack the input data
    metadata = input_data.get("metadata", {})
    df_list = input_data.get("data", [])

    if not df_list:
        print("Error: No data found in pickle file.")
        return

    df = pd.DataFrame(df_list)

    # --- Memory Management ---
    # The raw list of dicts can be huge. Now that we have a DataFrame,
    # we can free the raw list to save memory during analysis.
    del df_list
    if "data" in input_data:
        del input_data["data"]
    gc.collect()

    if "frame" not in df.columns:
        print("Error: No 'frame' column in data.")
        return

    df = df.set_index("frame").sort_index()

    frame_gaps = df.index.to_series().diff()

    # Cast the result to a Series explicitly
    frame_gaps_numeric = cast(Series, pd.to_numeric(frame_gaps, errors="coerce"))

    if (frame_gaps_numeric.fillna(0) > 1).any():
        print(f"Warning: Detected {(frame_gaps_numeric > 1).sum()} gaps.")

    # --- Metadata ---
    frame_width = metadata.get("frame_width", 1920)
    frame_height = metadata.get("frame_height", 1080)
    fps = metadata.get("fps", 30.0)

    df["frame_width"] = frame_width
    df["frame_height"] = frame_height

    # --- Unpack Landmark Data ---
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
            lambda x: x.get(name) if isinstance(x, dict) else None
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

    # --- Calculate Angles ---
    df["lifter_angle_deg"] = df["landmarks"].apply(calculate_lifter_angle)

    def get_pixel_pos(row, name):
        x_norm = row.get(f"{name}_x")
        y_norm = row.get(f"{name}_y")
        if pd.isna(x_norm) or pd.isna(y_norm):
            return np.array([np.nan, np.nan])
        return np.array([x_norm * frame_width, y_norm * frame_height])

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

    df["hip_y_avg"] = df[["left_hip_y", "right_hip_y"]].mean(axis=1) * frame_height

    # --- Calculate Stabilized Coordinates ---
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

    # --- Smooth Position Data ---
    # Apply Savitzky-Golay filter to stabilized position to remove jitter
    x_filled = df["barbell_x_stable"].interpolate(method="linear").bfill().ffill()
    y_filled = df["barbell_y_stable"].interpolate(method="linear").bfill().ffill()

    pos_window = min(11, len(x_filled) // 2 * 2 + 1)  # Must be odd
    if pos_window >= 5 and len(x_filled) >= pos_window:
        print(f"Applying position smoothing with window {pos_window}...")
        df["barbell_x_smooth"] = savgol_filter(x_filled, pos_window, 3)
        df["barbell_y_smooth"] = savgol_filter(y_filled, pos_window, 3)
    else:
        print("Warning: Not enough data to smooth position. Using unsmoothed values.")
        df["barbell_x_smooth"] = df["barbell_x_stable"]
        df["barbell_y_smooth"] = df["barbell_y_stable"]

    # --- NEW: Truncate data at beginning - discard frames before bar passes knee ---
    # Calculate average knee position
    df["knee_y_avg"] = df[["left_knee_y", "right_knee_y"]].mean(axis=1) * frame_height

    # Find when bar passes knee (bar Y > knee Y, since Y=0 is top)
    if bool(df["barbell_y_smooth"].notna().any()) and bool(
        df["knee_y_avg"].notna().any()
    ):
        # Create a boolean mask where bar is above (lower Y value than) knee
        bar_above_knee = df["barbell_y_smooth"] < df["knee_y_avg"]

        # Find the first frame where bar is above knee
        frames_above_knee = df[bar_above_knee].index.values

        if len(frames_above_knee) > 0:
            # Extract scalar values from pandas Index
            knee_pass_frame = int(frames_above_knee[0])

            # Calculate frame offset for 1 second before knee pass
            frames_before = int(fps)
            first_frame = int(df.index.values[0])

            start_frame = max(first_frame, knee_pass_frame - frames_before)

            print(
                f"Bar passes knee at frame {knee_pass_frame}. "
                f"Keeping data from frame {start_frame} onwards (1s before knee pass)."
            )

            # Truncate data before start_frame
            df = df.loc[start_frame:].copy()
        else:
            print("Warning: Bar never detected above knee. Keeping all data at start.")
    else:
        print("Warning: Cannot determine knee pass frame. Keeping all data at start.")

    # --- NEW: Truncate data at maximum height ---
    # Note: Y=0 is top, so max height is min Y value
    if bool(df["barbell_y_smooth"].notna().any()):
        # Find the index (frame) where the bar reaches its highest point (min Y)
        peak_height_idx = df["barbell_y_smooth"].idxmin()
        print(
            f"Peak height detected at frame {peak_height_idx}. Truncating data after this point."
        )

        # Slice the DataFrame to keep only data up to the peak
        # We use .loc which includes the endpoint
        df = df.loc[:peak_height_idx].copy()
    else:
        print("Warning: No barbell Y data found. Cannot truncate at peak height.")

    # --- Calculate Kinematics ---
    if df.index.is_monotonic_increasing:
        df["time_s"] = (df.index - df.index[0]) / fps
    else:
        print("Warning: Frame indices are not monotonic. Using sequential time.")
        df["time_s"] = np.arange(len(df)) / fps

    df["dt"] = df["time_s"].diff()
    df["dt"] = df["dt"].fillna(1 / fps)

    # Calculate velocity from smoothed position
    df["vel_y_px_s"] = (df["barbell_y_smooth"].diff() / df["dt"]) * -1

    # --- NEW: Calculate Bar Path Phases ---
    # 1. Interpolate and fill NaNs to create a continuous velocity signal for smoothing
    vel_filled = df["vel_y_px_s"].interpolate(method="linear").fillna(0)

    # 2. Smooth the velocity to remove noise. Window must be odd and less than data length.
    window_length = min(15, len(vel_filled) // 2 * 2 + 1)  # Must be odd
    if window_length >= 5:
        print(f"Applying Savitzky-Golay smoothing with window {window_length}...")
        df["vel_y_smooth"] = savgol_filter(vel_filled, window_length, 3)
    else:
        print("Warning: Not enough data to smooth velocity. Phases may be noisy.")
        df["vel_y_smooth"] = vel_filled

    # 3. Define a velocity threshold to ignore minor jitters (5% of peak, or 10px/s)
    vel_threshold = max(10, df["vel_y_smooth"].abs().max() * 0.05)
    print(f"Using velocity threshold of {vel_threshold:.2f} px/s for phase change.")

    # 4. Determine direction state (1=Up, -1=Down)
    df["direction_state"] = 0
    df.loc[df["vel_y_smooth"] > vel_threshold, "direction_state"] = 1
    df.loc[df["vel_y_smooth"] < -vel_threshold, "direction_state"] = -1

    # 5. Fill gaps (0s) with the previous valid state
    df["direction_state"] = (
        df["direction_state"].replace(0, np.nan).ffill().fillna(1)
    )  # Default to 1 (Up) at start

    # 6. Find where the state *changes*
    df["phase_change"] = df["direction_state"].diff().ne(0)

    # 7. Create the phase number by taking a cumulative sum of the changes
    df["bar_phase"] = df["phase_change"].cumsum()
    # --- End new block ---

    # --- Calculate Perspective Correction (if world landmarks available) ---
    # This happens after barbell_x_stable and barbell_y_stable are calculated

    # Check if world_landmarks column exists and has data
    has_world_landmarks = "world_landmarks" in df.columns and bool(
        df["world_landmarks"].notna().any()
    )

    if has_world_landmarks:
        print("Calculating perspective-corrected bar path...")
        df = calculate_perspective_correction(df, frame_width, frame_height)

        # Report statistics and quality checks
        valid_frames = df["barbell_x_corrected_px"].notna().sum()
        if valid_frames > 10:  # Need minimum frames for meaningful analysis
            print(
                f"  Perspective correction calculated for {valid_frames}/{len(df)} frames"
            )

            # Calculate displacement statistics
            corrected_range = (
                df["barbell_x_corrected_px"].max() - df["barbell_x_corrected_px"].min()
            )
            uncorrected_range = (
                df["barbell_x_smooth"].max() - df["barbell_x_smooth"].min()
            )
            print(
                f"  Horizontal displacement: {uncorrected_range:.1f} px (uncorrected) -> {corrected_range:.1f} px (corrected)"
            )
            avg_yaw = df["camera_yaw_deg"].mean()
            avg_yaw_val = float(avg_yaw) if not bool(pd.isna(avg_yaw)) else None
            if avg_yaw_val is not None:
                print(f"  Reference camera yaw: {avg_yaw_val:.1f}°")

            avg_factor = df["lateral_correction_factor"].mean()
            factor_val = float(avg_factor) if not bool(pd.isna(avg_factor)) else None
            if factor_val is not None:
                print(f"  Lateral correction factor: {factor_val:.3f}x")
        elif valid_frames > 0:
            print(
                f"  Warning: Only {valid_frames} frames with perspective correction (need >10)"
            )
    else:
        print("Skipping perspective correction (no world landmarks available)")

    # Y-Acceleration (px/s^2)
    # df['accel_y_px_s2'] = df['vel_y_px_s'].diff() / df['dt'] # OLD

    # NEW: Smoothed Acceleration
    df["accel_y_smooth"] = df["vel_y_smooth"].diff() / df["dt"]

    # Y-Jerk (px/s^3) - REMOVED
    # df['jerk_y_px_s3'] = df['accel_y_px_s2'].diff() / df['dt']

    # "Specific Power" (Power-to-Mass ratio, proxy)
    # df['specific_power_y'] = df['accel_y_px_s2'] * df['vel_y_px_s'] # OLD

    # NEW: Smoothed Specific Power
    df["specific_power_y_smooth"] = df["accel_y_smooth"] * df["vel_y_smooth"]

    # --- Preserve landmarks as string for video rendering ---
    df["landmarks_str"] = df["landmarks"].apply(
        lambda x: str(x) if isinstance(x, dict) else "{}"
    )

    def box_to_str(x):
        """Convert box coordinates to clean string format."""
        if isinstance(x, (list, tuple)):
            values = []
            for v in x:
                if hasattr(v, "item"):  # It's a tensor
                    values.append(v.item())
                else:
                    values.append(float(v))
            return ",".join(f"{v:.2f}" for v in values)
        return ""

    if "barbell_box" in df.columns:
        df["barbell_box_str"] = df["barbell_box"].apply(box_to_str)
    else:
        df["barbell_box_str"] = ""

    # --- Clean up and Save ---
    # Drop raw data columns that are no longer needed
    cols_to_drop = ["landmarks", "shake_dx", "shake_dy"] + list(LANDMARKS_TO_TRACK)
    if "barbell_center" in df.columns:
        cols_to_drop.append("barbell_center")
    if "barbell_box" in df.columns:
        cols_to_drop.append("barbell_box")

    # Also drop world landmark intermediate columns (keep only final results)
    world_landmark_cols = [col for col in df.columns if "world" in col]
    cols_to_drop.extend(world_landmark_cols)

    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    df.to_csv(output_path)
    print(f"Analysis complete. Enriched data saved to '{output_path}'")
    print(f"Saved {len(df)} frames with {len(df.columns)} columns")

    barbell_tracked = df["barbell_y_stable"].notna().sum()
    print(
        f"Barbell tracked in {barbell_tracked}/{len(df)} frames ({100 * barbell_tracked / len(df):.1f}%)"
    )


# --- Main Execution ---
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
