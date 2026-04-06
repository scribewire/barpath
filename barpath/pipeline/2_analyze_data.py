"""
Step 2: Analyze raw data and produce enriched CSV.

This step takes the raw data pickle from Step 1 and:
1. Unpacks landmark data into per-joint columns
2. Calculates joint angles
3. Stabilizes and smooths barbell position
4. Truncates data to relevant timeframe
5. Calculates kinematics (velocity, acceleration, power)
6. Detects lift phases
"""

import argparse
import gc
import os
import pickle
from typing import cast

import numpy as np
import pandas as pd
from analysis_utils import (
    calculate_max_specific_power,
    calculate_pixel_to_meter_conversion,
)
from config import BARBELL_ENDCAP_WIDTH_M
from pandas import Series
from step2_helpers import (
    assign_phases_from_classics,
    assign_phases_kinematic,
    calculate_hip_y_average,
    calculate_joint_angles,
    calculate_lifter_angle,
    calculate_stabilized_position,
    calculate_time_and_kinematics,
    detect_facing_direction,
    drop_intermediate_columns,
    smooth_barbell_position,
    truncate_at_knee_pass,
    truncate_at_peak_height,
    unpack_landmarks,
)


def step_2_analyze_data(input_data, output_path):
    print("--- Step 2: Analyzing Data ---")

    metadata = input_data.get("metadata", {})
    df_list = input_data.get("data", [])

    lift_type = str(metadata.get("lift_type", "none")).lower()

    if not df_list:
        print("Error: No data found in pickle file.")
        return

    df = pd.DataFrame(df_list)

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

    frame_width = metadata.get("frame_width", 1920)
    frame_height = metadata.get("frame_height", 1080)
    fps = metadata.get("fps", 30.0)

    df["frame_width"] = frame_width
    df["frame_height"] = frame_height

    df = unpack_landmarks(df)
    df = detect_facing_direction(df)
    df = calculate_joint_angles(df, frame_width, frame_height)
    df["lifter_angle"] = df["landmarks"].apply(
        lambda x: calculate_lifter_angle(x) if isinstance(x, dict) else np.nan
    )
    df = calculate_hip_y_average(df, frame_height)

    df = calculate_stabilized_position(df)
    df = smooth_barbell_position(df)

    df = truncate_at_knee_pass(df, fps, frame_height)
    df = truncate_at_peak_height(df)

    df = calculate_time_and_kinematics(df, fps)

    phases = None
    if lift_type in ("clean", "snatch"):
        from step2_helpers.classics_phase_detection import identify_classics_phases

        phases = identify_classics_phases(df)

    if phases is not None:
        df = assign_phases_from_classics(df, phases)
    else:
        if lift_type in ("clean", "snatch"):
            print(
                "Warning: Could not identify classics phases. "
                "Falling back to kinematic 3-phase detection."
            )
        # Use appropriate phase detection based on lift type
        df = assign_phases_kinematic(df, fps, lift_type)

    # Normalize the optional kinematic phase label column so downstream access is typed safely.
    if "bar_phase" in df.columns:
        df["bar_phase"] = pd.to_numeric(df["bar_phase"], errors="coerce")

    if phases is not None and lift_type in ("clean", "snatch"):
        print("\n--- Maximum Specific Power Analysis ---")
        px_to_m_factor = calculate_pixel_to_meter_conversion(df, BARBELL_ENDCAP_WIDTH_M)
        if px_to_m_factor is not None:
            df["px_to_m_conversion"] = px_to_m_factor

        max_power_result = calculate_max_specific_power(df, phases)
        if max_power_result is not None:
            max_power_real = max_power_result.get("max_power_real")
            max_power_px = max_power_result.get("max_power_px")
            if max_power_real is not None:
                print(
                    f"Peak power output (pull->pull-under): {float(max_power_real):.2f} W/kg"
                )
            elif max_power_px is not None:
                print(
                    f"Peak power output (pull->pull-under): {float(max_power_px):.2f} px^2/s^3 "
                    "(endcap detection failed)"
                )

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

    df = drop_intermediate_columns(df)

    df.to_csv(output_path)
    print(f"\nAnalysis complete. Enriched data saved to '{output_path}'")
    print(f"Saved {len(df)} frames with {len(df.columns)} columns")

    barbell_tracked = df["barbell_y_stable"].notna().sum()
    print(
        f"Barbell tracked in {barbell_tracked}/{len(df)} frames "
        f"({100 * barbell_tracked / len(df):.1f}%)"
    )

    if "bar_phase" in df.columns:
        phase_names = {0: "Pull", 1: "Pull-under", 2: "Recovery"}
        counts = df["bar_phase"].value_counts().sort_index()
        print("Phase breakdown:")
        for pid, count in counts.items():
            pid_int = int(float(str(pid)))
            label = phase_names.get(pid_int, f"Phase {pid_int}")
            print(f"  {label}: {count} frames ({100 * count / len(df):.1f}%)")


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
