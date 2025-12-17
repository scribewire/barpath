import argparse
import os

import numpy as np
import pandas as pd
from step5_helpers.classics_phase_detection import identify_classics_phases
from step5_helpers.clean import check_clean_faults
from step5_helpers.snatch import check_snatch_faults


def calculate_max_specific_power(df, phases):
    """
    Calculate maximum specific power between end of first pull (t1) and end of third pull (t3).
    Uses saved px_to_m_conversion factor from Step 2 to convert to W/kg.

    Args:
        df: DataFrame with calculated kinematics
        phases: ClassicsPhases dict with t0, t1, t2, t3, t4 frame indices

    Returns:
        dict: Dictionary with 'max_power_px' and optionally 'max_power_real' (W/kg),
              or None if cannot calculate
    """
    if phases is None or "t1" not in phases or "t3" not in phases:
        return None

    try:
        t1 = int(phases["t1"])
        t3 = int(phases["t3"])

        if "specific_power_y_smooth" not in df.columns:
            return None

        # Extract specific power data between t1 and t3
        power_segment = df.loc[t1:t3, "specific_power_y_smooth"]

        if power_segment.empty:
            return None

        # Get maximum absolute specific power in pixel units
        max_power_px = float(power_segment.abs().max())

        if np.isnan(max_power_px):
            return None

        result = {"max_power_px": max_power_px, "max_power_real": None}

        # Check if conversion factor is available from Step 2
        if "px_to_m_conversion" in df.columns:
            px_to_m = float(df["px_to_m_conversion"].iloc[0])
            if not np.isnan(px_to_m) and px_to_m > 0:
                # Convert from px²/s³ to m²/s³ (W/kg)
                max_power_real = max_power_px * (px_to_m**2)
                result["max_power_real"] = max_power_real

        return result
    except Exception as e:
        print(f"Warning: Could not calculate max specific power: {e}")
        return None


def write_analysis_md(critiques, phases, df, lift_type, output_path="analysis.md"):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Analysis Report: {lift_type.capitalize()}\n\n")

            # Calculate and display maximum specific power before phase timing
            max_power_result = calculate_max_specific_power(df, phases)
            if max_power_result is not None:
                f.write("## Maximum Specific Power\n")
                max_power_px = max_power_result["max_power_px"]
                max_power_real = max_power_result.get("max_power_real")

                if max_power_real is not None:
                    f.write(f"- **Peak Power (t1→t3):** {max_power_real:.2f} W/kg\n")
                    f.write(f"  *(Raw: {max_power_px:.2f} px²/s³)*\n\n")
                else:
                    f.write(f"- **Peak Power (t1→t3):** {max_power_px:.2f} px²/s³\n")
                    f.write(
                        "  *(Note: Real-world conversion unavailable - endcap not detected)*\n\n"
                    )

            f.write("## Phase Timing\n")
            if phases:

                def get_duration(start_idx, end_idx):
                    return df.loc[end_idx, "time_s"] - df.loc[start_idx, "time_s"]

                f.write(
                    f"- **First Pull:**  {get_duration(phases['t0'], phases['t1']):.2f}s\n"
                )
                f.write(
                    f"- **Second Pull:** {get_duration(phases['t1'], phases['t2']):.2f}s\n"
                )
                f.write(
                    f"- **Third Pull:**  {get_duration(phases['t2'], phases['t3']):.2f}s\n"
                )
                f.write(
                    f"- **Recovery:**    {get_duration(phases['t3'], phases['t4']):.2f}s\n"
                )
                f.write(
                    f"- **Total Time:**  {get_duration(phases['t0'], phases['t4']):.2f}s\n"
                )
            else:
                f.write("Could not identify phases.\n")

            f.write("\n## Critique\n")
            if not critiques:
                f.write("No major faults detected based on configured checks.\n")
            else:
                for c in critiques:
                    f.write(f"- {c}\n")
        print(f"Analysis report saved to '{output_path}'")
    except Exception as e:
        print(f"Error writing analysis.md: {e}")


def critique_lift(df, lift_type="clean", output_dir="."):
    phases = None
    if lift_type in ("clean", "snatch"):
        phases = identify_classics_phases(df)

    critiques = []
    if phases:
        if lift_type == "clean":
            critiques = check_clean_faults(df, phases)
        elif lift_type == "snatch":
            critiques = check_snatch_faults(df, phases)

        output_path = os.path.join(output_dir, "analysis.md")
        write_analysis_md(critiques, phases, df, lift_type, output_path)

        # Return formatted strings for CLI output
        results = []
        results.append(f"Phases identified. See {output_path} for details.")
        if critiques:
            results.extend(critiques)
        else:
            results.append("No faults detected.")
        return results
    else:
        return ["Could not identify lift phases."]


def main():
    parser = argparse.ArgumentParser(description="Step 5: Identify lift phases.")
    parser.add_argument(
        "--input", default="final_analysis.csv", help="Path to analysis CSV."
    )
    parser.add_argument(
        "--lift_type", required=True, choices=["clean", "snatch", "none"]
    )
    args = parser.parse_args()

    if args.lift_type == "none":
        return

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        return

    try:
        df = pd.read_csv(args.input)
        if "frame" in df.columns:
            df = df.set_index("frame")
        results = critique_lift(df, args.lift_type)
        for r in results:
            print(r)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
