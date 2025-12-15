import argparse
import os

import pandas as pd
from step5_helpers.classics_phase_detection import identify_classics_phases
from step5_helpers.clean import check_clean_faults


def write_analysis_md(critiques, phases, df, lift_type, output_path="analysis.md"):
    try:
        with open(output_path, "w") as f:
            f.write(f"# Analysis Report: {lift_type.capitalize()}\n\n")

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
    if lift_type == "clean":
        phases = identify_classics_phases(df)

    critiques = []
    if phases:
        if lift_type == "clean":
            critiques = check_clean_faults(df, phases)
        # Add snatch checks here if needed

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
