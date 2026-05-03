"""
Step 4: Technique Analysis using compiled rule-based analyzer.

The analysis reads from final_analysis.csv and produces analysis.md.
Uses the CompiledAnalyzer which runs in sub-millisecond time.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# When run as a standalone script, the pipeline dir is on sys.path
try:
    from step4_helpers import extract_technique_features, validate_phases
    from step4_helpers.compiled_analyzer import (
        CompiledAnalyzer,
        FAULT_DEFS,
        load_baselines_from_json,
    )
except ImportError:
    from barpath.pipeline.step4_helpers import (
        extract_technique_features,
        validate_phases,
    )
    from barpath.pipeline.step4_helpers.compiled_analyzer import (
        CompiledAnalyzer,
        FAULT_DEFS,
        load_baselines_from_json,
    )

PHASE_NAMES = {
    "snatch": {0: "Pull", 1: "Pull-under", 2: "Recovery"},
    "clean": {0: "Pull", 1: "Pull-under", 2: "Recovery"},
    "jerk": {0: "Dip", 1: "Drive", 2: "Recovery"},
}


def get_phase_names(lift_type: str) -> Dict[int, str]:
    """Get phase names for a specific lift type."""
    return PHASE_NAMES.get(
        lift_type.lower(), {0: "Phase 0", 1: "Phase 1", 2: "Phase 2"}
    )


def similarity_to_stars(score_pct: float) -> Tuple[str, str]:
    """Convert technique score (0-100) to star rating and label."""
    if score_pct >= 90:
        return "★★★★★", "Excellent"
    elif score_pct >= 80:
        return "★★★★", "Very Good"
    elif score_pct >= 70:
        return "★★★", "Good"
    elif score_pct >= 50:
        return "★★", "Fair"
    else:
        return "★", "Needs Work"


def _get_fault_description(fault_id: str) -> str:
    """Get human-readable description for a fault."""
    fdef = FAULT_DEFS.get(fault_id, {})
    return fdef.get("description", "Technique issue detected.")


def _get_fault_phase(fault_id: str, lift_type: str) -> str:
    """Get the phase where a fault typically occurs."""
    fdef = FAULT_DEFS.get(fault_id, {})
    phase = fdef.get("phase", "unknown")
    if isinstance(phase, int):
        names = get_phase_names(lift_type)
        return names.get(phase, f"Phase {phase}")
    return str(phase).replace("_", " ").title() if phase != "unknown" else "Unknown"


def _get_fault_coaching_cue(fault_id: str) -> str:
    """Get coaching cue for a fault."""
    fdef = FAULT_DEFS.get(fault_id, {})
    return fdef.get("coaching_cue", "")


def write_analysis_md(
    faults: Optional[List[Dict[str, Any]]],
    score: Optional[float],
    assessment: Optional[str],
    df: pd.DataFrame,
    lift_type: str,
    output_path: str,
) -> None:
    """Write the analysis.md report for a single segment."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Analysis Report: {lift_type.capitalize()}\n\n")

            f.write("## Technique Analysis\n\n")

            if faults:
                if score is not None:
                    stars, label = similarity_to_stars(score / 100.0)
                    f.write(
                        f"**Technique Score:** {score:.0f}/100 {stars} ({label})\n\n"
                    )

                if assessment:
                    f.write(f"_{assessment}_\n\n")

                f.write("**Detected Issues:**\n\n")
                for fault in faults:
                    fid = fault.get("id", "unknown")
                    name = fault.get("name", fid.replace("_", " ").title())
                    conf = fault.get("confidence", 0)
                    phase = fault.get("phase", "Unknown")
                    desc = fault.get("description", "")
                    cue = fault.get("coaching_cue", "")

                    f.write(f"### {name} ({conf}% confidence)\n\n")
                    f.write(f"**Phase:** {phase}\n\n")
                    if desc:
                        f.write(f"{desc}\n\n")
                    if cue:
                        f.write(f"**Coaching Cue:** {cue}\n\n")

                # Clear areas checklist
                all_checked = _get_all_fault_ids_for_lift(lift_type)
                flagged_ids = {f.get("id") for f in faults}
                clear = [fid for fid in all_checked if fid not in flagged_ids]
                if clear:
                    f.write("**No Issues Detected:**\n\n")
                    for fid in clear:
                        name = FAULT_DEFS.get(fid, {}).get(
                            "name", fid.replace("_", " ").title()
                        )
                        f.write(f"- [x] {name}\n")
                    f.write("\n")
            else:
                f.write("**No significant issues detected.**\n\n")
                f.write("All technique checks passed. Great lift!\n\n")

            # --- Kinematic Summary ---
            f.write("## Kinematic Summary\n\n")

            if "time_s" in df.columns:
                times = df["time_s"].dropna()
                if len(times) > 1:
                    total_time = float(times.iloc[-1] - times.iloc[0])
                    f.write(f"- **Total Lift Time:** {total_time:.2f}s\n")

            if "vel_y_smooth" in df.columns:
                vel = df["vel_y_smooth"].dropna()
                if len(vel) > 0:
                    vel_arr = np.asarray(vel, dtype=float)
                    f.write(
                        f"- **Peak Vertical Velocity:** {float(vel_arr.max()):.1f} px/s\n"
                    )

            if "accel_y_smooth" in df.columns:
                accel = df["accel_y_smooth"].dropna()
                if len(accel) > 0:
                    accel_arr = np.asarray(accel, dtype=float)
                    f.write(
                        f"- **Peak Acceleration:** {float(accel_arr.max()):.1f} px/s²\n"
                    )

            if "specific_power_y_smooth" in df.columns:
                power = df["specific_power_y_smooth"].dropna()
                if len(power) > 0:
                    power_arr = np.asarray(power, dtype=float)
                    f.write(
                        f"- **Peak Specific Power:** {float(power_arr.max()):.1f}\n"
                    )

            f.write("\n---\n\n")
            f.write("*Generated by BARPATH Technique Analysis Engine*\n")

        logger.info(f"Analysis report saved to '{output_path}'")
    except Exception as e:
        logger.error(f"Error writing analysis.md: {e}")


def write_unified_analysis_md(
    clean_result: Optional[Dict[str, Any]],
    jerk_result: Optional[Dict[str, Any]],
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """Write unified analysis.md for clean+jerk with separate sections."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Analysis Report: Clean & Jerk\n\n")

            # Clean section
            if clean_result:
                f.write("## Clean Analysis\n\n")
                faults = clean_result.get("faults", [])
                score = clean_result.get("score")
                assessment = clean_result.get("assessment")

                if faults:
                    if score is not None:
                        stars, label = similarity_to_stars(score / 100.0)
                        f.write(
                            f"**Technique Score:** {score:.0f}/100 {stars} ({label})\n\n"
                        )
                    if assessment:
                        f.write(f"_{assessment}_\n\n")
                    f.write("**Detected Issues:**\n\n")
                    for fault in faults:
                        conf = fault.get("confidence", 0)
                        f.write(
                            f"- **{fault.get('name', 'Unknown')}** ({conf}% confidence)\n"
                        )
                        f.write(f"  - {fault.get('description', '')}\n")
                        cue = fault.get("coaching_cue", "")
                        if cue:
                            f.write(f"  - *Cue:* {cue}\n")
                        f.write("\n")
                else:
                    f.write("No significant issues detected in the clean.\n\n")
            else:
                f.write("## Clean Analysis\n\nClean analysis not available.\n\n")

            # Jerk section
            if jerk_result:
                f.write("## Jerk Analysis\n\n")
                faults = jerk_result.get("faults", [])
                score = jerk_result.get("score")
                assessment = jerk_result.get("assessment")

                if faults:
                    if score is not None:
                        stars, label = similarity_to_stars(score / 100.0)
                        f.write(
                            f"**Technique Score:** {score:.0f}/100 {stars} ({label})\n\n"
                        )
                    if assessment:
                        f.write(f"_{assessment}_\n\n")
                    f.write("**Detected Issues:**\n\n")
                    for fault in faults:
                        conf = fault.get("confidence", 0)
                        f.write(
                            f"- **{fault.get('name', 'Unknown')}** ({conf}% confidence)\n"
                        )
                        f.write(f"  - {fault.get('description', '')}\n")
                        cue = fault.get("coaching_cue", "")
                        if cue:
                            f.write(f"  - *Cue:* {cue}\n")
                        f.write("\n")
                else:
                    f.write("No significant issues detected in the jerk.\n\n")
            else:
                f.write("## Jerk Analysis\n\nJerk analysis not available.\n\n")

            # Combined kinematic summary
            f.write("## Combined Kinematic Summary\n\n")
            if "time_s" in df.columns:
                times = df["time_s"].dropna()
                if len(times) > 1:
                    total_time = float(times.iloc[-1] - times.iloc[0])
                    f.write(f"- **Total Lift Time:** {total_time:.2f}s\n")

            if "vel_y_smooth" in df.columns:
                vel = df["vel_y_smooth"].dropna()
                if len(vel) > 0:
                    vel_arr = np.asarray(vel, dtype=float)
                    f.write(
                        f"- **Peak Vertical Velocity:** {float(vel_arr.max()):.1f} px/s\n"
                    )

            if "accel_y_smooth" in df.columns:
                accel = df["accel_y_smooth"].dropna()
                if len(accel) > 0:
                    accel_arr = np.asarray(accel, dtype=float)
                    f.write(
                        f"- **Peak Acceleration:** {float(accel_arr.max()):.1f} px/s²\n"
                    )

            f.write("\n---\n\n")
            f.write("*Generated by BARPATH Technique Analysis Engine*\n")

        logger.info(f"Unified analysis report saved to '{output_path}'")
    except Exception as e:
        logger.error(f"Error writing unified analysis.md: {e}")


def _get_all_fault_ids_for_lift(lift_type: str) -> List[str]:
    """Get all fault IDs applicable to a lift type."""
    return [
        fid
        for fid, fdef in FAULT_DEFS.items()
        if lift_type in fdef.get("lift_types", [])
    ]


def _load_baseline_for_lifter(
    models_base: Path, lifter: str, lift_type: str
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load per-lifter baselines, falling back to pooled report."""
    if lifter and lifter != "generic":
        lifter_json = models_base / f"pro_baseline_report_{lifter}.json"
        if lifter_json.exists():
            return load_baselines_from_json(lifter_json)

    pooled_json = models_base / "pro_baseline_report.json"
    return load_baselines_from_json(pooled_json)


def _analyze_segment(
    df: pd.DataFrame,
    lift_type: str,
    lifter: str,
    gender: str = "male",
) -> Dict[str, Any]:
    """Analyze a single lift segment using the CompiledAnalyzer."""
    models_base = Path(__file__).parent.parent / "models" / "analysis"
    features = extract_technique_features(df, lift_type)
    baselines = _load_baseline_for_lifter(models_base, lifter, lift_type)
    analyzer = CompiledAnalyzer(lift_type, gender, baselines)
    faults = analyzer.analyze(features, df)
    score, assessment = analyzer.get_technique_score(faults)
    return {
        "faults": faults,
        "score": score,
        "assessment": assessment,
        "features": features,
    }


def critique_lift(
    df: pd.DataFrame,
    lift_type: str = "clean",
    output_dir: str = ".",
    lifter: str = "generic",
    gender: str = "male",
) -> Dict[str, Any]:
    """Run technique critique using the compiled rule-based analyzer.

    For clean+jerk lifts, splits the trajectory and analyzes clean and
    jerk segments independently, producing a unified report.

    Args:
        df: DataFrame with kinematic data from final_analysis.csv
        lift_type: "clean", "snatch", "jerk", or "clean_jerk"
        output_dir: Directory to save analysis.md
        lifter: Lifter name for baseline selection (supports per-lifter baselines)
        gender: "male" or "female" for baseline selection

    Returns:
        Dict with analysis results
    """
    # Phase validation
    has_valid_phases = validate_phases(df)
    if not has_valid_phases:
        logger.warning(
            "CSV missing phases in bar_phase column — analysis may be unreliable"
        )

    output_path = str(Path(output_dir) / "analysis.md")

    if lift_type == "clean_jerk" and "lift_segment" in df.columns:
        # Split analysis
        df_clean = df[df["lift_segment"] == "clean"].copy()
        df_jerk = df[df["lift_segment"] == "jerk"].copy()

        clean_result = None
        jerk_result = None
        all_faults: List[Dict[str, Any]] = []

        if len(df_clean) > 0:
            clean_result = _analyze_segment(
                cast(pd.DataFrame, df_clean), "clean", lifter, gender
            )
            all_faults.extend(clean_result["faults"])

        if len(df_jerk) > 0:
            jerk_result = _analyze_segment(
                cast(pd.DataFrame, df_jerk), "jerk", lifter, gender
            )
            all_faults.extend(jerk_result["faults"])

        write_unified_analysis_md(clean_result, jerk_result, df, output_path)

        results: Dict[str, Any] = {
            "clean_result": clean_result,
            "jerk_result": jerk_result,
            "compiled_faults": all_faults,
            "lifter": lifter,
            "lift_type": lift_type,
        }
    else:
        # Single-segment analysis
        result = _analyze_segment(df, lift_type, lifter, gender)
        write_analysis_md(
            result["faults"],
            result["score"],
            result["assessment"],
            df,
            lift_type,
            output_path,
        )
        results = {
            "compiled_faults": result["faults"],
            "compiled_score": result["score"],
            "compiled_assessment": result["assessment"],
            "lifter": lifter,
            "lift_type": lift_type,
            "features": result["features"],
        }

    if results.get("compiled_faults"):
        results["flagged_faults"] = [f["id"] for f in results["compiled_faults"]]

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4: Technique critique using rule-based analysis."
    )
    parser.add_argument(
        "--input", default="final_analysis.csv", help="Path to analysis CSV."
    )
    parser.add_argument(
        "--lift_type",
        required=True,
        choices=["clean", "snatch", "jerk", "clean_jerk", "none"],
    )
    parser.add_argument(
        "--lifter", default="generic", help="Lifter name for baseline selection."
    )
    parser.add_argument(
        "--gender",
        default="male",
        choices=["male", "female"],
        help="Gender for baseline selection.",
    )
    parser.add_argument(
        "--output_dir", default=".", help="Output directory for analysis.md"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.lift_type == "none":
        return

    if not os.path.exists(args.input):
        logger.error(f"{args.input} not found")
        return

    try:
        df = pd.read_csv(args.input)
        if "frame" in df.columns:
            df = df.set_index("frame")

        critique_lift(
            df,
            args.lift_type,
            args.output_dir,
            args.lifter,
            gender=args.gender,
        )

        logger.info(
            f"Analysis complete. Results saved to {args.output_dir}/analysis.md"
        )

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
