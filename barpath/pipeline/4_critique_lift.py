"""
Step 4: Critique lift technique using ML-based analysis.

This module provides technique analysis using:
- Fast Analysis: DTW-based bar path similarity scoring
- Smart Analysis: Random Forest-based fault detection

The analysis reads from final_analysis.csv and produces analysis.md.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from step4_helpers import (
    extract_smart_features,
    extract_trajectory,
    load_fast_analysis_model,
    load_smart_analysis_model,
    run_fast_analysis,
    run_smart_analysis,
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


def resolve_model_path(
    lifter: str, lift_type: str, models_base: Path
) -> Optional[Path]:
    """
    Resolve the model directory for a given lifter and lift type.

    Falls back from lifter-specific to generic if not found.

    Args:
        lifter: Lifter name (e.g., "liao_hui", "lu_xiaojun", "generic")
        lift_type: "clean" or "snatch"
        models_base: Base path to models directory

    Returns:
        Path to model directory or None if not found.
    """
    if lifter and lifter != "generic":
        lifter_path = models_base / lifter / lift_type
        if lifter_path.exists():
            return lifter_path

    generic_path = models_base / "generic" / lift_type
    if generic_path.exists():
        return generic_path

    return None


def similarity_to_stars(similarity: float) -> Tuple[str, str]:
    """Convert similarity score to star rating and label."""
    if similarity >= 0.90:
        return "⭐⭐⭐⭐⭐", "Excellent"
    elif similarity >= 0.80:
        return "⭐⭐⭐⭐", "Very Good"
    elif similarity >= 0.70:
        return "⭐⭐⭐", "Good"
    elif similarity >= 0.50:
        return "⭐⭐", "Fair"
    else:
        return "⭐", "Needs Work"


def compute_phase_similarity_stats(
    df: pd.DataFrame, temporal_similarity: np.ndarray, lift_type: str = "snatch"
) -> Dict[int, Dict[str, Any]]:
    """Compute average similarity per phase."""
    stats: Dict[int, Dict[str, Any]] = {}

    if "bar_phase" not in df.columns:
        return stats

    phase_names = get_phase_names(lift_type)
    for phase_id, phase_name in phase_names.items():
        phase_mask = df["bar_phase"] == phase_id
        if phase_mask.any():
            phase_indices = df.index[phase_mask]
            valid_indices = [
                int(i) for i in phase_indices if int(i) < len(temporal_similarity)
            ]
            if valid_indices:
                phase_sims = [float(temporal_similarity[i]) for i in valid_indices]
                stats[phase_id] = {
                    "name": phase_name,
                    "mean": float(np.mean(phase_sims)),
                    "min": float(np.min(phase_sims)),
                    "max": float(np.max(phase_sims)),
                }

    return stats


def write_analysis_md(
    fast_result: Optional[Dict],
    smart_result: Optional[Dict],
    df: pd.DataFrame,
    lift_type: str,
    lifter: str,
    output_path: str,
    faults_config: Optional[Dict] = None,
) -> None:
    """Write the analysis.md report."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Analysis Report: {lift_type.capitalize()}\n\n")

            if fast_result and fast_result.get("available"):
                f.write("## Bar Path Similarity (Fast Analysis)\n\n")

                similarity = fast_result.get("similarity", 0.0)
                stars, label = similarity_to_stars(similarity)

                f.write(
                    f"**Overall Score:** {similarity * 100:.1f}% {stars} ({label})\n\n"
                )

                if lifter and lifter != "generic":
                    f.write(f"**Baseline:** {lifter.replace('_', ' ').title()}\n\n")

                temporal = fast_result.get("temporal_similarity")
                if temporal is not None and len(temporal) > 0:
                    phase_stats = compute_phase_similarity_stats(
                        df, temporal, lift_type
                    )

                    if phase_stats:
                        f.write("**Similarity by Phase:**\n\n")
                        f.write("| Phase | Avg Similarity | Min | Max |\n")
                        f.write("|-------|----------------|-----|-----|\n")

                        for phase_id in [0, 1, 2]:
                            if phase_id in phase_stats:
                                stats = phase_stats[phase_id]
                                avg = stats["mean"] * 100
                                min_s = stats["min"] * 100
                                max_s = stats["max"] * 100

                                indicator = (
                                    "🟢" if avg >= 70 else "🟡" if avg >= 50 else "🔴"
                                )
                                f.write(
                                    f"| {stats['name']} | {indicator} {avg:.1f}% | "
                                    f"{min_s:.1f}% | {max_s:.1f}% |\n"
                                )
                        f.write("\n")
            else:
                f.write("## Bar Path Similarity (Fast Analysis)\n\n")
                f.write("*Fast Analysis not available (no DTW model found).*\n\n")

            if smart_result and smart_result.get("available"):
                f.write("## Technique Critique (Smart Analysis)\n\n")

                flagged = smart_result.get("flagged_faults", [])
                probs = smart_result.get("fault_probabilities", {})

                if flagged:
                    f.write("**Detected Issues:**\n\n")
                    for fault_id in flagged:
                        prob = probs.get(fault_id, 0.0) * 100
                        desc = _get_fault_description(fault_id, faults_config)
                        phase = _get_fault_phase(fault_id, faults_config)
                        f.write(
                            f"- **{fault_id.replace('_', ' ').title()}** ({prob:.0f}% confidence)\n"
                        )
                        f.write(f"  - Phase: {phase}\n")
                        f.write(f"  - {desc}\n\n")
                else:
                    f.write("**No significant issues detected.**\n\n")

                all_faults = list(probs.keys())
                if all_faults:
                    f.write("**All Checks:**\n\n")
                    for fault_id in all_faults:
                        prob = probs.get(fault_id, 0.0) * 100
                        status = "✅" if prob < 50 else "⚠️"
                        f.write(
                            f"- {status} {fault_id.replace('_', ' ').title()}: {prob:.0f}%\n"
                        )
                    f.write("\n")
            else:
                f.write("## Technique Critique (Smart Analysis)\n\n")
                f.write("*Smart Analysis not available (no RF model found).*\n\n")

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

            f.write("\n---\n\n")
            f.write("*Generated by BARPATH ML Analysis Engine*\n")

        print(f"Analysis report saved to '{output_path}'")
    except Exception as e:
        print(f"Error writing analysis.md: {e}")


def _get_fault_description(fault_id: str, faults_config: Optional[Dict]) -> str:
    """Get human-readable description for a fault."""
    if not faults_config or "faults" not in faults_config:
        return "Technique issue detected."

    for fault in faults_config["faults"]:
        if fault.get("id") == fault_id:
            return fault.get("description", "Technique issue detected.")

    return "Technique issue detected."


def _get_fault_phase(fault_id: str, faults_config: Optional[Dict]) -> str:
    """Get the phase where a fault typically occurs."""
    if not faults_config or "faults" not in faults_config:
        return "Unknown"

    for fault in faults_config["faults"]:
        if fault.get("id") == fault_id:
            phase = fault.get("phase", "unknown")

            if isinstance(phase, int):
                return PHASE_NAMES.get("clean", {}).get(phase, f"Phase {phase}")

            if isinstance(phase, str):
                return phase.title()

            return "Unknown"

    return "Unknown"


def critique_lift(
    df: pd.DataFrame,
    lift_type: str = "clean",
    output_dir: str = ".",
    lifter: str = "generic",
    fast_analysis: bool = True,
    smart_analysis: bool = True,
) -> Dict:
    """
    Run technique critique using ML-based analysis.

    Args:
        df: DataFrame with kinematic data from final_analysis.csv
        lift_type: "clean" or "snatch"
        output_dir: Directory to save analysis.md
        lifter: Lifter name for model selection
        fast_analysis: Whether to run DTW-based similarity analysis
        smart_analysis: Whether to run RF-based fault detection

    Returns:
        Dict with analysis results
    """
    models_base = Path(__file__).parent.parent.parent / "models" / "analysis"

    model_dir = resolve_model_path(lifter, lift_type, models_base)

    fast_result = None
    smart_result = None
    faults_config = None

    if model_dir:
        if fast_analysis:
            trajectories, config = load_fast_analysis_model(model_dir)
            if trajectories is not None:
                user_traj = extract_trajectory(df)
                fast_result = run_fast_analysis(user_traj, trajectories, config)

        if smart_analysis:
            model, features_config, faults_config = load_smart_analysis_model(model_dir)
            if model is not None:
                features = extract_smart_features(df)
                smart_result = run_smart_analysis(
                    features, model, features_config, faults_config
                )
    else:
        print(
            f"No analysis models found for lifter='{lifter}', lift_type='{lift_type}'"
        )

    output_path = str(Path(output_dir) / "analysis.md")
    write_analysis_md(
        fast_result, smart_result, df, lift_type, lifter, output_path, faults_config
    )

    results = {
        "fast_analysis": fast_result,
        "smart_analysis": smart_result,
        "lifter": lifter,
        "lift_type": lift_type,
    }

    if fast_result and fast_result.get("available"):
        results["similarity"] = fast_result.get("similarity")

    if smart_result and smart_result.get("available"):
        results["flagged_faults"] = smart_result.get("flagged_faults", [])

    return results


def main():
    parser = argparse.ArgumentParser(description="Step 4: ML-based technique critique.")
    parser.add_argument(
        "--input", default="final_analysis.csv", help="Path to analysis CSV."
    )
    parser.add_argument(
        "--lift_type", required=True, choices=["clean", "snatch", "none"]
    )
    parser.add_argument(
        "--lifter", default="generic", help="Lifter name for model selection."
    )
    parser.add_argument(
        "--output_dir", default=".", help="Output directory for analysis.md"
    )
    parser.add_argument(
        "--no-fast", action="store_true", help="Disable Fast Analysis (DTW)"
    )
    parser.add_argument(
        "--no-smart", action="store_true", help="Disable Smart Analysis (RF)"
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

        critique_lift(
            df,
            args.lift_type,
            args.output_dir,
            args.lifter,
            fast_analysis=not args.no_fast,
            smart_analysis=not args.no_smart,
        )

        print(f"Analysis complete. Results saved to {args.output_dir}/analysis.md")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
