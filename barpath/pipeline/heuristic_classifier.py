"""
Heuristic lift detection classifier.

Uses kinematic features for reliable classification without requiring
a trained ML model. Serves as the fallback when no model pickle is found.

Feature map (from _MODEL_FEATURE_NAMES, alphabetical):
  [2] cj_two_phase_detected:  boolean (0.0/1.0) — two-phase C+J pattern
  [3] dip_depth_norm:         float — depth of catch/pull-under (snatch=~0.4, clean=~0.3, jerk=~0.09)
  [12] phase_n_peaks:         int — number of velocity peaks (jerk=2)
  [18] s_curve_detected:      boolean — S-curve in bar path (clean/snatch)
  [29] shape_y_start:         float — normalized starting bar position
"""

from typing import Any

import numpy as np

# Import classification thresholds from config (fallback if running outside pipeline)
try:
    from config import (
        HEURISTIC_CONFIDENCE_CEILING,
        HEURISTIC_CONFIDENCE_FLOOR,
        LIFT_CLASS_DIP_DEPTH_CLEAN,
        LIFT_CLASS_DIP_DEPTH_SNATCH,
        LIFT_CLASS_Y_START_JERK_MAX,
    )
except ImportError:
    LIFT_CLASS_DIP_DEPTH_SNATCH = 0.37
    LIFT_CLASS_DIP_DEPTH_CLEAN = 0.12
    LIFT_CLASS_Y_START_JERK_MAX = 0.85
    HEURISTIC_CONFIDENCE_FLOOR = 0.33
    HEURISTIC_CONFIDENCE_CEILING = 0.99

# Jerk-only vs clean+jerk boundary: isolated jerk lifts have a shallow dip
# (dip_depth < 0.15), while combined clean+jerk-lifts have deeper pull-under
# from the clean phase.
JERK_DIP_DEPTH_MAX = 0.18

# Clean+jerk heuristic thresholds (relaxed for live-preview data where
# the recording may capture a shorter window of the full lift).
CJ_PHASE_GAP_MIN = 0.45
CJ_TRAJ_LEN_MIN = 120
CJ_TRAJ_LEN_STRICT = 160
CJ_PHASE_GAP_STRICT = 0.55


def classify_with_heuristics(features: np.ndarray) -> dict[str, Any]:
    """Classify lift type using kinematic feature heuristics.

    Args:
        features: 37-element feature array from extract_model_features_as_array

    Returns:
        Dict with 'class', 'confidence', 'probabilities', and 'reason'.
    """
    if features is None or len(features) < 37:
        return {
            "class": "none",
            "confidence": 0.0,
            "probabilities": {},
            "reason": "insufficient_features",
        }

    # Extract diagnostic features (alphabetical order indices)
    cj_two_phase = float(features[2])  # 0=no, 1=yes — C+J pattern detected
    dip_depth = float(features[3])  # normalized depth of pull-under/catch
    cj_phase_gap = float(features[1])  # temporal gap between peaks (normalized)
    phase_n_peaks = float(features[12])  # number of velocity peaks
    s_curve = float(features[18])  # S-curve pattern detected
    shape_y_start = float(features[29])  # normalized bar position at start
    trajectory_len = float(features[30])  # total trajectory frame count
    vel_range = float(features[35])  # velocity range: max(vel) - min(vel)

    probs = {"snatch": 0.33, "clean": 0.33, "jerk": 0.33, "clean_jerk": 0.0}
    reason: str

    # --- Tier 1: Two-phase pattern detection (jerk or clean+jerk) ---
    # cj_two_phase_detected fires for both isolated jerk (dip+drive peaks)
    # and genuine clean+jerk combos. Distinguish by:
    #   - dip_depth: jerk alone has shallow dip (< 0.18)
    #   - cj_phase_gap: real C+J has widely separated peaks (> 0.55),
    #     pure cleans with double-pull have narrow gap (< 0.50)
    #   - trajectory_length: C+J is much longer than single lift
    if cj_two_phase > 0.5 and dip_depth < JERK_DIP_DEPTH_MAX:
        # Isolated jerk — two velocity peaks from dip+drive, shallow dip
        probs = {"snatch": 0.05, "clean": 0.10, "jerk": 0.85, "clean_jerk": 0.0}
        predicted = "jerk"
        confidence = min(
            0.85 + (JERK_DIP_DEPTH_MAX - dip_depth) * 1.2,
            HEURISTIC_CONFIDENCE_CEILING,
        )
        reason = "jerk_two_phase_shallow_dip"

    elif (
        cj_two_phase > 0.5
        and dip_depth >= JERK_DIP_DEPTH_MAX
        and cj_phase_gap > CJ_PHASE_GAP_STRICT
        and trajectory_len > CJ_TRAJ_LEN_STRICT
    ):
        # Genuine clean+jerk: two-phase with large gap + long trajectory
        probs = {"snatch": 0.03, "clean": 0.07, "jerk": 0.10, "clean_jerk": 0.80}
        predicted = "clean_jerk"
        confidence = 0.80 + min(cj_phase_gap - CJ_PHASE_GAP_STRICT, 0.15) / 0.15 * 0.15
        reason = "clean_jerk_large_gap_long_traj"

    elif (
        cj_two_phase > 0.5 and dip_depth >= JERK_DIP_DEPTH_MAX and trajectory_len > CJ_TRAJ_LEN_MIN
    ):
        # Two-phase pattern with moderate trajectory length and deep
        # dip — more likely a clean+jerk than a single lift mimicking one.
        # This relaxed path catches shorter recordings where the full
        # clean+jerk is captured but tighter thresholds aren't met.
        probs = {"snatch": 0.07, "clean": 0.13, "jerk": 0.10, "clean_jerk": 0.70}
        predicted = "clean_jerk"
        confidence = 0.65 + min((trajectory_len - CJ_TRAJ_LEN_MIN) / 40.0, 0.15)
        reason = "clean_jerk_moderate_traj"

    elif cj_two_phase > 0.5 and dip_depth >= JERK_DIP_DEPTH_MAX:
        # Two-phase pattern but not clearly C+J. Check vel_range
        # to distinguish snatch double-pull (symmetric, < 1.45)
        # from clean double-pull (asymmetric, >= 1.45).
        if vel_range <= 1.45:
            probs = {"snatch": 0.80, "clean": 0.15, "jerk": 0.03, "clean_jerk": 0.02}
            predicted = "snatch"
            confidence = 0.80
            reason = "snatch_double_pull_symmetric"
        else:
            probs = {"snatch": 0.10, "clean": 0.80, "jerk": 0.05, "clean_jerk": 0.05}
            predicted = "clean"
            confidence = 0.75
            reason = "clean_double_pull_mimics_cj"

    # --- Tier 3: Jerk (shallow dip + multi-peak, no two-phase flag) ---
    elif dip_depth < LIFT_CLASS_DIP_DEPTH_CLEAN and phase_n_peaks >= 1.5:
        probs["jerk"] = 0.85
        probs["clean"] = 0.10
        probs["snatch"] = 0.05
        predicted = "jerk"
        confidence = min(
            0.85 + (LIFT_CLASS_DIP_DEPTH_CLEAN - dip_depth) * 1.5,
            HEURISTIC_CONFIDENCE_CEILING,
        )
        reason = "jerk_shallow_dip_multi_peak"

    elif dip_depth < LIFT_CLASS_DIP_DEPTH_CLEAN:
        # No multi-peak but shallow dip — still likely jerk
        probs["jerk"] = 0.70
        probs["clean"] = 0.20
        probs["snatch"] = 0.10
        predicted = "jerk"
        confidence = 0.70 + (LIFT_CLASS_DIP_DEPTH_CLEAN - dip_depth) * 0.80
        reason = "jerk_shallow_dip_only"

    # --- Tier 2: Jerk detection via shoulder-start position ---
    # If bar starts at shoulders (shape_y_start < 0.85), it cannot be
    # snatch or clean regardless of dip_depth. This catches jerks with
    # slightly elevated dip_depth that would otherwise fall through.
    elif shape_y_start < LIFT_CLASS_Y_START_JERK_MAX:
        if dip_depth < LIFT_CLASS_DIP_DEPTH_CLEAN or phase_n_peaks >= 1.5:
            probs = {"snatch": 0.03, "clean": 0.07, "jerk": 0.90, "clean_jerk": 0.0}
        else:
            probs = {"snatch": 0.05, "clean": 0.10, "jerk": 0.85, "clean_jerk": 0.0}
        predicted = "jerk"
        confidence = min(
            0.85 + (LIFT_CLASS_Y_START_JERK_MAX - shape_y_start),
            HEURISTIC_CONFIDENCE_CEILING,
        )
        reason = "jerk_shoulder_start"

    # --- Tier 4: Snatch vs Clean (floor-start lifts) ---
    # dip_depth_norm has overlap between snatch (0.27-0.55) and clean
    # (0.20-0.50). Use vel_range as a tiebreaker: snatch has symmetric
    # velocity (vel_range < 1.55), clean has asymmetric (vel_range > 1.65).
    elif dip_depth >= LIFT_CLASS_DIP_DEPTH_SNATCH and vel_range > 1.65:
        # Deep pull-under but highly asymmetric velocity — clean
        probs["snatch"] = 0.10
        probs["clean"] = 0.87
        probs["jerk"] = 0.03
        predicted = "clean"
        confidence = min(
            0.80 + (vel_range - 1.65) * 0.30,
            HEURISTIC_CONFIDENCE_CEILING,
        )
        reason = "clean_deep_under_asymmetric"

    elif dip_depth >= LIFT_CLASS_DIP_DEPTH_SNATCH:
        # Deep pull-under with normal velocity symmetry — snatch
        probs["snatch"] = 0.85
        probs["clean"] = 0.12
        probs["jerk"] = 0.03
        predicted = "snatch"
        confidence = min(
            0.85 + (dip_depth - LIFT_CLASS_DIP_DEPTH_SNATCH),
            HEURISTIC_CONFIDENCE_CEILING,
        )
        reason = "snatch_deep_pull_under"

    elif dip_depth >= LIFT_CLASS_DIP_DEPTH_CLEAN and vel_range <= 1.55:
        # Shallow pull-under but symmetric velocity — snatch (e.g. Ilyin)
        probs["snatch"] = 0.80
        probs["clean"] = 0.17
        probs["jerk"] = 0.03
        predicted = "snatch"
        confidence = 0.80 + (1.55 - vel_range) * 0.30
        reason = "snatch_shallow_under_symmetric"

    elif dip_depth >= LIFT_CLASS_DIP_DEPTH_CLEAN:
        # Medium pull-under — clean
        # Boost confidence when S-curve is present
        if s_curve > 0.5:
            probs["clean"] = 0.88
            probs["snatch"] = 0.10
            probs["jerk"] = 0.02
            confidence = 0.88
            reason = "clean_medium_depth_s_curve"
        else:
            probs["clean"] = 0.80
            probs["snatch"] = 0.15
            probs["jerk"] = 0.05
            confidence = min(
                0.80 + abs(dip_depth - 0.25),
                HEURISTIC_CONFIDENCE_CEILING,
            )
            reason = "clean_medium_depth"
        predicted = "clean"

    else:
        # Fallback: use shape_y_start for jerk detection
        if shape_y_start < LIFT_CLASS_Y_START_JERK_MAX:
            probs["jerk"] = 0.65
            probs["clean"] = 0.25
            probs["snatch"] = 0.10
            predicted = "jerk"
            confidence = 0.65
            reason = "fallback_jerk_low_start"
        else:
            probs["clean"] = 0.50
            probs["snatch"] = 0.40
            probs["jerk"] = 0.10
            predicted = "clean"
            confidence = 0.50
            reason = "fallback_clean_floor_start"

    # Clamp confidence
    confidence = max(HEURISTIC_CONFIDENCE_FLOOR, min(confidence, HEURISTIC_CONFIDENCE_CEILING))

    return {
        "class": predicted,
        "confidence": confidence,
        "probabilities": probs,
        "reason": reason,
    }


def classify_with_heuristics_smoothed(
    features_list: list[np.ndarray],
) -> dict[str, Any]:
    """Classify using multiple frames for smoother results.

    Args:
        features_list: List of feature arrays from consecutive frames

    Returns:
        Averaged prediction result with consensus reason.
    """
    if not features_list:
        return {
            "class": "none",
            "confidence": 0.0,
            "probabilities": {},
            "reason": "no_frames",
        }

    results = [classify_with_heuristics(f) for f in features_list]

    # Average probabilities across frames
    avg_probs: dict[str, float] = {"snatch": 0.0, "clean": 0.0, "jerk": 0.0}
    if any(r.get("clean_jerk", 0.0) for r in results):
        avg_probs["clean_jerk"] = 0.0
    for r in results:
        probs = r["probabilities"]
        for k in avg_probs:
            avg_probs[k] += probs.get(k, 0.0)

    for k in avg_probs:
        avg_probs[k] /= len(results)

    # Get consensus
    predicted = max(avg_probs, key=lambda k: avg_probs[k])
    confidence = avg_probs[predicted]

    # Majority vote for reason
    reasons = [r.get("reason", "unknown") for r in results]
    reason = max(set(reasons), key=reasons.count) if reasons else "consensus"

    return {
        "class": predicted,
        "confidence": confidence,
        "probabilities": avg_probs,
        "reason": reason,
    }


if __name__ == "__main__":
    # Quick test
    import glob
    import sys

    sys.path.insert(0, "barpath")
    import pandas as pd
    from pipeline.lift_detection_features import extract_model_features_as_array

    print("Testing heuristic classifier:")
    for cat in ["snatch", "clean", "jerk"]:
        files = list(glob.glob(f"outputs/male/{cat}/*/final_analysis.csv"))[:10]
        if not files:
            print(f"  {cat}: no files found")
            continue
        correct = 0
        for f in files:
            df = pd.read_csv(f)
            features = extract_model_features_as_array(df)
            result = classify_with_heuristics(features)
            pred = result["class"]
            reason = result.get("reason", "?")
            if pred == cat:
                correct += 1
            else:
                print(
                    f"  **MISCLASSIFIED** {f.split('/')[-2]}: "
                    f"expected={cat}, predicted={pred}, reason={reason}"
                )
        acc = correct / len(files) if files else 0
        print(f"  {cat} accuracy: {correct}/{len(files)} = {acc:.1%}")
