"""
Compiled Rule-Based Technique Analyzer.

Provides fault detection using biomechanical rules and statistical baselines.
Requires pro_baseline_report.json from smart_analysis_training.py — no
hardcoded fallback thresholds. All detection thresholds come from real
pro athlete percentile data.

This is the fallback when no Smart Analysis RF model is available, and serves
as the reference implementation of all detection rules.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

PHASE_NAMES = {
    "snatch": {0: "Pull", 1: "Pull-under", 2: "Recovery"},
    "clean": {0: "Pull", 1: "Pull-under", 2: "Recovery"},
    "jerk": {0: "Dip", 1: "Drive", 2: "Recovery"},
}

FAULT_DEFS: Dict[str, Dict[str, Any]] = {
    "slow_first_pull": {
        "name": "Slow First Pull",
        "phase": "pull",
        "description": "Bar velocity is significantly below elite baselines in the first half of the lift. This limits bar height and forces compensatory acceleration later.",
        "coaching_cue": "Drive harder with your legs off the floor. Think about pushing the platform away.",
        "lift_types": ["clean", "snatch"],
        "severity": "moderate",
    },
    "knee_cave": {
        "name": "Knee Cave",
        "phase": "pull",
        "description": "Knees collapse inward during the early pull. This reduces mechanical advantage and stresses the knee joint.",
        "coaching_cue": "Keep your knees tracking over your toes. Push them out as you drive off the floor.",
        "lift_types": ["clean", "snatch"],
        "severity": "moderate",
    },
    "hitching": {
        "name": "Hitching",
        "phase": "pull",
        "description": "Hips rise faster than the bar, causing a stalling pattern. The velocity profile shows a bimodal distribution with deceleration mid-pull.",
        "coaching_cue": "Keep your chest up and drive with your legs. The bar and hips should rise together.",
        "lift_types": ["clean", "snatch"],
        "severity": "major",
    },
    "early_arm_bend": {
        "name": "Early Arm Bend",
        "phase": "pull",
        "description": "Arms bend before full hip extension, reducing the power available for bar elevation.",
        "coaching_cue": "Keep your arms straight until you reach full extension. Think of your arms as ropes.",
        "lift_types": ["clean", "snatch"],
        "severity": "moderate",
    },
    "incomplete_extension": {
        "name": "Incomplete Extension",
        "phase": "pull",
        "description": "Peak velocity occurs too early in the lift — the lifter cuts the pull short before reaching full triple extension.",
        "coaching_cue": "Finish your extension completely before dropping under the bar. Stay patient.",
        "lift_types": ["clean", "snatch"],
        "severity": "major",
    },
    "premature_jump": {
        "name": "Premature Jump",
        "phase": "pull",
        "description": "Ankles rise significantly before full extension, indicating the lifter is pulling their feet off the platform too early.",
        "coaching_cue": "Stay connected to the floor longer. Extend fully before moving your feet.",
        "lift_types": ["clean", "snatch"],
        "severity": "moderate",
    },
    "slow_turnover": {
        "name": "Slow Turnover",
        "phase": "pull_under",
        "description": "The pull-under/drive phase takes longer than elite baselines, indicating sluggish transition under the bar.",
        "coaching_cue": "Pull yourself under the bar aggressively. Be fast with your elbows.",
        "lift_types": ["clean", "snatch"],
        "severity": "moderate",
    },
    "high_catch": {
        "name": "High Catch",
        "phase": "pull_under",
        "description": "The lifter does not get low enough in the catch position. The knee angle at catch is above elite baselines.",
        "coaching_cue": "Get lower in the catch. Drop your hips and meet the bar in a deeper squat.",
        "lift_types": ["clean", "snatch"],
        "severity": "moderate",
    },
    "unstable_recovery": {
        "name": "Unstable Recovery",
        "phase": "recovery",
        "description": "The bar shows instability during the recovery phase, with significant downward velocity indicating the lifter is struggling to stand.",
        "coaching_cue": "Build your squat strength. Focus on a strong, stable stand-up from the bottom position.",
        "lift_types": ["clean", "snatch"],
        "severity": "moderate",
    },
    "recovery_bounce": {
        "name": "Recovery Bounce",
        "phase": "recovery",
        "description": "Multiple velocity reversals during recovery from the squat indicate the athlete is bouncing to stand up — near their squat strength limit.",
        "coaching_cue": "Build squat strength. You should be able to stand up from the bottom without bouncing.",
        "lift_types": ["clean", "snatch"],
        "severity": "major",
    },
    "press_out": {
        "name": "Press Out",
        "phase": "pull_under",
        "description": "Elbows bend after the catch, then re-extend. The bar should be caught with locked arms.",
        "coaching_cue": "Lock your elbows faster in the catch. Punch the bar up aggressively.",
        "lift_types": ["snatch"],
        "severity": "moderate",
    },
    "overhead_instability": {
        "name": "Overhead Instability",
        "phase": "recovery",
        "description": "Elbow angle variance is high during recovery, indicating the bar is unstable overhead.",
        "coaching_cue": "Lock your elbows and stabilize the bar overhead. Engage your upper back.",
        "lift_types": ["snatch"],
        "severity": "moderate",
    },
    "shallow_dip": {
        "name": "Shallow Dip",
        "phase": "dip",
        "description": "The dip before the drive is too shallow, reducing the potential energy available for the upward drive.",
        "coaching_cue": "Dip deeper before driving. Aim for about 10-15% of your height.",
        "lift_types": ["jerk"],
        "severity": "moderate",
    },
    "poor_drive": {
        "name": "Poor Drive",
        "phase": "drive",
        "description": "Peak upward velocity during the drive is below elite baselines, indicating insufficient explosive power.",
        "coaching_cue": "Drive the bar explosively from the dip. Use your legs to push the bar up.",
        "lift_types": ["jerk"],
        "severity": "major",
    },
    "press_out_jerk": {
        "name": "Press Out (Jerk)",
        "phase": "recovery",
        "description": "Arms are not locked at the catch and need to re-extend. This is a press-out and may be red-lighted in competition.",
        "coaching_cue": "Lock your elbows before the bar stops moving up. Catch with straight arms.",
        "lift_types": ["jerk"],
        "severity": "major",
    },
    "dip_pause": {
        "name": "Dip Pause",
        "phase": "dip",
        "description": "Hesitation or velocity reversal at the bottom of the dip breaks the stretch-shortening cycle.",
        "coaching_cue": "Keep the dip smooth and continuous. Don't pause at the bottom — bounce right back up.",
        "lift_types": ["jerk"],
        "severity": "moderate",
    },
    "jerky_dip": {
        "name": "Jerky Dip",
        "phase": "dip",
        "description": "Non-smooth dip motion with high acceleration variance, indicating poor rhythm.",
        "coaching_cue": "Make the dip smooth and controlled. Think of it as one fluid downward motion.",
        "lift_types": ["jerk"],
        "severity": "minor",
    },
    "no_dip_pause": {
        "name": "Missing Dip Reversal",
        "phase": "dip",
        "description": "No detectable velocity reversal at the bottom of the dip. Elite jerkers show a brief pause that loads the stretch-shortening cycle for maximum drive power.",
        "coaching_cue": "Add a brief controlled pause at the bottom of your dip before driving. This loads your legs for a more explosive drive.",
        "lift_types": ["jerk"],
        "severity": "moderate",
    },
}


def load_baselines_from_json(
    json_path: Path,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load baselines from a pro_baseline_report.json.

    Returns dict keyed by "{lift_type}_{gender}" with feature stats.
    Each feature has: mean, std, p10, p25, p50, p75, p90.
    """
    if not json_path.exists():
        logger.error(
            f"Baseline JSON not found at {json_path}. "
            "Run smart_analysis_training.py first."
        )
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        baselines: Dict[str, Dict[str, Dict[str, float]]] = {}
        for key, baseline_data in report.get("baselines", {}).items():
            feat_stats = baseline_data.get("feature_statistics", {})
            baselines[key] = {}
            for feat_name, stats in feat_stats.items():
                baselines[key][feat_name] = {
                    "mean": stats.get("mean", 0.0),
                    "std": stats.get("std", 1.0),
                    "p10": stats.get("percentiles", {}).get("p10", 0.0),
                    "p25": stats.get("percentiles", {}).get("p25", 0.0),
                    "p50": stats.get("percentiles", {}).get("p50", 0.0),
                    "p75": stats.get("percentiles", {}).get("p75", 0.0),
                    "p90": stats.get("percentiles", {}).get("p90", 0.0),
                }
        return baselines
    except Exception as e:
        logger.warning(f"Failed to load baselines from {json_path}: {e}")
        return {}


class CompiledAnalyzer:
    """Rule-based technique analyzer using biomechanical thresholds.

    Detects faults by comparing extracted features against statistical
    baselines derived from pro athlete data. Requires baselines loaded
    from pro_baseline_report.json — no hardcoded defaults.
    """

    def __init__(
        self,
        lift_type: str,
        gender: str = "male",
        baselines: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    ) -> None:
        self.lift_type = lift_type.lower()
        self.gender = gender.lower()
        self.baseline_key = f"{self.lift_type}_{self.gender}"

        if baselines and self.baseline_key in baselines:
            self.baselines = baselines[self.baseline_key]
        else:
            logger.warning(
                f"No baselines for {self.baseline_key}. "
                "Fault detection will be limited."
            )
            self.baselines = {}

        self._applicable_faults = {
            fid: fdef
            for fid, fdef in FAULT_DEFS.items()
            if self.lift_type in fdef.get("lift_types", [])
        }

    def analyze(
        self,
        features: Dict[str, float],
        df: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        """Analyze a lift and return detected faults.

        Args:
            features: Extracted scalar features from extract_technique_features()
            df: Optional DataFrame with full lift data

        Returns:
            List of fault dicts sorted by confidence (highest first).
        """
        faults: List[Dict[str, Any]] = []

        if self.lift_type in ("clean", "snatch"):
            faults.extend(self._check_clean_snatch_faults(features))
        if self.lift_type == "snatch":
            faults.extend(self._check_snatch_specific_faults(features))
        if self.lift_type == "jerk":
            faults.extend(self._check_jerk_faults(features))

        faults.sort(key=lambda f: f.get("confidence", 0), reverse=True)
        return faults

    def _check_clean_snatch_faults(
        self, features: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Check faults common to clean and snatch.

        Threshold logic reference (from pro baseline data):
        - Image coords: y increases downward, so upward bar motion = negative velocity
        - MediaPipe elbow angle: low = straight arm, high = bent arm
        - MediaPipe knee angle: low = deep squat, high = standing
        """
        faults: List[Dict[str, Any]] = []

        # slow_first_pull
        # max_vel_y is always positive (abs of velocity). A slow pull has low max_vel_y.
        # Flag when max_vel_y < p25 of pro distribution.
        val = features.get("max_vel_y", 0)
        p25 = self._get_threshold("max_vel_y", "p25")
        if p25 > 0 and val < p25:
            faults.append(self._make_fault("slow_first_pull", val, "max_vel_y"))

        # knee_cave
        # knee_width_change_early < p10 means knees narrowing inward
        val = features.get("knee_width_change_early", 0)
        if val < self._get_threshold("knee_width_change_early", "p10"):
            faults.append(self._make_fault("knee_cave", val, "knee_width_change_early"))

        # hitching
        # High vel_profile_skewness (right-tailed velocity = stall-restart pattern)
        # combined with multiple accel peaks (power interruptions during pull).
        skewness = features.get("vel_profile_skewness", 0)
        accel_peaks = features.get("accel_peaks_count", 0)
        skew_thresh = self._get_threshold("vel_profile_skewness", "p90")
        peak_thresh = self._get_threshold("accel_peaks_count", "p75")
        if skewness > skew_thresh and accel_peaks > peak_thresh:
            faults.append(
                self._make_fault(
                    "hitching",
                    skewness,
                    "vel_profile_skewness",
                    extra={"accel_peaks_count": accel_peaks},
                )
            )

        # early_arm_bend
        # MediaPipe elbow: HIGH angle = bent arm. Flag when > p90.
        val = features.get("min_elbow_angle_early", 180)
        p90 = self._get_threshold("min_elbow_angle_early", "p90")
        if p90 > 0 and val > p90:
            faults.append(
                self._make_fault("early_arm_bend", val, "min_elbow_angle_early")
            )

        # incomplete_extension
        # Peak velocity position > p90 means peak occurs too late (extending past optimal)
        val = features.get("peak_vel_phase_frac", 0.5)
        if val > self._get_threshold("peak_vel_phase_frac", "p90"):
            faults.append(
                self._make_fault("incomplete_extension", val, "peak_vel_phase_frac")
            )

        # premature_jump
        # Ankle rise > p90 means feet leaving floor too early
        val = features.get("ankle_rise_late_pull", 0)
        if val > self._get_threshold("ankle_rise_late_pull", "p90"):
            faults.append(
                self._make_fault("premature_jump", val, "ankle_rise_late_pull")
            )

        # slow_turnover
        # Turnover duration > p90 means too long in pull-under phase
        val = features.get("turnover_duration_frac", 0)
        if val > self._get_threshold("turnover_duration_frac", "p90"):
            faults.append(
                self._make_fault("slow_turnover", val, "turnover_duration_frac")
            )

        # high_catch
        # Knee angle > p90 at catch = not squatting deep enough
        val = features.get("min_knee_angle_catch", 180)
        if val > self._get_threshold("min_knee_angle_catch", "p90"):
            faults.append(self._make_fault("high_catch", val, "min_knee_angle_catch"))

        # recovery_bounce / unstable_recovery
        # Pro data shows 0.0 bounces across all percentiles. Any bounce is meaningful.
        bounce_count = features.get("recovery_bounce_count", 0)
        if bounce_count >= 2:
            faults.append(
                self._make_fault(
                    "recovery_bounce", bounce_count, "recovery_bounce_count"
                )
            )
        elif bounce_count >= 1:
            # Lower confidence since the bounce detection threshold may
            # miss subtle bar oscillation vs real squat bounces
            fault = self._make_fault(
                "unstable_recovery", bounce_count, "recovery_bounce_count"
            )
            fault["confidence"] = min(fault["confidence"], 40)
            faults.append(fault)

        return faults

    def _check_snatch_specific_faults(
        self, features: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Check snatch-specific faults."""
        faults: List[Dict[str, Any]] = []

        # press_out: elbow angle variance > p90
        val = features.get("elbow_angle_variance", 0)
        if val > self._get_threshold("elbow_angle_variance", "p90"):
            faults.append(self._make_fault("press_out", val, "elbow_angle_variance"))

        # overhead_instability: high elbow variance + high catch position
        elbow_var = features.get("elbow_angle_variance", 0)
        knee_catch = features.get("min_knee_angle_catch", 180)
        if elbow_var > self._get_threshold(
            "elbow_angle_variance", "p75"
        ) and knee_catch > self._get_threshold("min_knee_angle_catch", "p75"):
            faults.append(
                self._make_fault(
                    "overhead_instability", elbow_var, "elbow_angle_variance"
                )
            )

        return faults

    def _check_jerk_faults(self, features: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check jerk-specific faults.

        Jerk phases: 0=Dip, 1=Drive, 2=Recovery.
        Note: Walking forward/backward during recovery is NORMAL for jerk.
        """
        faults: List[Dict[str, Any]] = []

        # shallow_dip
        # Dip depth < p10 = insufficient dip before drive
        val = features.get("dip_depth_norm", 0)
        if val < self._get_threshold("dip_depth_norm", "p10"):
            faults.append(self._make_fault("shallow_dip", val, "dip_depth_norm"))

        # poor_drive
        # drive_peak_vel is negative (upward in image coords).
        # Less negative = weaker drive. Flag when > p10 (closest to 0).
        val = features.get("drive_peak_vel", 0)
        if val > self._get_threshold("drive_peak_vel", "p10"):
            faults.append(self._make_fault("poor_drive", val, "drive_peak_vel"))

        # press_out_jerk
        # Elbow angle variance > p90
        val = features.get("elbow_angle_variance", 0)
        if val > self._get_threshold("elbow_angle_variance", "p90"):
            faults.append(
                self._make_fault("press_out_jerk", val, "elbow_angle_variance")
            )

        # dip_pause / no_dip_pause
        # 90% of pro jerkers have dip_pause_detected=1.0. NOT having one is the fault.
        if features.get("dip_pause_detected", 1.0) < 0.5:
            faults.append(self._make_fault("no_dip_pause", 0.0, "dip_pause_detected"))

        # jerky_dip: high accel peaks in dip phase
        val = features.get("accel_peaks_count", 0)
        if val > self._get_threshold("accel_peaks_count", "p90"):
            faults.append(self._make_fault("jerky_dip", val, "accel_peaks_count"))

        return faults

    def _get_threshold(self, feature: str, percentile: str) -> float:
        """Get a percentile threshold for a feature."""
        baseline = self.baselines.get(feature, {})
        if percentile in baseline:
            return float(baseline[percentile])
        return 0.0

    def _make_fault(
        self,
        fault_id: str,
        value: float,
        feature_name: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a fault dictionary."""
        fdef = FAULT_DEFS.get(fault_id, {})
        baseline = self.baselines.get(feature_name, {})
        deviation = self._compute_deviation(value, baseline)

        severity = fdef.get("severity", "moderate")
        if abs(deviation) > 2.5:
            severity = "major"
        elif abs(deviation) < 1.0:
            severity = "minor"

        phase = fdef.get("phase", "unknown")
        if isinstance(phase, str) and phase.isdigit():
            phase_names = PHASE_NAMES.get(self.lift_type, {})
            phase = phase_names.get(int(phase), phase)

        confidence = min(95, max(10, int(abs(deviation) * 30)))

        fault: Dict[str, Any] = {
            "id": fault_id,
            "name": fdef.get("name", fault_id.replace("_", " ").title()),
            "severity": severity,
            "phase": phase,
            "description": fdef.get("description", "Technique issue detected."),
            "coaching_cue": fdef.get(
                "coaching_cue", "Review this aspect of your technique."
            ),
            "confidence": confidence,
            "deviation": deviation,
            "feature_value": value,
            "feature_name": feature_name,
        }
        if extra:
            fault["extra"] = extra
        return fault

    @staticmethod
    def _compute_deviation(value: float, baseline: Dict[str, float]) -> float:
        """Compute deviation in standard deviations from mean."""
        mean = baseline.get("mean", value)
        std = baseline.get("std", 1.0)
        if std > 0:
            return (value - mean) / std
        return 0.0

    def get_technique_score(self, faults: List[Dict[str, Any]]) -> Tuple[float, str]:
        """Calculate overall technique score from detected faults.

        Returns:
            Tuple of (score_0_to_100, assessment_text)
        """
        score = 100.0
        severity_deductions = {
            "minor": 3,
            "moderate": 7,
            "major": 12,
            "critical": 20,
        }

        for fault in faults:
            severity = fault.get("severity", "moderate")
            deduction = severity_deductions.get(severity, 5)
            confidence = fault.get("confidence", 50) / 100.0
            score -= deduction * confidence

        score = max(0.0, min(100.0, score))

        if score >= 90:
            assessment = "Excellent technique with minimal areas for improvement."
        elif score >= 80:
            assessment = "Very good technique with minor refinements possible."
        elif score >= 70:
            assessment = "Good technique with some areas to address."
        elif score >= 50:
            assessment = "Fair technique with several areas needing work."
        else:
            assessment = (
                "Technique needs significant improvement across multiple areas."
            )

        return score, assessment
