"""Live coaching tip computation — heuristic fault checks on buffered lift data."""

from typing import Dict, List, Optional

from barpath.pipeline.config import COACHING_TIP_CONFIDENCE_THRESHOLD, COACHING_TIP_FALLBACK
from .live_buffer import FrameData


def compute_coaching_tip(buffer_frames: List[FrameData], detected_class: str) -> Optional[str]:
    """Run heuristic checks on buffered lift data. Return fault name or fallback.

    Checks (simplified versions of compiled_analyzer rules):
    - early_arm_bend: avg elbow angle during pull > 150° (straight arm = good)
    - hitching: velocity sign changes (reversals) in barbell vertical velocity
    - incomplete_extension: peak velocity occurs before 70% of lift duration

    Returns fault name if confidence > COACHING_TIP_CONFIDENCE_THRESHOLD (0.6),
    otherwise COACHING_TIP_FALLBACK ("Lift looks good").
    """
    if not buffer_frames or len(buffer_frames) < 10:
        return COACHING_TIP_FALLBACK

    checks = {}

    # Early arm bend check
    elbow_check = _check_early_arm_bend(buffer_frames)
    if elbow_check:
        checks["early_arm_bend"] = elbow_check

    # Hitching check (velocity reversals)
    hitch_check = _check_hitching(buffer_frames)
    if hitch_check:
        checks["hitching"] = hitch_check

    # Incomplete extension check
    ext_check = _check_peak_velocity_timing(buffer_frames)
    if ext_check:
        checks["incomplete_extension"] = ext_check

    # Filter by class-appropriate checks (snatch/clean don't have jerk faults)
    if detected_class in ("snatch", "clean"):
        pass  # all checks applicable
    elif detected_class == "jerk":
        pass  # all checks applicable

    # Find highest confidence fault
    if checks:
        best_fault = max(checks, key=checks.get)
        if checks[best_fault] > COACHING_TIP_CONFIDENCE_THRESHOLD:
            # Convert fault_id to display name
            name_map = {
                "early_arm_bend": "Early Arm Bend",
                "hitching": "Hitching",
                "incomplete_extension": "Incomplete Extension",
                "knee_cave": "Knee Cave",
            }
            return name_map.get(best_fault, best_fault.replace("_", " ").title())

    return COACHING_TIP_FALLBACK


def _check_early_arm_bend(frames: List[FrameData]) -> Optional[float]:
    """Check if average elbow angle during lift > 150°. Return confidence 0.0-1.0."""
    # Extract elbow angles from frames
    angles = []
    for f in frames:
        left = f.joint_angles.get('left_elbow', 180.0)
        right = f.joint_angles.get('right_elbow', 180.0)
        angles.append((left + right) / 2.0)
    if not angles:
        return None
    avg_elbow = sum(angles) / len(angles)
    # > 150° = straight arm (fault). Confidence scales with deviation.
    if avg_elbow > 150:
        return min(1.0, (avg_elbow - 150) / 30.0)  # 150-180° maps to 0.0-1.0
    return None


def _check_hitching(frames: List[FrameData]) -> Optional[float]:
    """Check for velocity reversals (sign changes) in barbell vertical velocity."""
    import numpy as np
    positions = []
    for f in frames:
        if f.barbell_center:
            positions.append(f.barbell_center[1])  # y position
    if len(positions) < 3:
        return None
    velocities = np.diff(positions)
    # Count sign changes (reversals)
    sign_changes = int(np.sum(np.diff(np.signbit(velocities))))
    if sign_changes >= 2:
        return min(1.0, sign_changes / 4.0)  # 2-4+ reversals maps to 0.5-1.0
    return None


def _check_peak_velocity_timing(frames: List[FrameData]) -> Optional[float]:
    """Check if peak velocity occurs before 70% of lift duration."""
    import numpy as np
    positions = []
    for f in frames:
        if f.barbell_center:
            positions.append(f.barbell_center[1])
    if len(positions) < 3:
        return None
    velocities = np.abs(np.diff(positions))
    if len(velocities) == 0:
        return None
    peak_idx = int(np.argmax(velocities))
    peak_fraction = peak_idx / len(velocities)
    if peak_fraction < 0.7:
        return min(1.0, (0.7 - peak_fraction) / 0.3)  # earlier = higher confidence
    return None
