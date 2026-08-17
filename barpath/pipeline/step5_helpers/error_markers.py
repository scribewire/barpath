"""Error markers for fault visualization on bar path.

Draws colored triangles at fault frame positions on the barbell path,
with fault name labels below each marker.
"""

import cv2
import numpy as np

from barpath.pipeline.config import (
    ERROR_TEXT_Y_OFFSET,
    ERROR_TRIANGLE_SIZE,
    FAULT_COLORS_BGR,
)

from .overlay_metrics import OverlayMetrics

# Mapping of fault IDs to fault categories for color assignment
FAULT_CATEGORY_MAP = {
    "early_arm_bend": "arm",
    "press_out": "arm",
    "incomplete_extension": "extension",
    "premature_jump": "extension",
    "hitching": "path",
    "slow_turnover": "path",
    "knee_cave": "knee_leg",
    "unstable_recovery": "knee_leg",
    "recovery_bounce": "knee_leg",
    "high_catch": "catch",
    "slow_first_pull": "catch",
}


def find_fault_frame(df, fault_id, lift_type):
    """Find frame index where a fault is most visible.

    Uses per-fault heuristics based on the known detection logic.

    Args:
        df: DataFrame with per-frame kinematic data
        fault_id: Fault identifier string
        lift_type: Type of lift (snatch, clean, jerk, clean_jerk)

    Returns:
        Frame index (int) or None if column not available
    """
    if fault_id in ("early_arm_bend", "press_out"):
        # Find frame with max elbow angle during pull phase (phase 0)
        pull_mask = df["bar_phase"] == 0
        elbow_cols = [c for c in df.columns if "elbow_angle" in c]
        if elbow_cols and pull_mask.any():
            pull_frames = df[pull_mask]
            col = elbow_cols[0]
            if col in pull_frames.columns:
                return pull_frames[col].idxmax()

    elif fault_id in ("hitching", "slow_turnover"):
        # Find first velocity reversal (acceleration sign change)
        if "accel_y_smooth" in df.columns:
            accel = df["accel_y_smooth"].dropna()
            if len(accel) > 1:
                sign_changes = np.where(np.diff(np.signbit(accel)))[0]
                if len(sign_changes) > 0:
                    return accel.index[sign_changes[0]]

    elif fault_id in ("incomplete_extension", "premature_jump"):
        # Find frame where peak velocity occurs
        if "vel_y_smooth" in df.columns:
            vel = df["vel_y_smooth"].dropna()
            if len(vel) > 0:
                return vel.idxmax()

    elif fault_id in ("knee_cave", "unstable_recovery", "recovery_bounce"):
        # Place at midpoint of relevant phase (phase 0 for knee_cave, phase 2 for recovery)
        target_phase = 0 if fault_id == "knee_cave" else 2
        phase_mask = df["bar_phase"] == target_phase
        if phase_mask.any():
            phase_indices = df[phase_mask].index
            mid_idx = len(phase_indices) // 2
            return phase_indices[mid_idx]

    elif fault_id in ("high_catch", "slow_first_pull"):
        # Place at phase 1→2 transition midpoint
        phase1_mask = df["bar_phase"] == 1
        if phase1_mask.any():
            phase1_indices = df[phase1_mask].index
            return phase1_indices[-1]

    return None


def draw_error_markers(
    frame,
    analysis_result,
    df,
    path_points,
    path_phases,
    max_path_index,
    shake_x,
    shake_y,
    lift_type,
    overlay_metrics=None,
):
    """Draw fault error markers on bar path.

    Top 3 faults by confidence rendered as colored triangles.
    Remaining faults returned as list for legend extension.

    Args:
        frame: OpenCV frame/image to draw on (modified in place)
        analysis_result: Dict from Step 4 critique_lift (must have 'compiled_faults')
        df: DataFrame with per-frame kinematic data
        path_points: Nx2 array of barbell positions
        path_phases: N array of phase indices
        max_path_index: Number of path points to draw
        shake_x: X shake offset
        shake_y: Y shake offset
        lift_type: Type of lift for phase color mapping

    Returns:
        List of remaining fault names (beyond top 3) for legend
    """
    if analysis_result is None:
        return []

    metrics = overlay_metrics or OverlayMetrics.for_frame(frame.shape[1], frame.shape[0])
    faults = analysis_result.get("compiled_faults", [])
    if not faults:
        return []

    # Sort by confidence descending, take top 3
    sorted_faults = sorted(faults, key=lambda f: f.get("confidence", 0), reverse=True)
    top_faults = sorted_faults[:3]
    remaining_faults = [f.get("name", f.get("id", "Unknown")) for f in sorted_faults[3:]]

    for fault in top_faults:
        fault_id = fault.get("id", "")
        fault_name = fault.get("name", fault_id)
        confidence = fault.get("confidence", 0)

        # Validate confidence is numeric
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            continue

        # Find fault frame
        fault_frame_idx = find_fault_frame(df, fault_id, lift_type)
        if fault_frame_idx is None:
            continue

        # Map frame index to bar path coordinates
        # Find closest path point at that frame index
        frame_idx = int(fault_frame_idx)
        if frame_idx < 0 or frame_idx >= max_path_index:
            continue

        path_x = path_points[frame_idx, 0]
        path_y = path_points[frame_idx, 1]

        # Apply shake offsets
        px = int(path_x + shake_x)
        py = int(path_y + shake_y)

        # Determine fault category color
        category = FAULT_CATEGORY_MAP.get(fault_id, "catch")
        color = FAULT_COLORS_BGR.get(category, (0, 0, 255))

        # Draw filled triangle
        size = metrics.px(ERROR_TRIANGLE_SIZE)
        pts = np.array(
            [
                [px, py - size],  # top point (apex)
                [px - size // 2, py + size // 2],  # bottom-left
                [px + size // 2, py + size // 2],  # bottom-right
            ],
            np.int32,
        )
        cv2.fillPoly(frame, [pts], color)

        # Draw fault name label below triangle with black background
        label_y = py + metrics.px(ERROR_TEXT_Y_OFFSET)
        font_scale = metrics.font(0.6)
        text_thickness = metrics.px(2)
        text_size = cv2.getTextSize(
            fault_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )[0]
        pad = metrics.px(4)
        text_x = px - text_size[0] // 2
        text_x = max(pad, min(frame.shape[1] - text_size[0] - pad, text_x))
        label_y = min(frame.shape[0] - pad, max(text_size[1] + pad, label_y))
        bg_tl = (text_x - pad, label_y - text_size[1] - pad)
        bg_br = (text_x + text_size[0] + pad, label_y + pad)
        cv2.rectangle(frame, bg_tl, bg_br, (0, 0, 0), -1)
        cv2.putText(
            frame,
            fault_name,
            (text_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            text_thickness,
            cv2.LINE_AA,
        )

    return remaining_faults
