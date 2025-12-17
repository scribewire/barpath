from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .classics_phase_detection import ClassicsPhases


def compute_bar_phase_from_snatch_phases(
    df: pd.DataFrame, phases: ClassicsPhases, *, phase_col: str = "bar_phase"
) -> pd.Series:
    """
    Build a per-frame phase label series from snatch phase boundaries.

    Output matches the pipeline's convention of integer "phase IDs" that increment
    at each boundary:
      - Phase 0: [t0, t1)
      - Phase 1: [t1, t2)
      - Phase 2: [t2, t3)
      - Phase 3: [t3, t4]  (inclusive end)

    Frames before t0 (if present) are assigned phase 0 by default.
    """
    if df.empty:
        return pd.Series(dtype="int64", name=phase_col)

    idx = df.index
    phase = pd.Series(0, index=idx, dtype="int64", name=phase_col)

    t0 = int(phases["t0"])
    t1 = int(phases["t1"])
    t2 = int(phases["t2"])
    t3 = int(phases["t3"])
    t4 = int(phases["t4"])

    # Assign phases by index slices. `.loc` is inclusive on both ends for label-based indices,
    # so we use half-open ranges by subtracting 1 where appropriate when indices are integers.
    # Because frame indices may not be contiguous, we apply masks instead of slicing by position.
    phase.loc[idx >= t1] = 1
    phase.loc[idx >= t2] = 2
    phase.loc[idx >= t3] = 3

    # If t4 is earlier than some boundary (shouldn't happen, but guard), clamp.
    phase.loc[idx > t4] = phase.loc[idx <= t4].iloc[-1] if (idx <= t4).any() else 0

    # If there are frames before t0 (rare after Step 2 truncation), keep them as 0.
    # Optionally you could set them to -1, but that would break downstream assumptions.
    _ = t0  # kept for clarity

    return phase


def check_snatch_faults(df: pd.DataFrame, phases: ClassicsPhases) -> List[str]:
    """
    Snatch-specific critique checks.

    Returns a list of critique strings suitable for report output.
    """
    critiques: List[str] = []

    t0, t1, t2, t3, t4 = (
        phases["t0"],
        phases["t1"],
        phases["t2"],
        phases["t3"],
        phases["t4"],
    )

    try:
        frame_height = float(df["frame_height"].iloc[0])
    except Exception:
        frame_height = 0.0

    def get_phase_df(start: int, end: int) -> pd.DataFrame:
        return df.loc[start:end]

    # --- First Pull (t0 -> t1) ---
    p1 = get_phase_df(t0, t1)
    if not p1.empty:
        # Check 1: Hips rising before knees extend (hitching/tipping forward)
        if (
            "hip_y_avg" in p1.columns
            and "left_knee_y" in p1.columns
            and "right_knee_y" in p1.columns
        ):
            # Calculate rates of change for hips and knees
            hip_y = np.array(p1["hip_y_avg"].values)
            left_knee = np.array(p1["left_knee_y"].values)
            right_knee = np.array(p1["right_knee_y"].values)
            knee_y_avg = (left_knee + right_knee) / 2.0

            # If we have enough frames to check
            if len(hip_y) > 5:
                # Hip rising = hip_y decreasing (lower Y = higher in frame)
                # Knee extending = knee_y decreasing (knees moving down/extending)
                hip_change = hip_y[-1] - hip_y[0]  # negative = hips rising
                knee_change = (
                    knee_y_avg[-1] - knee_y_avg[0]
                )  # negative = knees extending

                # If hips rise significantly while knees barely extend
                # Relaxed thresholds: hips must rise a lot (-0.08) while knees don't extend at all (0.0)
                if hip_change < -0.08 and knee_change > 0.0:
                    critiques.append(
                        "First Pull: You are hitching (tipping forward) in the first pull"
                    )

        # Check 2: Bar drifting away from lifter
        if "barbell_x_stable" in p1.columns and "hip_x_avg" in p1.columns:
            x_start_bar = float(p1["barbell_x_stable"].iloc[0])
            x_start_hip = float(p1["hip_x_avg"].iloc[0])
            x_end_bar = float(p1["barbell_x_stable"].iloc[-1])

            # Bar drifting away = x distance from hip increasing
            drift = abs(x_end_bar - x_start_hip) - abs(x_start_bar - x_start_hip)
            if frame_height > 0 and drift > (frame_height * 0.05):
                critiques.append(
                    "First Pull: The bar is drifting away in the first pull"
                )

        # Check 3: Knees caving in (turning inward)
        if "left_knee_x" in p1.columns and "right_knee_x" in p1.columns:
            knee_width_start = abs(
                float(p1["left_knee_x"].iloc[0]) - float(p1["right_knee_x"].iloc[0])
            )
            knee_width_min = abs(p1["left_knee_x"] - p1["right_knee_x"]).min()

            # If knees move noticeably closer together
            if frame_height > 0 and (knee_width_start - knee_width_min) > (
                frame_height * 0.03
            ):
                critiques.append(
                    "First Pull: Your knees are caving (turning) in during the first pull"
                )

        # Check 4: Negative acceleration (slowing down)
        # Relaxed threshold: allow moderate negative acceleration which is normal
        if "accel_y_smooth" in p1.columns:
            if float(p1["accel_y_smooth"].min()) < -500:
                critiques.append(
                    "First Pull: You are slowing down from the floor, pull harder!"
                )

    # --- Second Pull (t1 -> t2) ---
    p2 = get_phase_df(t1, t2)
    if not p2.empty:
        # Check 1: Arms bending too early
        if "left_elbow_angle" in p2.columns and "right_elbow_angle" in p2.columns:
            # Check if arms bend significantly before the end of second pull
            # We'll check the first 50% of the phase (more permissive)
            check_length = int(len(p2) * 0.50)
            if check_length > 0:
                early_phase = p2.iloc[:check_length]
                min_elbow = float(
                    early_phase[["left_elbow_angle", "right_elbow_angle"]].min().min()
                )
                # Relaxed threshold: allow more arm bend (down to 130 degrees)
                if min_elbow < 130:
                    critiques.append("Second Pull: You bend your arms too early")

        # Check 2: Negative acceleration during second pull
        # Relaxed threshold: allow significant negative acceleration which is normal during transitions
        if "accel_y_smooth" in p2.columns:
            if float(p2["accel_y_smooth"].min()) < -800:
                critiques.append(
                    "Second Pull: You are slowing down from the floor, pull harder!"
                )

    # --- Combined First and Second Pull Check ---
    # Check for any significant negative acceleration from start to end of second pull
    p1_p2 = get_phase_df(t0, t2)
    if not p1_p2.empty and "accel_y_smooth" in p1_p2.columns:
        # Relaxed threshold: allow substantial negative acceleration (-600) which can occur normally
        if float(p1_p2["accel_y_smooth"].min()) < -600:
            # Only add if we haven't already added a similar critique
            if not any("slowing down" in c for c in critiques):
                critiques.append("You are slowing down from the floor, pull harder!")

    # --- Third Pull / Turnover (t2 -> t3) ---
    p3 = get_phase_df(t2, t3)
    if not p3.empty:
        # Check: Knees not bending shortly after second pull ends
        if "left_knee_angle" in df.columns and "right_knee_angle" in df.columns:
            # Get knee angles at end of second pull and shortly after
            t2_knee_angle = float(
                df.loc[t2, ["left_knee_angle", "right_knee_angle"]].mean()
            )

            # Check a small window after t2 (about 10 frames or 1/3 of phase)
            check_frames = min(10, len(p3) // 3) if len(p3) > 3 else len(p3)
            if check_frames > 0:
                early_t3 = p3.iloc[:check_frames]
                min_knee_angle_after = float(
                    early_t3[["left_knee_angle", "right_knee_angle"]].min().min()
                )

                # If knees don't bend significantly (angle should decrease)
                if t2_knee_angle - min_knee_angle_after < 10:
                    critiques.append(
                        "Third Pull: You are getting stuck at the top of extension, bend your knees sooner"
                    )

        # Check: Elbows bending after catch (pressing out)
        if "left_elbow_angle" in df.columns and "right_elbow_angle" in df.columns:
            # t3 is the bottom of the catch (hips at lowest)
            # Check if elbows bend significantly after this point but before recovery
            catch_elbow_angle = float(
                df.loc[t3, ["left_elbow_angle", "right_elbow_angle"]].mean()
            )

            # Check a small window after the catch
            post_catch_frames = min(10, len(p3) // 2) if len(p3) > 5 else 0
            if post_catch_frames > 0:
                post_catch_idx = min(t3 + post_catch_frames, t4)
                if post_catch_idx > t3:
                    post_catch = get_phase_df(t3 + 1, post_catch_idx)
                    if not post_catch.empty:
                        min_elbow_after_catch = float(
                            post_catch[["left_elbow_angle", "right_elbow_angle"]]
                            .min()
                            .min()
                        )

                        # If elbows bend more than 25 degrees after catch (relaxed from 15)
                        if catch_elbow_angle - min_elbow_after_catch > 25:
                            critiques.append(
                                "Third Pull: You are pressing out the catch (no lift)"
                            )

    # --- Recovery (t3 -> t4) ---
    p4 = get_phase_df(t3, t4)
    if not p4.empty:
        # Check: Walking forward during recovery
        if "hip_x_avg" in p4.columns or "left_ankle_x" in p4.columns:
            # Use hip or ankle position to detect forward movement
            x_col = "hip_x_avg" if "hip_x_avg" in p4.columns else "left_ankle_x"
            x_start = float(p4[x_col].iloc[0])
            x_end = float(p4[x_col].iloc[-1])

            # Forward movement (x decreasing in typical frame setup, or increasing depending on orientation)
            movement = abs(x_end - x_start)
            # Relaxed threshold: allow more movement (0.10 of frame height) which is normal during recovery
            if frame_height > 0 and movement > (frame_height * 0.10):
                critiques.append(
                    "Recovery: You are walking forward in the recovery, let the bar settle in the catch!"
                )

    return critiques
