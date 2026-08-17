"""Test clean+jerk sequence detection."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "barpath")

from barpath.pipeline.realtime_processing.live_lift_recognition import (
    LiftState,
    LiveLiftRecognizer,
)


def simulate_clean_jerk_sequence(clean_csv: Path, jerk_csv: Path) -> dict:
    """Simulate a clean followed by a jerk."""
    frame_h = 720
    frame_w = 1280

    rec = LiveLiftRecognizer(
        "barpath/models/lift_detection/lift_detection_model.pkl",
        fps=30.0,
    )

    # Simulate clean
    df_clean = pd.read_csv(clean_csv)
    for idx, row in df_clean.iterrows():
        bar_y = row["barbell_y_smooth"]
        bar_x = row.get("barbell_x_smooth", 350)
        lm = {}
        for name, mp in [
            ("left_shoulder", 11),
            ("right_shoulder", 12),
            ("left_hip", 23),
            ("right_hip", 24),
            ("left_knee", 25),
            ("right_knee", 26),
        ]:
            x, y = row.get(f"{name}_x"), row.get(f"{name}_y")
            if pd.notna(x) and pd.notna(y):
                lm[mp] = (float(x), float(y), 0.0, 1.0)
        rec.update(
            barbell_center=(float(bar_x), float(bar_y)),
            barbell_box=None,
            landmarks=lm,
            timestamp_ms=float(idx) * 33.33,
            frame_width=frame_w,
            frame_height=frame_h,
        )

    clean_state = rec.state.name
    clean_stack = list(rec._display_stack)

    # Simulate jerk (if we are in SHOULDER_WAIT, feed jerk frames)
    if rec.state == LiftState.SHOULDER_WAIT:
        df_jerk = pd.read_csv(jerk_csv)
        for idx, row in df_jerk.iterrows():
            bar_y = row["barbell_y_smooth"]
            bar_x = row.get("barbell_x_smooth", 350)
            lm = {}
            for name, mp in [
                ("left_shoulder", 11),
                ("right_shoulder", 12),
                ("left_hip", 23),
                ("right_hip", 24),
                ("left_knee", 25),
                ("right_knee", 26),
            ]:
                x, y = row.get(f"{name}_x"), row.get(f"{name}_y")
                if pd.notna(x) and pd.notna(y):
                    lm[mp] = (float(x), float(y), 0.0, 1.0)
            rec.update(
                barbell_center=(float(bar_x), float(bar_y)),
                barbell_box=None,
                landmarks=lm,
                timestamp_ms=float(idx) * 33.33 + 10000,  # Offset time
                frame_width=frame_w,
                frame_height=frame_h,
            )

            # Track any snatch predictions during jerk recording
            if rec.state == LiftState.RECORDING and rec._predicted_class:
                pass  # Will check at end

    return {
        "clean_state": clean_state,
        "clean_stack": clean_stack,
        "final_state": rec.state.name,
        "final_stack": rec._display_stack,
        "predicted": rec._predicted_class,
        "expecting_jerk": rec._expecting_jerk,
    }


if __name__ == "__main__":
    clean_csv = Path("outputs/dataset/male/clean/botev_10_clean/final_analysis.csv")
    jerk_csv = Path("outputs/dataset/male/jerk/botev_15_jerk/final_analysis.csv")

    result = simulate_clean_jerk_sequence(clean_csv, jerk_csv)
    print("Clean+Jerk Sequence Test:")
    print(f"  After clean: state={result['clean_state']} stack={result['clean_stack']}")
    print(
        f"  Final: state={result['final_state']} stack={result['final_stack']} predicted={result['predicted']}"
    )
    print(f"  expecting_jerk={result['expecting_jerk']}")

    # Also test standalone jerk
    print("\nStandalone Jerk Test:")
    df_jerk = pd.read_csv(jerk_csv)
    frame_h = int(df_jerk["frame_height"].iloc[0])
    frame_w = int(df_jerk["frame_width"].iloc[0])
    rec = LiveLiftRecognizer("barpath/models/lift_detection/lift_detection_model.pkl", fps=30.0)

    snatch_preds = 0
    for idx, row in df_jerk.iterrows():
        bar_y = row["barbell_y_smooth"]
        bar_x = row.get("barbell_x_smooth", 350)
        lm = {}
        for name, mp in [
            ("left_shoulder", 11),
            ("right_shoulder", 12),
            ("left_hip", 23),
            ("right_hip", 24),
            ("left_knee", 25),
            ("right_knee", 26),
        ]:
            x, y = row.get(f"{name}_x"), row.get(f"{name}_y")
            if pd.notna(x) and pd.notna(y):
                lm[mp] = (float(x), float(y), 0.0, 1.0)
        rec.update(
            barbell_center=(float(bar_x), float(bar_y)),
            barbell_box=None,
            landmarks=lm,
            timestamp_ms=float(idx) * 33.33,
            frame_width=frame_w,
            frame_height=frame_h,
        )
        if rec._predicted_class and "SNATCH" in rec._predicted_class:
            snatch_preds += 1

    print(
        f"  Final: stack={rec._display_stack} predicted={rec._predicted_class} state={rec.state.name}"
    )
    print(f"  Snatch predictions during recording: {snatch_preds}")
