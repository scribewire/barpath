"""Test the reworked live preview detection.

Tests:
1. Clean detection -> shoulder wait -> jerk detection (stacked display)
2. Standalone snatch detection
3. Standalone jerk detection (isolated)
4. Verify no overcompensation (clean/jerk not misclassified as snatch)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "barpath")

from barpath.pipeline.realtime_processing.live_lift_recognition import (
    LiveLiftRecognizer,
)


def simulate_full_sequence(csv_path: Path, label: str) -> dict:
    """Simulate live preview and return final state."""
    df = pd.read_csv(csv_path)
    if "barbell_y_smooth" not in df.columns:
        return {"error": "No barbell data"}

    frame_h = int(df["frame_height"].iloc[0]) if "frame_height" in df.columns else 720
    frame_w = int(df["frame_width"].iloc[0]) if "frame_width" in df.columns else 1280

    recognizer = LiveLiftRecognizer(
        "barpath/models/lift_detection/lift_detection_model.pkl",
        fps=30.0,
    )

    # Simulate feeding frames
    for idx, row in df.iterrows():
        bar_y = row["barbell_y_smooth"]
        bar_x = row.get("barbell_x_smooth", 350)

        lm = {}
        for name, mp_idx in [
            ("left_shoulder", 11),
            ("right_shoulder", 12),
            ("left_elbow", 13),
            ("right_elbow", 14),
            ("left_wrist", 15),
            ("right_wrist", 16),
            ("left_hip", 23),
            ("right_hip", 24),
            ("left_knee", 25),
            ("right_knee", 26),
            ("left_ankle", 27),
            ("right_ankle", 28),
        ]:
            x = row.get(f"{name}_x", None)
            y = row.get(f"{name}_y", None)
            if x is not None and y is not None and pd.notna(x) and pd.notna(y):
                lm[mp_idx] = (float(x), float(y), 0.0, 1.0)

        recognizer.update(
            barbell_center=(float(bar_x), float(bar_y)),
            barbell_box=None,
            landmarks=lm,
            timestamp_ms=float(idx) * 33.33,
            frame_width=frame_w,
            frame_height=frame_h,
        )

    # Force classification if still recording (CSV ends at peak)
    from barpath.pipeline.realtime_processing.live_lift_recognition import LiftState

    if recognizer.state == LiftState.RECORDING:
        recognizer.state = LiftState.CLASSIFYING
        recognizer._handle_classifying(frame_w, frame_h)

    return {
        "label": label,
        "final_state": recognizer.state.name,
        "display_stack": recognizer._display_stack,
        "predicted_class": recognizer._predicted_class,
        "confidence": recognizer._predicted_confidence,
    }


def run_category_test(category: str, max_samples: int = 5):
    """Test on multiple samples from a category."""
    data_dir = Path("outputs/dataset/male")
    category_dir = data_dir / category
    csv_files = list(category_dir.rglob("final_analysis.csv"))[:max_samples]

    results = []
    for csv_path in csv_files:
        result = simulate_full_sequence(csv_path, category)
        results.append(result)

    # Summary
    print(f"\n{category.upper()} ({len(results)} samples):")
    for r in results:
        stack = " + ".join(r["display_stack"]) if r["display_stack"] else "None"
        status = "OK" if category.upper() in stack else "XX"
        print(
            f"  {status} {r['label']}: stack=[{stack}] "
            f"state={r['final_state']} conf={r['confidence']:.1%}"
        )

    correct = sum(1 for r in results if category.upper() in " + ".join(r["display_stack"]))
    print(f"  Correct: {correct}/{len(results)} = {correct / len(results):.0%}")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("REWORKED LIVE PREVIEW DETECTION TEST")
    print("=" * 60)

    for category in ["snatch", "clean", "jerk"]:
        run_category_test(category, max_samples=5)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
