"""Test the live preview classifier on simulated data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "barpath")

from barpath.pipeline.realtime_processing.live_lift_recognition import LiveLiftRecognizer


def simulate_live_preview(csv_path: Path, label: str) -> dict:
    """Simulate live preview on a CSV and return classification results."""
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
    predictions = []
    for idx, row in df.iterrows():
        bar_y = row["barbell_y_smooth"]
        bar_x = row.get("barbell_x_smooth", 350)

        # Build landmarks dict
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

        # Track predictions during recording
        if recognizer.state.name == "RECORDING" and recognizer._predicted_class:
            predictions.append(
                {
                    "frame": idx,
                    "class": recognizer._predicted_class,
                    "confidence": recognizer._predicted_confidence,
                }
            )

    # Final prediction after lift completes
    final_class = recognizer._predicted_class
    final_conf = recognizer._predicted_confidence

    return {
        "label": label,
        "final_class": final_class,
        "final_confidence": final_conf,
        "num_predictions": len(predictions),
        "first_prediction": predictions[0] if predictions else None,
        "last_prediction": predictions[-1] if predictions else None,
    }


def run_multiple_samples(data_dir: Path, category: str, max_samples: int = 5):
    """Test on multiple samples from a category."""
    category_dir = data_dir / category
    csv_files = list(category_dir.rglob("final_analysis.csv"))[:max_samples]

    results = []
    for csv_path in csv_files:
        result = simulate_live_preview(csv_path, category)
        results.append(result)

    # Summary
    correct = sum(1 for r in results if r.get("final_class") == category.upper())
    print(f"\n{category.upper()} ({len(results)} samples):")
    print(f"  Correct: {correct}/{len(results)} = {correct/len(results):.0%}")
    for r in results:
        status = "OK" if r.get("final_class") == category.upper() else "XX"
        print(
            f"  {status} {r['label']}: {r['final_class']} "
            f"({r['final_confidence']:.1%})"
        )

    return results


if __name__ == "__main__":
    data_dir = Path("outputs/male")

    print("=" * 60)
    print("LIVE PREVIEW CLASSIFICATION TEST")
    print("=" * 60)

    for category in ["snatch", "clean", "jerk"]:
        run_multiple_samples(data_dir, category, max_samples=5)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
