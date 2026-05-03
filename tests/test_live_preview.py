import sys
from pathlib import Path
from typing import Dict, Iterator, List, Tuple
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from barpath.pipeline.realtime_processing.live_buffer import CircularFrameBuffer
from barpath.pipeline.realtime_processing.live_feature_extractor import LiveFeatureExtractor
from barpath.pipeline.lift_classifier import LiveLiftClassifier


def get_csv_files(category: str, max_per_class: int = 20) -> List[Path]:
    """Get CSV files from a category directory."""
    base = Path(f"outputs/male/{category}")
    if not base.exists():
        return []
    csv_files = []
    for subdir in base.iterdir():
        if subdir.is_dir():
            csv = subdir / "final_analysis.csv"
            if csv.exists():
                csv_files.append(csv)
    return csv_files[:max_per_class]


def csv_to_frames(csv_path: Path) -> Iterator[Dict]:
    """Iterate through CSV rows as simulated frame data."""
    df = pd.read_csv(csv_path)
    cols = df.columns.tolist()
    
    barbell_x_col = "barbell_x_smooth" if "barbell_x_smooth" in cols else None
    barbell_y_col = "barbell_y_smooth" if "barbell_y_smooth" in cols else None
    time_col = "time_s" if "time_s" in cols else None
    
    for i, row in df.iterrows():
        yield {
            "barbell_center": (
                (row[barbell_x_col], row[barbell_y_col]) 
                if barbell_x_col and barbell_y_col and pd.notna(row[barbell_x_col])
                else None
            ),
            "time_s": row[time_col] if time_col else i / 30.0,
            "frame_height": row.get("frame_height", 1080),
        }


def simulate_video_stream(csv_path: Path) -> Iterator[Dict]:
    """Simulate a video stream from CSV file."""
    yield from csv_to_frames(csv_path)


def test_category(
    classifier: LiveLiftClassifier,
    feature_extractor: LiveFeatureExtractor,
    csv_files: List[Path],
    expected_label: str,
    window_size: int = 30,
) -> Tuple[int, int, List[Dict]]:
    """Test classifier on a category of lifts."""
    correct = 0
    total = 0
    predictions: List[Dict] = []
    
    for csv_file in csv_files:
        stream = simulate_video_stream(csv_file)
        buffer = CircularFrameBuffer(window_size)
        
        # Collect frames into buffer
        frame_data_list = []
        for frame in stream:
            buffer.add_frame({
                "barbell_center": frame["barbell_center"],
                "joint_angles": {},
                "landmarks": None,
                "time_s": frame["time_s"],
                "frame_height": frame["frame_height"],
            })
            frame_data_list.append(buffer.get_frames())
        
        # Test every N frames (skip first 10 to get meaningful data)
        for i in range(10, len(frame_data_list), 5):
            frames = frame_data_list[i]
            if len(frames) < 10:
                continue
            
            features = feature_extractor.window_to_features(frames)
            if features is None:
                continue
            
            result = classifier.predict(features)
            pred = result.get("class", "unknown")
            conf = result.get("confidence", 0.0)
            
            total += 1
            if pred == expected_label:
                correct += 1
            
            predictions.append({
                "file": csv_file.name,
                "expected": expected_label,
                "predicted": pred,
                "confidence": conf,
            })
    
    return correct, total, predictions


def run_test_harness(
    model_path: str = "barpath/models/lift_detection/lift_detection_model.pkl",
    samples_per_class: int = 20,
) -> Dict:
    """Run the live preview test harness."""
    classifier = LiveLiftClassifier(model_path)
    feature_extractor = LiveFeatureExtractor(1920, 1080, 30.0)
    
    snatch_files = get_csv_files("snatch", samples_per_class)
    clean_files = get_csv_files("clean", samples_per_class)
    jerk_files = get_csv_files("jerk", samples_per_class)
    
    results = {}
    
    if snatch_files:
        correct, total, preds = test_category(
            classifier, feature_extractor, snatch_files, "snatch"
        )
        results["snatch"] = {"correct": correct, "total": total, "accuracy": correct/total if total else 0, "predictions": preds}
        print(f"Snatch: {correct}/{total} = {correct/total:.1%}")
    
    if clean_files:
        correct, total, preds = test_category(
            classifier, feature_extractor, clean_files, "clean"
        )
        results["clean"] = {"correct": correct, "total": total, "accuracy": correct/total if total else 0, "predictions": preds}
        print(f"Clean: {correct}/{total} = {correct/total:.1%}")
    
    if jerk_files:
        correct, total, preds = test_category(
            classifier, feature_extractor, jerk_files, "jerk"
        )
        results["jerk"] = {"correct": correct, "total": total, "accuracy": correct/total if total else 0, "predictions": preds}
        print(f"Jerk: {correct}/{total} = {correct/total:.1%}")
    
    return results


if __name__ == "__main__":
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    results = run_test_harness(samples_per_class=samples)
    total_correct = sum(r["correct"] for r in results.values())
    total_total = sum(r["total"] for r in results.values())
    print(f"\nOverall: {total_correct}/{total_total} = {total_correct/total_total:.1%}")