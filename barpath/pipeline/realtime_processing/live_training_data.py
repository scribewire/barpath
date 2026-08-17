"""Live Preview Training Data Generator.

Extracts simulated live-preview windows from existing full-lift CSVs.
Each window represents what the live preview would see at a given moment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def extract_live_windows_from_csv(
    csv_path: Path,
    fps: float = 30.0,
    buffer_seconds: float = 1.0,
    min_frames: int = 15,
    max_window_frames: int = 240,
) -> list[pd.DataFrame]:
    """Extract simulated live-preview windows from a full-lift CSV.

    For each trigger point (bar passes knees going up), extracts multiple
    windows of increasing length to simulate in-progress classification.

    Args:
        csv_path: Path to final_analysis.csv
        fps: Frames per second
        buffer_seconds: Pre-trigger buffer to include
        min_frames: Minimum window length
        max_window_frames: Maximum window length

    Returns:
        List of DataFrames, each representing a live-preview window
    """
    df = pd.read_csv(csv_path)
    if "barbell_y_smooth" not in df.columns:
        return []

    frame_h = float(df["frame_height"].iloc[0]) if "frame_height" in df.columns else 720.0

    bar_y = np.asarray(df["barbell_y_smooth"].values, dtype=float)
    if len(bar_y) < min_frames:
        return []

    # Knee y-position: use hip_y_avg * 0.7 as proxy, or estimate from frame
    if "hip_y_avg" in df.columns:
        hip_y = np.asarray(df["hip_y_avg"].values, dtype=float)
        # Filter out zeros (missing data)
        valid_hip = hip_y[hip_y > 0]
        knee_y = valid_hip.mean() * 0.9 if len(valid_hip) > 0 else frame_h * 0.75
    else:
        knee_y = frame_h * 0.75

    # Find triggers: bar passes knees going up (y decreases past knee_y)
    triggers = []
    for i in range(1, len(bar_y)):
        if bar_y[i - 1] > knee_y and bar_y[i] <= knee_y:
            # Additional check: ensure bar is actually moving up
            if bar_y[i] < bar_y[i - 1]:
                triggers.append(i)

    if not triggers:
        # Fallback: use first frame where bar is clearly above mid-frame
        mid_frame = frame_h * 0.5
        above_mid = np.where(bar_y < mid_frame)[0]
        if len(above_mid) > 10:
            triggers = [above_mid[0]]
        else:
            # Last resort: just use the midpoint
            triggers = [len(bar_y) // 3]

    windows = []
    buffer_frames = int(buffer_seconds * fps)

    for trigger in triggers:
        start = max(0, trigger - buffer_frames)

        # Simulate windows of different lengths (in-progress lift)
        # Step by 5 frames to create variety without too much redundancy
        step = max(5, (max_window_frames - min_frames) // 20)
        for end in range(
            start + min_frames,
            min(len(df), start + max_window_frames),
            step,
        ):
            window_df = df.iloc[start:end].copy().reset_index(drop=True)
            if len(window_df) >= min_frames:
                windows.append(window_df)

    return windows


def generate_live_training_dataset(
    data_dir: Path,
    categories: list[str] | None = None,
    **kwargs,
) -> list[tuple[pd.DataFrame, str]]:
    """Generate complete live-preview training dataset.

    Args:
        data_dir: Base directory (e.g., outputs/male)
        categories: List of lift types to include
        **kwargs: Passed to extract_live_windows_from_csv

    Returns:
        List of (window_df, label) tuples
    """
    if categories is None:
        categories = ["snatch", "clean", "jerk"]

    dataset: list[tuple[pd.DataFrame, str]] = []

    for category in categories:
        category_dir = data_dir / category
        if not category_dir.exists():
            continue

        for csv_path in category_dir.rglob("final_analysis.csv"):
            windows = extract_live_windows_from_csv(csv_path, **kwargs)
            for window in windows:
                dataset.append((window, category))

    return dataset


if __name__ == "__main__":
    # Quick test
    from pathlib import Path

    data_dir = Path("outputs/male")
    dataset = generate_live_training_dataset(data_dir)

    from collections import Counter

    labels = [label for _, label in dataset]
    counts = Counter(labels)
    print(f"Generated {len(dataset)} windows")
    print(f"  Snatch: {counts.get('snatch', 0)}")
    print(f"  Clean:  {counts.get('clean', 0)}")
    print(f"  Jerk:   {counts.get('jerk', 0)}")

    # Show sample window stats
    if dataset:
        sample_df, sample_label = dataset[0]
        print(f"\nSample window ({sample_label}):")
        print(f"  Length: {len(sample_df)} frames")
        print(f"  Start y: {sample_df['barbell_y_smooth'].iloc[0]:.1f}")
        print(f"  End y: {sample_df['barbell_y_smooth'].iloc[-1]:.1f}")
