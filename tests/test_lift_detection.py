"""Tests for lift detection and clean+jerk split point detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from barpath.pipeline.lift_detection_features import (
    detect_clean_jerk_split_point,
    predict_lift_type,
)


def _make_synthetic_df(
    y_values: list[float],
    vel_values: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal DataFrame for split-point testing."""
    n = len(y_values)
    df = pd.DataFrame(
        {
            "barbell_y_smooth": y_values,
            "frame_height": [1080.0] * n,
            "frame": list(range(n)),
        }
    )
    if vel_values is not None:
        df["vel_y_smooth"] = vel_values
    return df


def test_detect_clean_jerk_split_point_returns_none_for_short_trajectory():
    """A trajectory shorter than 80 frames should yield no split."""
    y = np.linspace(0, 1, 50).tolist()
    df = _make_synthetic_df(y)
    result = detect_clean_jerk_split_point(df)
    assert result is None


def test_detect_clean_jerk_split_point_finds_split_for_two_phase_pattern():
    """Synthetic clean+jerk: floor -> clean peak -> shoulder -> dip -> jerk lockout."""
    # Build a synthetic y-trajectory in image coords (higher y = lower physical position)
    # Phase 1: clean pull (y drops from 500 to 100)
    # Phase 2: recovery to shoulder (y rises to 250)
    # Phase 3: jerk dip (y rises to 300)
    # Phase 4: jerk lockout (y drops to 80)
    y = []
    # Floor hold (frames 0-29) — keep y high so first_third_mean is high
    y.extend([500] * 30)
    # Clean floor to extension (frames 30-50)
    y.extend(np.linspace(500, 100, 21).tolist())
    # Recovery to shoulder (frames 51-75)
    y.extend(np.linspace(100, 240, 25).tolist())
    # Shoulder plateau
    y.extend([240] * 10)
    # Jerk dip (frames 86-105)
    y.extend(np.linspace(240, 320, 20).tolist())
    # Jerk drive to lockout (frames 106-135)
    y.extend(np.linspace(320, 80, 30).tolist())
    # Add trailing stable frames so global max is not at the very end
    y.extend([85] * 15)

    df = _make_synthetic_df(y)
    split = detect_clean_jerk_split_point(df)
    assert split is not None
    # Split should be in the shoulder region (after clean peak ~50, before jerk dip ~105)
    assert 60 < split < 90


def test_detect_clean_jerk_split_point_none_for_single_snatch():
    """A single-phase snatch-like trajectory should not produce a split."""
    y = np.linspace(500, 80, 120).tolist()
    df = _make_synthetic_df(y)
    result = detect_clean_jerk_split_point(df)
    assert result is None


def test_predict_lift_type_clean_jerk_fallback():
    """Ensure predict_lift_type can fall back to clean_jerk with the split heuristic."""
    # Build a DataFrame that mimics a clean+jerk when the model only predicts 'clean'
    y = []
    y.extend([500] * 30)
    y.extend(np.linspace(500, 100, 21).tolist())
    y.extend(np.linspace(100, 240, 25).tolist())
    y.extend([240] * 10)
    y.extend(np.linspace(240, 320, 20).tolist())
    y.extend(np.linspace(320, 80, 30).tolist())
    y.extend([85] * 15)

    df = _make_synthetic_df(y)
    # Create a mock model that predicts 'clean' with high confidence
    mock_model = {
        "classifier": _MockClassifier("clean"),
        "scaler": _MockScaler(),
        "feature_names": ["cj_two_phase_detected", "cj_phase_gap"],
    }
    result = predict_lift_type(df, mock_model)
    assert result is not None
    assert result["is_clean_jerk"] is True


class _MockClassifier:
    """Mock sklearn classifier for testing."""

    def __init__(self, predicted_class: str):
        self._predicted = predicted_class
        self.classes_ = ["clean", "snatch", "jerk", "clean_jerk"]

    def predict(self, X: np.ndarray) -> list[str]:
        return [self._predicted]

    def predict_proba(self, X: np.ndarray) -> list[list[float]]:
        probs = [0.1] * len(self.classes_)
        idx = self.classes_.index(self._predicted)
        probs[idx] = 0.7
        return [probs]


class _MockScaler:
    """Mock sklearn scaler for testing."""

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X
