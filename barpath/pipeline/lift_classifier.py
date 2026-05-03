"""
Random Forest classifier for live lift type detection.
Wraps existing lift detection model and provides probability scores.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def load_lift_detection_model(model_path: str) -> Dict[str, Any]:
    """Load existing lift detection model."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(path, "rb") as f:
        model_data = pickle.load(f)

    return model_data


class LiftClassifier:
    """Random Forest wrapper with probability scoring.

    Uses existing lift detection model or trains new one.
    Provides confidence scores for peak detection.
    """

    # Confidence threshold for automatic detection
    CONFIDENCE_THRESHOLD = 0.50

    def __init__(self, model_path: Optional[str] = None):
        """Initialize classifier with optional model path."""
        self.model: Optional[RandomForestClassifier] = None
        self.model_data: Optional[Dict[str, Any]] = None
        self.classes: List[str] = ["clean", "clean_jerk", "jerk", "snatch"]
        self.feature_names: Optional[List[str]] = None

        if model_path:
            self.load(model_path)

    def load(self, model_path: str) -> bool:
        """Load model from pickle file."""
        try:
            self.model_data = load_lift_detection_model(model_path)
            self.model = self.model_data.get("classifier")
            self.classes = self.model_data.get("classes", self.classes)
            self.feature_names = self.model_data.get("feature_names")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Predict lift class and confidence.

        Args:
            features: 1D array of features

        Returns:
            {
                'class': 'snatch' | 'clean' | 'jerk' | 'clean_jerk' | 'none',
                'confidence': float (0-1),
                'probabilities': dict of class -> probability
            }
        """
        if self.model is None:
            return {"class": "none", "confidence": 0.0, "probabilities": {}}

        # Handle 1D or 2D input
        if features.ndim == 1:
            features_2d = features.reshape(1, -1)
        else:
            features_2d = features

        try:
            # Get probabilities
            probas = self.model.predict_proba(features_2d)[0]
            max_idx = np.argmax(probas)
            predicted_class = self.classes[max_idx]
            confidence = float(probas[max_idx])

            # Build probabilities dict
            prob_dict = {self.classes[i]: float(p) for i, p in enumerate(probas)}

            return {
                "class": predicted_class,
                "confidence": confidence,
                "probabilities": prob_dict,
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            return {"class": "none", "confidence": 0.0, "probabilities": {}}

    def predict_from_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract features from DataFrame and predict."""
        from lift_detection_features import extract_model_features_as_array

        try:
            features = extract_model_features_as_array(df)
            return self.predict(features)
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return {"class": "none", "confidence": 0.0, "probabilities": {}}

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def has_sufficient_confidence(self, confidence: float) -> bool:
        """Check if confidence is above threshold."""
        return confidence >= self.CONFIDENCE_THRESHOLD


class LiveLiftClassifier(LiftClassifier):
    """Enhanced classifier specifically tuned for live detection.

    Adds:
    - Sliding window smoothing for more stable predictions
    - Lower threshold for faster detection
    - Background class handling
    """

    # Lower thresholds for live detection (need fast response)
    LIVE_CONFIDENCE_THRESHOLD = 0.40
    MIN_FRAMES_FOR_PREDICTION = 20

    # Smoothing window for confidence
    CONFIDENCE_SMOOTHING_WINDOW = 5

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(model_path)
        self._confidence_history: List[float] = []
        self._last_prediction: Dict[str, Any] = {"class": "none", "confidence": 0.0}

    def predict_live(
        self, features: np.ndarray, apply_smoothing: bool = True
    ) -> Dict[str, Any]:
        """Predict with optional confidence smoothing."""
        prediction = self.predict(features)

        if apply_smoothing:
            # Apply exponential moving average smoothing
            smoothed = self._smooth_confidence(prediction["confidence"])
            prediction["confidence"] = smoothed
            prediction["class"] = (
                prediction["class"]
                if smoothed >= self.LIVE_CONFIDENCE_THRESHOLD
                else "none"
            )

        self._last_prediction = prediction
        return prediction

    def _smooth_confidence(self, confidence: float) -> float:
        """Apply exponential smoothing to confidence."""
        self._confidence_history.append(confidence)

        # Keep only recent history
        if len(self._confidence_history) > self.CONFIDENCE_SMOOTHING_WINDOW:
            self._confidence_history.pop(0)

        if not self._confidence_history:
            return confidence

        # Exponential moving average (more weight on recent)
        weights = np.exp(np.linspace(-1, 0, len(self._confidence_history)))
        weights /= weights.sum()

        return float(np.average(self._confidence_history, weights=weights))

    def reset_smoothing(self) -> None:
        """Reset confidence smoothing history."""
        self._confidence_history.clear()

    @property
    def last_prediction(self) -> Dict[str, Any]:
        """Get last prediction."""
        return self._last_prediction


def find_model_path() -> str:
    """Find the default lift detection model path."""
    # Check multiple locations
    candidates = [
        "barpath/models/lift_detection/lift_detection_model.pkl",
        "models/lift_detection/lift_detection_model.pkl",
        "../barpath/models/lift_detection/lift_detection_model.pkl",
    ]

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return str(path.absolute())

    # Default path
    return "barpath/models/lift_detection/lift_detection_model.pkl"


def create_classifier(model_path: Optional[str] = None) -> LiveLiftClassifier:
    """Create a classifier with default model path."""
    if model_path is None:
        model_path = find_model_path()

    return LiveLiftClassifier(model_path)
