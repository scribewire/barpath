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


def _load_lift_detection_model_backend(
    model_path: str, backend: str = "auto"
) -> tuple[Any, str]:
    """Load lift detection model with backend awareness.

    Tries ONNX/OpenVINO if available, falls back to pickle.

    Args:
        model_path: Path to the .pkl model file.
        backend: Backend preference ("auto", "openvino", "pytorch").

    Returns:
        Tuple of (model_object, backend_used).
    """
    model_dir = Path(model_path).parent
    model_name = Path(model_path).stem

    # Check for ONNX model first
    onnx_path = model_dir / f"{model_name}.onnx"
    ov_dir = model_dir / f"{model_name}_openvino_export"

    if backend in ("openvino", "auto") and ov_dir.is_dir():
        try:
            from openvino.runtime import Core

            core = Core()
            compiled_model = core.compile_model(str(ov_dir), "CPU")
            print(f"Loaded lift detection model with OpenVINO")
            return compiled_model, "openvino"
        except Exception as e:
            print(f"WARNING: OpenVINO lift detection model failed: {e}")

    if backend in ("auto", "openvino", "pytorch") and onnx_path.exists():
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(onnx_path))
            print(f"Loaded lift detection model with ONNX Runtime")
            return session, "onnxruntime"
        except Exception as e:
            print(f"WARNING: ONNX Runtime lift detection model failed: {e}")

    # Fallback to pickle
    with open(model_path, "rb") as f:
        sklearn_model = pickle.load(f)
    print(f"Loaded lift detection model with sklearn (pickle)")
    return sklearn_model, "sklearn"


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

    def __init__(self, model_path: Optional[str] = None, backend: str = "auto"):
        """Initialize classifier with optional model path."""
        self.model: Optional[RandomForestClassifier] = None
        self.model_data: Optional[Dict[str, Any]] = None
        self.classes: List[str] = ["clean", "clean_jerk", "jerk", "snatch"]
        self.feature_names: Optional[List[str]] = None
        self.model_backend: str = "sklearn"  # Track which backend is used
        self.onnx_session = None
        self.ov_compiled_model = None

        if model_path:
            self.load(model_path, backend=backend)

    def load(self, model_path: str, backend: str = "auto") -> bool:
        """Load model from pickle file or ONNX/OpenVINO if available."""
        try:
            # Try backend-aware loading first
            model_obj, backend_used = _load_lift_detection_model_backend(
                model_path, backend
            )
            self.model_backend = backend_used

            if backend_used == "openvino":
                self.ov_compiled_model = model_obj
                # Load config for class names
                config_path = Path(model_path).parent / "lift_detection_config.json"
                if config_path.exists():
                    import json
                    with open(config_path) as f:
                        cfg = json.load(f)
                    self.classes = cfg.get("classes", self.classes)
                    self.feature_names = cfg.get("feature_names")
            elif backend_used == "onnxruntime":
                self.onnx_session = model_obj
                config_path = Path(model_path).parent / "lift_detection_config.json"
                if config_path.exists():
                    import json
                    with open(config_path) as f:
                        cfg = json.load(f)
                    self.classes = cfg.get("classes", self.classes)
                    self.feature_names = cfg.get("feature_names")
            else:
                # sklearn pickle model
                self.model_data = model_obj
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
        # Handle 1D or 2D input
        if features.ndim == 1:
            features_2d = features.reshape(1, -1).astype(np.float32)
        else:
            features_2d = features.astype(np.float32)

        try:
            if self.model_backend == "openvino" and self.ov_compiled_model:
                # OpenVINO compiled model inference
                result = self.ov_compiled_model(features_2d)
                # Get the output tensor (probabilities)
                output_key = next(iter(result))
                probas = result[output_key][0]
                max_idx = int(np.argmax(probas))
                predicted_class = self.classes[max_idx] if max_idx < len(self.classes) else "none"
                confidence = float(probas[max_idx])
                prob_dict = {self.classes[i]: float(probas[i]) for i in range(min(len(probas), len(self.classes)))}

            elif self.model_backend == "onnxruntime" and self.onnx_session:
                # ONNX Runtime inference
                input_name = self.onnx_session.get_inputs()[0].name
                result = self.onnx_session.run(None, {input_name: features_2d})
                probas = result[0][0]
                max_idx = int(np.argmax(probas))
                predicted_class = self.classes[max_idx] if max_idx < len(self.classes) else "none"
                confidence = float(probas[max_idx])
                prob_dict = {self.classes[i]: float(probas[i]) for i in range(min(len(probas), len(self.classes)))}

            elif self.model is not None:
                # sklearn pickle model
                probas = self.model.predict_proba(features_2d)[0]
                max_idx = int(np.argmax(probas))
                predicted_class = self.classes[max_idx]
                confidence = float(probas[max_idx])
                prob_dict = {self.classes[i]: float(p) for i, p in enumerate(probas)}

            else:
                return {"class": "none", "confidence": 0.0, "probabilities": {}}

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
