"""
Smart Analysis: Comprehensive Critique using Multi-Label Random Forest.

This module provides ML-based fault detection with probability-weighted critiques.
"""

import json
import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None  # type: ignore
    MultiOutputClassifier = None  # type: ignore
    print("Warning: scikit-learn not installed. Smart Analysis will be disabled.")


def load_smart_analysis_model(
    model_dir: Path,
) -> tuple[Any | None, dict | None, dict | None]:
    """
    Load trained Random Forest model and metadata for Smart Analysis.

    Args:
        model_dir: Path to model directory containing:
            - smart_analysis_model.pkl
            - smart_analysis_features.json
            - smart_analysis_faults.json

    Returns:
        Tuple of (model, features_config, faults_config) or (None, None, None) if not found.
    """
    model_path = model_dir / "smart_analysis_model.pkl"
    features_path = model_dir / "smart_analysis_features.json"
    faults_path = model_dir / "smart_analysis_faults.json"

    if not model_path.exists():
        return None, None, None

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        features_config = None
        if features_path.exists():
            with open(features_path) as f:
                features_config = json.load(f)

        faults_config = None
        if faults_path.exists():
            with open(faults_path) as f:
                faults_config = json.load(f)

        return model, features_config, faults_config
    except Exception as e:
        print(f"Error loading Smart Analysis model: {e}")
        return None, None, None


def run_smart_analysis(
    features: dict[str, float],
    model: Any,
    features_config: dict | None = None,
    faults_config: dict | None = None,
    threshold: float = 0.5,
) -> dict:
    """
    Run fault detection using trained Random Forest model.

    Args:
        features: Dict of feature name -> value
        model: Trained MultiOutputClassifier
        features_config: Dict with feature order and metadata
        faults_config: Dict with fault definitions
        threshold: Probability threshold to flag a fault

    Returns:
        Dict with:
            - fault_probabilities: Dict of fault_id -> probability
            - flagged_faults: List of fault_ids above threshold
            - available: bool indicating if analysis was possible
    """
    if not SKLEARN_AVAILABLE:
        return {
            "fault_probabilities": {},
            "flagged_faults": [],
            "available": False,
            "error": "scikit-learn not installed",
        }

    if model is None:
        return {
            "fault_probabilities": {},
            "flagged_faults": [],
            "available": False,
            "error": "No model available",
        }

    try:
        if features_config and "feature_order" in features_config:
            feature_order = features_config["feature_order"]
        else:
            feature_order = sorted(features.keys())

        X = np.array([[features.get(f, 0.0) for f in feature_order]])

        if hasattr(model, "predict_proba"):
            probas = cast(Any, model.predict_proba)(X)

            if isinstance(probas, list):
                fault_probs = {}
                for i, proba in enumerate(probas):
                    if hasattr(proba, "shape") and proba.shape[1] >= 2:
                        fault_probs[f"fault_{i}"] = float(proba[0, 1])
                    else:
                        fault_probs[f"fault_{i}"] = (
                            float(proba[0]) if hasattr(proba, "__getitem__") else 0.5
                        )
            else:
                fault_probs = {"fault_0": float(probas[0, 1]) if probas.shape[1] >= 2 else 0.5}
        else:
            predictions = cast(Any, model.predict)(X)
            if isinstance(predictions[0], (list, np.ndarray)):
                fault_probs = {f"fault_{i}": float(p) for i, p in enumerate(predictions[0])}
            else:
                fault_probs = {"fault_0": float(predictions[0])}

        if faults_config and "faults" in faults_config:
            fault_names = [f.get("id", f"fault_{i}") for i, f in enumerate(faults_config["faults"])]
            named_probs = {}
            for i, (key, val) in enumerate(fault_probs.items()):
                if i < len(fault_names):
                    named_probs[fault_names[i]] = val
                else:
                    named_probs[key] = val
            fault_probs = named_probs

        flagged = [fid for fid, prob in fault_probs.items() if prob >= threshold]

        return {
            "fault_probabilities": fault_probs,
            "flagged_faults": flagged,
            "available": True,
        }

    except Exception as e:
        print(f"Error running Smart Analysis: {e}")
        return {
            "fault_probabilities": {},
            "flagged_faults": [],
            "available": False,
            "error": str(e),
        }
