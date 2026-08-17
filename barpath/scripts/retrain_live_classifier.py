"""Retrain Live Preview Lift Classifier.

Trains a dedicated RandomForest model on simulated live-preview windows
from existing full-lift CSVs. The resulting model is designed for
real-time classification of partial trajectories.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from barpath.pipeline.realtime_processing.live_training_data import (
    generate_live_training_dataset,
)
from barpath.pipeline.realtime_processing.live_window_features import (
    extract_window_features,
)


def train_live_classifier(
    data_dir: Path,
    output_path: Path,
    categories: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a live-preview lift classifier.

    Args:
        data_dir: Base directory with lift subdirectories (e.g., outputs/male)
        output_path: Where to save the trained model pickle
        categories: List of lift types to include
        test_size: Fraction of data to hold out for testing
        random_state: Random seed for reproducibility

    Returns:
        Dict with training results and model data
    """
    if categories is None:
        categories = ["snatch", "clean", "jerk"]

    print(f"Loading training data from {data_dir}...")
    dataset = generate_live_training_dataset(data_dir, categories=categories)
    print(f"Generated {len(dataset)} windows")

    if len(dataset) < 100:
        raise ValueError(f"Too few training samples: {len(dataset)}")

    # Extract features from all windows
    print("Extracting features...")
    X_rows: list[dict[str, float]] = []
    y_labels: list[str] = []

    for window_df, label in dataset:
        features = extract_window_features(window_df)
        if features:
            X_rows.append(features)
            y_labels.append(label)

    # Build feature matrix
    df_features = pd.DataFrame(X_rows)
    df_features["label"] = y_labels

    print(f"Feature matrix shape: {df_features.shape}")
    print(f"Features: {list(df_features.columns[:-1])}")

    # Class distribution
    print("\nClass distribution:")
    for cat in categories:
        count = sum(1 for label in y_labels if label == cat)
        print(f"  {cat}: {count}")

    # Train/test split
    X = df_features.drop("label", axis=1)
    y = df_features["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RandomForest
    print("\nTraining RandomForest...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred))

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    cm = confusion_matrix(y_test, y_pred, labels=categories)
    print("Rows: True, Cols: Predicted")
    print(f"Labels: {categories}")
    print(cm)

    # Per-class accuracy
    print("\n" + "=" * 60)
    print("PER-CLASS ACCURACY")
    print("=" * 60)
    for cat in categories:
        mask = y_test == cat
        if mask.sum() > 0:
            acc = (y_pred[mask] == cat).mean()
            print(f"  {cat}: {acc:.1%} ({(y_pred[mask] == cat).sum()}/{mask.sum()})")

    # Feature importances
    print("\n" + "=" * 60)
    print("TOP 15 FEATURE IMPORTANCES")
    print("=" * 60)
    importances = list(zip(X.columns, clf.feature_importances_, strict=False))
    importances.sort(key=lambda x: x[1], reverse=True)
    for name, imp in importances[:15]:
        print(f"  {name}: {imp:.4f}")

    # Confidence distribution
    print("\n" + "=" * 60)
    print("CONFIDENCE DISTRIBUTION")
    print("=" * 60)
    max_probs = y_proba.max(axis=1)
    print(f"  Mean confidence: {max_probs.mean():.1%}")
    print(f"  Median confidence: {np.median(max_probs):.1%}")
    print(f"  >90%: {(max_probs > 0.9).mean():.1%}")
    print(f"  >70%: {(max_probs > 0.7).mean():.1%}")
    print(f"  >50%: {(max_probs > 0.5).mean():.1%}")

    # Save model
    model_data = {
        "classifier": clf,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "categories": categories,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved to {output_path}")

    return {
        "model_data": model_data,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": (y_pred == y_test).mean(),
    }


if __name__ == "__main__":
    data_dir = Path("outputs/male")
    output_path = Path("barpath/models/lift_detection/live_lift_model.pkl")

    results = train_live_classifier(data_dir, output_path)
    print(f"\n{'=' * 60}")
    print(f"FINAL TEST ACCURACY: {results['accuracy']:.1%}")
    print(f"{'=' * 60}")
