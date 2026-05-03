"""
Retrain lift detection classifier using all available labeled data.
"""

from pathlib import Path
from typing import Dict, List, Optional
import sys
import pickle

# Add barpath to path
sys.path.insert(0, str(Path(__file__).parent.parent / "barpath"))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def get_all_csv_files() -> Dict[str, List[Path]]:
    """Get all labeled CSV files by category."""
    base = Path("outputs/male")
    categories = {}

    for cat in ["snatch", "clean", "jerk"]:
        cat_dir = base / cat
        if not cat_dir.exists():
            categories[cat] = []
            continue

        csv_files = []
        for subdir in cat_dir.iterdir():
            if subdir.is_dir():
                csv = subdir / "final_analysis.csv"
                if csv.exists():
                    csv_files.append(csv)
        categories[cat] = csv_files

    return categories


def load_features(category: str, csv_files: List[Path]) -> tuple:
    """Load features and labels from CSV files."""
    from pipeline.lift_detection_features import extract_model_features_as_array

    all_features = []
    all_labels = []
    errors = 0

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            if len(df) < 30:
                print(f"  Skipping {csv_file.parent.name}: only {len(df)} rows")
                continue

            features = extract_model_features_as_array(df)

            if features is None or len(features) == 0:
                print(f"  Skipping {csv_file.parent.name}: no features extracted")
                continue

            if features.sum() == 0:
                print(f"  Skipping {csv_file.parent.name}: all-zero features")
                continue

            all_features.append(features)
            all_labels.append(category)

        except Exception as e:
            print(f"  Error loading {csv_file.parent.name}: {e}")
            errors += 1

    return all_features, all_labels


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    classes: List[str],
) -> RandomForestClassifier:
    """Train a RandomForest classifier."""

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(X, y)

    return clf


def save_model(
    classifier: RandomForestClassifier,
    scaler: Optional[StandardScaler],
    feature_names: List[str],
    classes: List[str],
    output_path: str,
) -> bool:
    """Save the trained model to a pickle file."""

    model_data = {
        "classifier": classifier,
        "scaler": scaler,
        "feature_names": feature_names,
        "classes": classes,
        "version": "1.0",
    }

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(model_data, f)
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False


def main():
    """Main retraining function."""
    print("=" * 60)
    print("LIFT DETECTION MODEL RETRAINING")
    print("=" * 60)

    category_files = get_all_csv_files()

    print("\nFound files:")
    for cat, files in category_files.items():
        print(f"  {cat}: {len(files)} files")

    total = sum(len(f) for f in category_files.values())
    print(f"\nTotal: {total} labeled samples")

    if total == 0:
        print("ERROR: No training data found!")
        return

    print("\n" + "=" * 60)
    print("LOADING FEATURES")
    print("=" * 60)

    all_features = []
    all_labels = []

    for category, files in category_files.items():
        if not files:
            continue

        print(f"\nProcessing {category} ({len(files)} files)...")
        features, labels = load_features(category, files)

        print(f"  Loaded {len(features)} samples")
        all_features.extend(features)
        all_labels.extend(labels)

    if not all_features:
        print("ERROR: No features loaded!")
        return

    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"\nTotal training samples: {len(X)}")
    print(f"Feature dimensions: {X.shape[1]}")

    classes = ["snatch", "clean", "jerk"]
    print(f"Classes: {classes}")

    # Check class distribution
    for c in classes:
        count = np.sum(y == c)
        print(f"  {c}: {count} samples")

    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)

    # Train classifier
    clf = train_classifier(X, y, classes)

    # Cross-validation
    print("\nRunning cross-validation...")
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std() * 2:.1%})")

    # Train on full data for final model
    print("\nTraining on full dataset...")
    clf = train_classifier(X, y, classes)

    # Feature importances
    from pipeline.lift_detection_features import get_model_feature_names

    feature_names = get_model_feature_names()
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("\nTop 10 features:")
    for i in top_idx:
        print(f"  {feature_names[i]}: {importances[i]:.3f}")

    # Save model
    output_path = "barpath/models/lift_detection/lift_detection_model.pkl"
    print(f"\nSaving model to: {output_path}")

    success = save_model(clf, None, feature_names, classes, output_path)

    if success:
        print("\n" + "=" * 60)
        print("RETRAINING COMPLETE")
        print("=" * 60)
        print(f"Model saved: {output_path}")
        print(f"CV Accuracy: {cv_scores.mean():.1%}")
    else:
        print("\nERROR: Failed to save model!")


if __name__ == "__main__":
    main()
