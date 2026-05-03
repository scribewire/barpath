from __future__ import annotations

import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "barpath" / "models"
OUT = ROOT / "outputs" / "plans" / "ios" / "barpath" / "barpath" / "Resources"


def export_baselines() -> None:
    src = MODELS / "analysis" / "pro_baseline_report.json"
    report = json.loads(src.read_text(encoding="utf-8"))
    slim = {"version": report["report_metadata"]["version"], "baselines": {}}
    for baseline_key, baseline in report["baselines"].items():
        slim["baselines"][baseline_key] = {}
        for feature, stats in baseline["feature_statistics"].items():
            percentiles = stats.get("percentiles", {})
            slim["baselines"][baseline_key][feature] = {
                "mean": stats.get("mean", 0.0),
                "std": stats.get("std", 1.0),
                "p10": percentiles.get("p10", 0.0),
                "p25": percentiles.get("p25", 0.0),
                "p50": percentiles.get("p50", 0.0),
                "p75": percentiles.get("p75", 0.0),
                "p90": percentiles.get("p90", 0.0),
            }
    baselines_dir = OUT / "Baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    (baselines_dir / "analysis_baselines_v2.json").write_text(
        json.dumps(slim, indent=2), encoding="utf-8"
    )
    print(f"Exported baselines to {baselines_dir / 'analysis_baselines_v2.json'}")


def export_fault_definitions() -> None:
    sys_path = ROOT / "barpath"
    import sys

    sys.path.insert(0, str(sys_path))
    from pipeline.step4_helpers.compiled_analyzer import FAULT_DEFS

    fault_list = []
    for fid, fdef in FAULT_DEFS.items():
        fault_list.append(
            {
                "id": fid,
                "name": fdef["name"],
                "phase": fdef["phase"],
                "description": fdef["description"],
                "coachingCue": fdef["coaching_cue"],
                "liftTypes": fdef["lift_types"],
                "severity": fdef["severity"],
            }
        )

    fault_dir = OUT / "Baselines"
    fault_dir.mkdir(parents=True, exist_ok=True)
    (fault_dir / "fault_defs_v2.json").write_text(
        json.dumps({"version": "2.0.0", "faults": fault_list}, indent=2),
        encoding="utf-8",
    )
    print(f"Exported fault definitions to {fault_dir / 'fault_defs_v2.json'}")


def export_tree(tree, feature_names: list[str]) -> dict:
    raw = tree.tree_
    nodes = []
    for node_id in range(raw.node_count):
        left = int(raw.children_left[node_id])
        right = int(raw.children_right[node_id])
        is_leaf = left == right or left < 0
        if is_leaf:
            nodes.append(
                {
                    "id": node_id,
                    "featureIndex": None,
                    "featureName": None,
                    "threshold": None,
                    "left": None,
                    "right": None,
                    "classCounts": raw.value[node_id][0].astype(float).tolist(),
                }
            )
        else:
            feature_index = int(raw.feature[node_id])
            nodes.append(
                {
                    "id": node_id,
                    "featureIndex": feature_index,
                    "featureName": feature_names[feature_index],
                    "threshold": float(raw.threshold[node_id]),
                    "left": left,
                    "right": right,
                    "classCounts": None,
                }
            )
    return {"nodes": nodes}


def export_lift_classifier() -> None:
    model_path = MODELS / "lift_detection" / "lift_detection_model.pkl"
    if not model_path.exists():
        print(f"WARNING: {model_path} not found, skipping lift classifier export")
        return

    with model_path.open("rb") as f:
        model_data = pickle.load(f)

    classifier = model_data["classifier"]
    scaler = model_data["scaler"]
    feature_names = list(model_data["feature_names"])
    classes = [str(cls) for cls in classifier.classes_]

    exported = {
        "version": "2.0.0",
        "source": "barpath/models/lift_detection/lift_detection_model.pkl",
        "classes": classes,
        "featureNames": feature_names,
        "scaler": {
            "mean": scaler.mean_.astype(float).tolist(),
            "scale": scaler.scale_.astype(float).tolist(),
        },
        "trees": [export_tree(tree, feature_names) for tree in classifier.estimators_],
    }

    models_dir = OUT / "Models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "lift_classifier_rf_v2.json").write_text(
        json.dumps(exported, indent=2), encoding="utf-8"
    )
    print(
        f"Exported lift classifier ({len(classifier.estimators_)} trees, {len(feature_names)} features) "
        f"to {models_dir / 'lift_classifier_rf_v2.json'}"
    )


def export_yolo_manifest() -> None:
    manifest = {
        "modelName": "yolo26",
        "input": {
            "name": "image",
            "width": 640,
            "height": 640,
            "colorSpace": "RGB",
            "normalization": "0_1",
        },
        "output": {
            "kind": "multiArray",
            "name": "var_1440",
            "layout": "xyxy_conf_class",
            "coordinates": "model_input_pixels",
            "classIndexForBarbellEndcap": 0,
            "shape": [1, -1, 6],
            "endToEnd": True,
        },
        "postprocess": {
            "confidenceThreshold": 0.25,
            "iouThreshold": 0.45,
            "maxDetections": 20,
        },
    }
    manifests_dir = OUT / "Manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / "yolo_barbell_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Exported YOLO manifest to {manifests_dir / 'yolo_barbell_manifest.json'}")


def export_analysis_manifest() -> None:
    manifest = {
        "version": "1.0.0",
        "assets": {
            "yoloModel": "yolo26.mlpackage",
            "liftClassifier": "Models/lift_classifier_rf_v2.json",
            "liftClassifierConfig": "lift_detection_config.json",
            "baselines": "Baselines/analysis_baselines_v2.json",
            "faultDefinitions": "Baselines/fault_defs_v2.json",
        },
    }
    manifests_dir = OUT / "Manifests"
    (manifests_dir / "analysis_asset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"Exported analysis manifest to {manifests_dir / 'analysis_asset_manifest.json'}"
    )


if __name__ == "__main__":
    export_baselines()
    export_fault_definitions()
    export_lift_classifier()
    export_yolo_manifest()
    export_analysis_manifest()
    print("\nAll exports complete!")
