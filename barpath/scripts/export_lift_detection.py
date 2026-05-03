#!/usr/bin/env python3
"""Export lift detection sklearn model to ONNX and OpenVINO formats.

Usage:
    python barpath/scripts/export_lift_detection.py              # Export both
    python barpath/scripts/export_lift_detection.py --skip-onnx  # OpenVINO only
    python barpath/scripts/export_lift_detection.py --skip-openvino  # ONNX only
"""

import argparse
import json
import pickle
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Export lift detection model to ONNX/OpenVINO"
    )
    parser.add_argument(
        "--skip-onnx", action="store_true", help="Skip ONNX export"
    )
    parser.add_argument(
        "--skip-openvino", action="store_true", help="Skip OpenVINO export"
    )
    args = parser.parse_args()

    # Locate model directory
    model_dir = Path(__file__).parent.parent / "models" / "lift_detection"
    pkl_path = model_dir / "lift_detection_model.pkl"
    config_path = model_dir / "lift_detection_config.json"

    if not pkl_path.exists():
        print(f"Error: Model file not found: {pkl_path}", file=sys.stderr)
        sys.exit(1)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Load model and config
    print(f"Loading model from {pkl_path}")
    with open(pkl_path, "rb") as f:
        sklearn_model = pickle.load(f)
    with open(config_path) as f:
        config = json.load(f)

    n_features = config.get("n_features", len(config.get("feature_names", [])))
    classes = config.get("classes", [])
    print(f"Model: {type(sklearn_model).__name__}, {n_features} features, {classes}")

    # Export to ONNX
    onnx_path = None
    if not args.skip_onnx:
        print("\n--- ONNX Export ---")
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError:
            print(
                "Error: skl2onnx not installed. Run: pip install skl2onnx",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            import onnx
        except ImportError:
            print(
                "Error: onnx not installed. Run: pip install onnx",
                file=sys.stderr,
            )
            sys.exit(1)

        initial_type = [("input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(
            sklearn_model,
            initial_types=initial_type,
            target_opset=15,
        )

        onnx_path = model_dir / "lift_detection_model.onnx"
        onnx.save(onnx_model, str(onnx_path))
        print(f"Saved ONNX model: {onnx_path}")

    # Export ONNX to OpenVINO IR
    if not args.skip_openvino:
        print("\n--- OpenVINO Export ---")
        if onnx_path is None:
            onnx_path = model_dir / "lift_detection_model.onnx"
            if not onnx_path.exists():
                print(
                    "Error: ONNX model not found and --skip-onnx not set. "
                    "Run without --skip-onnx first, or provide an existing ONNX model.",
                    file=sys.stderr,
                )
                sys.exit(1)

        try:
            from openvino.tools import mo
        except ImportError:
            print(
                "Error: openvino not installed. Run: pip install openvino",
                file=sys.stderr,
            )
            sys.exit(1)

        ov_dir = model_dir / "lift_detection_openvino_export"
        ov_dir.mkdir(exist_ok=True)

        ov_model = mo.convert_model(
            onnx_path,
            compress_to_fp16=True,
            output_dir=str(ov_dir),
        )
        print(f"Saved OpenVINO model: {ov_dir}")

        # Verify output files
        xml_files = list(ov_dir.glob("*.xml"))
        bin_files = list(ov_dir.glob("*.bin"))
        if xml_files and bin_files:
            print(f"  .xml files: {len(xml_files)}")
            print(f"  .bin files: {len(bin_files)}")
        else:
            print("Warning: OpenVINO export may be incomplete (missing .xml or .bin)")

    # Summary
    print("\n--- Export Summary ---")
    if not args.skip_onnx:
        print(f"  ONNX: {onnx_path}")
    if not args.skip_openvino:
        print(f"  OpenVINO: {ov_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
