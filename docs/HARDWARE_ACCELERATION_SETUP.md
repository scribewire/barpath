# Hardware Acceleration Setup

barpath runs on CPU out of the box. This guide explains the optional hardware-accelerated inference backends and how to install them.

## What's optional

| Backend | Benefit | Applies to |
|---------|---------|-----------|
| **ONNX Runtime** (`onnxruntime`) | ~1–2× faster CPU inference for exported `.onnx` models | all platforms |
| **OpenVINO** (`openvino`) | ~2–4× faster inference on Intel CPUs | Intel CPU, Linux/Windows |
| **TensorRT** (`tensorrt`) | 3–5× faster inference on NVIDIA GPUs | NVIDIA GPU, Linux/Windows |

None are required. The pipeline detects what's installed and falls back to PyTorch CPU inference.

## Quick install

```bash
# Universal CPU optimization
pip install onnxruntime

# Intel CPU
pip install openvino

# NVIDIA GPU
pip install tensorrt        # + export your model to .engine
```

Or use the interactive installer, which detects your hardware (OS, CPU brand, GPU) and prints the exact command:

```bash
python barpath/briefcase_hardware_installer.py
```

## How detection works

`barpath/hardware_detection.py`:
- `get_hardware_profile()` — detects OS, CPU brand (Intel/AMD), and NVIDIA GPU presence.
- `get_optional_packages(profile)` — returns recommended ONNX and OpenVINO packages for the profile.
- `detect_installed_runtimes()` — probes which runtimes are actually installed.
- `get_available_runtimes_for_model()` — maps a model file (`.pt`/`.onnx`/`.engine`/OpenVINO dir) to the runtimes that can serve it.

The model-loading code in `barpath/pipeline/1_collect_data.py` then chooses a device in priority order (CUDA GPU → whatever optional runtime is installed → CPU) and prints which acceleration is active.

## Verifying

```bash
python -c "
from barpath.hardware_detection import get_hardware_profile, get_optional_packages
p = get_hardware_profile()
print('Hardware Profile:', p)
o, v = get_optional_packages(p)
print('Recommended packages:', o + v)
"
```

## Using an exported model

**ONNX:**
```bash
pip install onnxruntime
yolo export model=barpath/models/std_nano.pt format=onnx
python barpath/barpath_cli.py --input_video lift.mp4 --model models/best.onnx --lift_type clean
```

**OpenVINO (Intel CPU):**
```bash
pip install openvino
yolo export model=barpath/models/std_nano.pt format=openvino
python barpath/barpath_cli.py --input_video lift.mp4 --model barpath/models/std_nano_openvino_model --lift_type clean
```

**TensorRT (NVIDIA GPU):**
```bash
pip install tensorrt
yolo export model=barpath/models/std_nano.pt format=engine
python barpath/barpath_cli.py --input_video lift.mp4 --model models/best.engine --lift_type clean
```

## Notes

- Model files included in the repo: `barpath/models/std_nano.pt` (PyTorch) and `barpath/models/std_nano_openvino_model/` (OpenVINO export, ready to use).
- `requirements-hardware.txt` documents the full matrix of optional packages. It is intentionally all-commented so nothing is force-installed.
- If hardware acceleration packages are missing, barpath simply runs on CPU — no error, no special configuration.