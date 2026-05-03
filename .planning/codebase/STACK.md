# Technology Stack

**Analysis Date:** 2026-05-01

## Languages

**Primary:**
- Python 3.8+ (setup.py specifies `python_requires=">=3.8"`; README recommends 3.12+) — All application logic, pipeline, CLI, GUI

**Secondary:**
- None detected (no JavaScript, TypeScript, Rust, or other languages in the repo)

## Runtime

**Environment:**
- Python 3.8+ (CPython)

**Package Manager:**
- pip (Python Package Installer)
- Lockfile: Not detected (no `requirements.lock`, `Pipfile.lock`, or `poetry.lock`)
- Dependencies specified in `requirements.txt` and `requirements-hardware.txt`
- Optional: `setup.py` with extras (`[onnx]`, `[openvino]`, `[hardware]`, `[dev]`)

## Frameworks

**Core Computer Vision & ML:**
- **OpenCV** (`opencv-python>=4.10.0`) — Video capture, frame processing, optical flow stabilization, video rendering — `barpath/pipeline/1_collect_data.py`, `barpath/pipeline/5_render_video.py`
- **MediaPipe** (`mediapipe>=0.10.25`) — Pose landmark estimation (Tasks API with PoseLandmarker) — `barpath/pipeline/1_collect_data.py`
- **Ultralytics YOLO** (`ultralytics>=8.3.0`) — Barbell endcap detection via YOLO26 NMS-free models — `barpath/pipeline/1_collect_data.py`
- **PyTorch** (`torch>=2.5.0`, `torchvision>=0.20.0`) — Backend for YOLO model inference (CUDA GPU support) — `barpath/pipeline/1_collect_data.py`

**Data Processing & Analysis:**
- **pandas** (`>=2.2.0`) — CSV analysis data, phase detection — `barpath/pipeline/2_analyze_data.py`
- **numpy** (`>=1.26.0`) — Array operations throughout pipeline — Multiple files
- **scipy** (`>=1.14.0`) — Savitzky-Golay signal smoothing — `barpath/pipeline/analysis_utils.py`
- **scikit-learn** (`>=1.5.0`) — Random Forest classifier for Smart Analysis fault detection — `barpath/pipeline/step4_helpers/smart_analysis.py`
- **dtw-python** (`>=1.4.0`) — Dynamic Time Warping for bar path similarity comparison — `barpath/pipeline/step4_helpers/compiled_analyzer.py`

**Visualization:**
- **matplotlib** (`>=3.9.0`) — Kinematic graphs (bar path, velocity, acceleration, power) — `barpath/pipeline/3_generate_graphs.py`

**GUI:**
- **Toga** (`toga>=0.4.7`, BeeWare project) — Cross-platform Python GUI framework — `barpath/barpath_gui.py`

**CLI:**
- **Rich** (`rich>=13.9.0`) — Terminal formatting, progress bars, markdown rendering — `barpath/barpath_cli.py`

**Testing:**
- **pytest** (`>=7.0.0`) — Test runner — `tests/` directory
- Config: `tests/conftest.py` adds project root to sys.path

**Linting/Formatting:**
- **ruff** (`>=0.4.0`) — Linter and formatter — configured via `pyproject.toml` `[tool.basedpyright]` and GitHub Actions workflow `.github/workflows/ruff.yml`
- **basedpyright** (`>=1.15.0`) — Type checking — `pyproject.toml` config with relaxed settings

**Build/Packaging:**
- **Briefcase** (BeeWare project) — Standalone installer creation for Windows (.msi), macOS (.dmg), Linux (.deb) — `docs/BUILD_INSTRUCTIONS.md`
- **setuptools** — Package distribution via `setup.py`

## Key Dependencies

**Critical (core pipeline):**
| Package | Minimum Version | Why It Matters |
|---------|----------------|----------------|
| `opencv-python` | 4.10.0 | Video I/O, frame decoding, rendering |
| `mediapipe` | 0.10.25 | Full-body pose landmark estimation (12 joints) |
| `ultralytics` | 8.3.0 | YOLO26 NMS-free barbell detection model |
| `torch` | 2.5.0 | PyTorch backend for YOLO inference |
| `pandas` | 2.2.0 | CSV data storage and analysis |
| `scipy` | 1.14.0 | Savitzky-Golay smoothing of joint/bar data |
| `matplotlib` | 3.9.0 | All kinematic graphs and comparisons |
| `numpy` | 1.26.0 | Fundamental array/matrix operations |

**Infrastructure & Optional:**
| Package | Purpose | Notes |
|---------|---------|-------|
| `rich` | CLI progress bars and formatted output | Core dependency |
| `toga` | Cross-platform GUI | Core dependency |
| `pycairo` | Low-level drawing for some GUI backends | Core dependency |
| `scikit-learn` | Random Forest fault detection | Step 4 Smart Analysis |
| `dtw-python` | DTW bar path similarity | Step 4 Fast Analysis |
| `onnxruntime` / `onnxruntime-gpu` | ONNX model inference | Optional hardware acceleration |
| `openvino` | Intel CPU model optimization | Optional hardware acceleration |
| `tensorrt` + `pycuda` | NVIDIA GPU TensorRT | Optional hardware acceleration |
| `pytest` | Unit/integration testing | Dev dependency |
| `ruff` | Linting and formatting | Dev dependency |
| `basedpyright` | Type checking | Dev dependency |

## Configuration

**Environment:**
- No `.env` files detected (`.env*` files absent from repo)
- Configuration is file-based: pipeline parameters in `barpath/pipeline/config.py`
- Hardware detection is runtime: `barpath/hardware_detection.py` auto-detects OS, CPU, GPU

**Key Configuration Files:**
- `barpath/pipeline/config.py` — Central config for all pipeline thresholds, smoothing windows, phase detection, graph sizes, phase colors
- `pyproject.toml` — basedpyright type checker settings
- `.gitignore` — Ignores `__pycache__/`, `venv/`, `.vscode/`, `.idea/`, `outputs/`, `build/`, `*.mp4`, `datasets/`

**System Dependencies (not pip-installable):**
- **FFmpeg** — Video processing and audio muxing (system install required)
- **Git LFS** — Large file support for model files

## Platform Requirements

**Development:**
- Python 3.8+ (3.12+ recommended)
- pip, venv or equivalent
- FFmpeg installed and available in PATH
- Git LFS for model files
- Optional: NVIDIA GPU with CUDA for GPU-accelerated inference
- Optional: Intel CPU for OpenVINO optimization

**Production:**
- Standalone installers via Briefcase (Windows .msi, macOS .dmg, Linux .deb)
- No server deployment — fully offline/desktop application
- ~500MB–1GB installer size (includes Python runtime + all dependencies)

---

*Stack analysis: 2026-05-01*
