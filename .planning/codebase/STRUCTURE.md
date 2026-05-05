# Codebase Structure

**Analysis Date:** 2026-05-01

## Directory Layout

```
barpath/
├── .github/workflows/          # CI: ruff lint
├── .planning/                  # Codebase analysis documents
├── barpath/                    # ★ Main Python package (installable via setup.py)
│   ├── __init__.py             # Package version marker
│   ├── barpath_core.py         # Pipeline orchestrator (generator-based)
│   ├── barpath_cli.py          # CLI entry point (argparse + rich)
│   ├── barpath_gui.py          # GUI entry point (toga desktop app)
│   ├── briefcase_hardware_installer.py  # Briefcase post-install helper
│   ├── hardware_detection.py   # OS/CPU/GPU detection logic
│   ├── assets/                 # Static assets (SVG logo, sample images)
│   ├── gui_helpers/            # GUI rendering utilities
│   │   ├── log_renderer.py     # Rich-markup to HTML converter
│   │   ├── markdown_renderer.py # Markdown to HTML converter
│   │   └── templates/          # HTML templates for WebView
│   ├── models/                 # Pre-trained models and baselines
│   │   ├── analysis/           # Pro lifter baselines (trajectory npy + config)
│   │   │   ├── generic/        # Pooled baselines (clean, snatch, jerk)
│   │   │   ├── liao/           # Lifter-specific baselines
│   │   │   ├── lu/             # (and more lifters)
│   │   │   └── .../
│   │   ├── lift_detection/     # ML models for lift type classification
│   │   ├── pose_landmarker_heavy.task  # MediaPipe pose model
│   │   ├── std_nano.pt         # YOLO barbell detection model
│   │   └── std_nano_openvino_model/  # OpenVINO export
│   └── pipeline/               # ★ Core analysis pipeline
│       ├── __init__.py         # Version + dynamic import notice
│       ├── config.py           # Central constants and thresholds
│       ├── utils.py            # Shared video drawing helpers
│       ├── analysis_utils.py   # Shared Savitzky-Golay and power calc
│       ├── 1_collect_data.py   # Step 1: Video → raw_data.pkl
│       ├── 2_analyze_data.py   # Step 2: raw_data.pkl → final_analysis.csv
│       ├── 3_generate_graphs.py# Step 3: CSV → kinematic graphs (.png)
│       ├── 4_critique_lift.py  # Step 4: CSV → analysis.md (critique)
│       ├── 5_render_video.py   # Step 5: CSV + video → output.mp4
│       ├── step1_helpers/      # Step 1: stabilization + landmarks
│       │   ├── stabilization.py  # Lucas-Kanade optical flow
│       │   ├── landmarks.py      # MediaPipe pose landmark extraction
│       │   └── __init__.py       # Re-exports
│       ├── step2_helpers/      # Step 2: kinematics + phase detection
│       │   ├── kinematics.py           # Barbell kinematics, phase assignment
│       │   ├── landmark_processing.py  # Joint angle calculations
│       │   ├── classics_phase_detection.py  # Hip-based phase detection
│       │   └── __init__.py
│       ├── step4_helpers/      # Step 4: technique analysis
│       │   ├── compiled_analyzer.py    # Rule-based fault detector
│       │   ├── feature_extraction.py   # Feature extraction for analysis
│       │   ├── smart_analysis.py       # Random Forest fault detection
│       │   └── __init__.py
│       ├── lift_detection_features.py  # Lift type feature extraction
│       ├── lift_classifier.py          # Random Forest lift classifier
│       ├── heuristic_classifier.py     # Heuristic lift type detection
│       ├── kinematic_completion.py     # Live completion detection
│       ├── live_buffer.py              # Circular frame buffer
│       ├── live_detection_system.py    # Live detection state machine
│       ├── live_feature_extractor.py   # Live frame feature extraction
│       ├── live_lift_recognition.py    # Live lift recognition
│       ├── live_training_data.py       # Live training dataset generation
│       └── live_window_features.py     # Sliding window features
├── docs/                    # Documentation
│   ├── USAGE_GUIDE.md
│   ├── BUILD_INSTRUCTIONS.md
│   ├── HARDWARE_ACCELERATION_SETUP.md
│   ├── PROGRESS_FLOW_VERIFICATION.md
│   └── analysis_engine_v1_report.md
├── outputs/                 # Pipeline output directory (gitignored)
│   ├── diagrams/            # Architecture SVG diagrams
│   ├── plans/               # Refactoring plans
│   └── v1_analysis/         # Legacy v1 analysis artifacts
├── scripts/                 # Standalone utility scripts
│   ├── retrain_lift_classifier.py
│   ├── retrain_live_classifier.py
│   └── export_ios_assets.py
├── tests/                   # Pytest test suite
│   ├── conftest.py          # Adds project root to sys.path
│   ├── test_phases.py
│   ├── test_jerk_*.py       # Jerk-specific phase tests
│   ├── test_debug*.py       # Debug/exploratory tests
│   ├── test_stabilization.py
│   ├── test_lift_detection.py
│   ├── test_live_*.py       # Live detection tests
│   ├── test_cuda.py
│   └── test_clean_jerk_sequence.py
├── pyproject.toml           # basedpyright config
├── setup.py                 # Package definition + entry points
├── requirements.txt         # Core dependencies
├── requirements-hardware.txt # Optional accelerator packages
├── CLAUDE.md                # GitNexus instructions
├── AGENTS.md                # Agent instructions
└── README.md                # Project overview
```

## Directory Purposes

**`barpath/` (package root):**
- Purpose: Main installable Python package containing all application code
- Contains: Entry points, pipeline orchestrator, all pipeline modules, models, assets, GUI helpers
- Key files: `barpath_core.py` (orchestrator), `barpath_cli.py` (CLI), `barpath_gui.py` (GUI), `hardware_detection.py`

**`barpath/pipeline/` (core logic):**
- Purpose: All analysis pipeline code including numbered steps, helpers, live detection, and shared utilities
- Contains: 5 numbered pipeline step scripts (1 through 5), 3 step-helper packages, live detection subsystem files, configuration, utilities
- Key files: `config.py` (central constants), `1_collect_data.py` (video processing), `2_analyze_data.py` (kinematics/analysis), `3_generate_graphs.py` (~881 lines — largest single file), `4_critique_lift.py`, `5_render_video.py`

**`barpath/pipeline/step1_helpers/`:**
- Purpose: Video processing algorithms used by Step 1
- Contains: `stabilization.py` (optical flow), `landmarks.py` (MediaPipe pose)
- Dependency: OpenCV, MediaPipe, NumPy

**`barpath/pipeline/step2_helpers/`:**
- Purpose: Kinematics and phase detection algorithms used by Step 2
- Contains: `kinematics.py` (barbell smoothing, velocity/acceleration/power, phase assignment), `landmark_processing.py` (joint angles, facing direction), `classics_phase_detection.py` (hip-based phase detection)
- Dependency: NumPy, SciPy, Pandas

**`barpath/pipeline/step4_helpers/`:**
- Purpose: Technique analysis algorithms used by Step 4
- Contains: `compiled_analyzer.py` (rule-based fault detection with pro baselines, ~538 lines), `feature_extraction.py` (trajectory/technique features), `smart_analysis.py` (Random Forest multi-label classifier)
- Dependency: NumPy, Pandas, scikit-learn, DTW

**`barpath/models/`:**
- Purpose: Pre-trained models, baselines, and configuration for analysis
- Contains: YOLO barbell detection model (`.pt`), OpenVINO export, MediaPipe pose model (`.task`), lift detection ML models (`.pkl`), pro lifter baseline trajectories (`analysis/{lifter}/{lift}/{trajectory.npy,config.json}`)
- Committed: Yes — required for application to function

**`barpath/gui_helpers/`:**
- Purpose: Rendering utilities for the Toga-based GUI
- Contains: `log_renderer.py` (Rich→HTML conversion), `markdown_renderer.py` (Markdown→HTML), `templates/` (HTML templates)
- Dependency: Markdown Python library

**`tests/`:**
- Purpose: Pytest test files for pipeline components
- Contains: Phase detection tests, live detection tests, debug/exploratory tests, CUDA detection test, stabilization test
- Pattern: Tests are one file per concept area, not per pipeline step
- Note: Several tests are exploratory scripts rather than structured unit tests (e.g., `test_phases.py` runs inline on real output data)

**`scripts/`:**
- Purpose: Standalone utility scripts for model retraining and iOS export
- Contains: `retrain_lift_classifier.py`, `retrain_live_classifier.py`, `export_ios_assets.py`
- Not part of the main package — run separately as needed

**`docs/`:**
- Purpose: User and developer documentation
- Contains: Usage guide, build instructions, hardware acceleration setup, progress flow verification, legacy v1 analysis report

**`outputs/`:**
- Purpose: Default directory for pipeline output artifacts
- Contains: Generated graphs (`.png`), analysis reports (`.md`), CSVs, pickle files, videos, diagrams, architecture plans
- Not committed to git (except `diagrams/` and `plans/` subdirectories which contain project planning artifacts)

## Key File Locations

**Entry Points:**
- `barpath/barpath_cli.py`: CLI entry point (console_script: `barpath`)
- `barpath/barpath_gui.py`: GUI entry point (console_script: `barpath-gui`)
- `barpath/briefcase_hardware_installer.py`: Briefcase installer script
- `setup.py`: Package definition defining console_scripts

**Configuration:**
- `barpath/pipeline/config.py`: Central constants and thresholds for all 5 pipeline steps (188 lines)
- `setup.py`: Package metadata, dependencies, extras (hardware, dev)
- `pyproject.toml`: Type checking configuration (basedpyright)
- `requirements.txt`: Core runtime dependencies
- `requirements-hardware.txt`: Optional hardware acceleration dependencies
- `.github/workflows/ruff.yml`: CI pipeline (ruff lint on push/PR to main)

**Core Logic:**
- `barpath/barpath_core.py`: Pipeline orchestrator — coordinates all 5 steps (545 lines)
- `barpath/pipeline/1_collect_data.py`: YOLO + MediaPipe + stabilization (409 lines)
- `barpath/pipeline/2_analyze_data.py`: Kinematics + phase detection (251 lines)
- `barpath/pipeline/3_generate_graphs.py`: Matplotlib graph generation (881 lines — largest)
- `barpath/pipeline/4_critique_lift.py`: Technique critique with rule-based analyzer (483 lines)
- `barpath/pipeline/5_render_video.py`: Video overlay rendering (421 lines)
- `barpath/pipeline/step4_helpers/compiled_analyzer.py`: Rule-based fault detection engine (538 lines)
- `barpath/pipeline/step2_helpers/kinematics.py`: Barbell kinematics calculations
- `barpath/pipeline/live_detection_system.py`: Live detection state machine (383 lines)
- `barpath/hardware_detection.py`: OS/CPU/GPU detection for install (376 lines)

**Testing:**
- `tests/conftest.py`: Pytest config — adds project root to sys.path
- `tests/test_phases.py`: Phase detection comparison tests
- `tests/test_lift_detection.py`: Lift type classification tests
- `tests/test_live_classifier.py`: Live classifier tests
- `tests/test_stabilization.py`: Camera stabilization tests
- `tests/test_cuda.py`: CUDA availability test
- `tests/test_progress_flow.py`: Pipeline progress flow tests

## Naming Conventions

**Files:**
- Pipeline step scripts: `{number}_{description}.py` (e.g., `1_collect_data.py`, `2_analyze_data.py`)
- Helper sub-packages: `step{number}_helpers/` (e.g., `step1_helpers/`, `step2_helpers/`)
- Live detection files: `live_{component}.py` (e.g., `live_buffer.py`, `live_feature_extractor.py`)
- Test files: `test_{subject}.py` (e.g., `test_phases.py`, `test_stabilization.py`)
- Script files: `{function}.py` (e.g., `retrain_lift_classifier.py`, `export_ios_assets.py`)
- Configuration files: Standard names (`config.py`, `setup.py`, `pyproject.toml`)
- HTML templates: `{viewer}_viewer.html` (e.g., `log_viewer.html`, `analysis_viewer.html`)

**Directories:**
- Sub-packages use `snake_case` (e.g., `step1_helpers/`, `gui_helpers/`, `lift_detection/`)
- Model directories use lifter names in lowercase (e.g., `liao/`, `talakhadze/`, `ilyin/`)
- Output subdirectories use video filename stem (e.g., `outputs/botev_10_clean/`)

**Functions:**
- Pipeline step functions: `step_{number}_{action}()` (e.g., `step_1_collect_data()`, `step_2_analyze_data()`)
- Class names: `PascalCase` (e.g., `LiftDetectionSystem`, `CompiledAnalyzer`, `CircularFrameBuffer`, `LogRenderer`, `BarpathTogaApp`)
- Helper functions: `snake_case` (e.g., `calculate_joint_angles()`, `smooth_barbell_position()`, `unpack_landmarks()`)
- Private helper functions: `_leading_underscore_snake_case` (e.g., `_import_step_function()`, `_detect_lift_type_auto()`, `_get_model_path()`)
- Event handlers in GUI: `on_{action}_{noun}()` (e.g., `on_run_analysis()`, `on_browse_video()`, `on_select_output_dir()`)

**Variables:**
- `snake_case` throughout
- Type hints used consistently (Python 3.8+ compatible with `from __future__ import annotations` in some files)

## Where to Add New Code

**New Feature (e.g., new metric in analysis):**
- If it changes kinematic calculations: add to `barpath/pipeline/step2_helpers/kinematics.py`
- If it's a new graph: add to `barpath/pipeline/3_generate_graphs.py`
- If it's a new technique fault: add to `barpath/pipeline/step4_helpers/compiled_analyzer.py` (rule) or `smart_analysis.py` (ML)
- If it's a pipeline step: create `barpath/pipeline/{N}_{description}.py` with a function named `step_{N}_{description}`, then register it in `barpath/barpath_core.py` via `_import_step_function()`
- Tests: `tests/test_{feature}.py`

**New Live Detection Feature:**
- New component: `barpath/pipeline/live_{component}.py`
- State: add new state to `DetectionState` enum in `live_detection_system.py`
- Tests: `tests/test_live_{component}.py`

**New Entry Point:**
- Create file in `barpath/` (e.g., `barpath_web.py`)
- Register in `setup.py` under `entry_points.console_scripts`

**New Helper Package:**
- Create `barpath/pipeline/step{N}_helpers/` with `__init__.py` re-exporting public API
- Import from step scripts with `from step{N}_helpers import ...`

**New Model Baseline:**
- Add trajectory data: `barpath/models/analysis/{lifter_name}/{lift_type}/trajectory.npy`
- Add config: `barpath/models/analysis/{lifter_name}/{lift_type}/config.json`
- Update lifter discovery in `barpath/barpath_gui.py:_populate_lifter_options()`

**New GUI Tab:**
- Add tab button in `barpath/barpath_gui.py:BarpathTogaApp.startup()` sidebar
- Add `_build_{tab}_page()` method returning a `toga.Box`
- Add page switching logic in `_select_tab()`

**Shared Utilities:**
- If used across multiple steps: add to `barpath/pipeline/analysis_utils.py` or `barpath/pipeline/utils.py`
- If a new constant/threshold: add to `barpath/pipeline/config.py`

## Special Directories

**`barpath/models/`:**
- Purpose: All pre-trained models, baselines, and model config
- Generated: No (manually placed, except `lift_detection_model.pkl` which can be retrained)
- Committed: Yes — required for application function

**`outputs/`:**
- Purpose: Default pipeline output location
- Generated: Yes — created at runtime
- Committed: No (in `.gitignore`), except `outputs/diagrams/` and `outputs/plans/` which contain planning artifacts

**`barpath/__pycache__/` and `**/__pycache__/`:**
- Purpose: Python bytecode cache
- Generated: Yes
- Committed: No (in `.gitignore`)

**`venv/`:**
- Purpose: Python virtual environment (local development)
- Generated: Yes
- Committed: No (in `.gitignore`)

---

*Structure analysis: 2026-05-01*
