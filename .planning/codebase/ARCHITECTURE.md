<!-- refreshed: 2026-05-01 -->
# Architecture

**Analysis Date:** 2026-05-01

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────┐
│                        ENTRY POINTS                                  │
│   barpath_cli.py  (argparse + rich)    barpath_gui.py  (toga)        │
└───────────────────────┬──────────────────────────────────────────────┘
                        │ calls run_pipeline / run_pipeline_from_folder
                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR                                     │
│               barpath_core.py  (Generator-based)                     │
│   run_pipeline() — yields (step_name, progress, message) tuples      │
│   run_pipeline_from_folder() — re-runs steps 2–5 from raw_data.pkl   │
│   run_batch_postprocess() — superimposed graphs across multiple lifts │
│   run_pipeline_simple() — synchronous wrapper                         │
└───────┬──────────┬──────────┬──────────┬──────────┬──────────────────┘
        │          │          │          │          │
        ▼          ▼          ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ STEP 1   │ │ STEP 2   │ │ STEP 3   │ │ STEP 4   │ │ STEP 5   │
│ Collect  │→│ Analyze  │→│ Generate │→│ Critique │→│ Render   │
│ Data     │ │ Data     │ │ Graphs   │ │ Lift     │ │ Video    │
│ .py .py  │ │ 2_analyze│ │ 3_generate│ │ 4_critique│ │ 5_render │
│          │ │ _data.py │ │ _graphs  │ │ _lift.py │ │ _video   │
└────┬─────┘ └────┬─────┘ └──────────┘ └────┬─────┘ └────┬─────┘
     │            │                          │            │
     ▼            ▼                          ▼            ▼
┌──────────┐ ┌─────────────┐          ┌──────────────┐ ┌────────┐
│step1_    │ │step2_helpers│          │step4_helpers  │ │utils.py│
│helpers   │ │ kinematics  │          │ compiled_     │ │config  │
│.py .py   │ │ .py .py.phase│         │ analyzer.py   │ │.py     │
│stabiliza-│ │ landmark_   │          │ feature_      │ │analysis│
│tion   │ │ processing│          │ extraction.py │ │_utils  │
│landmarks │ │ classics_   │          │ smart_analysis│ │.py     │
│          │ │ phase_detect│          │ .py           │ │        │
└──────────┘ └─────────────┘          └──────────────┘ └────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| CLI | Parse args, run pipeline, display rich progress | `barpath/barpath_cli.py` |
| GUI | Toga desktop app, tab-based UI, background thread for pipeline | `barpath/barpath_gui.py` |
| Orchestrator | 5-step pipeline state machine, cancel support, auto-detect lift type | `barpath/barpath_core.py` |
| Step 1 | YOLO barbell detection + MediaPipe pose + optical flow stabilization | `barpath/pipeline/1_collect_data.py` |
| Step 2 | Kinematics, joint angles, phase detection, CSV output | `barpath/pipeline/2_analyze_data.py` |
| Step 3 | matplotlib kinematic graphs (bar path, velocity, acceleration, power) | `barpath/pipeline/3_generate_graphs.py` |
| Step 4 | Rule-based + ML technique critique, Markdown report | `barpath/pipeline/4_critique_lift.py` |
| Step 5 | OpenCV overlay video with skeleton, bar-path, HUD | `barpath/pipeline/5_render_video.py` |
| Config | Centralized constants and thresholds | `barpath/pipeline/config.py` |
| Hardware Detection | OS/CPU/GPU detection for acceleration packages | `barpath/hardware_detection.py` |
| Live Detection | Real-time lift classification state machine | `barpath/pipeline/live_detection_system.py` |

## Pattern Overview

**Overall:** Sequential 5-stage pipeline (Generator-based producer-consumer)

**Key Characteristics:**
- Each pipeline step is a standalone Python script importable by function name (`step_1_collect_data`, `step_2_analyze_data`, etc.)
- Dynamic import via `importlib` in `barpath_core.py` loads step functions by path (`_import_step_function()`)
- Progress is communicated via Python generators yielding `(step_name, progress_float, message)` tuples
- File-based data passing: raw pickle → CSV → graphs/reports/video
- The GUI runs the pipeline on a background `ThreadPoolExecutor` to keep the Toga event loop responsive
- Cancellation is supported via `threading.Event` passed through from the frontend to each step
- Separate "live detection" subsystem (`live_*.py`) uses a state machine pattern for real-time webcam inference, independent of the offline pipeline
- Step 1 uses a producer-consumer pattern: background thread decodes frames into a bounded queue (size 8), main thread runs inference
- Step helpers are organized into sub-packages by step number: `step1_helpers/`, `step2_helpers/`, `step4_helpers/`

## Layers

**Entry Points:**
- Purpose: User-facing interfaces for running analysis
- Location: `barpath/barpath_cli.py` and `barpath/barpath_gui.py`
- Contains: Argument parsing, UI widgets, user interaction logic
- Depends on: `barpath_core.py` (orchestrator)
- Used by: End users via terminal or desktop app

**Orchestration Layer:**
- Purpose: Coordinate the 5-step pipeline, manage data flow between steps
- Location: `barpath/barpath_core.py`
- Contains: `run_pipeline()`, `run_pipeline_from_folder()`, `run_batch_postprocess()`, `run_pipeline_simple()`
- Depends on: Pipeline step scripts (loaded dynamically via `importlib`)
- Used by: CLI and GUI entry points

**Pipeline Steps:**
- Purpose: Execute domain-specific analysis operations
- Location: `barpath/pipeline/1_collect_data.py` through `barpath/pipeline/5_render_video.py`
- Contains: Video processing, kinematics, graph generation, technique critique, video rendering
- Depends on: Helper packages, `config.py`, `utils.py`, `analysis_utils.py`
- Used by: Orchestrator (`barpath_core.py`)

**Helper Layer:**
- Purpose: Provide reusable algorithms to pipeline steps
- Location: `barpath/pipeline/step1_helpers/`, `step2_helpers/`, `step4_helpers/`
- Contains: Stabilization, landmark processing, kinematics, phase detection, technique analysis, feature extraction
- Depends on: External libraries (OpenCV, MediaPipe, NumPy, SciPy, scikit-learn)
- Used by: Pipeline step scripts

**Shared Utilities:**
- Purpose: Cross-cutting functions and configuration
- Location: `barpath/pipeline/config.py`, `utils.py`, `analysis_utils.py`, `barpath/hardware_detection.py`
- Contains: Constants, video drawing helpers, Savitzky-Golay smoothing, hardware detection
- Depends on: Minimal external dependencies
- Used by: All pipeline layers

**Live Detection Subsystem:**
- Purpose: Real-time lift type classification during live webcam preview
- Location: `barpath/pipeline/live_detection_system.py`, `live_buffer.py`, `live_feature_extractor.py`, `live_lift_recognition.py`, `live_window_features.py`, `live_training_data.py`, `lift_classifier.py`, `kinematic_completion.py`, `heuristic_classifier.py`
- Contains: Circular frame buffer, state machine, feature extractor, completion detector, heuristic+machine-learning classifiers
- Depends on: External libraries (OpenCV, NumPy, scikit-learn, MediaPipe)
- Used by: GUI live preview feature, independent of offline pipeline

## Data Flow

### Primary Request Path (Full Pipeline)

1. **Step 1 — Data Collection** (`barpath/pipeline/1_collect_data.py:step_1_collect_data()`)
   - Opens input video, runs producer-consumer frame pipeline
   - Producer thread: reads frames from disk into bounded queue (`DECODE_QUEUE_SIZE=8`)
   - Consumer main thread: runs YOLO barbell detection + MediaPipe pose estimation + optical flow stabilization
   - Output: `raw_data.pkl` (pickle with metadata dict + list of per-frame data dicts)

2. **Step 2 — Data Analysis** (`barpath/pipeline/2_analyze_data.py:step_2_analyze_data()`)
   - Loads `raw_data.pkl`, unpacks landmarks to per-joint columns
   - Calculates joint angles, smooths barbell position (Savitzky-Golay)
   - Detects lift phases (classic or kinematic approach)
   - Output: `final_analysis.csv` (pandas DataFrame with kinematics, phases)

3. **Step 3 — Generate Graphs** (`barpath/pipeline/3_generate_graphs.py:step_3_generate_graphs()`)
   - Reads `final_analysis.csv`
   - Generates bar-path plot, velocity/acceleration/power time-series
   - Output: `.png` graph files in output directory

4. **Step 4 — Technique Critique** (`barpath/pipeline/4_critique_lift.py:critique_lift()`)
   - Reads `final_analysis.csv` (as DataFrame)
   - Runs `CompiledAnalyzer` (rule-based fault detection with pro baselines)
   - Optionally runs `SmartAnalysis` (Random Forest model)
   - Output: `analysis.md` (Markdown report with star ratings, fault descriptions)

5. **Step 5 — Render Video** (`barpath/pipeline/5_render_video.py:step_5_render_video()`)
   - Reads `final_analysis.csv` and source video
   - Draws skeleton overlay, phase-colored bar path, legend, HUD
   - Output: `output.mp4`

### Reanalysis Path (Steps 2–5 Only)

`run_pipeline_from_folder()` in `barpath/barpath_core.py` skips Step 1 and uses an existing `raw_data.pkl` to re-run Steps 2–5. Useful for tweaking lift type or lifter settings without re-decoding video.

### Batch Postprocess Path

`run_batch_postprocess()` reads multiple `final_analysis.csv` files and produces superimposed bar-path comparison graphs with DTW similarity scores.

**State Management:**
- No long-lived in-memory state between pipeline runs
- Intermediate data persisted as files on disk (`raw_data.pkl` → `final_analysis.csv` → graphs/reports/video)
- Pipeline progress is yielded as generator tuples — callers (CLI/GUI) consume these as they arrive, no shared mutable state
- Cancellation is cooperative: `threading.Event` passed into each step, checked explicitly at yield points

## Key Abstractions

**Pipeline Step Function:**
- Purpose: Each step is a callable function dynamically imported by name from a numbered script
- Examples: `barpath/pipeline/1_collect_data.py` → `step_1_collect_data`, `barpath/pipeline/2_analyze_data.py` → `step_2_analyze_data`
- Pattern: Functions loaded via `_import_step_function()` using `importlib.util.spec_from_file_location`

**Generator Progress Tuple:**
- Purpose: All pipeline steps communicate progress via `(step_name: str, progress: float | None, message: str)` tuples
- Pattern: `yield ("step2", 0.5, "Processing frame 150/300")`
- Used by: CLI (rich progress bars), GUI (progress bar + log HTML)

**Detection State Machine:**
- Purpose: The live detection subsystem uses an enum-based state machine
- Location: `barpath/pipeline/live_detection_system.py:DetectionState` (IDLE → DETECTING → COMPLETE → JERK_WATCH → DISPLAYING)
- Pattern: States drive which processing logic runs each frame

**Circular Frame Buffer:**
- Purpose: Stores recent frames for sliding-window classification in live detection
- Location: `barpath/pipeline/live_buffer.py:CircularFrameBuffer`
- Pattern: Bounded-size deque that evicts oldest entries when full

**CompiledAnalyzer:**
- Purpose: Rule-based technique fault detection driven by pro athlete baseline percentiles
- Location: `barpath/pipeline/step4_helpers/compiled_analyzer.py:CompiledAnalyzer`
- Pattern: No hardcoded thresholds; all comparison values from `pro_baseline_report.json`
- Fallback when no Random Forest Smart Analysis model is available

**Config Module:**
- Purpose: All magic numbers, thresholds, and tunable parameters in one place
- Location: `barpath/pipeline/config.py`
- Pattern: Simple module-level constants imported by pipeline steps

## Entry Points

**CLI:**
- Location: `barpath/barpath_cli.py:main()`
- Triggers: Command-line invocation via `python barpath/barpath_cli.py` or `barpath` pip console script
- Responsibilities: Parse arguments, validate model/video paths, create output directories, run pipeline with rich progress display, handle batch processing with skip/rerun logic, display final analysis report

**GUI:**
- Location: `barpath/barpath_gui.py:BarpathTogaApp.startup()`
- Triggers: `barpath-gui` pip console script
- Responsibilities: Display tabbed interface (Files, Settings, Analyze, Analysis), manage input videos/folders, configure model/lift-type, run pipeline in background thread, render logs as HTML, display analysis results as styled HTML

**Console Scripts (setup.py):**
- `barpath=barpath.barpath_cli:main` — runs the CLI
- `barpath-gui=barpath.barpath_gui:main` — runs the GUI

## Architectural Constraints

- **Threading:** Step 1 uses a producer-consumer model with a decode thread + inference thread. The GUI runs pipeline on a single `ThreadPoolExecutor` worker thread. The live detection system runs on a separate `threading.Thread`. No thread safety issues because pipeline steps communicate via files and generators (not shared memory).
- **Global state:** `sys.path` is modified at import time in `barpath/barpath_core.py` (append pipeline and barpath dirs). `importlib` caches loaded modules globally. No other significant global mutable state.
- **Circular imports:** Step scripts import from `config.py`, `utils.py`, `analysis_utils.py`, and their respective helper packages. The helper packages import from each other only via their `__init__.py` re-exports. No known circular dependency chains.
- **File system coupling:** All data passes through the filesystem (pickle → CSV → graphs/video). This means concurrent runs on the same output directory will conflict. Skip/rerun logic exists for this reason.
- **Dynamic imports:** Step functions are loaded via string-based `importlib` — this means type checkers and linters cannot statically verify the step function signatures. The module `barpath/pipeline/__init__.py` explicitly documents this design choice.

## Anti-Patterns

### sys.path Manipulation

**What happens:** Four different files mutate `sys.path` by inserting project directories: `barpath/barpath_core.py` (line 44-46), `barpath/barpath_cli.py` (line 29), `barpath/barpath_gui.py` (line 36), `tests/conftest.py` (line 7), `scripts/retrain_lift_classifier.py` (line 11), `barpath/briefcase_hardware_installer.py` (line 16).

**Why it's wrong:** Fragile and redundant — any one of these changes could break module resolution depending on import order. If running as an installed package (via `pip install -e .`), these are unnecessary.

**Do this instead:** Install the package in development mode (`pip install -e .`) and rely on the package metadata for imports. The entry points in `setup.py` already handle this. Remove inline sys.path modifications from files other than `tests/conftest.py` (which is standard practice).

### Dynamic Import of Pipeline Steps

**What happens:** `barpath/barpath_core.py` uses `importlib.util.spec_from_file_location()` and `exec_module()` to load step functions from numbered filenames (`1_collect_data.py`, etc.) that are not valid Python identifiers.

**Why it's wrong:** Circumvents static analysis — no IDE, type checker, or linter can verify these imports. Any refactoring of step function signatures will silently break at runtime.

**Do this instead:** Rename step files to valid identifiers (e.g., `step1_collect_data.py`) or create a factory/registry pattern. See the existing approach in `pipeline/__init__.py` which explicitly acknowledges this.

## Error Handling

**Strategy:** Error propagation via generator yield, with fallback defaults and warnings.

**Patterns:**
- Step 1 yields `("_insufficient_data_", None, message)` for unrecoverable errors (e.g., no barbell detected)
- The CLI captures this and skips the video; the GUI shows it in the log
- Lift type auto-detection wraps in try/except, defaults to "clean" with warning printed
- Step functions print progress directly (not through a logging framework) — `print("--- Step 2: Analyzing Data ---")`
- `InsufficientDataError` from `kinematics.py` is explicitly caught in the orchestrator

## Cross-Cutting Concerns

**Logging:** Print-based for pipeline steps (stderr/stdout). `rich` library for CLI progress. `LogRenderer` in the GUI converts Rich-like markup to HTML. `logging` module is imported but not consistently configured — only `step4_helpers` modules use `logger.debug()`/`logger.warning()`.

**Validation:** CLI validates video extension, model file/directory existence, and OpenVINO directory structure. GUI validates input selection before enabling pipeline. Step 1 does YOLO model validation. No schema validation on pickle/CSV data passing between steps.

**Hardware Detection:** `barpath/hardware_detection.py` detects OS, CPU brand, GPU presence at install time to recommend the right accelerator packages. Used during `pip install` via `setup.py` and by the Briefcase installer helper. Not used at runtime — the YOLO model format (`.pt`, `.onnx`, `.engine`, OpenVINO XML) determines which backend is used.

---

*Architecture analysis: 2026-05-01*
