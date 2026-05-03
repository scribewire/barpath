# Phase 1: Code Reorganization — Context

## Domain
File/folder restructuring — consolidating live preview code into `barpath/pipeline/realtime_processing/` and moving training/model generation scripts into `barpath/scripts/`, while preserving all imports, functionality, and test coverage.

## Decisions

### Import Strategy
- **`__init__.py` re-exports** for `realtime_processing/` package — all public classes and functions are re-exported from `barpath/pipeline/realtime_processing/__init__.py`
- Consumers (e.g., `barpath_gui.py`) import via `from barpath.pipeline.realtime_processing import LiveDetectionSystem`
- Within the package, files use relative imports (`from . import X`)

### Dependency Boundaries
- **Move only `live_*.py` files** to `realtime_processing/`:
  - `live_lift_recognition.py`
  - `live_window_features.py`
  - `live_training_data.py`
  - `live_feature_extractor.py`
  - `live_detection_system.py`
  - `live_buffer.py`
- **Keep in `pipeline/` root** (shared with offline pipeline):
  - `lift_classifier.py`
  - `heuristic_classifier.py`
  - `kinematic_completion.py`
  - `lift_detection_features.py`
- Moved files will import shared classifiers via `from barpath.pipeline import lift_classifier` (absolute imports for cross-package dependencies)

### Scripts Structure
- **`barpath/scripts/` as package** with `__init__.py`
- Scripts moved from repo root `scripts/` to `barpath/scripts/`:
  - `retrain_lift_classifier.py`
  - `retrain_live_classifier.py`
- `export_ios_assets.py` also moves to `barpath/scripts/` (it's a utility script)
- Scripts run via `python -m barpath.scripts.retrain_lift_classifier` or direct execution
- Remove `sys.path` manipulation from scripts — rely on package installation (`pip install -e .`)

### Test File Placement
- **Keep tests/ in place**, update imports to point to new `realtime_processing` location
- Affected test files:
  - `tests/test_live_classifier.py`
  - `tests/test_live_detection.py`
  - Any other `test_live_*.py` files

## Canonical Refs
- `.planning/PROJECT.md` — Project roadmap with phase goals
- `.planning/codebase/STRUCTURE.md` — Current directory layout and naming conventions
- `.planning/codebase/ARCHITECTURE.md` — Pipeline architecture, live detection subsystem, import patterns
- `.planning/codebase/STACK.md` — Technology stack and dependencies
- `barpath/pipeline/__init__.py` — Existing dynamic import documentation
- `barpath/barpath_gui.py` — Primary consumer of live detection modules
- `barpath/barpath_core.py` — Pipeline orchestrator with `_import_step_function()`
- `scripts/retrain_lift_classifier.py` — Current script with sys.path manipulation
- `scripts/retrain_live_classifier.py` — Current script with sys.path manipulation
- `scripts/export_ios_assets.py` — iOS asset export script

## Code Context

### Reusable Assets
- `barpath/pipeline/live_detection_system.py:DetectionState` — Enum-based state machine (IDLE → DETECTING → COMPLETE → JERK_WATCH → DISPLAYING)
- `barpath/pipeline/live_buffer.py:CircularFrameBuffer` — Bounded-size deque for frame storage
- `barpath/pipeline/live_feature_extractor.py` — Per-frame feature extraction
- `barpath/pipeline/live_lift_recognition.py` — Live lift recognition logic
- `barpath/pipeline/live_window_features.py` — Sliding window feature computation
- `barpath/pipeline/live_training_data.py` — Training dataset generation

### Established Patterns
- Pipeline step scripts use numbered naming (`1_collect_data.py`) with dynamic import via `importlib`
- Helper sub-packages use `step{N}_helpers/` naming with `__init__.py` re-exports
- Live detection files use `live_{component}.py` naming convention
- Test files use `test_{subject}.py` naming in `tests/` directory
- Scripts use `{function}.py` naming at repo root

### Integration Points
- `barpath/barpath_gui.py` imports live detection modules for webcam preview
- `barpath/pipeline/__init__.py` may need updates for new package structure
- Test files in `tests/` import from `barpath.pipeline` — need import path updates
- Scripts use `sys.path.insert()` to access pipeline modules — should be replaced with package imports

### Anti-Patterns to Address
- **sys.path manipulation** in scripts — replace with proper package imports after move
- **Redundant sys.path modifications** across multiple files — opportunity to clean up during reorganization

## Deferred Ideas
- (none — all scope captured in phase goal)
