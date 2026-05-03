# Coding Conventions

**Analysis Date:** 2026-05-01

## Languages

**Primary:** Python 3.8+ (project requires `>=3.8` in `setup.py`)

## Naming Patterns

**Files:**
- `snake_case.py` for all source files — e.g., `barpath_core.py`, `live_lift_recognition.py`, `analysis_utils.py`
- `test_*.py` for test files — e.g., `test_lift_detection.py`, `test_phases.py`
- Numbered pipeline scripts with underscores: `1_collect_data.py`, `2_analyze_data.py`, `3_generate_graphs.py`, `4_critique_lift.py`, `5_render_video.py`
- `step{N}_helpers/` subdirectories for module-specific helpers — e.g., `step1_helpers/`, `step2_helpers/`, `step4_helpers/`
- Config files use standard names (`setup.py`, `pyproject.toml`, `requirements.txt`, `requirements-hardware.txt`)

**Functions:**
- `snake_case` exclusively — e.g., `run_pipeline()`, `detect_cpu_brand()`, `safe_savgol_smooth()`, `_import_step_function()`
- Leading underscore for private/package-internal functions — e.g., `_is_openvino_model_dir()`, `_detect_lift_type_auto()`, `_safe_savgol()`, `_MockClassifier`
- Verb-oriented names: `run_pipeline()`, `calculate_stabilized_position()`, `extract_trajectory()`, `classify_with_heuristics()`
- Helper/predicate functions named descriptively: `_is_openvino_model_dir()`, `validate_phases()`, `detect_clean_jerk_split_point()`

**Variables:**
- `snake_case` — e.g., `output_folder`, `input_video`, `lift_type`, `csv_path`, `prev_gray`
- Leading underscore for "private" module-level variables: `_run_pipeline`, `_QUEUE_DONE`, `_source_video_abs`
- Short names (t, x, y, df, n) acceptable in tight mathematical/array contexts — e.g., in `compiled_analyzer.py` line ~246 area where `a`, `b`, `x`, `y` are local

**Types:**
- `PascalCase` for classes and type aliases — e.g., `BarpathTogaApp`, `LiveLiftRecognizer`, `LiftState`, `FrameData`, `InsufficientDataError`
- `PascalCase` for `Enum` classes: `LiftState(Enum)` with `UPPER_CASE` members like `IDLE`, `RECORDING`, `CLASSIFYING`

**Constants:**
- `UPPER_SNAKE_CASE` for module-level constants — e.g., `DECODE_QUEUE_SIZE = 8`, `YOLO_CONFIDENCE_THRESHOLD = 0.25`, `PHASE_COLORS`, `LANDMARK_NAMES`, `GRID_FEATURE_NAMES`

## Code Style

**Formatting:**
- **Ruff** (v0.15.6, per `.ruff_cache/`) for both linting and auto-formatting
- CI enforces: `ruff check --fix . && ruff format .` in `.github/workflows/ruff.yml`
- No custom Ruff config in `pyproject.toml` — defaults are used
- No `pyproject.toml` `[tool.ruff]` section detected, so default line length (88) and rules apply

**Linting:**
- CI uses `ruff check --fix .` — failing lint blocks CI
- basedpyright for static type checking in `pyproject.toml` with `typeCheckingMode = "standard"` and many relaxed settings (`reportUnknownParameterType = "none"`, `reportUnknownVariableType = "none"`, `reportAttributeAccessIssue = "none"`, etc.)
- Many inline type-ignore comments: `# type: ignore[import-untyped]`, `# type: ignore[attr-defined]`, `# type: ignore[arg-type]` — common pattern for third-party library interop
- `# noqa: F401` used to suppress unused-import warnings

## Import Organization

**Order:**
1. Standard library: `import sys`, `import json`, `from pathlib import Path`, `from __future__ import annotations` (always first)
2. Third-party: `import numpy as np`, `import pandas as pd`, `import cv2`, `from scipy.signal import savgol_filter`
3. Local application: `from pipeline.lift_detection_features import ...`, `from config import ...`, `from utils import ...`

**Key patterns:**
- `from __future__ import annotations` is used at the top of most source files (enables PEP 604-style annotations)
- `TYPE_CHECKING` guard for circular-import-safe type imports (see `barpath_core.py`):
  ```python
  from typing import TYPE_CHECKING, Any
  if TYPE_CHECKING:
      from barpath.pipeline.step2_helpers.kinematics import InsufficientDataError
  ```
- Conditional imports for optional dependencies:
  ```python
  try:
      import cv2
  except ImportError:
      cv2 = None
  ```
- Dynamic imports via `importlib` for pipeline step files (numbered scripts, not valid Python identifiers):
  ```python
  spec = importlib.util.spec_from_file_location(module_name, step_file)
  ```
- Module-level `sys.path.insert(0, ...)` to resolve intra-package imports — used in `barpath_core.py`, `barpath_cli.py`, `barpath_gui.py`, and many test files

## Docstrings and Comments

**Docstrings:**
- Triple-quoted `"""` strings consistently, never `'''`
- Module-level docstrings at the top of every file, describing purpose and contents — e.g., in `config.py`:
  ```python
  """
  Central configuration for the barpath analysis pipeline.
  All magic numbers, thresholds, and tunable parameters are defined here
  so they can be adjusted in one place and easily tested.
  """
  ```
- Google-style function docstrings with `Args:` and `Returns:` sections — e.g., in `analysis_utils.py`:
  ```python
  def safe_savgol_smooth(series: pd.Series, window: int = 11, poly: int = 3) -> pd.Series:
      """
      Apply Savitzky-Golay smoothing with automatic window adjustment.

      Handles NaN and Inf values by interpolation and automatically clamps the window
      to be valid for the data length.

      Args:
          series: Input data series
          window: Desired window size
          poly: Polynomial order

      Returns:
          Smoothed series with same index as input
      """
  ```
- Returning `None` or `Optional` types documented precisely in `Returns:` section
- One-liner docstrings for simple functions:
  ```python
  def check_cancel() -> None:
      if cancel_event and cancel_event.is_set():
          raise InterruptedError("Pipeline cancelled by user")
  ```

**Comments:**
- Section dividers used in larger files — e.g., `# ============================================================================` blocks in `live_lift_recognition.py` separate sections
- `# ---` sub-section dividers (e.g., in `barpath_gui.py`, `test_stabilization.py`)
- Inline comments explaining non-obvious logic, thresholds, or biomechanical rationale
- In `config.py`: comments explain what each magic-number group controls and which pipeline step it belongs to
- Trailing `# type: ignore[...]` comments are common for third-party library type-checking interop

## Error Handling

**Patterns:**
- `try/except` with specific exception types — e.g., `except ImportError:`, `except FileNotFoundError:`, `except InsufficientDataError as e:`
- Bare `except:` rarely used; most use `except Exception:` as a catch-all with logging
- Print-based error reporting via `print()` — no structured logging framework in most modules (only `compiled_analyzer.py` uses `logging`)
- Warnings printed and silently handled: `print(f"Warning: ... {e}")` then return a fallback value
- Generator-based progress: pipeline functions yield `(step_name, progress, message)` tuples; the caller (CLI/GUI) consumes them for progress reporting
- Cancellation via `threading.Event`: `cancel_event.is_set()` checked between pipeline steps to support graceful user cancellation
- Error recovery with fallback values is common — e.g., `return None`, `return "clean"`, `return {}`

**Example error-handling patterns:**
```python
try:
    from some_module import SomeClass
except ImportError:
    class SomeClass(Exception):
        """Fallback when module cannot be imported."""
        pass
```

```python
try:
    result = some_operation()
except SomeSpecificError as e:
    print(f"Warning: {e}")
    return default_fallback
```

## Function Design

**Size:**
- Functions range from simple one-liner helpers (e.g., `check_cancel()`) to large orchestrators (e.g., `run_pipeline()` at ~155 lines, `test_stabilization` test at ~210 lines, `simulate_video_stream()` at ~40 lines)
- Generally well-scoped for their purpose — no extreme god functions beyond the pipeline orchestration functions

**Parameters:**
- Named parameters with defaults — e.g., `def run_pipeline(input_video, model_path, output_video=None, ...):`
- Keyword-argument style is standard; positional parameters used for required inputs
- Complex config passed as simple scalar parameters, not config objects

**Return Values:**
- `Generator[tuple[str, float | None, str], None, None]` for pipeline functions (yield progress tuples)
- `dict[str, Any]` for result bundles
- `Optional[float]`, `Optional[dict]` for functions that may fail (return `None`)
- `pd.DataFrame` returned (and modified in-place with new columns added)

**Type Annotations:**
- Heavy use of `from typing import Any, Dict, List, Optional, Tuple, Sequence, cast` (pre-3.9 style)
- Some files use `dict[str, Any]` (3.9+ syntax) when `from __future__ import annotations` is present
- `numpy.typing.NDArray` used for typed numpy arrays in newer files
- `cast()` used where type narrowing is needed: `cast(NDArray[np.float64], arr)`

## Module Design

**Exports:**
- `__all__` defined only in `barpath/pipeline/__init__.py`
- Most modules expose specific functions by name; consumers import what they need

**Barrel Files:**
- `step1_helpers/__init__.py`, `step2_helpers/__init__.py`, `step4_helpers/__init__.py` — all present but contents not read (likely empty or minimal re-exports)

**Layout pattern:**
- Each `step{N}_helpers/` module is a sub-package grouping related logic for a pipeline step
- Constants are centralized in `config.py` to avoid magic number duplication
- Shared utilities in `analysis_utils.py` and `utils.py`

---

*Convention analysis: 2026-05-01*
