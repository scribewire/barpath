# Testing Patterns

**Analysis Date:** 2026-05-01

## Test Framework

**Runner:**
- **pytest** (declared as dev dependency in `setup.py`: `"pytest>=7.0.0"`)
- No explicit pytest configuration file detected (`pytest.ini`, `pyproject.toml` `[tool.pytest]`, or `conftest.py` with options)
- Test runner uses default pytest discovery

**Assertion Library:**
- Python built-in `assert` statements exclusively — no third-party assertion libraries (e.g., no `unittest.TestCase`, no `assertpy`)
- `numpy` comparisons used for array assertions: `np.sum(old_phases == new_phases) / len(old_phases)`

**Run Commands:**
```bash
pytest                           # Run all tests (no config detected)
pytest -v                        # Verbose output
pytest tests/                    # Run specific test directory
python tests/test_lift_detection.py  # Run as script (many tests support this)
```

## Test File Organization

**Location:**
- All test files live in `tests/` directory at project root — co-located with `barpath/` source package
- No tests co-located with source files (no `tests/` subdirectory within `barpath/`)

**Naming:**
- `test_*.py` prefix for all test files — e.g., `test_lift_detection.py`, `test_phases.py`, `test_clean_jerk_sequence.py`, `test_stabilization.py`
- No `*_test.py` or `*.spec.py` patterns used

**Structure:**
```
tests/
├── conftest.py
├── test_clean_jerk_sequence.py
├── test_cuda.py
├── test_debug.py
├── test_debug2.py
├── test_jerk_debug.py
├── test_jerk_debug2.py
├── test_jerk_phases_detail.py
├── test_jerk_torso.py
├── test_lift_detection.py
├── test_live_classifier.py
├── test_live_preview.py
├── test_live_rework.py
├── test_phases.py
├── test_progress_flow.py
└── test_stabilization.py
```

**Total:** 15 test files

## Test Structure

**Suite Organization:**
Tests follow two distinct patterns:

**Pattern 1 — pytest-style (valid pytest tests):**
Files: `test_lift_detection.py`, `test_cuda.py`, `test_progress_flow.py`
```python
"""Docstring describing the test module."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Helper factory
def _make_synthetic_df(y_values, vel_values=None) -> pd.DataFrame:
    ...

# Test functions with descriptive names
def test_detect_clean_jerk_split_point_returns_none_for_short_trajectory():
    """A trajectory shorter than 80 frames should yield no split."""
    y = np.linspace(0, 1, 50).tolist()
    df = _make_synthetic_df(y)
    result = detect_clean_jerk_split_point(df)
    assert result is None

def test_detect_clean_jerk_split_point_finds_split_for_two_phase_pattern():
    ...
    assert split is not None
    assert 60 < split < 90
```

**Pattern 2 — script-style (runnable with `python test_*.py`):**
Files: `test_stabilization.py`, `test_live_classifier.py`, `test_live_preview.py`, `test_live_rework.py`, `test_clean_jerk_sequence.py`, `test_phases.py`, `test_debug.py`, `test_debug2.py`, `test_jerk_debug.py`, `test_jerk_debug2.py`, `test_jerk_phases_detail.py`, `test_jerk_torso.py`

```python
"""Docstring."""

import sys
sys.path.insert(0, "barpath")

import pandas as pd
from pipeline.live_lift_recognition import LiveLiftRecognizer

def simulate_full_sequence(csv_path, label):
    """Helper function for testing."""
    ...
    return {"final_state": ..., "predicted_class": ...}

if __name__ == "__main__":
    # Top-level script logic runs on import AND when executed directly
    result = simulate_full_sequence(Path("outputs/..."), "clean")
    print(result)
```

**Critical distinction:** The majority of test files (10 of 15) are script-style, not pytest-style. They execute immediately on import and print results to stdout. Only `test_lift_detection.py` (4 test functions), `test_cuda.py`, and `test_progress_flow.py` are structured as proper pytest test modules.

## Setup and Teardown

- **`conftest.py`** — single setup file at `tests/conftest.py`:
  ```python
  """Pytest configuration to add project root to sys.path."""
  import sys
  from pathlib import Path

  project_root = Path(__file__).parent.parent
  sys.path.insert(0, str(project_root))
  ```
- No pytest fixtures, no `setup_module`/`teardown_module`, no `setUp`/`tearDown` patterns
- No factory fixtures — helpers are plain functions (e.g., `_make_synthetic_df()`)

## Mocking

**Framework:**
- **No mocking library** — no `unittest.mock`, no `pytest-mock`, no `monkeypatch`

**Patterns:**
- Manual mock classes defined in test files:
  ```python
  class _MockClassifier:
      """Mock sklearn classifier for testing."""
      def __init__(self, predicted_class: str):
          self._predicted = predicted_class
          self.classes_ = ["clean", "snatch", "jerk", "clean_jerk"]

      def predict(self, X: np.ndarray) -> list[str]:
          return [self._predicted]

      def predict_proba(self, X: np.ndarray) -> list[list[float]]:
          probs = [0.1] * len(self.classes_)
          idx = self.classes_.index(self._predicted)
          probs[idx] = 0.7
          return [probs]

  class _MockScaler:
      """Mock sklearn scaler for testing."""
      def transform(self, X: np.ndarray) -> np.ndarray:
          return X
  ```

**What to Mock:**
- External/ML model dependencies (sklearn classifiers, scalers)
- File-based model loading where test should not depend on real model files

**What NOT to Mock:**
- Inner functions from the same module being tested
- pandas DataFrames (synthetic data generation used instead)

## Fixtures and Factories

**Test Data:**
- Synthetic data generation via helper functions — e.g., `_make_synthetic_df()` in `test_lift_detection.py`:
  ```python
  def _make_synthetic_df(y_values, vel_values=None) -> pd.DataFrame:
      n = len(y_values)
      df = pd.DataFrame({
          "barbell_y_smooth": y_values,
          "frame_height": [1080.0] * n,
          "frame": list(range(n)),
      })
      if vel_values is not None:
          df["vel_y_smooth"] = vel_values
      return df
  ```
- Real CSV data loaded from `outputs/` directory — many script-style tests read existing pipeline output files:
  ```python
  df = pd.read_csv("outputs/male/clean/botev_10_clean/final_analysis.csv")
  ```

**Location:**
- Helpers and synthetic data factories are defined inline in each test file (not shared across files)
- Real test data lives in `outputs/` directory tree (not checked into tests/)

## Coverage

**Requirements:**
- **No coverage enforcement** — no `.coveragerc`, no `--cov` flags in CI, no `pytest-cov` dependency in `setup.py`

**View Coverage:**
```bash
# Would need to install pytest-cov and run:
pip install pytest-cov
pytest --cov=barpath tests/
```
(Not currently configured)

## Test Types

**Unit Tests:**
- `test_lift_detection.py` — tests `detect_clean_jerk_split_point()` and `predict_lift_type()` with synthetic data and mock models
- `test_progress_flow.py` — tests the generator yield pattern using mock pipeline functions

**Integration/Validation Tests (script-style):**
- `test_phases.py` — reads real pipeline CSVs, strips phases, re-runs phase detection, compares results to original
- `test_stabilization.py` — runs full stabilization pipeline on a real video, generates diagnostics and reports
- `test_live_classifier.py`, `test_live_rework.py` — simulate live lift recognition using real CSV data and the actual `LiveLiftRecognizer` class
- `test_clean_jerk_sequence.py` — end-to-end clean+jerk sequence simulation
- `test_live_preview.py` — runs full test harness across multiple sample categories
- `test_cuda.py` — CUDA hardware verification test (calls `torch.cuda.is_available()` etc.)

**Debug Scripts (manual investigation):**
- `test_debug.py`, `test_debug2.py` — step-through debugging of phase detection logic
- `test_jerk_debug.py`, `test_jerk_debug2.py` — detailed jerk phase detection debugging
- `test_jerk_phases_detail.py` — frame-by-frame phase comparison
- `test_jerk_torso.py` — torso length calculation debugging

**E2E Tests:**
- **Not used** — no Selenium, Playwright, or browser-based tests

## Common Patterns

**Synthetic CSV/DataFrame creation for tests:**
```python
# Pattern: build synthetic DataFrame row by row
y = []
y.extend([500] * 30)                    # floor hold
y.extend(np.linspace(500, 100, 21))     # clean pull
y.extend(np.linspace(100, 240, 25))     # recovery
df = _make_synthetic_df(y)
result = detect_clean_jerk_split_point(df)
assert result is not None
```

**`sys.path.insert(0, ...)` pattern (script-style tests):**
```python
import sys
sys.path.insert(0, "barpath")          # or str(Path(__file__).parent.parent)
# Now can import: from pipeline.lift_detection_features import ...
```

**Comparing old vs new output (regression validation):**
```python
old_phases = df["bar_phase"].values.astype(int)
# ... run new detection ...
new_phases = df["bar_phase"].values.astype(int)
match = np.sum(old_phases == new_phases) / len(old_phases)
print(f"Phase match: {match:.1%}")
```

**`if __name__ == "__main__":` guard in script-style tests:**
```python
if __name__ == "__main__":
    data_dir = Path("outputs/male")
    for category in ["snatch", "clean", "jerk"]:
        run_multiple_samples(data_dir, category, max_samples=5)
```

**Hardware verification test pattern:**
```python
def test_pytorch_cuda():
    """Test basic PyTorch CUDA functionality."""
    try:
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            # ... run GPU tensor ops ...
            return True
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
```

## Key Gaps

- **No mocking library** — tests use hand-rolled mock classes, making test setup repetitive and fragile
- **No pytest fixtures** — no reusable test state, no parametrized tests, no shared mocks
- **No coverage tracking** — no way to measure what's tested vs not
- **Heavy reliance on real data files** — script-style tests depend on `outputs/` directory with pre-processed CSVs; tests fail if those files are missing
- **Debug scripts in test suite** — `test_debug*.py` and `test_jerk_debug*.py` are mixed in with actual tests; they exist for manual debugging, not automated CI runs
- **No CI test job** — the only CI workflow (`.github/workflows/ruff.yml`) lints but does not run tests
- **No E2E testing of GUI** — `barpath_gui.py` (~2100 lines) has no test coverage

---

*Testing analysis: 2026-05-01*
