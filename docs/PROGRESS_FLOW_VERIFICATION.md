# Progress Yielding Flow

## Status: VERIFIED

The progress-yielding flow from the pipeline steps through `barpath_core.py` to the CLI and GUI consumers works correctly.

## Flow architecture

```
Step 1 (1_collect_data.py)
    └─> yields: ('step1', 0.0-1.0, 'Collecting data: frame X/Y')
         └─> barpath_core.py
              └─> yield from step_1_collect_data(...)
                   └─> CLI/GUI consumers receive progress

Step 5 (5_render_video.py)
    └─> yields: ('step5', 0.0-1.0, 'Rendering video: frame X/Y')
         └─> barpath_core.py
              └─> yield from step_5_render_video(...)
                   └─> CLI/GUI consumers receive progress
```

## Contract

All pipeline functions are **generators** yielding `(step_name, progress, message)` tuples:

- `step_name`: `'step1'` … `'step5'`, plus `'complete'`.
- `progress`: `float` in [0, 1] for fine-grained steps (1, 5); `None` for steps that report start/end messages only (2, 3, 4).
- `message`: human-readable status text.

`barpath_core.run_pipeline` forwards the yields with `yield from`; batch post-processing and reanalyze mode follow the same pattern.

## Verified properties

- ✅ Each step yields once per progress increment (no duplicates)
- ✅ Progress is monotonically increasing from 0.0 to 1.0
- ✅ `yield from` passes all yields through unchanged
- ✅ Consistent `(step_name, progress|None, message)` format everywhere
- ✅ Generator-based throughout — steps never block the consumer

## Consumers

**CLI** (`barpath_cli.py`): maps each `step_name` to a Rich progress task; updates the bar with `progress * 100`.

**GUI** (`barpath_gui.py`): reads tuples from a thread-safe queue in the background worker, updates the Toga progress bar and appends color-coded HTML log lines.

## Verification test

`tests/test_progress_flow.py` mocks the step generators and asserts:

```
✓ Step 1 yielded 10 progress updates
✓ Step 5 yielded 10 progress updates
✓ Exactly one yield at 100% per step (no duplicates)
✓ Progress is monotonically increasing
```

Run it with:

```bash
python -m pytest tests/test_progress_flow.py -q
```

## Files involved

1. `barpath/pipeline/1_collect_data.py` — yields per processed frame.
2. `barpath/pipeline/5_render_video.py` — yields per rendered frame.
3. `barpath/barpath_core.py` — `yield from` passthrough in `run_pipeline`.
4. `barpath/barpath_cli.py` — Rich progress bars.
5. `barpath/barpath_gui.py` — Toga progress + HTML log.
6. `tests/test_progress_flow.py` — automated verification.