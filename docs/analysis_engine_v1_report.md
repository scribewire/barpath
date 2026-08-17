# Technique Analysis Engine — Current State

This document describes the current state of BARPATH's technique analysis engine: the compiled rule-based analyzer, the lift-type classifier, and the supporting model data shipped in the repo.

> This replaces the earlier "Analysis Engine v1 — Implementation Report", which described a DTW-based *Fast Analysis* and a Random Forest *Smart Analysis*. Those modules are no longer wired into the pipeline; the **CompiledAnalyzer** is the active critique engine. Historical detail is noted where relevant.

## Overview

The engine has two active components:

1. **CompiledAnalyzer** (`barpath/pipeline/step4_helpers/compiled_analyzer.py`) — rule-based fault detection against percentile baselines built from professional lifter data.
2. **Lift-type classifier** (`barpath/pipeline/lift_detection_features.py` + `models/lift_detection/lift_detection_model.pkl`) — a Random Forest that identifies the lift from trajectory shape features when `--lift_type auto` is used.

## 1. CompiledAnalyzer

The active critique engine in Step 4 (`4_critique_lift.py`) and in live recognition (`realtime_processing/live_lift_recognition.py`).

### How it works

1. **Feature extraction** (`step4_helpers/feature_extraction.py`) — ~26 scalar features per lift, camera-angle-invariant by design:
   - Velocity / power scalars (peak, mean, timing of `vel_y_smooth`).
   - Joint angle scalars (elbow, knee) derived from smoothed angles.
   - Body-position scalars (lean, hip rise).
   - Phase-timing scalars (durations as fractions of total).
   - Time-series profile scalars (skewness, peak counts).
   - Lift-specific signals (recovery bounce for clean/snatch; dip/drive/pause for jerk).
2. **Baseline comparison** — features are compared against percentile bands (p10–p90) from `pro_baseline_report.json`. A feature outside the expected band flags a fault; confidence scales with deviation.
3. **Fault detection** — 17+ fault definitions across clean (11), snatch (15, a superset), and jerk. Each fault maps to a phase, a coaching cue, and a severity.
4. **Technique score** — 0–100, with deductions per fault severity × confidence.

### Fault taxonomy (clean)

| Fault ID | Name | Phase |
|----------|------|-------|
| `slow_first_pull` | Slow First Pull | Pull (early) |
| `bar_drift_early` | Bar Drift (Early Pull) | Pull (early) |
| `knee_cave` | Knee Cave (Valgus) | Pull (early) |
| `hitching` | Hitching (Hips Rise Early) | Pull (early) |
| `early_arm_bend` | Early Arm Bend | Pull (late) |
| `incomplete_extension` | Incomplete Extension | Pull (late) |
| `premature_jump` | Premature Jump | Pull (late) |
| `slow_turnover` | Slow Turnover | Pull-under |
| `high_catch` | High Catch Position | Pull-under |
| `forward_chase` | Forward Chase in Recovery | Recovery |
| `unstable_recovery` | Unstable Recovery | Recovery |

Snatch adds `wide_grip_early_bend`, `press_out`, `overhead_instability`, `excessive_forward_lean`. Jerk has its own dip/drive/recovery checks (e.g. `no_dip_pause`, `poor_drive`).

### Baseline loading

`4_critique_lift.py::_load_baseline_for_lifter` resolves:
1. `models/analysis/pro_baseline_report_{lifter}.json` (per-lifter), else
2. `models/analysis/pro_baseline_report.json` (pooled), else
3. critique is skipped with a clear message.

### Known design decisions

- **Camera-angle invariance**: horizontal bar position is unreliable across camera angles, so detection uses vertical kinematics, joint angles, phase timing, and symmetric ratios only.
- **Direction normalization**: athletes may face either way; left/right comparisons use `abs()` / symmetric operations.
- **Elbow-angle convention**: MediaPipe reports near-straight arms at low angles (~0–10°) and bent arms at high angles (~170–180°), so early-arm-bend checks are `val > p90` (not `< p10`).
- **Dip pause (jerk)**: a dip pause is normal pro technique (stretch-shortening cycle); only its *absence* is flagged (`no_dip_pause`).
- **Recovery bounce**: pro data shows near-zero bounce counts; a detected bounce is capped at 40% confidence to avoid oscillation false positives.

## 2. Lift-Type Classifier

`barpath/models/lift_detection/lift_detection_model.pkl` is a Random Forest (37 trajectory-shape features; classes `clean`, `clean_jerk`, `jerk`, `snatch`) used by `run_pipeline` auto-detection (`--lift_type auto`). Config/metadata: `models/lift_detection/lift_detection_config.json`, `lift_detection_report.json`.

## 3. Model data in the repo

| Path | Purpose | Consumed by |
|------|---------|-------------|
| `models/analysis/pro_baseline_report.json` | pooled pro-lifter percentiles | CompiledAnalyzer |
| `models/analysis/{lifter}/` (trajectory.npy, config.json) | per-lifter reference data | drives GUI lifter dropdown; per-lifter baseline JSON not yet generated |
| `models/lift_detection/lift_detection_model.pkl` | lift-type classifier | `run_pipeline` auto-detect |
| `models/lift_detection/live_lift_model.pkl` | live webcam classification | `LiveLiftRecognizer` |

## Historical notes

- **DTW Fast Analysis**: a weighted multi-channel DTW comparison against per-lifter `trajectory.npy` files was designed and its reference data generated, but the consumer was never wired in. The `DTW_SIMILARITY_K` constant and the `dtw-python` dependency have been removed. The trajectory files remain as reference data for a future re-implementation.
- **Random Forest Smart Analysis** (`step4_helpers/smart_analysis.py`): now integrated into `4_critique_lift.py` as an optional ML tier — `_analyze_segment` tries to load `models/analysis/{lifter}/{lift_type}/smart_analysis_model.pkl` and uses the model's fault probabilities when present, falling back to the CompiledAnalyzer. No trained fault model ships yet, so the CompiledAnalyzer is the default path.
- **Heuristic classifier** (`heuristic_classifier.py`): its config constants (`HEURISTIC_CONFIDENCE_*`, `LIFT_CLASS_*`) are now defined in `config.py`, so the module imports cleanly. It remains a fallback utility for live lift classification, not part of the main pipeline.

## What still needs work

- Train a `smart_analysis_model.pkl` (Random Forest fault detector) so the Smart Analysis tier activates.
- Generate per-lifter baseline JSONs (`pro_baseline_report_{lifter}.json`) so lifter selection uses real per-lifter data instead of always pooling.
- Female baselines are not trained (the classifier and baselines were built from male data).
- The GUI lifter dropdown is driven by directory names; it should reflect actual baseline availability.