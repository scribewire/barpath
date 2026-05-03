# Analysis Engine v1 — Implementation Report

## Overview

This report documents the planning, design, and implementation of the first version of BARPATH's technique analysis engine. The engine replaces a hardcoded rule-based system with a data-driven approach that uses statistical baselines from professional athlete data to detect technique faults in Olympic weightlifting.

The system has three components: a **compiled rule-based analyzer** that detects faults using biomechanical thresholds, a **lift detection classifier** that identifies lift types, and a **DTW trajectory comparison system** that compares user bar paths against professional references.

---

## Part 1: Planning Phase

### Source Materials

Four planning documents were provided:

1. **`technique training guide.md`** — Described a two-stage approach: training scripts compile pro athlete statistics into JSON reports, then an LLM reads those reports and generates Python detection rules. This eliminates the need for labeled error samples, heavy ML models, or runtime LLM inference.

2. **`analysis planning doc.md`** — The master architecture document. Defined the 5-step pipeline (collect → analyze → graphs → critique → video), specified that Steps 4 and 5 would be swapped so analysis runs before video rendering. Detailed the fault taxonomy (10 clean faults, 12 snatch faults, 7 jerk faults), the 26 Smart Analysis features, and the DTW-based Fast Analysis approach.

3. **`smart_analysis_training.py`** — A 2,158-line script that loads pro athlete data, extracts features, computes statistical baselines (percentiles, correlations, phase statistics), and generates LLM-ready reports. Served as the reference implementation for feature extraction.

4. **`lift_detection_training.py`** — A 1,773-line script that extracts trajectory shape features, trains a Random Forest classifier for lift type detection, and generates biomechanical signature reports.

### Key Design Decisions

**Camera angle invariance**: Camera angle compensation had been removed from the pipeline, making horizontal position features (`barbell_x_smooth`) unreliable. All fault detection features were specified to use only vertical kinematics, joint angles, phase timing, and symmetric ratios.

**Direction normalization**: Athletes may face left or right in the video frame. All left/right joint comparisons were required to use `abs()` or symmetric operations.

**Bar oscillation handling**: At the end of a clean or beginning of a jerk, the bar oscillates sinusoidally (2-4 Hz, small amplitude). This normal biomechanical phenomenon must not be confused with recovery bounces. A velocity threshold of ~20% of peak recovery velocity filters oscillation while catching real bounces.

**Phase validation**: Any CSV missing any of the 3 phases (0, 1, 2) must be discarded. Incomplete data produces unreliable features.

---

## Part 2: Existing Code Assessment

Before writing new code, the existing pipeline files were examined:

**`step4_helpers/feature_extraction.py`**: Used 2-channel `(x_norm, y_norm)` trajectories with `barbell_x_smooth` (unreliable). Had only ~15 features instead of the specified ~26. Missing lift-specific features (recovery bounce, jerk dip/drive). No phase validation.

**`step4_helpers/fast_analysis.py`**: Used 2-channel trajectories with simple Euclidean distance. Had a broken heuristic for matching the best pro trajectory index.

**`step4_helpers/smart_analysis.py`**: Generic but untested against real data.

**`4_critique_lift.py`**: Good structure but lacked jerk support, had no fallback analysis when ML models weren't available.

**Training scripts**: Were reference implementations for the planning docs, not runnable against the actual data directory structure.

**Data**: Located at `outputs/male/{lift_type}/{lifter}_{num}_{lift}/final_analysis.csv` with 111 clean lifts, 118 snatch lifts, and 69 jerk lifts across 12-13 unique lifters. Not in the `data/pro/` structure the planning docs assumed.

---

## Part 3: Implementation

### Phase 1: Pipeline Helpers (Feature Extraction + DTW)

The `step4_helpers/feature_extraction.py` was rewritten from scratch:

- `extract_trajectory()` produces 3-channel `(N, 3)` arrays: `[y_position_normalized, velocity_normalized, acceleration_normalized]`. This captures the full lift profile (shape + speed + power) while being independent of camera angle.
- `extract_smart_features()` computes all ~26 scalar features organized into 6 categories: velocity/power scalars (6), joint angle scalars (6), body position scalars (4), phase timing scalars (4), time-series profile scalars (2), and lift-specific scalars (recovery bounce for clean/snatch, dip/drive/pause for jerk).
- `validate_phases()` rejects CSVs that don't contain all 3 phases.

The `step4_helpers/fast_analysis.py` was rewritten to use weighted multi-channel DTW with channel weights `[0.5, 0.3, 0.2]` (position most important, then velocity, then acceleration). The pro-trajectory matching logic was fixed to correctly identify the best-aligned trajectory.

### Phase 2: Compiled Analyzer (The Critique Engine)

A new `compiled_analyzer.py` was created as the core fault detection engine. It contains:

- **`FAULT_DEFS`**: A dictionary of 17 fault definitions, each with name, phase, description, coaching cue, applicable lift types, and severity level. Covers all faults from the planning taxonomy: 10 for clean, 12 for snatch, 7 for jerk.
- **`CompiledAnalyzer` class**: Takes a lift type, gender, and optional baselines dict. The `analyze()` method extracts features and runs type-specific fault checks. The `get_technique_score()` method computes a 0-100 score with deductions based on fault severity and confidence.
- **`load_baselines_from_json()`**: Loads real percentile data from `pro_baseline_report.json`. No hardcoded fallback — if the JSON is missing, the analyzer logs a warning and produces no detections.

The initial version had hardcoded `DEFAULT_BASELINES` with fabricated values. Testing revealed these were off by orders of magnitude from real data (e.g., `max_vel_y p10=5.0` vs actual `p10=236.2`). The defaults were removed entirely.

### Phase 3: Threshold Bug Fixes

Testing against pro athlete lifts revealed 5 threshold direction bugs:

**Bug 1 — `slow_first_pull`**: The check `if val < p10` of `mean_vel_y_first_half` was inverted. In image coordinates, upward bar motion produces negative velocity values. A slow pull has velocity closer to 0 (less negative), but p10 was the most negative value (fastest pull). The comparison was catching fast pulls instead of slow ones. Fixed by using `abs(max_vel_y) < p25` instead.

**Bug 2 — `early_arm_bend`**: MediaPipe elbow angles use a convention where low values (~0-10°) represent straight arms and high values (~170-180°) represent bent arms. The check `if val < p10` (p10=0.34°) was flagging when arms were nearly perfectly straight — the opposite of arm bend. Fixed by flipping to `val > p90` (high angle = bent arm).

**Bug 3 — `hitching`**: The `hip_rise_vs_bar_rise_early` feature had extreme variance (std=30,079, max=318,336) due to near-zero bar displacement in early frames causing division blowups. Replaced with a combination of `vel_profile_skewness > p90` (right-tailed velocity = stall pattern) and `accel_peaks_count > p75` (multiple power interruptions).

**Bug 4 — `dip_pause` (jerk)**: The original logic flagged when `dip_pause_detected == 1`. But 90% of professional jerkers have a dip pause — it's normal technique that loads the stretch-shortening cycle. Flipped to flag when `dip_pause_detected == 0` (absence of dip pause = poor rhythm). Added a new `no_dip_pause` fault definition.

**Bug 5 — `recovery_bounce`**: All pro data showed 0.0 bounces across all percentiles, suggesting the feature extractor's velocity threshold is calibrated conservatively. Any detected bounce is meaningful but potentially a false positive from bar oscillation. Confidence was capped at 40% for single-bounce detections.

### Phase 4: Pipeline Integration

The `4_critique_lift.py` was rewritten to:

- Support `jerk` as a lift type (previously only clean/snatch)
- Accept a `--gender` argument for baseline selection
- Automatically fall back to the `CompiledAnalyzer` when no ML model exists
- Write comprehensive `analysis.md` reports with fault details, coaching cues, technique scores, and a "No Issues Detected" checklist

The fallback chain works as: ML model (Random Forest) → compiled rule-based analyzer → skip with message.

### Phase 5: Training Scripts

Both training scripts were rewritten to work against the actual data directory structure (`outputs/male/{lift_type}/...`) instead of the assumed `data/pro/{gender}/...`:

**`smart_analysis_training.py`**: Loads 298 lifts across 3 lift types, computes per-feature statistics (mean, std, p10-p90), and generates `pro_baseline_report.json` with real percentile data. Also generates `analyzer_template.py` (skeleton for LLM) and `llm_prompt.txt` (instructions for LLM).

**`lift_detection_training.py`**: Extracts 37 features per lift, trains a Random Forest classifier (300 trees, balanced classes), achieves 98% accuracy across clean/snatch/jerk. Supports `train`, `predict`, and `report` subcommands.

### Phase 6: DTW Trajectory Generation

A new `generate_dtw_trajectories.py` was created to produce per-lifter reference trajectories using DTW Barycenter Averaging (DBA):

1. Scans all lift directories, groups by lifter name (parsed from `{lifter}_{num}_{lift}` folder names)
2. For each lifter+lift_type, collects all individual 3-channel trajectories
3. Computes DBA: iteratively aligns each trajectory to a centroid via DTW and averages aligned points. Repeats 10 iterations.
4. Computes a generic group (DBA of all lifter centroids)
5. Calibrates the DTW distance scale as the median of pairwise distances between lifter centroids
6. Saves per-lifter `trajectory.npy` and `config.json` files

A scanning bug was discovered during implementation: the `_scan_lift_dirs` function used a three-branch logic (match, name-match-recurse, else-recurse) that accidentally treated the `elif` branch as "recurse into this directory" rather than "this directory is the target." The recursive call would search inside `botev_10_clean/` for more `_clean` directories, find none, and return 0. This was fixed by simplifying to a two-branch approach: match-and-add, or recurse.

Output: 12 lifters × 3 lift types = 36 DBA average trajectories, plus generic groups.

---

## Part 4: Testing and Verification

### False Positive Testing

The compiled analyzer was tested on 3 professional lifts:

| Lift | Score | Faults | Assessment |
|------|-------|--------|------------|
| Botev clean | 100/100 | 0 | Excellent technique |
| Botev snatch | 97/100 | 1 (slow_turnover, 41% conf) | Excellent technique |
| Botev jerk | 96/100 | 2 (press_out 31%, poor_drive 17%) | Excellent technique |
| Ilyin clean | 98/100 | 1 (slow_first_pull, 31% conf) | Excellent technique |

All pro lifts scored 96+, with faults appearing only at low confidence (17-41%), correctly indicating marginal deviations rather than significant issues. No excessive false positives.

### Lift Detection Verification

The classifier was tested on 3 held-out samples:

| Actual | Predicted | Confidence |
|--------|-----------|------------|
| clean | clean | 100.0% |
| snatch | snatch | 100.0% |
| jerk | jerk | 98.5% |

All predictions correct with no false positives.

### Code Quality

All 8 modified/new files pass `ruff check` and `ruff format` with zero errors.

---

## Part 5: What Still Needs Work

**LLM integration**: The `compiled_analyzer.py` currently uses detection rules written by hand during this implementation. The intended workflow is to feed `pro_baseline_report.json` + `llm_prompt.txt` to GPT-4 or Claude, which generates more nuanced detection rules from the actual percentile data. The hand-written rules are functional but less sophisticated than what an LLM could produce with the full statistical context.

**DTW integration in step 5**: The per-lifter trajectory files exist but are not yet consumed by `5_render_video.py` for the temporal similarity heatmap overlay. This was intentionally deferred per the user's request.

**Female baselines**: Only male data was trained. Female baselines can be generated by running the training scripts with `--female`.

**Feature extraction edge cases**: The `hip_rise_vs_bar_rise_early` feature has extreme outliers (std=30,079) that make it unreliable for hitching detection. The `recovery_bounce_count` feature shows 0.0 across all pro percentiles, suggesting the velocity threshold in the feature extractor may need tuning to better distinguish bar oscillation from genuine squat bounces.

**Smart Analysis ML model**: No trained Random Forest fault detection model exists yet. This would require labeled error samples (lifts with known faults), which the compiled approach was specifically designed to avoid. The rule-based analyzer serves as the working substitute.

---

## File Inventory

### Pipeline files (8 files)

| File | Status | Lines |
|------|--------|-------|
| `barpath/pipeline/step4_helpers/feature_extraction.py` | Rewritten | 374 |
| `barpath/pipeline/step4_helpers/fast_analysis.py` | Rewritten | 228 |
| `barpath/pipeline/step4_helpers/compiled_analyzer.py` | New | 470 |
| `barpath/pipeline/step4_helpers/__init__.py` | Updated | 21 |
| `barpath/pipeline/step4_helpers/smart_analysis.py` | Unchanged | 167 |
| `barpath/pipeline/4_critique_lift.py` | Rewritten | 422 |

### Training/generation scripts (3 files)

| File | Status | Lines |
|------|--------|-------|
| `outputs/smart_analysis_training.py` | Rewritten | 538 |
| `outputs/lift_detection_training.py` | Rewritten | 407 |
| `outputs/generate_dtw_trajectories.py` | New | 453 |

### Generated model files

| File | Size | Description |
|------|------|-------------|
| `barpath/models/analysis/pro_baseline_report.json` | ~50KB | Statistical baselines for 298 lifts |
| `barpath/models/analysis/analyzer_template.py` | ~3KB | LLM template |
| `barpath/models/analysis/llm_prompt.txt` | ~2KB | LLM instructions |
| `barpath/models/analysis/{lifter}/{type}/trajectory.npy` | ~2KB each | 36 DBA trajectories |
| `barpath/models/analysis/{lifter}/{type}/config.json` | ~200B each | 36 calibration configs |
| `barpath/models/analysis/generic/{type}/fast_analysis_trajectories.npy` | ~5KB each | 3 generic trajectory sets |
| `barpath/models/analysis/generic/{type}/fast_analysis_config.json` | ~200B each | 3 generic configs |
| `barpath/models/lift_detection/lift_detection_model.pkl` | ~500KB | Trained RF classifier |
| `barpath/models/lift_detection/lift_detection_config.json` | ~1KB | Feature/class metadata |
| `barpath/models/lift_detection/lift_detection_report.json` | ~5KB | Biomechanical signatures |
