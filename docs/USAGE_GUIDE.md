# BARPATH Usage Guide

A comprehensive guide to using barpath for AI-powered weightlifting technique analysis.

## Table of Contents

1. [Quick Start](#quick-start)
   - [Using the GUI](#using-the-gui-recommended)
   - [Using the Command Line](#using-the-command-line)
   - [Re-Running Analysis on an Existing Output Folder](#re-running-analysis-on-an-existing-output-folder)
2. [How It Works](#how-it-works)
3. [Technique Analysis](#technique-analysis)
4. [Batch Processing, Reanalysis, and Hardware Acceleration](#batch-processing-reanalysis-and-hardware-acceleration)
5. [Model Formats and Models Included](#model-formats-and-models-included)
6. [Output Files](#output-files)
7. [Recording Best Practices](#recording-best-practices)
8. [Tips for Best Results](#tips-for-best-results)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Using the GUI (Recommended)

The easiest way to analyze a video:

```bash
python barpath/barpath_gui.py
```

The GUI features a tabbed interface with four main sections:

#### **📂 Files Tab**
- Add one or more video files for analysis using **Add Videos**.
- Re-run analysis on existing output folders using **Add Folders (Reanalyze)** — skips the slow video decoding step and re-runs steps 2–5 from the saved `raw_data.pkl`.
- **Mutually exclusive modes**: once you add videos, the "Add Folders" button is disabled, and vice versa — clear the list to switch modes.
- Clear the entire input list with **Clear**.
- Select the output directory for results.
- View and individually remove queued items before running.
- Supports MP4, AVI, MOV, MKV, and WebM formats.

#### **⚙️ Settings Tab**
- Automatically discovers YOLO models from `barpath/models` (`.pt`, `.onnx`, `.engine`, OpenVINO directories). The shipped model is `std_nano.pt`.
- **Lift type**:
  - **auto** (default): detects the lift type automatically after Step 2 using the trained lift classifier.
  - **clean**: Pull / Pull-under / Recovery phases, power calculation, technique critique.
  - **snatch**: same 3-phase analysis with snatch-specific fault checks.
  - **jerk**: Dip / Drive / Recovery phases and jerk-specific technique critique.
  - **clean_jerk**: splits into two segments, 6-phase detection, unified report.
  - **none**: kinematics only, no lift-specific analysis or critique.
- **Lifter baseline**: choose which professional lifter to compare against for Technique Analysis. Options are populated from `barpath/models/analysis/`; the default is `generic` (pooled pro data). Falls back to the pooled baseline if a per-lifter baseline is not found.
- **Analysis options**: toggle **Technique Analysis** on/off.
- Settings are **disabled automatically** in Reanalyze (folders) mode — the model and video are not re-run.

#### **▶️ Analyze Tab**
- Press **Analyze** to start — the pipeline runs in a **background worker thread**; the GUI never hangs.
- Cancel at any time.
- Monitor progress with a real-time progress bar and a color-coded, HTML-rendered log.
- In Reanalyze mode the log shows steps 2–5 only.
- For batch runs (2+ videos or folders), a **batch post-processing** phase runs automatically, generating the superimposed comparison graphs.

#### **📊 Analysis Tab**
- Displays the generated `analysis.md` report as formatted HTML after each run.
- Includes phase timing, maximum specific power, technique score, and detected faults with coaching cues.

#### **🎥 Live Preview (Alpha)**
- Toggle the webcam preview in the Analyze tab to run real-time YOLO + MediaPipe detection on the camera feed.
- The preview uses `LiveLiftRecognizer` to detect and classify lifts live, draw the bar path, and run technique analysis on the recorded lift.

### Using the Command Line

```bash
python barpath/barpath_cli.py \
  --input_video "lift.mp4" \
  --model "barpath/models/std_nano.pt" \
  --lift_type clean
```

### Command Line Options

```
Required Arguments:
  --input_video PATH         Path to video file(s) to analyze.
                             Accepts multiple files for batch processing:
                               vid1.mp4 vid2.mp4 vid3.mp4
                             Formats: .mp4, .avi, .mov, .mkv, .webm

  --model PATH               Path to the YOLO model or OpenVINO export directory.
                             Examples:
                               barpath/models/std_nano.pt             (PyTorch)
                               models/best.onnx                       (ONNX)
                               models/best.engine                     (TensorRT)
                               barpath/models/std_nano_openvino_model (OpenVINO dir)

Optional Arguments:
  --lift_type {auto,clean,snatch,jerk,clean_jerk,none}
                             Type of lift to analyze (default: auto).
                               auto        - Automatically detects lift type after Step 2
                               clean       - Pull/Pull-under/Recovery phases, power calc, critique
                               snatch      - Same 3-phase analysis with snatch-specific faults
                               jerk        - Dip/Drive/Recovery phases and technique critique
                               clean_jerk  - 6-phase detection, unified report
                               none        - Kinematics only, no lift-specific analysis

  --lifter NAME              Pro lifter baseline for Technique Analysis (default: generic).
                             Examples: 'liao', 'botev', 'ilyin', 'generic'.
                             Falls back to the pooled baseline if per-lifter data is missing.

  --output_video PATH        Path to save the annotated video.
                             (Default: outputs/output.mp4)

  --no-video                 Skip video rendering (Step 5) — much faster.
                             Useful when you only need CSV data and graphs.

  --output_dir PATH          Directory for all outputs (default: outputs).
                             Batch processing creates one subdirectory per video.

  --force                    Force reprocessing even if output folders exist (skips prompt).
  --skip-existing            Automatically skip videos that are already processed.

  HUD toggles:
  --no-skeleton              Hide the skeleton overlay on the rendered video
  --no-sparkline             Hide the velocity sparkline HUD element
  --no-power-zones           Hide the power zone band HUD element
  --no-error-markers         Hide the fault error markers on the bar path

  -h, --help                 Show detailed help with examples.
```

### Quick Examples

**Basic clean analysis (no video output):**
```bash
python barpath/barpath_cli.py \
  --input_video "lift.mp4" \
  --model "barpath/models/std_nano.pt" \
  --lift_type clean \
  --no-video
```

**Snatch analysis with full video rendering:**
```bash
python barpath/barpath_cli.py \
  --input_video "lift.mp4" \
  --model "barpath/models/std_nano.pt" \
  --lift_type snatch
```

**Auto-detect lift type:**
```bash
python barpath/barpath_cli.py \
  --input_video "lift.mp4" \
  --model "barpath/models/std_nano.pt" \
  --lift_type auto
```

**With OpenVINO acceleration (Intel CPU):**
```bash
python barpath/barpath_cli.py \
  --input_video "lift.mp4" \
  --model "barpath/models/std_nano_openvino_model" \
  --lift_type clean
```

**Batch processing with a custom output directory:**
```bash
python barpath/barpath_cli.py \
  --input_video lift1.mp4 lift2.mp4 lift3.mp4 \
  --model "barpath/models/std_nano.pt" \
  --lift_type clean \
  --output_dir "batch_results"
```

### Running Individual Pipeline Steps

For debugging or custom workflows, each pipeline step can be invoked directly:

```bash
# Step 1: Collect raw tracking data (YOLO + MediaPipe + producer-consumer)
python barpath/pipeline/1_collect_data.py \
  --input video.mp4 \
  --model barpath/models/std_nano.pt \
  --output raw_data.pkl \
  --lift_type clean

# Step 2: Analyze kinematics, smooth joints, detect phases, generate CSV
python barpath/pipeline/2_analyze_data.py \
  --input raw_data.pkl \
  --output final_analysis.csv

# Step 3: Generate kinematic graphs
python barpath/pipeline/3_generate_graphs.py \
  --input final_analysis.csv \
  --output_dir graphs

# Step 4: Technique analysis
python barpath/pipeline/4_critique_lift.py \
  --input final_analysis.csv \
  --lift_type clean \
  --lifter generic \
  --output_dir .

# Step 5: Render annotated video with skeleton, bar path, and HUD
python barpath/pipeline/5_render_video.py \
  --input_csv final_analysis.csv \
  --input_video video.mp4 \
  --output_video final.mp4
```

### Re-Running Analysis on an Existing Output Folder

If you already ran step 1 and want to re-run steps 2–5 (e.g. after a code update or to change the lift type), use `run_pipeline_from_folder` from `barpath_core.py`:

```python
from barpath.barpath_core import run_pipeline_from_folder

for step, progress, msg in run_pipeline_from_folder(
    output_folder="outputs/my_lift",
    lift_type="clean",
    encode_video=True,  # set False to skip video re-render
):
    print(step, msg)
```

The folder must contain a `raw_data.pkl` produced by step 1. The source video path is read automatically from the pickle's metadata — if the file no longer exists, video rendering is skipped with a warning rather than raising an error.

In the **GUI**, this is exposed as the **Add Folders (Reanalyze)** button in the Files tab.

---

## How It Works

### Producer-Consumer Video Pipeline

barpath uses a **producer-consumer architecture** to maximize throughput:

- **Producer (I/O thread)**: decodes video frames in a background thread into a bounded queue (size 8).
- **Consumer (main thread)**: pulls frames and runs YOLO + MediaPipe inference.
- **Benefit**: the main thread never stalls waiting for disk I/O.

This is especially effective for slow storage, high-resolution video, and CPU-intensive inference.

### YOLO26 NMS-Free Barbell Detection

barpath uses **Ultralytics YOLO26** for barbell endcap detection:

- **NMS-free architecture**: the model outputs final detections directly; no separate Non-Maximum Suppression step.
- **Inference**: uses only `conf` (confidence threshold); no `iou` parameter.
- **Shipped model**: `barpath/models/std_nano.pt` (YOLO26n) — small, fast, accurate.

**Supported model formats:**

| Format | Path | Notes |
|--------|------|-------|
| PyTorch | `*.pt` | default; GPU via CUDA when available |
| ONNX | `*.onnx` | requires `onnxruntime` |
| TensorRT | `*.engine` | NVIDIA GPU; requires TensorRT |
| OpenVINO | directory with `.xml` + `.bin` | Intel CPU; shipped as `std_nano_openvino_model/` |

**Adjusting detection sensitivity:** lower `conf` (e.g. `0.3`) detects more barbells but may add false positives; higher (e.g. `0.7`) is stricter. Default is `0.5`.

### MediaPipe Pose Estimation

- **Model complexity**: 1 (Medium) by default. 0 = Light/fast, 2 = Heavy/most accurate.
- **Detection / tracking confidence**: `0.5` each.
- **Segmentation**: enabled (used for background feature detection in stabilization).
- **Tracked joints**: 12 — shoulders, hips, knees, ankles, elbows, wrists.
- Outputs both normalized (0–1) and world (meters) coordinates.

The heavy pose model (`pose_landmarker_heavy.task`, 30 MB) ships in `barpath/models/` and is auto-downloaded if missing.

### Camera Shake Stabilization

1. Detect background features using MediaPipe's person-segmentation mask.
2. Track features across frames with Lucas-Kanade optical flow.
3. Estimate motion (translation + rotation) from feature correspondences.
4. Apply correction to the barbell position.

The result is a smooth, stabilized bar path even when the camera moves slightly.

### Joint Smoothing and Kinematics

1. Extract raw MediaPipe joint positions per frame.
2. Apply a **Savitzky-Golay filter** (11-frame window, cubic polynomial) to all joint coordinates and angles.
3. Compute bar velocity and acceleration; smooth derivatives with a wider 15-frame window.
4. Compute **specific power** = (bar velocity)² × (bar mass) / (lifter mass); in W/kg when pixel-to-meter scaling is available, otherwise in px²/s³.

`final_analysis.csv` contains only smoothed, analysis-ready data.

### Phase Detection

**Classic lifts (clean / snatch)** — Pull → Pull-under → Recovery:

| Phase | Label | Color | Definition |
|-------|-------|-------|------------|
| Pull | 0 | 🔴 Red | barbell moves upward; hips reach full extension |
| Pull-under | 1 | 🟠 Orange | hips drop from extension; lifter catches the bar |
| Recovery | 2 | 🟢 Green | lifter rises from the catch to standing; bar reaches peak height |

Detection uses MediaPipe landmarks plus kinematic heuristics (hip velocity, hip Y position, barbell signals) to find the key timepoints.

**Jerk** — Dip → Drive → Recovery (3 phases).

**Clean & jerk** — auto-split into two segments with 6-phase detection and a unified report.

**Lift type `none`** — no phase detection; kinematics only.

Phase timing is reported in `analysis.md` with frame ranges and durations:

```
Phase Timing:
  Pull (t0→t2):        Frames 0–45 (1500 ms)
  Pull-under (t2→t3):  Frames 45–52 (233 ms)
  Recovery (t3→t4):    Frames 52–80 (933 ms)
```

---

## Technique Analysis

barpath's critique engine (`step4_helpers/compiled_analyzer.py`) is a **compiled rule-based analyzer**: it extracts ~26 biomechanical features from `final_analysis.csv`, compares them against percentile baselines built from professional lifter data (`pro_baseline_report.json`), and reports deviations as faults with confidence scores.

**How it works:**
1. Extract features (velocity/power scalars, joint angle scalars, body position, phase timing, time-series profiles, lift-specific signals).
2. Compare against pro percentiles (p10–p90) per lifter or pooled (`generic`).
3. Flag faults where the feature falls outside the expected band; assign confidence from the deviation magnitude.
4. Generate a 0–100 **technique score** (deductions per fault severity and confidence).
5. Write coaching-style cues into `analysis.md`.

**Fault taxonomy:**

**Clean (11 faults):**

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

**Snatch (15 faults):** all clean faults plus `wide_grip_early_bend`, `press_out`, `overhead_instability`, `excessive_forward_lean`.

**Jerk:** dedicated checks for the dip/drive/recovery rhythm (e.g. `no_dip_pause`, `poor_drive`, `press_out`).

**Model files:**
- `barpath/models/analysis/pro_baseline_report.json` — pooled pro-lifter percentiles (read at runtime).
- `barpath/models/analysis/{lifter}/` — per-lifter baseline directories (used for lifter selection; fall back to pooled).

### Lifter Selection

```bash
# Compare against the pooled pro baseline
python barpath/barpath_cli.py --input_video lift.mp4 --model barpath/models/std_nano.pt --lift_type clean --lifter generic

# Compare against a specific lifter's technique
python barpath/barpath_cli.py --input_video lift.mp4 --model barpath/models/std_nano.pt --lift_type clean --lifter botev
```

**Baseline resolution:**
1. `models/analysis/pro_baseline_report_{lifter}.json` if present.
2. Otherwise the pooled `models/analysis/pro_baseline_report.json`.
3. If neither exists, critique is skipped with a clear message (analysis still runs).

### Training / Customization (internal)

Baseline generation and lift-classifier training live in `outputs/` (gitignored) and are documented for maintainers in `docs/analysis_engine_v1_report.md`. They operate on a dataset of `final_analysis.csv` files and are not required to run the pipeline.

---

## Batch Processing, Reanalysis, and Hardware Acceleration

### Batch Processing

```bash
python barpath/barpath_cli.py \
  --input_video video1.mp4 video2.mp4 video3.mp4 \
  --model "barpath/models/std_nano.pt" \
  --lift_type clean \
  --output_dir "batch_results"
```

**Behavior:**
- Each video is processed sequentially into its own subdirectory:
  ```
  batch_results/
  ├── lift_1/
  │   ├── raw_data.pkl
  │   ├── final_analysis.csv
  │   ├── graphs/
  │   ├── output.mp4
  │   └── analysis.md
  ├── lift_2/
  │   └── ...
  └── superimposed_bar_paths.png   ← batch post-processing
  ```
- After all videos complete, **batch post-processing** generates a superimposed comparison graph (see below).

### Reanalyze Existing Output Folders

To re-run steps 2–5 on an already-processed folder (after a code update, to change `lift_type`, or to regenerate graphs):

**GUI:** Files tab → **Add Folders (Reanalyze)** → select folders containing `raw_data.pkl` → **Analyze**. Settings are auto-disabled; video is re-rendered only if the original source path still exists.

**Notes:**
- The source video path is stored inside `raw_data.pkl` when step 1 runs.
- Reanalyze mode and Videos mode are mutually exclusive — clear the list to switch.

### Superimposed Comparison Graphs

After processing 2+ lifts, barpath generates `superimposed_bar_paths.png` in the top-level output directory:

1. Each lift's path is **origin-normalized** at its pull-under start point, so all paths share a common origin at (0,0).
2. Non-reference lifts are **uniformly scaled** — a single least-squares scale factor aligns matching phase-transition markers (arc-length ratio as fallback). X and Y are scaled together, preserving path shape.
3. Paths are overlaid with phase-based color coding.

### Hardware Acceleration

| Backend | Install | Benefit | Shipped model |
|---------|---------|---------|---------------|
| ONNX Runtime | `pip install onnxruntime` | ~1–2× faster CPU inference | any exported `.onnx` |
| OpenVINO | `pip install openvino` | ~2–4× faster on Intel CPU | `std_nano_openvino_model/` |
| TensorRT | `pip install tensorrt` | 3–5× faster on NVIDIA GPU | export your own `.engine` |

The interactive installer detects your hardware and prints the right command:

```bash
python barpath/briefcase_hardware_installer.py
```

---

## Model Formats and Models Included

| Model | Path | Purpose |
|-------|------|---------|
| YOLO26n barbell detector | `barpath/models/std_nano.pt` (LFS) | barbell endcap detection |
| YOLO26n OpenVINO export | `barpath/models/std_nano_openvino_model/` | Intel CPU inference |
| MediaPipe pose (heavy) | `barpath/models/pose_landmarker_heavy.task` | full-body joint tracking |
| Lift-type classifier | `barpath/models/lift_detection/lift_detection_model.pkl` | auto-detects clean/snatch/jerk/clean_jerk |
| Live lift classifier | `barpath/models/lift_detection/live_lift_model.pkl` | webcam live recognition |
| Pro baselines | `barpath/models/analysis/pro_baseline_report.json` + per-lifter dirs | technique critique |

**Exporting your own OpenVINO model:**
```bash
pip install openvino
yolo export model=barpath/models/std_nano.pt format=openvino
```
This creates a directory with `.xml` and `.bin` files that barpath loads directly.

---

## Output Files

### Summary

```
outputs/
├── raw_data.pkl              # Step 1: raw detections + source video path
├── final_analysis.csv        # Step 2: smoothed kinematic data
├── graphs/
│   ├── barbell_xy_stable_path.png
│   ├── barbell_xy_stable_path_unsmoothed.png
│   ├── barbell_velocity.png
│   ├── barbell_acceleration.png
│   └── barbell_specific_power.png
├── output.mp4                # Step 5: annotated video (unless --no-video)
└── analysis.md               # Step 4: technique analysis report
```

### Analysis Report (`analysis.md`)

1. **Technique Critique** — flagged faults with confidence, phase, and coaching-style description; an "All Checks" list showing every fault's probability.
2. **Kinematic Summary** — total lift time, peak velocity, peak acceleration.
3. **Footer** — BARPATH version, baseline lifter.

Example:

```markdown
# Analysis Report: Clean

## Technique Critique

**Detected Issues:**

- **Early Arm Bend** (92% confidence)
  - Phase: Pull (late)
  - Elbow angle decreases (arms bend) before the end of the pull phase.

**All Checks:**

- ✅ Slow First Pull: 18%
- ⚠️ Early Arm Bend: 92%
- ✅ Knee Cave: 12%
- ✅ Hitching: 8%
- ...

## Kinematic Summary

- **Total Lift Time:** 1.82s
- **Peak Vertical Velocity:** 1245.3 px/s
- **Peak Acceleration:** 8921.4 px/s²
```

### Graphs (in `graphs/`)

| Graph | Description |
|-------|-------------|
| `barbell_xy_stable_path.png` | smoothed, stabilized bar path (pixel units) |
| `barbell_xy_stable_path_unsmoothed.png` | raw stabilized bar path before smoothing |
| `barbell_velocity.png` | vertical bar velocity over time, colored by phase |
| `barbell_acceleration.png` | vertical bar acceleration over time, colored by phase |
| `barbell_specific_power.png` | specific power proxy over time |

All path graphs include generous horizontal padding for legend legibility. Phase colors: 🔴 Pull · 🟠 Pull-under · 🟢 Recovery.

### CSV Data Structure (`final_analysis.csv`)

All position/angle columns are **smoothed** (Savitzky-Golay, 11-frame window, cubic polynomial).

| Column Group | Example Columns | Description |
|---|---|---|
| Barbell position | `barbell_x_smooth`, `barbell_y_smooth` | smoothed barbell position (pixels) |
| Stabilization | `barbell_x_stable`, `barbell_y_stable` | shake-corrected barbell position |
| Joint positions | `{joint}_x`, `{joint}_y`, `{joint}_z`, `{joint}_vis` | smoothed normalized joints for 12 tracked joints; `_vis` = MediaPipe visibility |
| Joint angles | `left_knee_angle`, `right_knee_angle`, `left_elbow_angle`, `right_elbow_angle` | smoothed degrees |
| Bar kinematics | `vel_y_smooth`, `accel_y_smooth`, `specific_power_y_smooth` | smoothed vertical velocity (px/s), acceleration (px/s²), specific power proxy (px²/s³) |
| Lifter angle | `lifter_angle` | body forward-lean angle (degrees), smoothed |
| Phase & timing | `bar_phase`, `time_s` | 0=Pull, 1=Pull-under, 2=Recovery; elapsed seconds |
| Stabilization | `total_shake_x`, `total_shake_y` | cumulative camera-shake correction (pixels) |

Tracked joints: `left/right_shoulder`, `left/right_hip`, `left/right_knee`, `left/right_ankle`, `left/right_elbow`, `left/right_wrist`.

---

## Recording Best Practices

### 1. Camera Position

- **Lateral (side) view** — captures full bar-path range of motion.
- Camera at **hip height**, capturing the entire lift from floor to overhead.
- Keep 3–4 feet away so the lifter fills ~60% of frame width.

**Avoid:** front-facing camera, top-down angle, too close, too far.

### 2. Camera Stability

- Use a **tripod or fixed mount**. barpath stabilizes small shake, but a steady camera gives the best results.
- Avoid handheld or panned shots.

### 3. Visibility

- Full body visible: shoulders, hips, knees, ankles all in frame.
- Even, diffuse lighting; the lifter should contrast with the background.
- Minimum **24 fps** (60 fps recommended).

### 4. Video Quality

- Resolution: minimum 720p; **1080p+ recommended**.
- Codec: H.264 in MP4 is most compatible.
- Bitrate: ~5–8 Mbps for 1080p.

---

## Tips for Best Results

1. **Use `--no-video` during development** — get CSV and graphs fast; render the video only after validating the analysis.
2. **Start with one video** before batch processing.
3. **Inspect the CSV first** — check for high NaN counts (visibility issues) and confirm phase labels span the lift.
4. **Use hardware acceleration for batch jobs** — OpenVINO or ONNX give 2–4× speedups on CPU.
5. **Validate phase timing in `analysis.md`** — Pull ≈ 1–2 s, Pull-under ≈ 200–400 ms, Recovery 0.5–2 s.
6. **Compare multiple attempts** — batch-process several takes of the same lift to check consistency.
7. **Re-run analysis after code updates** — use "Add Folders (Reanalyze)" (GUI) or `run_pipeline_from_folder` to skip the slow step 1.

---

## Troubleshooting

### Runtime Errors

**"Error: Model path not found"**
- Check the model path: `ls barpath/models/std_nano.pt`.
- For OpenVINO, the directory must contain `.xml` and `.bin` files.

**"Error: No valid video files provided"**
- Verify the file exists and has a supported extension (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`).
- Convert if needed: `ffmpeg -i input.mov -c:v h264 -c:a aac output.mp4`.

**"ImportError: No module named 'mediapipe'"**
- `pip install -r requirements.txt` in your venv.

**"FFmpeg not found"**
- `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Debian/Ubuntu).

**"CUDA out of memory" (GPU users)**
- Use a smaller model, reduce resolution, or fall back to CPU (`.pt`/`.onnx`).

### Performance Issues

**Pipeline is slow**
- Use `--no-video` to skip rendering.
- Install hardware acceleration (OpenVINO for Intel CPU, ONNX Runtime otherwise).
- Use **Reanalyze mode** to skip step 1 entirely on already-collected folders.

**Video rendering takes forever (Step 5)**
- Normal for long videos; use `--no-video` if you don't need the annotated video.

**Batch processing is slow**
- Videos are processed sequentially; parallelization is not yet implemented. Expect roughly ~2 min per minute of footage on CPU.

### Technique Analysis Issues

**"Technique Analysis not available (no baseline found)"**
- Check `ls barpath/models/analysis/` — a `pro_baseline_report.json` must exist.

**"No faults detected"**
- The lift may be within pro baseline ranges — that's a good result. Verify `--lifter` and `--lift_type` match available baselines.

**All fault probabilities are ~0%**
- Baselines may be permissive or trained on insufficient data. Re-generate baselines with more data (maintainers only).

---

**For more details, see [README.md](../README.md) or file an issue on GitHub.**