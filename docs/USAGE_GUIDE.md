# BARPATH Usage Guide

A comprehensive guide to using barpath for AI-powered weightlifting technique analysis.

## Table of Contents

1. [Quick Start](#quick-start)
   - [Using the GUI](#using-the-gui-recommended)
   - [Using the Command Line](#using-the-command-line)
   - [Re-Running Analysis on an Existing Output Folder](#re-running-analysis-on-an-existing-output-folder)
2. [How It Works](#how-it-works)
   - [Superimposed Comparison Graphs](#superimposed-comparison-graphs)
   - [ML-Based Technique Analysis](#ml-based-technique-analysis)
3. [Batch Processing, Reanalysis, and Hardware Acceleration](#batch-processing-reanalysis-and-hardware-acceleration)
4. [Model Format Support](#model-format-support)
5. [Output Files](#output-files)
6. [Recording Best Practices](#recording-best-practices)
7. [Tips for Best Results](#tips-for-best-results)
8. [Troubleshooting](#troubleshooting)

> **What's new:** Unified Technique Analysis using CompiledAnalyzer (rule-based + pro baseline comparison), automatic lift type detection (`auto`), Clean & Jerk splitting (`clean_jerk`), per-lifter baselines with pooled fallback, 6-phase coloring for Clean & Jerk, and reorganized pipeline (analysis before video rendering).

---

## Quick Start

### Using the GUI (Recommended)

The easiest way to analyze a video:

```bash
python barpath/barpath_gui.py
```

The GUI features a clean tabbed interface with four main sections:

#### **📂 Files Tab**
- Add one or more video files for analysis using **Add Videos**
- Re-run analysis on existing output folders using **Add Folders (Reanalyze)** — skips the slow video decoding step and re-runs steps 2–5 from the saved `raw_data.pkl`
- **Mutually exclusive modes**: once you add videos the "Add Folders" button is disabled, and vice versa — clear the list to switch modes
- Clear the entire input list with **Clear**
- Select output directory for results
- View and individually remove queued items before running analysis
- Supports MP4, AVI, MOV, MKV, WebM formats

#### **⚙️ Settings Tab**
- Automatically detects available YOLO26 models from `barpath/models` directory
- Select a model from available options (displayed as horizontally scrollable buttons)
- Choose lift type:
  - **auto**: Automatically detects the lift type after Step 2 (default)
  - **clean**: Power clean analysis with Pull/Pull-under/Recovery phases, power calculation, and technique critique
  - **snatch**: Snatch analysis with Pull/Pull-under/Recovery phases, power calculation, and technique critique
  - **jerk**: Jerk analysis with Dip/Drive/Recovery phases and technique critique
  - **clean_jerk**: Clean & Jerk — splits into two segments, 6-phase detection, unified report
  - **none**: Kinematics only, no lift-specific analysis or critique
- **Lifter Selection**: Choose which pro baseline to compare against for Technique Analysis
  - Options are populated from `barpath/models/analysis/` directory
  - Default is "generic" (uses pooled pro lifter data)
  - Select specific lifters (e.g., "liao_hui", "lu_xiaojun") for more targeted baselines
  - Falls back to the pooled report if per-lifter baselines are not found
- **Analysis Options**:
  - **Technique Analysis**: Toggle on/off — detects specific technique faults using biomechanical rules and pro baselines
- **Disabled automatically** when "Reanalyze" (folders) mode is active — settings are not needed because the model and video are not re-run

#### **▶️ Analyze Tab**
- Press **Analyze** to start — the entire pipeline runs in a **background worker thread**
- The GUI never hangs or freezes; remains fully responsive during analysis
- Cancel analysis at any time with the Cancel button
- Monitor progress with a real-time progress bar and log viewer
- Logs are rendered as formatted HTML with color coding
- Progress messages are delivered from the background thread to the UI via a thread-safe queue
- In **Reanalyze mode**, the progress log shows steps 2–5 only (step 1 / data collection is skipped)
- For batch runs (2+ videos or folders), a **batch post-processing** phase runs automatically after all items complete, generating the superimposed comparison graphs

#### **📊 Analysis Tab**
- View the generated lift analysis report automatically after completion
- Report is rendered as beautifully formatted HTML from `analysis.md`
- Includes phase timing (Pull / Pull-under / Recovery), maximum specific power, and technique findings
- Updates automatically when a new analysis completes
- Can be manually refreshed to view results

### Using the Command Line

For scripting and batch processing, use the CLI:

```bash
python barpath/barpath_cli.py \
  --input_video "lift.mp4" \
  --model "models/yolo26n.pt" \
  --lift_type clean
```

### Command Line Options

```
Required Arguments:
  --input_video PATH         Path to video file(s) to analyze
                             (Supports multiple files: vid1.mp4 vid2.mp4 vid3.mp4)
                             Accepted formats: .mp4, .avi, .mov, .mkv, .webm

  --model PATH               Path to trained YOLO26 model or OpenVINO export directory
                             Examples:
                               'models/yolo26n.pt'           (PyTorch)
                               'models/best.onnx'            (ONNX)
                               'models/best.engine'          (TensorRT)
                               'models/yolo26_openvino_model/' (OpenVINO directory)

Optional Arguments:
  --lift_type {auto,clean,snatch,jerk,clean_jerk,none}
                              Type of lift to analyze (default: auto)
                              'auto'        - Automatically detects lift type after Step 2
                              'clean'       - Power clean: Pull/Pull-under/Recovery phases,
                                              power calculation, technique critique
                              'snatch'      - Snatch: Pull/Pull-under/Recovery phases,
                                              power calculation, technique critique
                              'jerk'        - Jerk: Dip/Drive/Recovery phases and technique critique
                              'clean_jerk'  - Clean & Jerk: 6-phase detection, unified report
                              'none'        - Kinematics only, no lift-specific analysis

  --lifter NAME              Lifter name for model selection (default: generic)
                             Determines which pro baseline to compare against
                             Examples: 'liao_hui', 'lu_xiaojun', 'generic'
                             Models are loaded from models/analysis/{lifter}/{lift_type}/

  --lifter NAME              Lifter name for baseline selection (default: generic)
                              Determines which pro baseline to compare against
                              Falls back to pooled report if lifter-specific baselines not found

  --output_video PATH        Path to save annotated video output
                             (Default: outputs/output.mp4 for single video,
                              outputs/{video_name}/output.mp4 for batch)
                             Set implicitly when using --no-video (skips Step 5)

  --no-video                 Skip video rendering (Step 5) — much faster for quick analysis
                             Useful when you only need CSV data and graphs, not the
                             annotated video output

  --output_dir PATH          Directory to save all outputs
                             (Default: outputs)
                             For batch processing: creates subdirectories for each video

  -h, --help                 Show detailed help message with examples
```

### Quick Examples

**Basic analysis — clean lift with technique critique (YOLO26 NMS-free):**
```bash
python barpath/barpath_cli.py \
  --input_video "lift.mp4" \
  --model "models/yolo26n.pt" \
  --lift_type clean
```

**Snatch analysis with full video rendering:**
```bash
python barpath/barpath_cli.py \
  --input_video "lift.mp4" \
  --model "models/yolo26n.pt" \
  --lift_type snatch \
  --output_video "output.mp4"
```

**Skip video rendering (kinematics and CSV analysis only — faster):**
```bash
python barpath/barpath_cli.py \
  --input_video "lift.mp4" \
  --model "models/yolo26n.pt" \
  --lift_type clean \
  --no-video
```

**Batch processing multiple videos:**
```bash
python barpath/barpath_cli.py \
  --input_video lift1.mp4 lift2.mp4 lift3.mp4 \
  --model "models/yolo26n.pt" \
  --lift_type clean \
  --no-video
```

**With OpenVINO acceleration (Intel CPU optimization, YOLO26 export):**
```bash
python barpath/barpath_cli.py \
  --input_video "lift.mp4" \
  --model "models/yolo26n_openvino_model" \
  --lift_type snatch
```

**Custom output directory:**
```bash
python barpath/barpath_cli.py \
  --input_video "lift.mp4" \
  --model "models/yolo26n.pt" \
  --lift_type clean \
  --output_dir "my_results/"
```

### Running Individual Pipeline Steps

For debugging or custom workflows, you can run pipeline steps individually:

```bash
# Step 1: Collect raw tracking data (YOLO26 + MediaPipe + producer-consumer)
python barpath/pipeline/1_collect_data.py \
  --input video.mp4 \
  --model models/yolo26n.pt \
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

# Step 5: Render annotated video with skeleton, bar path, and similarity overlay
python barpath/pipeline/5_render_video.py \
  --input_csv final_analysis.csv \
  --input_video video.mp4 \
  --output_video final.mp4
```

### Re-Running Analysis on an Existing Output Folder

If you have already run step 1 and want to re-run steps 2–5 (e.g. after a code update or to change the lift type), use `run_pipeline_from_folder` from `barpath_core.py`:

```python
from barpath.barpath_core import run_pipeline_from_folder

for step, progress, msg in run_pipeline_from_folder(
    output_folder="outputs/my_lift",
    lift_type="clean",
    encode_video=True,   # set False to skip video re-render
):
    print(step, msg)
```

The folder must contain a `raw_data.pkl` produced by step 1. The source video path is read automatically from the pickle's metadata — if the file no longer exists, video rendering is skipped with a warning rather than raising an error.

In the **GUI**, this is exposed as the **Add Folders (Reanalyze)** button in the Files tab.

---

## How It Works

### Producer-Consumer Video Pipeline

barpath uses a **producer-consumer architecture** to maximize throughput:

- **Producer (I/O thread)**: Decodes video frames in a background thread and places them in a bounded queue (size 8)
- **Consumer (main thread)**: Pulls frames from the queue and runs YOLO26 + MediaPipe inference
- **Benefit**: The main thread never stalls waiting for disk I/O; frames are pre-decoded and ready

This is especially effective for:
- Slow storage (network drives, USB 3.0, older SSDs)
- High-resolution videos (4K, 60+ fps)
- CPU-intensive inference (larger YOLO26 models)

### YOLO26 NMS-Free Barbell Detection

barpath uses **Ultralytics YOLO26** for barbell endcap detection:

- **NMS-Free Architecture**: The model outputs final detections directly; no separate Non-Maximum Suppression step is needed
- **Speed**: Eliminates post-processing overhead compared to traditional YOLO
- **Inference Call**: Uses only `conf` (confidence threshold); no `iou` parameter is passed
- **Default Model**: `yolo26n.pt` (nano variant) balances accuracy and speed

**Supported Model Formats:**
- `.pt` (PyTorch native)
- `.onnx` (ONNX Runtime)
- `.engine` (TensorRT for NVIDIA GPUs)
- OpenVINO directory export (Intel CPU optimization)

**Adjusting Detection Sensitivity:**
- **Lower confidence** (e.g., `--conf 0.3`): Detects more barbells, but may include false positives
- **Higher confidence** (e.g., `--conf 0.7`): Stricter detection, fewer false positives, but may miss frames
- Default is `0.5` (reasonable balance)

### MediaPipe Pose Estimation

barpath uses **MediaPipe Pose** for full-body joint tracking:

**Current Configuration:**
- **Model Complexity**: `1` (Medium) — recommended default
- **Min Detection Confidence**: `0.5`
- **Min Tracking Confidence**: `0.5`
- **Segmentation**: Enabled (used for background feature detection in stabilization)
- **Tracked Joints**: 12 landmarks (shoulders, hips, knees, ankles, elbows, wrists)

**Complexity Levels:**
| Level | Complexity Param | Speed | Accuracy | Best For |
|-------|---|---|---|---|
| Light | 0 | Fastest | Lower | Real-time, low-end devices |
| Medium | 1 | Balanced | Good | **Default** — standard analysis |
| Heavy | 2 | Slowest | Highest | Offline analysis, difficult angles |

**Output Coordinates:**
- **Normalized landmarks**: x, y, z values in [0, 1] range (relative to image dimensions); z is pseudo-depth
- **World landmarks**: Real-world 3D coordinates in meters (relative to hip center)
- **Visibility**: Confidence score (0–1) for each joint

### Camera Shake Stabilization

barpath uses **Lucas-Kanade optical flow** to stabilize the bar path:

1. Detect features in the **background** (using MediaPipe's person segmentation mask)
2. Track features across frames using Lucas-Kanade algorithm
3. Estimate motion (translation + rotation) from feature correspondences
4. Apply correction to barbell position to remove camera shake

**Result**: A smooth, stabilized bar path even if the camera moves or shakes during the lift.

### Joint Smoothing and Kinematic Calculations

**Smoothing Pipeline:**
1. Extract raw MediaPipe joint positions for every frame
2. Apply **Savitzky-Golay filter** (11-frame window, cubic polynomial) to all joint coordinates
3. Calculate joint angles from smoothed positions
4. Smooth velocity signals with larger window (15 frames) to suppress noise in derivatives

**Result**: `final_analysis.csv` contains only smoothed, analysis-ready data with no raw noisy frames.

### 3-Phase Lift Analysis

barpath analyzes Olympic lifts using a **3-phase system** based on biomechanics:

#### The Three Phases

| Phase | Label | Color | Duration | Definition |
|-------|-------|-------|----------|------------|
| **Pull** | 0 | 🔴 Red | t0 → t2 | Barbell begins moving upward; lifter extends through ankles, knees, hips in sequence; ends when hips reach maximum extension |
| **Pull-under** | 1 | 🟠 Orange | t2 → t3 | Hips transition from extending to actively dropping; lifter begins knee bend for catch; turnaround point where bar is fastest |
| **Recovery** | 2 | 🟢 Green | t3 → t4 | Lifter in catch position (squat or split); drives upward to standing position; lift is complete when bar reaches peak height |

#### Phase Detection Methods

**For Classic Lifts (clean/snatch):**
- Uses MediaPipe-detected joint landmarks to identify key timepoints (t0, t1, t2, t3, t4)
- Applies kinematic heuristics:
  - Hip velocity and hip Y-position to find extension peak (t2)
  - Barbell and hip signals to identify catch completion (t3)
- Robust to video artifacts and lighting variation

**For Other Lift Types (lift_type=none):**
- No phase detection is performed
- Only kinematics are analyzed (bar path, velocity, acceleration)

#### Phase Timing in Analysis Report

The generated `analysis.md` report includes:
- **Phase Start/End Frames**: Frame numbers where each phase begins and ends
- **Phase Duration**: Time in milliseconds for each phase
- **Transition Points**:
  - **Extension Peak (t2)**: Maximum hip height during drive
  - **Catch Point (t3)**: When lifter's hips stop descending
  - **Bar Peak (t4)**: Maximum bar height after recovery

Example output:
```
Phase Timing:
  Pull (t0→t2):        Frames 0–45 (1500 ms)
  Pull-under (t2→t3):  Frames 45–52 (233 ms)
  Recovery (t3→t4):    Frames 52–80 (933 ms)
```

#### Maximum Specific Power

Calculated as maximum power output during the Pull and Pull-under phases (t1→t3):

**Calculation:**
```
Specific Power = (bar velocity)² × (bar mass) / (lifter mass)
```

When pixel-to-meter conversion is available, reported in **watts per kilogram (W/kg)**.

Alternatively, reported in pixel units (px²/s³) when real-world scaling is unavailable.

**Interpretation:**
- Higher values indicate explosive power production
- Occurs typically during mid-Pull phase (bar accelerating fastest)
- Useful for comparing athlete performance across sessions

#### Technique Critique

barpath analyzes lift technique using **rule-based checks**:

**For Clean:**
- Early arm bend (should stay straight until after extension)
- Incomplete extension (hips must reach full extension before Pull-under)
- Poor turnover timing (arms should bend only after bar reaches shoulder height)
- Weight shift (should stay mid-foot throughout)

**For Snatch:**
- Similar checks, adapted for overhead lockout
- Checks for head position during Pull-under
- Evaluates stability in overhead position

**Output:**
- Findings are reported in `analysis.md` with specific frame ranges and severity
- Recommendations are provided for each fault

---

## Technique Analysis

barpath includes **Technique Analysis** powered by a CompiledAnalyzer that combines rule-based biomechanical checks with comparison against professional lifter baselines. This provides specific, confidence-weighted fault detection and coaching feedback.

**How it works:**
1. Extracts biomechanical features from `final_analysis.csv` (bar path shape, velocity profiles, joint angles, phase timings)
2. Compares against pro baselines using configurable thresholds and ratios
3. Detects specific faults with confidence scores based on deviation from pro ranges
4. Generates coaching-style recommendations for each detected issue

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

**Snatch (15 faults):** All clean faults plus:
| Fault ID | Name | Phase |
|----------|------|-------|
| `wide_grip_early_bend` | Early Arm Bend (Snatch-Specific) | Pull (late) |
| `press_out` | Press Out (No Lift) | Pull-under |
| `overhead_instability` | Overhead Instability | Pull-under |
| `excessive_forward_lean` | Excessive Forward Lean | Recovery |

**Model files:**
- `compiled_analyzer_config.json` — Analyzer configuration

### Lifter Selection

You can choose which pro baseline to compare against:

```bash
# Compare against generic baseline (all pros pooled)
python barpath/barpath_cli.py --input_video lift.mp4 --model models/yolo26n.pt --lift_type clean --lifter generic

# Compare against specific lifter's technique
python barpath/barpath_cli.py --input_video lift.mp4 --model models/yolo26n.pt --lift_type clean --lifter liao_hui
```

**Model resolution:**
1. First checks `models/analysis/{lifter}/{lift_type}/`
2. Falls back to `models/analysis/generic/{lift_type}/`
3. If neither exists, analysis is skipped gracefully

### Training Custom Models

To train your own analysis baselines, use the training scripts in `outputs/`:

```bash
# Technique Analysis baseline generation
python outputs/technique_analysis_training.py \
    --pro_dir data/pro \
    --output_dir barpath/models/analysis \
    --lift_type clean

# Lift Detection (lift type classifier)
python outputs/lift_detection_training.py train \
    --data_dir data/pro \
    --output_dir barpath/models/lift_detection
```

**Data requirements:**
- Pro lifts: 30+ lifts per lifter-lift-type (100+ recommended)
- Each lift must have `final_analysis.csv` from running through the pipeline

**Directory structure (gender-based, flat folders):**
```
data/pro/
├── male/
│   ├── liao_hui_001_clean/final_analysis.csv
│   ├── liao_hui_002_snatch/final_analysis.csv
│   ├── lu_xiaojun_001_clean/final_analysis.csv
│   └── ...
└── female/
    ├── lifter_001_snatch/final_analysis.csv
    └── ...
```

**Folder naming:** `{lifter_name}_{sequence}_{lift_type}`
- Examples: `liao_hui_001_snatch`, `lu_xiaojun_003_clean`
- Lift types: `snatch`, `clean`, `jerk` (not `clean_jerk`)
- Gender is inferred from the folder path (`male/` or `female/`)
- Clean & Jerk is detected automatically when a lifter has both `clean` and `jerk` folders

---

## Batch Processing, Reanalysis, and Hardware Acceleration

### Batch Processing

Process multiple videos in a single command:

```bash
python barpath/barpath_cli.py \
  --input_video video1.mp4 video2.mp4 video3.mp4 \
  --model "models/yolo26n.pt" \
  --lift_type clean \
  --output_dir "batch_results"
```

**Behavior:**
- Each video is processed sequentially
- A subdirectory is created for each video in `output_dir`:
  ```
  batch_results/
  ├── lift_1/
  │   ├── raw_data.pkl
  │   ├── final_analysis.csv
  │   ├── graphs/
  │   ├── output.mp4
  │   └── analysis.md
  ├── lift_2/
  │   ├── raw_data.pkl
  │   ...
  └── superimposed_bar_paths.png   ← batch post-processing
  ```
- Progress is shown for each video with frame count
- After all videos complete, **batch post-processing** automatically generates superimposed comparison graphs (see [Superimposed Comparison Graphs](#superimposed-comparison-graphs) below)
- Useful for analyzing workout sessions or comparing multiple attempts of the same lift

### Reanalyze Existing Output Folders

If you have already run barpath on a video and want to re-run steps 2–5 (e.g. after a code update, to change `lift_type`, or to regenerate graphs), you can reanalyze without re-processing the video:

**GUI:**
1. Open the **Files Tab**
2. Click **Add Folders (Reanalyze)** and select one or more existing output folders (each must contain `raw_data.pkl`)
3. The Settings tab is automatically disabled — model and video settings are not needed
4. Click **Analyze** — steps 2–5 run on each folder; video is re-rendered if the original source path is still accessible

**Notes:**
- The source video path is stored inside `raw_data.pkl` when step 1 runs; if the file has moved or been deleted, the video render step is skipped with a warning rather than an error
- You can add multiple folders and compare them using the batch post-processing superimposed graphs
- Reanalyze mode and Videos mode are mutually exclusive — clear the input list to switch between them

### Superimposed Comparison Graphs

After processing 2 or more videos (or reanalyzing 2 or more folders), barpath automatically generates a superimposed comparison graph in the top-level output directory:

| File | Description |
|------|-------------|
| `superimposed_bar_paths.png` | All lifts overlaid in pixel space; non-reference lifts uniformly scaled |

**How the overlay works:**
1. Each lift's path is **origin-normalised** at its pull-under start point (the first frame where phase transitions from Pull → Pull-under), so all paths share a common reference origin at (0, 0).
2. Non-reference lifts are **uniformly scaled** — a single scale factor is found by least-squares minimisation of the distance between matching phase-transition markers (e.g. Pull→Pull-under and Pull-under→Recovery points). If no matching markers exist, arc-length ratio is used as a fallback. Both x and y are multiplied by the same factor (no distortion of the path shape).
3. Paths are overlaid with phase-based color coding for visual comparison.

**Example legend entry:** `Lift 2`

### Hardware Acceleration with OpenVINO

**Intel CPUs only** — use OpenVINO for faster inference:

1. **Export your YOLO26 model to OpenVINO format:**
   ```bash
   yolo export model=models/yolo26n.pt format=openvino
   ```
   This creates a directory `yolo26n_openvino_model/` with `.xml` and `.bin` files.

2. **Use the exported model in barpath:**
   ```bash
   python barpath/barpath_cli.py \
     --input_video "lift.mp4" \
     --model "models/yolo26n_openvino_model" \
     --lift_type clean
   ```

3. **Installation:**
   ```bash
   pip install openvino onnxruntime
   ```

**Performance:**
- On modern Intel CPUs: ~2–4× speedup vs. PyTorch CPU inference
- Especially effective for batch processing multiple videos
- Requires `.xml` and `.bin` files in the model directory

---

## Model Format Support

barpath supports multiple YOLO26 model formats:

| Format | File Extension | Requirement | Installation | Speed | Use Case |
|--------|---|---|---|---|---|
| **PyTorch** | `.pt` | ultralytics | Built-in | Baseline | Development, testing |
| **ONNX** | `.onnx` | onnxruntime | `pip install onnxruntime` | ~1–2× faster | CPU optimization |
| **TensorRT** | `.engine` | TensorRT | `pip install tensorrt` | 3–5× faster | NVIDIA GPU only |
| **OpenVINO** | `{dir}/*.xml` + `*.bin` | openvino | `pip install openvino` | 2–4× faster | Intel CPU only |

**Selecting a Model:**
- **For CPU (any brand)**: Use `.pt` or `.onnx` formats
- **For Intel CPU**: Export to OpenVINO for best performance
- **For NVIDIA GPU**: Use TensorRT (`.engine` export)

---

## Output Files

### Summary

After running the pipeline, you'll find:

```
outputs/
├── raw_data.pkl              # Step 1: Raw detections
├── final_analysis.csv        # Step 2: Enriched kinematic data
├── graphs/
│   ├── barbell_xy_path.png
│   ├── barbell_velocity.png
│   ├── barbell_acceleration.png
│   ├── barbell_specific_power.png
│   └── ...
├── output.mp4                # Step 5: Annotated video (if not --no-video)
└── analysis.md               # Step 4: Technique analysis report (rendered as HTML in GUI)
```

### Analysis Report (`analysis.md`)

The analysis report includes Technique Analysis results:

**Structure:**
1. **Technique Critique** — Flagged faults with confidence percentages
   - Fault name, phase, and coaching-style description
   - "All Checks" section showing probability for each fault type
2. **Kinematic Summary** — Total lift time, peak velocity, peak acceleration
3. **Footer** — BARPATH version, baseline lifter

**Example output:**
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

---

*Generated by BARPATH ML Analysis Engine*
```

### Graph Files (in `graphs/` subdirectory)

| Graph | Description | Axes |
|-------|-------------|------|
| `barbell_xy_stable_path.png` | Smoothed, stabilized bar path (shake-corrected, pixel units) | X px (lateral) vs Y px (vertical) |
| `barbell_xy_stable_path_unsmoothed.png` | Raw stabilized bar path before Savitzky-Golay smoothing | X px (lateral) vs Y px (vertical) |
| `barbell_velocity.png` | Vertical bar velocity over time | Time (s) vs Velocity (px/s) — colored by phase |
| `barbell_acceleration.png` | Vertical bar acceleration over time | Time (s) vs Acceleration (px/s²) — colored by phase |
| `barbell_specific_power.png` | Specific power (proxy) over time | Time (s) vs Specific power (px²/s³) |

### Superimposed Graphs (in top-level batch output directory)

These are generated automatically after batch processing 2 or more videos (or reanalyzing 2 or more folders):

| Graph | Description | Axes |
|-------|-------------|------|
| `superimposed_bar_paths.png` | All lifts overlaid in pixel space; non-reference lifts uniformly scaled | px (lateral) vs px (vertical) |

**All path graphs:**
- Include generous horizontal padding (≥ 30% of the vertical range on each side) so the legend and axis labels are never crowded
- Use the same phase color coding

**All graphs use color coding:**
- 🔴 **Red**: Pull phase
- 🟠 **Orange**: Pull-under phase
- 🟢 **Green**: Recovery phase

### CSV Data Structure (`final_analysis.csv`)

The CSV contains per-frame kinematic data with the following columns:

#### Barbell Position
| Column | Description | Units | Notes |
|--------|-------------|-------|-------|
| `barbell_x_smooth` | Smoothed horizontal bar position | pixels | Savitzky-Golay smoothed |
| `barbell_y_smooth` | Smoothed vertical bar position | pixels | Top=0, bottom=max |
| `barbell_x_stable` | Stabilized horizontal bar position | pixels | Camera shake removed, pre-smooth |
| `barbell_y_stable` | Stabilized vertical bar position | pixels | Camera shake removed, pre-smooth |

#### Kinematics
| Column | Description | Units | Notes |
|--------|-------------|-------|-------|
| `vel_y_smooth` | Vertical bar velocity | px/s | Positive=upward |
| `accel_y_smooth` | Vertical bar acceleration | px/s² | Positive=upward acceleration |
| `specific_power_y_smooth` | Specific power proxy | px²/s³ | Power output indicator |

#### Joint Positions (raw, normalized 0–1)
| Column Pattern | Description | Units | Notes |
|---|---|---|---|
| `{joint}_x` | Joint horizontal position | 0–1 (normalized) | Tracked for 12 joints (shoulders, hips, knees, ankles, elbows, wrists) |
| `{joint}_y` | Joint vertical position | 0–1 (normalized) | |
| `{joint}_z` | Joint depth (pseudo-3D) | 0–1 (normalized) | MediaPipe's pseudo-depth estimate |
| `{joint}_vis` | Joint visibility | 0–1 | Confidence that the joint is visible and correctly detected |

**Joints tracked:**
- Shoulders: `left_shoulder`, `right_shoulder`
- Hips: `left_hip`, `right_hip`
- Knees: `left_knee`, `right_knee`
- Ankles: `left_ankle`, `right_ankle`
- Elbows: `left_elbow`, `right_elbow`
- Wrists: `left_wrist`, `right_wrist`

#### Joint Angles (smoothed)
| Column | Description | Units | Notes |
|---|---|---|---|
| `left_knee_angle` | Left knee angle | degrees | 0° = fully extended, 90° = right angle |
| `right_knee_angle` | Right knee angle | degrees | |
| `left_elbow_angle` | Left elbow angle | degrees | |
| `right_elbow_angle` | Right elbow angle | degrees | |

#### Lifter Angle (per-frame, smoothed)
| Column | Description | Units | Notes |
|---|---|---|---|
| `lifter_angle` | Body forward lean angle | degrees | Derived from shoulder-hip alignment; measures how far the lifter leans forward |

#### Phase and Timing
| Column | Description | Values | Notes |
|---|---|---|---|
| `bar_phase` | Current lift phase | 0 (Pull), 1 (Pull-under), 2 (Recovery) | Color-coded in graphs |
| `time_s` | Time from start of analyzed clip | seconds | Useful for aligning with video |

#### Stabilization
| Column | Description | Units | Notes |
|---|---|---|---|
| `total_shake_x` | Cumulative camera shake correction (horizontal) | pixels | Derived from Lucas-Kanade optical flow |
| `total_shake_y` | Cumulative camera shake correction (vertical) | pixels | Derived from Lucas-Kanade optical flow |

---

## Recording Best Practices

To get the best results from barpath analysis, follow these recording guidelines:

### 1. Camera Position

- **Lateral view (side angle)** is best — captures full range of motion in bar path
- Position camera at **hip height** for optimal skeleton visibility
- Aim camera to capture the **entire lift from floor to overhead**
- Keep **3–4 feet away** so the lifter fills ~60% of frame width

**Avoid:**
- Front-facing camera (hides bar path behind lifter)
- Top-down angle (distorts joint positions)
- Too close (limbs cut off frame)
- Too far (joints become too small)

### 2. Camera Stability

- Use a **tripod or fixed mount** (stable camera is essential for accurate analysis)
- **Avoid handheld or panned shots** — creates motion artifacts
- If camera does move, barpath will attempt to stabilize, but results are best with truly fixed camera
- Test: Roll 2–3 seconds of footage; look for camera movement

### 3. Visibility Requirements

- **Full-body visibility**: All joints (shoulders, hips, knees, ankles) must be visible
- **Lighting**: Even, diffuse lighting (avoid harsh shadows or backlighting)
- **Contrast**: Lifter should contrast with background (no white lifter on white wall)
- **Speed**: Minimum **24 fps** (60 fps recommended for smooth motion analysis)

### 4. Lighting

- **Bright, even illumination**: Outdoors or well-lit gym works well
- **Avoid:**
  - Backlighting (lifter appears as silhouette)
  - Harsh shadows (MediaPipe may lose joint tracking)
  - Flickering lights (causes jitter)
  - Reflective surfaces near camera (creates glare)

### 5. Video Quality

- **Resolution**: Minimum 720p; 1080p or higher recommended
- **Frame Rate**: Minimum 24 fps; 60 fps ideal for smooth motion analysis
- **Codec**: H.264 (mp4) is most compatible; avoid lossy compression artifacts
- **Bitrate**: ~5 Mbps for 1080p (high enough to preserve detail)

**Recommended Settings:**
- 1080p @ 60 fps, H.264, 6–8 Mbps bitrate
- If processing is slow, try 720p @ 60 fps

---

## Tips for Best Results

1. **Use the `--no-video` flag during development** — saves time by skipping Step 5 (video rendering)
   ```bash
   python barpath/barpath_cli.py --input_video lift.mp4 --model models/yolo26n.pt --lift_type clean --no-video
   ```
   This gives you CSV and graphs much faster; only render video after you've validated the analysis.

2. **Start with one video** — test the pipeline end-to-end before batch processing
   ```bash
   python barpath/barpath_cli.py --input_video test_lift.mp4 --model models/yolo26n.pt --lift_type clean --no-video
   ```

3. **Check the CSV first** — inspect `final_analysis.csv` to verify data quality before looking at graphs or video
   - Open in Excel or use pandas in a Jupyter notebook
   - Look for gaps in joint positions (high NaN counts = visibility issues)
   - Check if phase labels (0, 1, 2) appear throughout the lift

4. **Use hardware acceleration for batch jobs** — OpenVINO or TensorRT can 2–5× speedup
   ```bash
   pip install openvino onnxruntime
   yolo export model=models/yolo26n.pt format=openvino
   ```

5. **Validate phase detection** — Review `analysis.md` to check phase timing is reasonable
   - Pull phase should be ~1–2 seconds (1000–2000 ms) for most lifts
   - Pull-under should be fast (~200–400 ms)
   - Recovery depends on lifter strength; typically 0.5–2 seconds

6. **Compare multiple attempts** — Batch process several takes of the same lift to identify consistency
   ```bash
   python barpath/barpath_cli.py --input_video attempt1.mp4 attempt2.mp4 attempt3.mp4 --model models/yolo26n.pt --lift_type clean --no-video
   ```
    The superimposed graphs are generated automatically for visual comparison of bar path shape and consistency across attempts.

7. **Re-run analysis after code updates** — Use "Add Folders (Reanalyze)" in the GUI (or `run_pipeline_from_folder` in code) to re-run steps 2–5 on any previously processed folder without repeating the slow step 1 video decoding. This is especially useful when:
   - You want to change the `lift_type` without re-processing the video
   - You have updated the analysis or graphing code and want to refresh outputs
   - You want to regenerate graphs or the critique report with new settings

---

## Troubleshooting

### Runtime Errors

**"Error: Model path not found"**
- Check that the model file exists: `ls barpath/models/yolo26n.pt`
- Use absolute path if relative path doesn't work
- For OpenVINO, ensure directory contains `.xml` and `.bin` files

**"Error: No valid video files provided"**
- Verify video file exists and has correct extension (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`)
- Convert video if needed: `ffmpeg -i input.mov -c:v h264 -c:a aac output.mp4`

**"ImportError: No module named 'mediapipe'"**
- Install missing dependencies: `pip install -r requirements.txt`
- Ensure you're using the correct Python environment

**"FFmpeg not found"**
- Install FFmpeg: `sudo apt install ffmpeg` (Ubuntu) or `brew install ffmpeg` (macOS)
- Verify: `ffmpeg -version`

**"CUDA out of memory" (GPU users)**
- Use a smaller model variant (e.g., `yolo26n` instead of `yolo26s`)
- Reduce video resolution before processing
- Use CPU inference instead: remove TensorRT and use `.pt` or `.onnx`

### Performance Issues

**Pipeline is very slow**
- Check if CPU is maxed out: use `top` or Task Manager
- Try `--no-video` to skip the expensive video rendering step
- Use hardware acceleration: install OpenVINO (Intel) or TensorRT (NVIDIA)
- Process smaller videos first to validate the pipeline
- If you only changed analysis code (not detection), use **Reanalyze mode** to skip step 1 entirely — this re-runs only steps 2–5 on the saved `raw_data.pkl`

**Video rendering takes forever (Step 5)**
- This is normal for long videos (several minutes)
- Use `--no-video` if you don't need the annotated video
- Or render at lower resolution: edit `5_render_video.py` and reduce output resolution

**Batch processing is slow**
- Videos are processed sequentially; parallelization is not yet implemented
- Each video must complete fully before the next starts
- Estimated time: ~2 min per minute of footage (varies by hardware)

### Verifying Hardware Acceleration

Check if hardware acceleration packages are installed:

```bash
python -c "
from barpath.hardware_detection import get_hardware_profile, get_optional_packages
p = get_hardware_profile()
print('Hardware Profile:', p)
o, v = get_optional_packages(p)
print('Recommended packages:', o + v)
"
```

If OpenVINO/ONNX/TensorRT are recommended but not installed:

```bash
# Intel CPU
pip install onnxruntime openvino

# NVIDIA GPU
pip install tensorrt

# All platforms (ONNX is universal)
pip install onnxruntime
```

### FFmpeg Errors

**"Could not initialize video writer"**
- Check output directory exists and is writable
- Verify sufficient disk space
- Try a different output format (change file extension)

**"FFmpeg: Unknown encoder 'aac'"**
- FFmpeg variant may not include AAC encoder; install full version
- Ubuntu: `sudo apt install ffmpeg libavcodec-extra`
- macOS: `brew install ffmpeg --with-options-for-aac`

**"Encoder (h264) not found"**
- Install H.264 encoder support
- Ubuntu: `sudo apt install libx264-dev`
- macOS: `brew install x264`

**Video output is corrupted or silent**
- Try using a different codec or container format
- Edit `5_render_video.py` to adjust codec/quality settings

### Superimposed Graph Issues

**Superimposed graphs are not generated**
- Superimposed graphs require at least 2 lifts (videos or folders) in the same batch run
- Each lift must have a valid `final_analysis.csv` in its output folder
- Check the Analyze tab log for "Skipping superimposed graphs: fewer than 2 lifts with valid data"

**Paths look misaligned in superimposed graphs**
- Ensure lifts are recorded from similar camera angles for meaningful visual comparison
- Large differences in recording angle or athlete body position between takes can make overlay comparison difficult

### Technique Analysis Issues

**"Technique Analysis not available (no baseline found)"**
- Check that baseline files exist: `ls barpath/models/analysis/generic/clean/`
- Required files: `pro_baseline_report.json` (pooled) or `pro_baseline_report_{lifter}.json` (per-lifter)
- Train baselines using: `python outputs/technique_analysis_training.py --lift_type clean`

**"No faults detected"**
- Technique Analysis may return no faults if the lift is within pro baseline ranges for all checked criteria
- Ensure `--lifter` and `--lift_type` match available baselines
- Verify `final_analysis.csv` contains valid kinematic data

**All fault probabilities are 0% or near 0%**
- Baseline may be too permissive or trained on insufficient data
- Check `training_metadata.json` for sample counts
- Ensure training data includes both pro and error lifts with diverse technique profiles

**"CompiledAnalyzer import error"**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Technique Analysis requires `numpy`, `pandas`, and `scipy`

---

**For more details, see [README.md](../README.md) or file an issue on GitHub.**
