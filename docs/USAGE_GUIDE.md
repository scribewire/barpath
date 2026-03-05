# BARPATH Usage Guide

<<<<<<< HEAD
A comprehensive guide to using the BARPATH system for AI-powered weightlifting technique analysis. 
=======
A comprehensive guide to using barpath for weightlifting technique analysis.
>>>>>>> 27734bd (Upgrade to YOLO26 models, nano and small, also increase mediapipe)

## Table of Contents

1. [Quick Start](#quick-start)
2. [How It Works](#how-it-works)
3. [Batch Processing and Hardware Acceleration](#batch-processing-and-hardware-acceleration)
4. [Model Format Support](#model-format-support)
5. [Output Files](#output-files)
6. [Recording Best Practices](#recording-best-practices)
7. [Tips for Best Results](#tips-for-best-results)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Using the GUI (Recommended)

The easiest way to analyze a video:

```bash
python barpath/barpath_gui.py
```

The GUI features a clean tabbed interface with four main sections:

#### **📂 Files Tab**
- Add one or more video files for analysis
- Clear videos from the queue
- Select output directory for results
- View selected videos before running analysis
- Supports MP4, AVI, MOV, MKV, WebM formats

#### **⚙️ Settings Tab**
- Automatically detects available YOLO26 models from `barpath/models` directory
- Select a model from available options (displayed as horizontally scrollable buttons)
- Choose lift type:
  - **clean**: Power clean analysis with Pull/Pull-under/Recovery phases, power calculation, and technique critique
  - **snatch**: Snatch analysis with Pull/Pull-under/Recovery phases, power calculation, and technique critique
  - **none**: Kinematics only, no lift-specific analysis or critique

#### **▶️ Analyze Tab**
- Press **Analyze** to start — the entire pipeline runs in a **background worker thread**
- The GUI never hangs or freezes; remains fully responsive during analysis
- Cancel analysis at any time with the Cancel button
- Monitor progress with a real-time progress bar and log viewer
- Logs are rendered as formatted HTML with color coding
- Progress messages are delivered from the background thread to the UI via a thread-safe queue

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
  --lift_type {clean,snatch,none}
                             Type of lift to analyze (default: none)
                             'clean'  - Power clean: Pull/Pull-under/Recovery phases,
                                        power calculation, technique critique
                             'snatch' - Snatch: Pull/Pull-under/Recovery phases,
                                        power calculation, technique critique
                             'none'   - Kinematics only, no lift-specific analysis

  --output_video PATH        Path to save annotated video output
                             (Default: outputs/output.mp4 for single video,
                              outputs/{video_name}/output.mp4 for batch)
                             Set implicitly when using --no-video (skips Step 4)

  --no-video                 Skip video rendering (Step 4) — much faster for quick analysis
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

# Step 4: Render annotated video with skeleton and bar path
python barpath/pipeline/4_render_video.py \
  --input_csv final_analysis.csv \
  --input_video video.mp4 \
  --output_video final.mp4

# Step 5: Generate critique and analysis report (Markdown)
python barpath/pipeline/5_critique_lift.py \
  --input final_analysis.csv \
  --lift_type clean
```

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

### Lifter Angle Tracking

The lifter's **orientation relative to the camera** is calculated per-frame:

- Derived from MediaPipe shoulder and hip landmarks
- Represents camera-facing (yaw) angle in degrees
- Smoothed using Savitzky-Golay filter (11-frame window, cubic polynomial)
- Recorded in `final_analysis.csv` as `lifter_angle` for every frame

**Use case**: Detect if lifter is angled away from camera (which can distort bar path analysis).

### Angle-Compensated Bar Path

barpath produces a physically accurate bar path by correcting for camera angle using **MediaPipe world landmarks**:

**How it works (per frame):**
1. Measure the shoulder width in **pixel space** — how wide the shoulders appear on screen
2. Measure the shoulder width in **world space** — the true 3D metric distance in metres from MediaPipe's world landmarks
3. Derive a `px_to_m` scale factor: `world_width_m / pixel_width_px`
4. Convert the bar's pixel displacement from its starting position into **centimetres** using that scale
5. Apply a **Savitzky-Golay smooth** to the resulting cm-space path to suppress per-frame landmark jitter

**Why this is better than a trigonometric correction:**
- The old `1/cos(yaw)` approach amplifies errors badly — a 60° angle doubles the horizontal scale; 70° nearly triples it
- The shoulder-geometry method is numerically stable at any camera angle: a camera placed at an angle foreshortens the shoulders in pixel space, so the `px_to_m` ratio naturally compensates without any blowup
- Missing shoulder frames fall back to the median scale computed over all valid frames

**Output:**
- `barbell_x_corrected_cm` / `barbell_y_corrected_cm` — bar displacement in real-world centimetres from the starting position, stored in `final_analysis.csv`
- `barbell_lateral_corrected_path.png` — angle-compensated bar path graph with both axes in centimetres, equal aspect ratio, and a camera yaw + scale annotation
- `camera_yaw_deg` — estimated camera yaw in degrees (informational only; not used in the correction)
- `px_to_m_scale` — per-frame metres-per-pixel scale factor

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

## Batch Processing and Hardware Acceleration

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
  ├── video1/
  │   ├── raw_data.pkl
  │   ├── final_analysis.csv
  │   ├── graphs/
  │   ├── output.mp4
  │   └── analysis.md
  ├── video2/
  │   ├── raw_data.pkl
  │   ...
  ```
- Progress is shown for each video with frame count
- Useful for analyzing workout sessions or comparing multiple athletes

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
├── output.mp4                # Step 4: Annotated video (if not --no-video)
└── analysis.md               # Step 5: Technique critique report (rendered as HTML in GUI)
```

### Graph Files (in `graphs/` subdirectory)

| Graph | Description | Axes |
|-------|-------------|------|
| `barbell_xy_stable_path.png` | Smoothed, stabilized bar path (shake-corrected, pixel units) | X px (lateral) vs Y px (vertical) |
| `barbell_xy_stable_path_unsmoothed.png` | Raw stabilized bar path before Savitzky-Golay smoothing | X px (lateral) vs Y px (vertical) |
| `barbell_lateral_corrected_path.png` | **Angle-compensated bar path** — both axes in real-world centimetres, equal aspect ratio, SG-smoothed after conversion; annotated with camera yaw and mm/px scale | X cm (lateral displacement) vs Y cm (vertical displacement) |
| `barbell_velocity.png` | Vertical bar velocity over time | Time (s) vs Velocity (px/s) — colored by phase |
| `barbell_acceleration.png` | Vertical bar acceleration over time | Time (s) vs Acceleration (px/s²) — colored by phase |
| `barbell_specific_power.png` | Specific power (proxy) over time | Time (s) vs Specific power (px²/s³) |

**All path graphs:**
- Include generous horizontal padding (≥ 30% of the vertical range on each side) so the legend and axis labels are never crowded
- Use the same phase color coding

**All graphs use color coding:**
- 🔴 **Red**: Pull phase
- 🟠 **Orange**: Pull-under phase
- 🟢 **Green**: Recovery phase

> `barbell_lateral_corrected_path.png` is only generated when MediaPipe world landmarks are available (i.e. when pose estimation returns 3D joint data). It is produced for all lift types, not just clean/snatch.

### CSV Data Structure (`final_analysis.csv`)

The CSV contains per-frame kinematic data with the following columns:

#### Barbell Position
| Column | Description | Units | Notes |
|--------|-------------|-------|-------|
| `barbell_x_smooth` | Smoothed horizontal bar position | pixels | Savitzky-Golay smoothed |
| `barbell_y_smooth` | Smoothed vertical bar position | pixels | Top=0, bottom=max |
| `barbell_x_stable` | Stabilized horizontal bar position | pixels | Camera shake removed, pre-smooth |
| `barbell_y_stable` | Stabilized vertical bar position | pixels | Camera shake removed, pre-smooth |
| `barbell_x_corrected_cm` | Angle-compensated horizontal displacement from start | centimetres | Derived from shoulder geometry; SG-smoothed |
| `barbell_y_corrected_cm` | Angle-compensated vertical displacement from start | centimetres | Derived from shoulder geometry; SG-smoothed |

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
| `lifter_angle` | Lifter orientation (camera yaw) | degrees | 0° = facing camera directly, ±90° = perpendicular |

#### Phase and Timing
| Column | Description | Values | Notes |
|---|---|---|---|
| `bar_phase` | Current lift phase | 0 (Pull), 1 (Pull-under), 2 (Recovery) | Color-coded in graphs |
| `time_s` | Time from start of analyzed clip | seconds | Useful for aligning with video |

#### Stabilization and Scale
| Column | Description | Units | Notes |
|---|---|---|---|
| `total_shake_x` | Cumulative camera shake correction (horizontal) | pixels | Derived from Lucas-Kanade optical flow |
| `total_shake_y` | Cumulative camera shake correction (vertical) | pixels | Derived from Lucas-Kanade optical flow |
| `camera_yaw_deg` | Estimated camera yaw angle | degrees | Informational only — not used in the correction calculation |
| `px_to_m_scale` | Per-frame pixel→metre scale factor | m/px | Derived from shoulder pixel width vs world-space width; median used for frames with missing landmarks |
| `lateral_correction_factor` | Legacy field | — | Always 1.0; retained for backwards compatibility |

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

1. **Use the `--no-video` flag during development** — saves time by skipping Step 4 (video rendering)
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

**Video rendering takes forever (Step 4)**
- This is normal for long videos (several minutes)
- Use `--no-video` if you don't need the annotated video
- Or render at lower resolution: edit `4_render_video.py` and reduce output resolution

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

<<<<<<< HEAD
**"Could not initialize video writer"**
- Check output directory exists and is writable
- Verify sufficient disk space
- Try a different output format (change file extension)
=======
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
- Edit `4_render_video.py` to adjust codec/quality settings

---

**For more details, see [README.md](../README.md) or file an issue on GitHub.**
>>>>>>> 27734bd (Upgrade to YOLO26 models, nano and small, also increase mediapipe)
