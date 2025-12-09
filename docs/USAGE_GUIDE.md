# BARPATH Usage Guide

A comprehensive guide to using the BARPATH system for AI-powered weightlifting technique analysis.

## Table of Contents

- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
  - [Camera Stabilization](#camera-stabilization)
  - [Perspective Angle Compensation](#perspective-angle-compensation)
- [Batch Processing and Hardware Acceleration](#batch-processing-and-hardware-acceleration)
  - [Batch Processing](#batch-processing)
  - [Hardware Acceleration with OpenVINO](#hardware-acceleration-with-openvino)
- [Output Files](#output-files)
- [Recording Best Practices](#recording-best-practices)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Using the GUI (Recommended)

The easiest way to analyze a video:

```bash
python barpath/barpath_gui.py
```

The GUI provides:
- 📂 Interactive file/directory selection
- 🎯 Automatic model detection
- ⚡ Hardware acceleration selection (CPU, OpenVINO, etc.)
- 📊 Real-time progress tracking
- 📝 Live log output
- 👁️ View analysis reports

### Using the Command Line

For scripting and batch processing:

```bash
python barpath/barpath_cli.py \
  --input_video "lift.mp4" \
  --model "models/yolo11s.pt" \
  --lift_type clean
```

### Command Line Options

```
Required Arguments:
  --input_video PATH         Path to source video file
                            (e.g., 'videos/clean.mp4')
  
  --model PATH              Path to trained YOLO model file
                            (e.g., 'models/yolo11s.pt')

Optional Arguments:
  --output_video PATH       Path to save annotated video
                            (Default: outputs/output.mp4)
  
  --lift_type {clean,snatch,none}
                            Type of lift to analyze
                            'clean'  - Power clean technique critique
                            'snatch' - Snatch technique critique
                            'none'   - Skip critique (default)
  
  --no-video                Skip video rendering (faster)
  
  --output_dir PATH         Directory to save outputs
                            (Default: outputs)
```

### Quick Examples

**Basic analysis with clean technique critique:**
```bash
python barpath/barpath_cli.py \
  --input_video "lift.mp4" \
  --model "models/yolo11s.pt" \
  --lift_type clean
```

**With OpenVINO acceleration (Intel CPUs):**
```bash
python barpath/barpath_cli.py \
  --input_video "lift.mp4" \
  --model "models/yolo11s_openvino_model" \
  --lift_type snatch
```

**Skip video rendering (analyze only, much faster):**
```bash
python barpath/barpath_cli.py \
  --input_video "lift.mp4" \
  --model "models/yolo11s.pt" \
  --lift_type clean \
  --no-video
```

### Running Individual Pipeline Steps

For debugging or custom workflows:

```bash
# Step 1: Collect raw tracking data
python barpath/pipeline/1_collect_data.py \
  --input video.mp4 \
  --model models/yolo11s.pt \
  --output raw_data.pkl \
  --lift_type clean

# Step 2: Analyze kinematics and angles
python barpath/pipeline/2_analyze_data.py \
  --input raw_data.pkl \
  --output final_analysis.csv

# Step 3: Generate graphs
python barpath/pipeline/3_generate_graphs.py \
  --input final_analysis.csv \
  --output_dir graphs

# Step 4: Render annotated video
python barpath/pipeline/4_render_video.py \
  --input_csv final_analysis.csv \
  --input_video video.mp4 \
  --output_video final.mp4

# Step 5: Generate critique
python barpath/pipeline/5_critique_lift.py \
  --input final_analysis.csv \
  --lift_type clean
```

---

## How It Works

The BARPATH system analyzes weightlifting videos in two complementary ways:

### Camera Stabilization

**Problem:** Videos shot with handheld cameras contain motion (shake) that distorts measurements.

**Solution:** The system uses optical flow-based motion estimation to detect and compensate for camera movement.

**Technical Details:**

1. **Feature Detection & Tracking** (Step 1: `1_collect_data.py`)
   - Detects corner features in the background of each frame
   - Tracks features between consecutive frames using Lucas-Kanade optical flow
   - Excludes the person (using MediaPipe segmentation) to focus on background motion
   - Estimates camera translation from the median displacement of tracked features

2. **Motion Compensation** (Step 2: `2_analyze_data.py`)
   - Accumulates frame-by-frame motion to calculate total camera shake
   - Removes this accumulated motion from barbell position measurements
   - Results in a "stabilized" barbell path that represents true movement, not camera movement

3. **Implementation Details:**
   - Uses `cv2.goodFeaturesToTrack()` to detect corner features
   - Employs Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK()`) for tracking
   - Filters out the person using MediaPipe segmentation
   - Applies median filtering to robustly reject outliers
   - Refines features at 50-feature threshold to maintain tracking quality

**Output Columns:**
- `shake_dx`, `shake_dy` - Per-frame camera motion in pixels
- `total_shake_x`, `total_shake_y` - Cumulative camera motion
- `barbell_x_stable`, `barbell_y_stable` - Barbell position with camera motion removed

### Perspective Angle Compensation

**Problem:** Videos shot at an angle (not perpendicular to the lifter) compress horizontal bar movements. A 45° camera angle makes lateral movements appear ~30% smaller than they actually are.

**Solution:** Uses 3D world landmarks from MediaPipe Pose to calculate the camera angle and apply a perspective correction factor.

**Technical Details:**

1. **Reference Angle Calculation** (First frame, 1 second before knee pass)
   - Extracts shoulder positions from MediaPipe world landmarks (3D coordinates in meters)
   - Calculates lateral axis: vector from left shoulder to right shoulder
   - Determines how this axis is oriented relative to the camera
   - Computes camera yaw angle using: `yaw = arctan2(|z|, |x|)` where X and Z are the shoulder axis components in world space

2. **Correction Factor Derivation**
   - Camera angle compresses horizontal displacement by a factor of `cos(yaw)`
   - To correct this, we scale observed displacement by: `correction = 1 / cos(yaw)`
   - Examples:
     - 0° angle (perpendicular): correction = 1.0 (no correction needed)
     - 30° angle: correction = 1.155 (expand by 15.5%)
     - 45° angle: correction = 1.414 (expand by 41.4%)
     - 60° angle: correction = 2.0 (expand by 100%)

3. **Application to All Frames**
   - Uses the reference angle from the first frame (constant throughout lift)
   - Applies the same correction factor to all frames for consistency
   - Prevents drift by correcting only the displacement from baseline, not absolute position

4. **Implementation Details:**
   - World landmarks provided by MediaPipe Pose in meters relative to hip center
   - Extracts (x, y, z) for both shoulders from first frame
   - Filters landmarks by visibility score (> 0.1)
   - Projects lateral axis onto horizontal plane (XZ components only)
   - Stores correction factor and reference angle with output data

**Output Columns:**
- `barbell_x_corrected_px` - Horizontally corrected barbell position
- `camera_yaw_deg` - Detected camera angle (from reference frame)
- `lateral_correction_factor` - Scaling factor applied (1.0 = no correction)

**When It Works Best:**
- ✅ 30-60° camera angles
- ✅ Stationary camera on tripod
- ✅ Both shoulders visible throughout lift
- ✅ Good lighting for reliable landmark detection

**When Correction Is Skipped:**
- ❌ `lift_type = "none"` (pose landmarks not collected)
- ❌ Insufficient world landmark data
- ❌ Unrealistic shoulder width (< 0.3m or > 0.6m)

---

## Batch Processing and Hardware Acceleration

### Batch Processing

BARPATH supports processing multiple videos in a single run, which is useful for analyzing multiple lifts or sessions efficiently.

**How it works:**
- In the **GUI**: Add multiple video files using the "Add Video" button. The system will process them sequentially, one after another.
- In the **CLI**: Use the `--input_video` argument multiple times or pass a list of files.

**Behavior:**
- Videos are processed **sequentially**, not in parallel. This means the total time is the sum of individual video processing times.
- For each video, a separate subfolder is created in the output directory, named after the video file (e.g., `video1.mp4` → `video1/`).
- Progress is tracked per video, and the GUI updates the progress bar accordingly.
- If any video fails, the others continue processing.
- Output files (graphs, CSV, video) are generated in each video's subfolder.

**Example CLI batch processing:**
```
python barpath_cli.py --input_video lift1.mp4 lift2.mp4 lift3.mp4 --model yolo.pt --lift_type clean --no-video
```
This will create `lift1/`, `lift2/`, `lift3/` subfolders in the output directory.

**When to use batch processing:**
- Analyzing multiple attempts of the same lift for consistency.
- Processing a session's worth of videos overnight.
- Comparing different lifters or techniques.

### Hardware Acceleration with OpenVINO

BARPATH can leverage Intel's OpenVINO toolkit for faster inference on Intel CPUs, providing 2-5x speed improvements over standard PyTorch models.

**How it works:**
- Export your YOLO model to OpenVINO format using Ultralytics:
  ```
  yolo export model=yolo11n.pt format=openvino
  ```
  This creates a directory like `yolo11n_openvino_model/` containing `.xml` and `.bin` files.

- Point BARPATH to this directory instead of the `.pt` file:
  - GUI: Select the OpenVINO directory in the model dropdown.
  - CLI: `--model yolo11n_openvino_model`

**Detection and Usage:**
- BARPATH automatically detects OpenVINO models by checking for directories with "openvino" in the name containing `.xml` and `.bin` files.
- On Intel CPUs, it uses the OpenVINO runtime for inference.
- If OpenVINO is not installed or the CPU is not Intel, it falls back to standard PyTorch/ONNX Runtime.
- No GPU support currently; OpenVINO here refers to CPU optimization.

**Requirements:**
- Intel CPU (detection is automatic).
- Install OpenVINO: `pip install openvino`
- Export model as above.

**Benefits:**
- Faster processing, especially for long videos.
- Lower CPU usage during inference.
- Compatible with existing BARPATH workflows.

**Verification:**
Check if OpenVINO is active:
```bash
python -c "import openvino; print('OpenVINO available')"
```

---

## Output Files

After running the pipeline, you'll find these files in the output directory:

### Summary

| File | Description | When Generated |
|------|-------------|-----------------|
| `raw_data.pkl` | Serialized tracking data (pose landmarks, barbell detections, stabilization) | Step 1 |
| `final_analysis.csv` | Processed data with kinematics, angles, barbell positions | Step 2 |
| `output.mp4` | Annotated video with skeleton overlay and bar path | Step 4 |
| `analysis.md` | Technique critique report | Step 5 |

### Graph Files

Located in `graphs/` subdirectory (generated in Step 3):

#### Standard Graphs (Always Generated)

- **`barbell_xy_stable_path.png`**
  - 2D bar path diagram (stabilized)
  - Colored by movement phase (up, down)
  - Shows true bar trajectory as detected
  - Same horizontal scale as corrected graph (for comparison)

- **`vel_y_smooth.png`**
  - Vertical velocity over time
  - Smoothed with Savitzky-Golay filter
  - Useful for identifying pull phases

- **`accel_y_smooth.png`**
  - Vertical acceleration over time
  - Shows force application timing
  - Peaks indicate maximum acceleration effort

- **`specific_power_y_smooth.png`**
  - Power-to-mass ratio proxy
  - Product of velocity × acceleration
  - High values indicate explosive phases

#### Perspective-Corrected Graph (When `lift_type != "none"`)

- **`barbell_lateral_corrected_path.png`**
  - 2D bar path with perspective correction applied
  - **Uses same horizontal scale as `barbell_xy_stable_path.png`** for direct comparison
  - Shows true lateral displacement if camera were perpendicular
  - Displays detected camera angle (reference frame)
  - Colored by movement phase
  - Difference between this and uncorrected path shows camera angle effect

### CSV Data Structure

`final_analysis.csv` contains the following key columns:

#### Position Data
- `barbell_x_smooth` - Horizontal position (pixels, stabilized)
- `barbell_y_smooth` - Vertical position (pixels, stabilized)
- `barbell_x_corrected_px` - Perspective-corrected horizontal position (when available)

#### Kinematics
- `vel_y_smooth` - Vertical velocity (px/s)
- `accel_y_smooth` - Vertical acceleration (px/s²)
- `specific_power_y_smooth` - Power-to-mass ratio proxy
- `bar_phase` - Movement phase (1 = up, -1 = down, 0 = transitional)

#### Stabilization
- `shake_dx`, `shake_dy` - Per-frame camera motion (pixels)
- `total_shake_x`, `total_shake_y` - Cumulative camera motion

#### Perspective Correction (when `lift_type != "none"`)
- `camera_yaw_deg` - Detected camera angle (constant across all frames)
- `lateral_correction_factor` - Applied correction multiplier

#### Body Angles
- `lifter_angle_deg` - Lifter's orientation relative to camera
- `left_knee_angle`, `right_knee_angle` - Knee flexion angle
- `left_elbow_angle`, `right_elbow_angle` - Elbow flexion angle

#### Joint Positions
- Normalized coordinates: `left_shoulder_x`, `left_shoulder_y`, etc.
- All normalized to [0, 1] range (multiply by frame dimensions for pixels)

### Video Overlay

The output video (`output.mp4`) includes:

- **Skeleton overlay**: Person's joints and connections (when `lift_type != "none"`)
- **Bar path**: Red line showing barbell trajectory
- **Grid**: Reference grid for spatial orientation
- **Statistics**: Frame number, timestamp, velocity, phase information

---

## Recording Best Practices

To get the best results from BARPATH, follow these recording guidelines:

### 1. Camera Position

**Ideal Setup:**
- **Angle**: 45° side view (perpendicular to lifter creates perspective errors, 45° is optimal)
- **Height**: Camera at hip/waist level
- **Distance**: 8-12 feet from lifter

**Why it matters:**
- Perspective angle compensation works best at 30-60°
- Hip-level height captures full body motion clearly
- Sufficient distance keeps entire body in frame

### 2. Camera Stability

**Requirements:**
- ✅ Use a tripod or stable surface
- ✅ Some minor camera shake is OK (system compensates)
- ✅ Keep camera still—no panning or zooming
- ❌ Avoid handheld recording
- ❌ Don't track the lifter

**Why it matters:**
- Camera motion is extracted and removed from measurements
- Excessive motion (> 30 pixels/frame) can overwhelm the stabilization algorithm
- Consistent camera position ensures angle detection works correctly

### 3. Visibility Requirements

**Essential:**
- ✅ Entire body visible from head to feet throughout the entire lift
- ✅ Nearest barbell endcap clearly visible
- ✅ No occlusions (people, equipment blocking view)
- ✅ Both shoulders visible for angle compensation

**Why it matters:**
- MediaPipe Pose needs to see the full body to detect landmarks
- YOLO barbell detector needs to see the endcap
- Occlusions cause tracking failures and data gaps
- Shoulder visibility is required for perspective correction

### 4. Lighting

**Best Practice:**
- ✅ Consistent, even lighting (avoid shadows across body)
- ✅ Natural daylight or studio lighting
- ✅ Bright enough to see details clearly
- ❌ Avoid backlighting (silhouettes)
- ❌ Avoid flickering lights

**Why it matters:**
- Poor lighting degrades landmark detection accuracy
- Shadows can be misidentified as joint positions
- Inconsistent lighting causes tracking jitter

### 5. Video Quality

**Recommended Settings:**
- **Resolution**: 1080p (1920×1080)
- **Frame Rate**: 30 fps (24 fps minimum)
- **Format**: MP4, MOV, MKV, WebM, or AVI
- **Codec**: H.264 (standard, widely compatible)

**Why it matters:**
- 1080p provides good detail without excessive processing time
- 30 fps captures motion dynamics adequately
- Lower resolution/fps may miss important details
- Higher resolution/fps increases processing time significantly

### 6. Example Setup

```
        Camera (tripod)
           /  45°
          /
         / ______
        /|      |
       / |      |
      /  | Lift |
     /   |______|
    /
   /___________
     Lifter

Position:
- Camera on tripod at hip height
- 8-12 feet away from lifter
- 45° angle to lifter's sagittal plane
- Stable, level surface
- Good lighting from the side
```

---

## Tips for Best Results

1. **Always start with `--no-video`** when testing a new video setup. It runs much faster and lets you see if detection is working before spending time on video rendering.

2. **Check the graphs first** - They show if the analysis worked well. Bad graphs indicate detection issues.

3. **Compare corrected and uncorrected paths** - The horizontal distance between `barbell_xy_stable_path.png` and `barbell_lateral_corrected_path.png` shows how much your camera angle affected the measurements.

4. **Use `lift_type clean` or `lift_type snatch`** to enable perspective correction and shoulder-based stabilization. Use `lift_type none` for quick analysis without pose landmarks.

5. **Check console output** for the detected camera angle. Typical values are 30-60°. If it shows 0° or >75°, the camera angle may be too steep or pose detection failed.

6. **Enable hardware acceleration** if available (OpenVINO, CUDA, etc.) for 2-5x speed improvement.

---

## Troubleshooting

### Runtime Errors

**"Error loading YOLO model"**
- ✅ Verify model path is correct
- ✅ Check model file is a valid `.pt` file (not a pointer)
- ✅ Ensure model was trained with Ultralytics YOLO
- ✅ Try a different model from `models/` directory

**"Could not detect barbell"**
- ✅ Ensure barbell endcap is visible in video
- ✅ Check class definition matches model, default models use class `endcap`

**"KeyError: 'barbell_center'"**
- This indicates barbell was not detected in any frame
- Solution: Check video quality and barbell visibility
- Fallback: Analysis still runs, but bar path will be missing

**"Missing required data columns"**
- Usually indicates MediaPipe pose detection failed
- ✅ Ensure lifter's full body is visible
- ✅ Check lighting conditions
- ✅ Verify no occlusions blocking the person

### Performance Issues

**Slow inference or video processing**
- ✅ Install hardware acceleration packages (see Installation section in README)
- ✅ Use a faster YOLO model: `yolo11s50e.pt` instead of `yolo11l60e.pt`
- ✅ Reduce video resolution before processing
- ✅ Use `--no-video` flag to skip video rendering

**To check if hardware acceleration is active:**
```bash
python -c "from barpath.pipeline import _get_yolo_device; print(_get_yolo_device())"
```
- Should show: `cpu` (CPU inference is used)

### Verifying Hardware Acceleration Installation

After installing acceleration packages, verify they're working:

```bash
# Check ONNX Runtime providers
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"

# Check OpenVINO installation (if installed)
python -c "import openvino; print('OpenVINO version:', openvino.__version__)"
```

Expected output for your hardware:
- **All platforms**: `['CPUExecutionProvider']`
- **Intel CPUs with OpenVINO**: OpenVINO available as separate runtime option

### FFmpeg Errors

**"Could not initialize video writer"**
- Check output directory exists and is writable
- Verify sufficient disk space
- Try a different output format (change file extension)