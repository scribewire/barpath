# 🏋️ barpath - AI-Powered Weightlifting Technique Analysis

**barpath** is an advanced biomechanical analysis tool that acts as your virtual weightlifting coach. Using computer vision and pose estimation, it analyzes Olympic lifts (clean, snatch, jerk) to provide detailed kinematic feedback, visualizations, and technique critiques.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

## ✨ Features

- **🎯 Camera Shake Stabilization**: Uses Lucas-Kanade optical flow on background features to create perfectly stabilized bar path tracking
- **📐 3D Orientation Detection**: Automatically detects lifter orientation (90° side view vs. 45° corner view) using MediaPipe's pseudo-depth (z-coordinate)
- **📊 Comprehensive Kinematic Analysis**:
  - Vertical velocity, acceleration, jerk, and specific power graphs
  - Frame-by-frame joint angle measurements (knees, elbows, hips)
  - Temporal analysis of movement phases
- **🎥 Annotated Video Output**: Skeleton overlay with stabilized bar path visualization
- **🔍 Rule-Based Technique Critique**: Identifies common faults in Olympic lifts:
  - Early arm bend
  - Incomplete extension
  - Poor timing
  - Catching errors

## 🏗️ Proposed Project Structure (not yet complete)

```
barpath/
├── README.md                          # Main documentation
├── LICENSE
├── .gitignore
├── .gitattributes                     # For git-lfs (YOLO models)
├── requirements.txt                   # Core dependencies
├── setup.py                          # Package installation
│
├── barpath/                          # Core package
│   ├── __init__.py
│   ├── collect_data.py               # Refactored from 1_collect_data.py
│   ├── analyze_data.py               # Refactored from 2_analyze_data.py
│   ├── generate_graphs.py            # Refactored from 3_generate_graphs.py
│   ├── render_video.py               # Refactored from 4_render_video.py
│   ├── critique_lift.py              # Refactored from 5_critique_lift.py
│   └── utils.py                      # Shared utilities
│
├── cli/                              # Command-line interface
│   ├── __init__.py
│   └── barpath_cli.py                # CLI entry point (current barpath.py)
│
├── gui/                              # Graphical interface
│   ├── __init__.py
│   ├── barpath_gui.py                # Main GUI application
│   ├── requirements.txt              # GUI-specific dependencies (PyQt6/tkinter)
│   └── assets/                       # GUI assets (icons, images)
│       ├── icon.png
│       └── logo.png
│
├── models/                     # Pre-trained YOLO models
│   ├── yolo11s50e.pt      # Small (fast)
│   ├── yolo11m50e.pt      # Medium (recommended)
│   └── yolo11l60e.pt      # Large (high accuracy)
│
├── examples/                         # Example videos and outputs
│   ├── sample_clean.mp4
│   └── expected_output/
│
├── docs/                             # Documentation
│   ├── QUICK_START.md
│   ├── CLI_GUIDE.md
│   └── GUI_GUIDE.md
│
└── tests/                            # Unit tests
    ├── test_collect_data.py
    ├── test_analyze_data.py
    └── test_critique_lift.py
```

## 🔧 Requirements

### System Dependencies

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| **Python 3.8+** | Runtime environment | [python.org](https://www.python.org/downloads/) |
| **FFmpeg** | Video processing | See below |
| **Git LFS** | Large file support | See below |

**FFmpeg Installation:**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

**Git LFS Installation** (required to clone models):
```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt-get install git-lfs

# Windows
# Download from https://git-lfs.github.com/

# Initialize (run once)
git lfs install
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

**Key libraries:**
- `opencv-python` (≥4.8.0) - Video processing and computer vision
- `mediapipe` (≥0.10.0) - Human pose estimation (33 landmarks)
- `ultralytics` (≥8.0.0) - YOLOv11 object detection
- `pandas` (≥2.0.0) - Data analysis and manipulation
- `numpy` (≥1.24.0) - Numerical computing
- `matplotlib` (≥3.7.0) - Graph generation
- `scipy` (≥1.10.0) - Signal processing (smoothing)
- `tqdm` (≥4.65.0) - Progress bars

## 📦 Installation

### 1. Clone the Repository

```bash
# Clone with Git LFS (downloads models automatically)
git clone https://github.com/yourusername/barpath.git
cd barpath
```

**Important:** If you already have Git installed but models aren't downloading:
```bash
git lfs install
git lfs pull
```

### 2. Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg python3-pip git-lfs

# macOS
brew install ffmpeg git-lfs python3
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Check barpath CLI
python barpath.py --help

# Verify models downloaded (should be ~20-50 MB each, not tiny)
ls -lh models/*.pt
```

If model files show as only a few KB, Git LFS didn't work. Run:
```bash
git lfs pull
```

## 🚀 Quick Start

### Basic Analysis

```bash
python barpath.py \
  --input_video "path/to/your/clean.mp4" \
  --model "models/yolo11s-barbell.pt" \
  --output_video "output.mp4" \
  --lift_type clean
```

### Fast Analysis (Skip Video Rendering)

For quick feedback without rendering the annotated video:

```bash
python barpath.py \
  --input_video "my_lift.mp4" \
  --model "models/yolo11s-barbell.pt" \
  --output_video "final.mp4" \
  --lift_type clean \
  --no-video
```

This generates graphs and critique in seconds, perfect for rapid iteration.

## 📖 Usage

### Command Line Options

```
Required Arguments:
  --input_video PATH         Path to source video file
                            (e.g., 'videos/clean.mp4')
  
  --model PATH              Path to trained YOLO model file
                            (e.g., 'models/yolo11s-barbell.pt')
  
  --output_video PATH       Path to save annotated video
                            (e.g., 'output/final.mp4')

Optional Arguments:
  --lift_type {clean,none}  Type of lift to analyze
                            'clean' - Power clean analysis
                            'none'  - Skip technique critique
                            Default: none
  
  --no-video                Skip Step 4 (video rendering)
                            Saves 60-80% processing time
                            Graphs and critique still generated
```

### Available Models

| Model File | Size | Speed | Accuracy | Use Case |
|------------|------|-------|----------|----------|
| `yolo11n-barbell.pt` | ~7 MB | Very Fast | Good | Testing, quick analysis |
| `yolo11s-barbell.pt` | ~22 MB | Fast | Better | **Recommended for general use** |
| `yolo11m-barbell.pt` | ~52 MB | Medium | Best | High-accuracy requirements |
| `best.pt` | Varies | - | - | Your custom-trained model |

**Recommendation:** Start with `yolo11s-barbell.pt` for the best balance of speed and accuracy.

### Running Individual Pipeline Steps

For debugging or custom workflows, run steps independently:

```bash
# Step 1: Collect raw tracking data
python 1_collect_data.py \
  --input video.mp4 \
  --model models/yolo11s-barbell.pt \
  --output raw_data.pkl

# Step 2: Analyze kinematics and angles
python 2_analyze_data.py \
  --input raw_data.pkl \
  --output final_analysis.csv

# Step 3: Generate kinematic graphs
python 3_generate_graphs.py \
  --input final_analysis.csv \
  --output_dir graphs

# Step 4: Render annotated video
python 4_render_video.py \
  --input_csv final_analysis.csv \
  --input_video video.mp4 \
  --output_video final.mp4

# Step 5: Generate technique critique
python 5_critique_lift.py \
  --input final_analysis.csv \
  --lift_type clean
```

## 📂 Output Files

After running the pipeline, you'll find:

### Generated Files

| File | Description |
|------|-------------|
| `raw_data.pkl` | Serialized tracking data (pose landmarks, barbell detections, optical flow) |
| `final_analysis.csv` | Processed data with kinematics, angles, and stabilized coordinates |
| `graphs/` | Directory containing kinematic plots |
| `output.mp4` | Annotated video with skeleton and bar path overlay (if `--no-video` not used) |

### Graph Files (in `graphs/` directory)

- `vel_y_px_s_graph.png` - Vertical velocity over time
- `accel_y_px_s2_graph.png` - Vertical acceleration over time
- `jerk_y_px_s3_graph.png` - Vertical jerk (rate of acceleration change)
- `specific_power_y_graph.png` - Specific power (acceleration × velocity)

### Console Output

The technique critique is printed to the console with:
- Identified technical issues
- Severity levels (Major, Moderate, Minor)
- Specific recommendations for improvement

## 🎥 Recording Best Practices

For optimal tracking results:

### 1. Camera Position
- **Ideal**: 90° side view (perpendicular to bar)
- **Acceptable**: 20-45° offset from side
- **Height**: Camera at hip level
- **Distance**: 2-4 meters from lifter
- **Framing**: Lifter fills 70-90% of frame vertically

### 2. Camera Stability
- ✅ Use a tripod or stable surface
- ✅ Some camera shake is OK (pipeline compensates)
- ❌ Avoid handheld recording
- ❌ Don't pan or zoom during lift

### 3. Visibility Requirements
- ✅ Entire body visible throughout lift (head to feet)
- ✅ At least **one barbell endcap** clearly visible
- ✅ No occlusions (people, equipment in foreground)
- ✅ Consistent lighting (no shadows obscuring body)

### 4. Video Quality
- **Resolution**: 1080p minimum, 4K ideal
- **Frame Rate**: 30 fps minimum, 60 fps recommended
- **Format**: MP4, MOV, or AVI
- **Compression**: Avoid heavy compression (quality > file size)
- **Lighting**: Bright, even lighting (avoid backlighting)

## 🔬 Technical Architecture

### 5-Step Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Video File (.mp4, .mov, .avi)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────────┐
    │  STEP 1: Data Collection                       │
    │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
    │  • MediaPipe Pose (33 landmarks)              │
    │  • YOLO Barbell Detection                     │
    │  • Optical Flow (camera shake estimation)     │
    │  Output: raw_data.pkl                         │
    └────────────────┬───────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────────┐
    │  STEP 2: Data Analysis                         │
    │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
    │  • Angle calculations (knees, elbows, hips)   │
    │  • Kinematic derivatives (v, a, j, power)     │
    │  • Coordinate stabilization                   │
    │  Output: final_analysis.csv                   │
    └────────────────┬───────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────────┐
    │  STEP 3: Graph Generation                      │
    │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
    │  • Matplotlib visualization                    │
    │  • 4 kinematic graphs (velocity, accel, etc.) │
    │  Output: graphs/*.png                         │
    └────────────────┬───────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────────┐
    │  STEP 4: Video Rendering (Optional)            │
    │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
    │  • Skeleton overlay                            │
    │  • Stabilized bar path visualization          │
    │  • Joint angle annotations                     │
    │  Output: output.mp4                            │
    └────────────────┬───────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────────┐
    │  STEP 5: Technique Critique                    │
    │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
    │  • Rule-based fault detection                  │
    │  • Issue severity classification               │
    │  • Actionable recommendations                  │
    │  Output: Console report                        │
    └────────────────────────────────────────────────┘
```

### Key Algorithms

| Component | Algorithm | Purpose |
|-----------|-----------|---------|
| **Pose Estimation** | MediaPipe BlazePose | 33 3D landmarks + segmentation mask |
| **Object Detection** | YOLOv11 | Barbell endcap detection |
| **Stabilization** | Lucas-Kanade Optical Flow | Background feature tracking for shake removal |
| **Bar Selection** | Proximity-based | Selects endcap nearest to detected hand |
| **Angle Calculation** | Vector dot product | 3-point joint angle measurement |
| **Kinematics** | Central difference | Numerical derivatives (velocity, acceleration, jerk) |
| **Power Estimation** | Specific power | Power-to-mass ratio proxy (a × v) |

### Data Flow

```
Video Frame → [MediaPipe] → 33 Pose Landmarks (x, y, z, visibility)
           ↓
           → [YOLO] → Barbell Endcap Bounding Boxes
           ↓
           → [Optical Flow] → Camera Shake Vectors (dx, dy)
           ↓
           → [Analysis] → Angles, Kinematics, Stabilized Path
           ↓
           → [Critique] → Technical Fault Report
```

## 🐛 Troubleshooting

### Installation Issues

**"ModuleNotFoundError: No module named 'cv2'"**
```bash
pip install opencv-python
```

**"ModuleNotFoundError: No module named 'mediapipe'"**
```bash
pip install mediapipe
```

**Models are only a few KB (pointer files)**
```bash
# Git LFS didn't download models
git lfs install
git lfs pull
```

### Runtime Errors

**"Error loading YOLO model"**
- ✅ Verify model path is correct
- ✅ Check model file is a valid `.pt` file (not a pointer)
- ✅ Ensure model was trained with Ultralytics YOLOv11
- ✅ Try a different model from `models/` directory

**"Could not detect barbell"**
- ✅ Ensure barbell endcap is visible in video
- ✅ Check lighting and contrast
- ✅ Verify camera angle (side view recommended)
- ✅ Try a higher-accuracy model

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

**Video rendering is very slow**
- This is expected - Step 4 processes every frame
- **Solution 1**: Use `--no-video` flag (saves 60-80% time)
- **Solution 2**: Use lower resolution video for testing
- **Solution 3**: Run on a machine with better CPU/GPU

**Analysis takes longer than expected**
- Check video length 
- MediaPipe and YOLO are computationally intensive
- Consider shorter clips for testing

### FFmpeg Errors

**"Error: Could not open video file"**
```bash
# Verify FFmpeg is installed
ffmpeg -version

# On Windows, ensure FFmpeg is in PATH
# On Linux/Mac, reinstall if needed
sudo apt-get install ffmpeg  # Ubuntu
brew install ffmpeg          # macOS
```

**"Could not initialize video writer"**
- Check output directory exists and is writable
- Verify sufficient disk space
- Try a different output format (change file extension)

## 📊 Project Status

**Current Status: Alpha (v0.9)**

### ✅ Implemented
- Complete 5-step pipeline
- Camera shake stabilization
- Clean lift critique engine
- Multi-model support (nano/small/medium)
- Command-line interface

### 🚧 In Development
- Graphical user interface (GUI)
- Additional lift types (snatch, jerk)
- Advanced critique rules
- Real-time analysis mode

### 🔮 Planned Features
- Mobile app (iOS/Android)
- Cloud processing option
- Athlete progress tracking
- Comparative analysis (vs. elite lifters)
- Export to coaching platforms

### Known Limitations
- Only "clean" lift fully supported for critique
- Requires stable camera position
- Barbell endcap must be visible
- Single-person tracking only
- No real-time processing (yet)

## 🤝 Contributing

This project is in active development. Contributions welcome!

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution
- [ ] Additional lift type critiques (snatch, jerk)
- [ ] GUI development (PyQt6 or Tkinter)
- [ ] Improved YOLO training datasets
- [ ] Documentation and tutorials
- [ ] Bug fixes and error handling
- [ ] Unit tests and CI/CD

## 📄 License

This project is licensed under the GPL-v3 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with amazing open-source tools:

- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** - State-of-the-art object detection
- **[MediaPipe](https://google.github.io/mediapipe/)** - Real-time pose estimation by Google
- **[OpenCV](https://opencv.org/)** - Computer vision and video processing
- **[pandas](https://pandas.pydata.org/)** - Data analysis and manipulation
- **[matplotlib](https://matplotlib.org/)** - Visualization and graphing


- **Issues**: [GitHub Issues](https://github.com/scribewire/barpath/issues)

---

**Made with ❤️ for weightlifters, by weightlifters**