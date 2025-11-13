<img src="barpath/gui/assets/barpath_logo.svg" alt="Logo" width = "300" />

# BARPATH: AI-Powered Weightlifting Technique Analysis

**barpath** is an advanced biomechanical analysis tool that acts as a powerful training tool. Using computer vision and pose estimation, it analyzes Olympic lifts (clean, snatch, jerk) to provide detailed kinematic feedback, visualizations, and technique critiques.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

## ✨ Features

- **🎯 Camera Shake Stabilization**: Uses Lucas-Kanade optical flow on background features to create perfectly stabilized bar path tracking
- **📐 3D Orientation Detection**: Automatically detects lifter orientation using MediaPipe's pseudo-depth (z-coordinate)
- **📊 Comprehensive Kinematic Analysis**:
  - Vertical velocity, acceleration, and bar path graphs
  - Frame-by-frame joint angle measurements (knees, elbows, hips)
  - Temporal analysis of movement phases
- **🎥 Annotated Video Output**: Skeleton overlay with stabilized bar path visualization
- **🔍 Rule-Based Technique Critique**: Identifies common faults in Olympic lifts:
  - Early arm bend
  - Incomplete extension
  - Poor timing
  - Catching errors
  - and more!

## 🏗️ Proposed Project Structure (not yet complete)

```
barpath/
├── README.md                         # Main documentation
├── LICENSE
├── .gitignore
├── .gitattributes                    # For git-lfs (YOLO models)
├── requirements.txt                  # Core dependencies
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
├── models/                           # Pre-trained YOLO models
│   ├── yolo11s50e.pt                 # Small (fast)
│   ├── yolo11m50e.pt                 # Medium (recommended)
│   └── yolo11l60e.pt                 # Large (high accuracy)
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

## 📦 Installation

### 1. Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg python3-pip git-lfs

# macOS
brew install ffmpeg git-lfs python3

# For Windows, install:
# git-lfs https://git-lfs.github.com/
# ffmpeg https://ffmpeg.org/download.html
# python https://ffmpeg.org/download.html
```

### 2. Clone the Repository

```bash
# Clone with Git LFS (downloads models automatically)
git clone https://github.com/yourusername/barpath.git
cd barpath
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Check barpath CLI
python barpath/cli/barpath_cli.py --help

# Verify models downloaded (should be ~20-50 MB each, not tiny)
ls -lh barpath/models/*.pt
```

## 🚀 Quick Start

### Analysis

```bash
python barpath/cli/barpath_cli.py \
  --input_video "path/to/your/clean.mp4" \
  --model "barpath/models/yolo11m50e.pt" \
  --output_video "output.mp4" \
  --lift_type clean
```

For quick feedback without rendering the annotated video, use the --no-video option  
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
| `yolo11s50e.pt` | ~19 MB | Fast | Good | Testing, quick analysis |
| `yolo11m50e.pt` | ~40 MB | Medium | Better | **Recommended for general use** |
| `yolo11l60e.pt` | ~50 MB | Slow | Best | High-accuracy requirements |

**Recommendation:** Start with `yolo11m50e.pt` for the best balance of speed and accuracy.

### Running Individual Pipeline Steps

For debugging or custom workflows, run steps independently:

```bash
# Step 1: Collect raw tracking data
python barpath/1_collect_data.py \
  --input video.mp4 \
  --model barpath/models/yolo11m50e.pt \
  --output raw_data.pkl

# Step 2: Analyze kinematics and angles
python barpath/2_analyze_data.py \
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

- `barbell_xy_stable_path.png` - 2D barbell path graph
- `vel_y_px_s_graph.png` - Vertical velocity over time (raw output)
- `vel_y_smooth_graph.png` - Smoothed velocity graph

### Console Output

The technique critique is printed to the console with:
- Identified technical issues
- Severity levels (Major, Moderate, Minor)
- Specific recommendations for improvement

## 🎥 Recording Best Practices

For optimal tracking results:

### 1. Camera Position
- **Ideal**: 45° side view
- **Height**: Camera at hip level

### 2. Camera Stability
- ✅ Use a tripod or stable surface
- ✅ Some camera shake is OK (pipeline compensates)
- ❌ Avoid handheld recording
- ❌ Don't pan or zoom during lift

### 3. Visibility Requirements
- ✅ Entire body visible throughout lift (head to feet)
- ✅ Nearest **barbell endcap** clearly visible
- ✅ No occlusions (people, equipment in foreground)
- ✅ Consistent lighting

### 4. Video Quality
- For a good balance between quality and processing speed:
    - **Resolution**: 1080p recommended
    - **Frame Rate**: 30 fps recommended
- **Format**: MP4, MOV, mkv, webm, or AVI

## 🐛 Troubleshooting

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

### FFmpeg Errors

**"Could not initialize video writer"**
- Check output directory exists and is writable
- Verify sufficient disk space
- Try a different output format (change file extension)

## 📊 Project Status

**Current Status: Alpha (v0.9)**

### 🚧 In Development
- Graphical user interface (GUI)
- Additional lift types (snatch, jerk)
- Advanced critique rules
- Option to select video segment for analysis

### 🔮 Planned Features
- Cloud processing option
- Athlete progress tracking
- Comparative analysis (vs. elite lifters)
- Export to coaching platforms

### Known Limitations
- Only "clean" lift fully supported for critique
- Requires stable camera position
- Barbell endcap must be visible
- No real-time processing (yet)

## 🤝 Contributing

This project is in active development. Contributions welcome!

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

Built with amazing open-source tools:

- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** - State-of-the-art object detection
- **[MediaPipe](https://google.github.io/mediapipe/)** - Real-time pose estimation by Google
- **[OpenCV](https://opencv.org/)** - Computer vision and video processing
- **[pandas](https://pandas.pydata.org/)** - Data analysis and manipulation
- **[matplotlib](https://matplotlib.org/)** - Visualization and graphing
---

**Made with ❤️ for weightlifters, by weightlifters**
