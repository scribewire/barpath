<img src="barpath/assets/barpath.svg" alt="Logo" width = "300" />

# BARPATH: AI-Powered Weightlifting Technique Analysis

**barpath** is an advanced biomechanical analysis tool that acts as a powerful training tool. Using computer vision and pose estimation, it analyzes Olympic lifts (clean, snatch, jerk) to provide detailed kinematic feedback, visualizations, and technique critiques.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
    <img src="barpath/assets/sample_video.gif" alt="Sample Output Video";>
    <img src="barpath/assets/sample_graph.png" alt="Sample Output Graph" height="360">
</div>

## ✨ Features

- **🖥️ Dual Interface**: Command-line tool for batch processing and GUI for interactive analysis
  - **GUI**: Modern tabbed interface with Files, Settings, Analyze, and Analysis tabs
  - **CLI**: Script-friendly command-line tool for batch processing
- **🎯 Camera Shake Stabilization**: Uses Lucas-Kanade optical flow on background features to create perfectly stabilized bar path tracking
- **📐 3D Orientation Detection**: Automatically detects lifter orientation using MediaPipe's pseudo-depth (z-coordinate)
- **⚙️ Hardware-Accelerated Inference**: CPU-optimized inference with optional acceleration:
  - ONNX Runtime for cross-platform CPU optimization
  - OpenVINO support for Intel CPUs
- **📊 Comprehensive Kinematic Analysis**:
  - Smoothed bar position graph for technical analysis
  - Data automatically truncated at peak height
  - Frame-by-frame joint angle measurements (knees, elbows, hips)
  - Temporal analysis of movement phases
- **🎥 Annotated Video Output**: 
  - Skeleton overlay with stabilized bar path visualization
  - Color-coded bar path phases (first/second/third pull)
  - Persistent barpath overlay at the end of the lift for easier review
- **📋 Beautiful Analysis Reports**: Markdown-based reports rendered as formatted HTML
  - Kinematic data and graphs
  - Technique findings and recommendations
  - Automatically displayed in the GUI Analysis tab
- **🔍 Rule-Based Technique Critique**: Identifies common faults in Olympic lifts:
  - Early arm bend
  - Incomplete extension
  - Poor timing
  - and more!

## 🔧 Requirements

### System Dependencies

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| **Python 3.12+** | Runtime environment | [python.org](https://www.python.org/downloads/) |
| **FFmpeg** | Video processing | See below |
| **Git LFS** | Large file support | See below |

Python packages required by barpath are listed in `requirements.txt`.

## 📦 Installation

### 1. Install System Dependencies

Ubuntu
```bash
sudo apt update
sudo apt install ffmpeg python3-pip git git-lfs libcairo2-dev pkg-config libgirepository-2.0-dev gir1.2-gtk-3.0 libgirepository-2.0-0
```
macOS
```
brew install ffmpeg git git-lfs python
```
On Windows, install:  
[git](https://github.com/git-guides/install-git%20#install-git-on-windows) 
[ffmpeg](https://ffmpeg.org/download.html) 
[python](https://www.python.org/downloads/windows)

### 2. Clone the Repository

```bash
# Clone with Git LFS (downloads models automatically)
git clone https://github.com/scribewire/barpath
cd barpath
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs the core pipeline libraries and the Toga GUI dependency. 

### 3.5. Optional: Install Hardware Acceleration (Recommended)

barpath can use hardware-accelerated inference for faster model processing. The specific packages depend on your OS and hardware.

#### Automatic Setup (Interactive, WIP)

Run the interactive hardware detector:

```bash
python barpath/briefcase_hardware_installer.py
```

This will:
1. Detect your OS and CPU brand
2. Show available acceleration options for your hardware
3. Prompt you to select which packages to install
4. Provide the exact pip command to run

#### Manual Setup

See `requirements-hardware.txt` for all available options, or install based on your hardware:

**Windows**
- `pip install onnxruntime`
- Intel CPU: `pip install onnxruntime openvino` (optional, adds Intel optimization)

**macOS**
- `pip install onnxruntime`

**Linux**
- `pip install onnxruntime`
- Intel CPU: `pip install onnxruntime openvino` (optional, adds Intel optimization)

#### Using setup.py extras

If you installed barpath via `setup.py`, you can install hardware acceleration with:

```bash
pip install .[hardware]      # Install all recommended for your hardware
pip install .[onnx]          # Install ONNX acceleration only
pip install .[openvino]      # Install OpenVINO only
```

### 4. Verify Installation

```bash
# Check barpath CLI
python barpath/barpath_cli.py --help

# Verify models downloaded (should be ~20-50 MB each, not tiny)
ls -lh barpath/models/*.pt

# (Optional) Verify hardware acceleration is available
python -c "from barpath.hardware_detection import get_hardware_profile, get_optional_packages; p=get_hardware_profile(); print('Hardware Profile:', p); o,v=get_optional_packages(p); print('Recommended packages:', o+v)"
```

### 5. Launch the GUI

Once dependencies are installed, you can run the desktop GUI:

```bash
python barpath/barpath_gui.py
```

## 🚀 Quick Start

### GUI (Recommended)

```bash
python barpath/barpath_gui.py
```

Then:
1. **Files Tab** → Select video(s) and output directory
2. **Settings Tab** → Choose model and lift type
3. **Analyze Tab** → Run analysis and monitor progress
4. **Analysis Tab** → View the generated report

### Command Line

```bash
python barpath/barpath_cli.py \
  --input_video "lift.mp4" \
  --model "models/yolo11s.pt" \
  --lift_type clean
```

For comprehensive usage instructions, including detailed GUI workflow, CLI options, examples, and running individual pipeline steps, see the [**USAGE_GUIDE.md**](docs/USAGE_GUIDE.md).

## 📦 Building Installers

For comprehensive instructions on building standalone installers for Windows, macOS, and Linux using Briefcase, see the [**BUILD_INSTRUCTIONS.md**](docs/BUILD_INSTRUCTIONS.md).

## 📊 Project Status

**Current Status: Alpha**

### ✅ Recently Completed
- **Automatic Lift Truncation** - The beginning and end of the lift is automatically detected (clean only so far)
- **Angle compensation** - When using mediapipe pose estimation, angle compensation is applied to improve accuracy

### 🔮 Planned Features
- Cloud processing option
- Athlete progress tracking
- Comparative analysis (vs. elite lifters)
- Live preview to test models

### Known Limitations
- Only "clean" lift fully supported for critique
- Requires stable camera position
- Barbell endcap must be visible for whole lift
- No real-time processing

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
- Barbell detection trained on:
    - Our dataset: Bar path (2025) bar path detection unified (v6) [Dataset]. Roboflow. [Source](https://universe.roboflow.com/bar-path/bar-path-detection-unified-cyusm/dataset/6). Accessed 16 December 2025.
    - Which contains: barbelldetection (2024) barbell detection (v2) [Dataset]. Roboflow. [Source](https://universe.roboflow.com/barbelldetection-8kean/barbell-detection-gjsrc/dataset/2). Accessed 4 December 2025

---

**Made with ❤️ for weightlifters, by weightlifters**
