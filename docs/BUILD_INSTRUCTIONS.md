# BARPATH Build & Packaging Guide

This guide covers packaging barpath as a Python package and building standalone installers.

## Table of Contents

- [Running from Source (Recommended)](#running-from-source-recommended)
- [Installing as a Package](#installing-as-a-package)
- [Building Installers with Briefcase](#building-installers-with-briefcase)
- [Hardware Acceleration in Installers](#hardware-acceleration-in-installers)
- [Troubleshooting](#troubleshooting)

---

## Running from Source (Recommended)

The simplest way to run barpath is from a source checkout:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python barpath/barpath_cli.py --help
python barpath/barpath_gui.py
```

No build step is required.

## Installing as a Package

barpath is pip-installable. `pyproject.toml` declares the `[project]` metadata, dependencies, and the `barpath` / `barpath-gui` console scripts:

```bash
pip install .            # core dependencies
pip install .[hardware]  # adds ONNX Runtime + OpenVINO
pip install .[dev]       # pytest, ruff, basedpyright
```

After install, the `barpath` and `barpath-gui` entry points are available:

```bash
barpath --help
barpath-gui
```

## Building Installers with Briefcase

[Briefcase](https://briefcase.readthedocs.io/) packages Python GUI apps into native installers. To use it:

### Prerequisites

```bash
pip install briefcase
```

Platform tools:
- **Windows**: no additional setup.
- **macOS**: `xcode-select --install`.
- **Linux**: build tools and GTK dev packages.

### Build flow

```bash
briefcase create <platform>   # one-time scaffold (windows | macos | linux)
briefcase build <platform>
briefcase package <platform>
```

Output installers:
- **Windows**: `build/barpath/windows/msi/*.msi`
- **macOS**: `build/barpath/macos/dmg/*.dmg`
- **Linux**: `build/barpath/linux/deb/*.deb`

`build/` is gitignored, so artifacts are never committed.

### Configuring the app

Briefcase reads app configuration from `pyproject.toml`. This project has **not yet added** a `[tool.briefcase.app.barpath]` section; add one like the following before building:

```toml
[tool.briefcase.app.barpath]
formal_name = "Barpath - Weightlifting Analysis"
bundle = "com.scribewire"
version = "1.0.0"
description = "AI-powered biomechanical analysis for Olympic lifts"
sources = ['barpath']
icon = "barpath/assets/barpath"
```

## Hardware Acceleration in Installers

Options for shipping hardware-accelerated inference:

1. **During install**: run `python barpath/briefcase_hardware_installer.py` — detects hardware and prints the right `pip install` command.
2. **Extras**: `pip install barpath[hardware]`.
3. **Manual**: `pip install onnxruntime` (all platforms) and/or `pip install openvino` (Intel CPU).

See `requirements-hardware.txt` for the full matrix.

## Troubleshooting

**"Could not find Python installation"**
- Ensure Python is on PATH; use `python3.12 -m briefcase` instead of `briefcase`.

**"Missing dependencies"**
- `pip install -r requirements.txt` in your venv. Hardware packages are optional.

**Build fails with missing dev headers (Linux)**
- Debian/Ubuntu: `sudo apt-get install python3-dev build-essential`.

**Installer crashes on launch**
- Run the app from source first (`python barpath/barpath_gui.py`) to confirm it works; Briefcase packaging is still experimental.

---

**Questions?** Open an issue on the [BARPATH GitHub repository](https://github.com/scribewire/barpath).