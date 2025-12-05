# BARPATH Build Instructions

This guide covers building standalone installers for BARPATH using [Briefcase](https://briefcase.readthedocs.io/).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Building Installers](#building-installers)
- [Build Output](#build-output)
- [Configuration](#configuration)
- [Hardware Acceleration in Installers](#hardware-acceleration-in-installers)
- [Advanced Topics](#advanced-topics)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Install Briefcase

```bash
pip install briefcase
```

### 2. Platform-Specific Requirements

**Windows:**
- No additional setup needed

**macOS:**
- Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```

**Linux:**
- Build tools and development dependencies (varies by distribution)

---

## Building Installers

BARPATH can be packaged as standalone installers for Windows, macOS, and Linux.

### Quick Start: All-In-One Build

For Windows:
```bash
briefcase create windows
briefcase build windows
briefcase package windows
```

For macOS:
```bash
briefcase create macos
briefcase build macos
briefcase package macos
```

For Linux:
```bash
briefcase create linux
briefcase build linux
briefcase package linux
```

### Step-by-Step Build Process

#### Step 1: Create the App (First Time Only)

```bash
# Create the Briefcase app structure for your target platform
briefcase create windows      # or macos, linux
```

This sets up the app directory structure needed for building. You only need to run this once. Subsequent builds can skip this step.

#### Step 2: Build the Application

```bash
# Build the executable/app bundle
briefcase build windows       # or macos, linux
```

The built app will be in `build/barpath/windows/app/` (or your platform equivalent).

#### Step 3: Package as an Installer

```bash
# Create the installer package
briefcase package windows     # or macos, linux
```

This creates platform-specific installers:
- **Windows**: `.msi` file in `build/barpath/windows/msi/`
- **macOS**: `.dmg` file in `build/barpath/macos/dmg/`
- **Linux**: `.deb` file in `build/barpath/linux/deb/`

#### Step 4: Run the Installer

After packaging, users can install with:

**Windows:**
- Double-click the `.msi` file

**macOS:**
- Double-click the `.dmg` file
- Drag the app to Applications folder

**Linux:**
```bash
sudo dpkg -i build/barpath/linux/deb/barpath*.deb
```

### Full Build Workflow Example

```bash
# All-in-one build (for Windows)
briefcase create windows
briefcase build windows
briefcase package windows

# Find your installer in:
# build/barpath/windows/msi/barpath-1.0.0.msi
```

---

## Build Output

### Directory Structure

```
build/
├── barpath/
│   ├── windows/
│   │   ├── app/              # Built application files
│   │   └── msi/              # Windows installer (.msi)
│   ├── macos/
│   │   ├── app/              # Built application bundle
│   │   └── dmg/              # macOS installer (.dmg)
│   └── linux/
│       ├── app/              # Built application files
│       └── deb/              # Linux installer (.deb)
```

**Note**: The `build/` directory is already in `.gitignore`, so installers won't be committed to the repository.

### Installer Sizes

Typical installer sizes for Python GUI applications:
- **Windows (.msi)**: 500 MB - 1 GB
- **macOS (.dmg)**: 500 MB - 1 GB
- **Linux (.deb)**: 500 MB - 1 GB

Large size is normal and includes:
- Python runtime
- All dependencies (PyTorch, MediaPipe, OpenCV, etc.)
- Application code and assets

---

## Configuration

### Briefcase Configuration Files

Briefcase uses `pyproject.toml` or `setup.cfg` for configuration.

### Key Settings for BARPATH

```toml
[tool.briefcase.app.barpath]
formal_name = "Barpath - Weightlifting Analysis"
bundle = "com.scribewire"
version = "1.0.0"
description = "AI-powered biomechanical analysis for Olympic lifts"
sources = ['barpath']
icon = "barpath/assets/barpath"

# Windows-specific
[tool.briefcase.app.barpath.windows]
installer_icon = "barpath/assets/barpath.ico"

# macOS-specific  
[tool.briefcase.app.barpath.macos]
universal_build = false
requires = [
    "PyYAML>=5.3.1",
]
```

### Customizing the Installer

To customize the installer, edit your `pyproject.toml`:

1. **Application name**: `formal_name`
2. **Bundle identifier**: `bundle`
3. **Version**: `version`
4. **Description**: `description`
5. **Icon**: `icon` (points to image file)

Example customization:
```toml
[tool.briefcase.app.barpath]
formal_name = "My Custom Barpath"
version = "2.0.0"
description = "Custom biomechanical analysis tool"
```

---

## Hardware Acceleration in Installers

The Briefcase installer can include hardware acceleration support for end users.

### Option 1: During Installation

Users can run the hardware setup wizard after installation:

```bash
python -m barpath.briefcase_hardware_installer
```

This interactive wizard helps users install appropriate acceleration packages for their hardware.

### Option 2: Using setup.py Extras

After installation, users can add acceleration packages:

```bash
pip install barpath[hardware]
```

### Option 3: Manual Selection

Users can manually select hardware packages from `requirements-hardware.txt`:

```bash
pip install -r requirements-hardware.txt
```

### What Gets Installed

Depending on the user's hardware:
- **All platforms**: ONNX Runtime (already included)
- **Intel CPUs**: OpenVINO support (optional)
- **NVIDIA GPUs**: CUDA support (optional)

---

## Advanced Topics

### Cross-Platform Building

To build for multiple platforms, run Briefcase on each operating system:

```bash
# On Windows machine
briefcase create windows && briefcase build windows && briefcase package windows

# On macOS machine
briefcase create macos && briefcase build macos && briefcase package macos

# On Linux machine
briefcase create linux && briefcase build linux && briefcase package linux
```

Alternatively, use CI/CD (GitHub Actions, etc.) to automate cross-platform builds.

### Updating Installers

When you update barpath code:

```bash
# Just rebuild and repackage (no need to create again)
briefcase build windows
briefcase package windows
```

Briefcase will detect changes and rebuild only what's necessary.

### Signing Installers (Advanced)

For production releases, you may want to code-sign installers:

**Windows (.msi):**
- Use a code-signing certificate from a trusted CA
- Configure signing in Briefcase settings

**macOS (.dmg):**
- Use Apple Developer certificate
- Configure signing in Briefcase settings

**Linux (.deb):**
- Sign with GPG key
- Configure signing in Briefcase settings

Consult [Briefcase documentation](https://briefcase.readthedocs.io/) for detailed signing instructions.

---

## Troubleshooting

### "Could not find Python installation"

**Problem**: Briefcase can't locate Python executable.

**Solutions:**
- Ensure Python is in your system PATH
- Use `python -m briefcase` instead of `briefcase` command
- Verify Python installation: `python --version`

### "Missing dependencies"

**Problem**: Build fails due to missing Python packages.

**Solutions:**
- All dependencies in `requirements.txt` are automatically included
- Hardware packages must be added manually or via the hardware installer script
- Verify requirements: `pip list`

### "Icon not found"

**Problem**: Build fails because icon file is missing.

**Solutions:**
- Ensure `barpath/assets/barpath.png` exists
- For Windows .msi, also create or convert to `.ico` format
- Icon should be at least 256×256 pixels

**Creating a Windows .ico file:**
```bash
# Using PIL (install with: pip install pillow)
from PIL import Image
img = Image.open("barpath/assets/barpath.png")
img.save("barpath/assets/barpath.ico")
```

### Large Installer Size

**Problem**: Installer is very large (>500 MB).

**Causes:**
- Normal for Python GUI apps with heavy dependencies
- Includes Python runtime (~100-150 MB)
- Includes PyTorch, MediaPipe, OpenCV

**Potential optimizations:**
- Consider using compression in Briefcase settings
- Remove unused dependencies from `requirements.txt`
- Use lighter model files if available

### Build Fails on Linux

**Problem**: Build fails with missing development headers.

**Solutions:**
- Install build tools:
  ```bash
  # Debian/Ubuntu
  sudo apt-get install python3-dev build-essential

  # Red Hat/Fedora
  sudo yum install python3-devel gcc gcc-c++

  # Arch
  sudo pacman -S base-devel
  ```

### macOS Code Signing Errors

**Problem**: `Signature invalid` or `certificate not found` errors.

**Solutions:**
- For development: Briefcase handles code signing automatically
- For distribution: Use an Apple Developer certificate
- Verify Xcode installation: `xcode-select --install`

### Build Succeeds but Installer Crashes on Launch

**Problem**: Installer runs but application fails to start.

**Solutions:**
- Check log files in the application directory
- Run the app from command line to see error output
- Verify all dependencies are included in `requirements.txt`
- Test with `briefcase build` and `briefcase run` before packaging

### FFmpeg Not Found

**Problem**: Video processing fails in the packaged app.

**Solutions:**
- Add `ffmpeg-python` to `requirements.txt` (if not already present)
- Verify FFmpeg is installed on your build machine
- Consider bundling FFmpeg directly in the app (advanced)

---

## Additional Resources

- [Briefcase Documentation](https://briefcase.readthedocs.io/)
- [Briefcase GitHub Repository](https://github.com/beeware/briefcase)
- [BeeWare Project](https://beeware.org/)

---

**Questions?** Open an issue on the [BARPATH GitHub repository](https://github.com/your-org/barpath).