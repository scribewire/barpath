#!/usr/bin/env python3
"""
Briefcase installer helper for barpath.

This script provides an interactive UI for users to select hardware-accelerated
packages during Briefcase-based installation. It detects the user's hardware
and provides a checkbox-style selection of available acceleration options.

This can be called as part of a Briefcase post-install hook.
"""

import sys
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from barpath.hardware_detection import (
    get_hardware_profile,
    get_optional_packages,
    get_hardware_description,
    detect_os,
)


def install_with_input():
    """
    Interactive installation helper for hardware acceleration selection.
    Prompts user and prints install commands.
    """
    print("\n" + "=" * 70)
    print("barpath: Hardware Acceleration Setup")
    print("=" * 70 + "\n")
    
    # Detect hardware
    profile = get_hardware_profile()
    os_type = profile.get('os', 'unknown')
    
    print("Detected Hardware:")
    print(f"  {get_hardware_description(profile)}\n")
    
    if os_type == 'unknown':
        print("⚠️  Warning: Could not detect OS. Skipping hardware acceleration.\n")
        return
    
    onnx_packages, openvino_packages = get_optional_packages(profile)
    
    # Display available options
    print("Available Hardware Acceleration Options:\n")
    
    has_options = False
    
    if onnx_packages:
        has_options = True
        pkg_str = ', '.join(onnx_packages)
        print(f"  [1] ONNX Runtime Acceleration")
        print(f"      Package(s): {pkg_str}")
        if os_type == 'windows' and 'directml' in pkg_str:
            print(f"      GPU acceleration via DirectML (supports NVIDIA/AMD/Intel GPU)")
        elif os_type == 'macos':
            print(f"      GPU acceleration via Apple Metal (faster inference)")
        elif os_type == 'linux' and 'gpu' in pkg_str:
            print(f"      NVIDIA GPU acceleration via CUDA")
        elif os_type == 'linux' and 'rocm' in pkg_str:
            print(f"      AMD GPU acceleration via ROCm")
        else:
            print(f"      CPU optimization via ONNX Runtime")
        print()
    
    if openvino_packages:
        has_options = True
        pkg_str = ', '.join(openvino_packages)
        print(f"  [2] OpenVINO Acceleration")
        print(f"      Package(s): {pkg_str}")
        print(f"      Intel CPU optimization (faster inference on Intel processors)")
        print()
    
    if not has_options:
        print("  No hardware acceleration packages available for your system.")
        print("  (Only Intel CPUs support OpenVINO.)\n")
        return
    
    # Prompt for selection
    print("Installation Options:\n")
    print("  [0] None (core packages only)")
    if onnx_packages and openvino_packages:
        print("  [1] ONNX Runtime only")
        print("  [2] OpenVINO only")
        print("  [3] Both ONNX and OpenVINO (recommended)")
        prompt_msg = "Select an option [0-3]: "
        valid_choices = ['0', '1', '2', '3']
    elif onnx_packages:
        print("  [1] ONNX Runtime (recommended)")
        prompt_msg = "Select an option [0-1]: "
        valid_choices = ['0', '1']
    else:  # openvino only
        print("  [1] OpenVINO (recommended)")
        prompt_msg = "Select an option [0-1]: "
        valid_choices = ['0', '1']
    
    print()
    choice = input(prompt_msg).strip()
    
    if choice not in valid_choices:
        print("\n❌ Invalid choice. No additional packages will be installed.")
        return
    
    print()
    
    packages_to_install = []
    
    if onnx_packages and openvino_packages:
        # Both available
        if choice == '1':
            packages_to_install = onnx_packages
            print("✓ Selected: ONNX Runtime acceleration")
        elif choice == '2':
            packages_to_install = openvino_packages
            print("✓ Selected: OpenVINO acceleration")
        elif choice == '3':
            packages_to_install = onnx_packages + openvino_packages
            print("✓ Selected: ONNX Runtime + OpenVINO acceleration")
    elif onnx_packages:
        # ONNX only
        if choice == '1':
            packages_to_install = onnx_packages
            print("✓ Selected: ONNX Runtime acceleration")
    else:
        # OpenVINO only
        if choice == '1':
            packages_to_install = openvino_packages
            print("✓ Selected: OpenVINO acceleration")
    
    if choice == '0':
        print("✓ Skipping hardware acceleration packages")
        print("\nYou can install them later with:")
        print(f"  pip install -r requirements-hardware.txt\n")
        return
    
    # Show install commands
    print("\nTo complete installation, run one of the following:\n")
    
    if packages_to_install:
        print("Option A: Using pip directly")
        cmd = " ".join(packages_to_install)
        print(f"  pip install {cmd}\n")
        
        print("Option B: Using setup.py")
        if 'openvino' in ' '.join(packages_to_install):
            if 'onnxruntime' in ' '.join(packages_to_install) or 'onnxruntime-directml' in ' '.join(packages_to_install) or 'onnxruntime-metal' in ' '.join(packages_to_install) or 'onnxruntime-gpu' in ' '.join(packages_to_install) or 'onnxruntime-rocm' in ' '.join(packages_to_install):
                print(f"  pip install .[hardware]\n")
            else:
                print(f"  pip install .[openvino]\n")
        else:
            print(f"  pip install .[onnx]\n")
    
    print("=" * 70 + "\n")


def get_install_command_for_briefcase():
    """
    Get the pip install command for Briefcase to use (non-interactive).
    
    Returns:
        str: pip install command to append to requirements, or empty string if none
    """
    profile = get_hardware_profile()
    onnx_packages, openvino_packages = get_optional_packages(profile)
    
    packages = onnx_packages + openvino_packages
    
    if packages:
        return " ".join(packages)
    return ""


if __name__ == '__main__':
    # Run interactive mode
    install_with_input()
