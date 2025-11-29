"""
Hardware detection utility for barpath.

Detects OS, CPU type (Intel/AMD), and GPU availability to recommend
appropriate hardware-accelerated dependencies for ONNX and OpenVINO.
"""

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def detect_os() -> str:
    """
    Detect the operating system.

    Returns:
        str: 'windows', 'macos', or 'linux'
    """
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    elif system == "darwin":
        return "macos"
    elif system == "linux":
        return "linux"
    else:
        return "unknown"


def detect_cpu_brand() -> Optional[str]:
    """
    Detect CPU brand (Intel or AMD).

    Returns:
        str: 'intel', 'amd', or None if undetectable
    """
    try:
        if sys.platform == "win32":
            # Windows: Use wmic
            result = subprocess.run(
                ["wmic", "cpu", "get", "manufacturer"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout.lower()
            if "intel" in output:
                return "intel"
            elif "amd" in output:
                return "amd"
        elif sys.platform == "darwin":
            # macOS: Use sysctl (M-series are ARM, not relevant here)
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout.lower()
            if "intel" in output:
                return "intel"
            elif "amd" in output:
                return "amd"
        elif sys.platform == "linux":
            # Linux: Check /proc/cpuinfo
            with open("/proc/cpuinfo", "r") as f:
                content = f.read().lower()
                if "intel" in content:
                    return "intel"
                elif "amd" in content:
                    return "amd"
    except Exception:
        pass

    return None


def detect_nvidia_gpu() -> bool:
    """
    Detect NVIDIA GPU availability.

    Returns:
        bool: True if NVIDIA GPU detected, False otherwise
    """
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def detect_amd_gpu() -> bool:
    """
    Detect AMD GPU availability (rocm).

    Returns:
        bool: True if AMD GPU with ROCm support detected, False otherwise
    """
    try:
        result = subprocess.run(["rocm-smi"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def detect_intel_gpu() -> bool:
    """
    Detect Intel GPU availability.

    Returns:
        bool: True if Intel GPU detected, False otherwise
    """
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["wmic", "path", "win32_videocontroller", "get", "name"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return "intel" in result.stdout.lower()
        elif sys.platform == "linux":
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5
            )
            return "intel" in result.stdout.lower() and "vga" in result.stdout.lower()
    except Exception:
        pass

    return False


def get_hardware_profile() -> Dict[str, Any]:
    """
    Get complete hardware profile.

    Returns:
        dict: Hardware profile with os, cpu_brand, has_nvidia_gpu, has_amd_gpu, has_intel_gpu
    """
    return {
        "os": detect_os(),
        "cpu_brand": detect_cpu_brand(),
        "has_nvidia_gpu": detect_nvidia_gpu(),
        "has_amd_gpu": detect_amd_gpu(),
        "has_intel_gpu": detect_intel_gpu(),
    }


def get_optional_packages(
    hardware_profile: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """
    Get list of optional hardware-accelerated packages based on hardware profile.

    Args:
        hardware_profile (dict): Hardware profile from get_hardware_profile()

    Returns:
        tuple: (onnx_packages, openvino_packages) - lists of recommended packages
    """
    os_type = hardware_profile.get("os", "unknown")
    cpu_brand = hardware_profile.get("cpu_brand")
    has_nvidia = hardware_profile.get("has_nvidia_gpu", False)
    has_amd = hardware_profile.get("has_amd_gpu", False)

    onnx_packages = []
    openvino_packages = []

    # ONNX Runtime options
    if os_type == "windows":
        # Windows: CPU default, but offer DirectML for any GPU
        if has_nvidia or has_amd or hardware_profile.get("has_intel_gpu"):
            onnx_packages = ["onnxruntime-directml"]
        else:
            onnx_packages = ["onnxruntime"]
    elif os_type == "macos":
        # macOS: Metal acceleration
        onnx_packages = ["onnxruntime-metal"]
    elif os_type == "linux":
        # Linux: CPU default, but offer GPU if available
        if has_nvidia:
            onnx_packages = ["onnxruntime-gpu"]
        elif has_amd:
            onnx_packages = ["onnxruntime-rocm"]
        else:
            onnx_packages = ["onnxruntime"]

    # OpenVINO options (Intel CPU only)
    if cpu_brand == "intel":
        openvino_packages = ["openvino"]

    return onnx_packages, openvino_packages


def get_hardware_description(hardware_profile: Dict[str, Any]) -> str:
    """
    Get a human-readable description of the detected hardware.

    Args:
        hardware_profile (dict): Hardware profile from get_hardware_profile()

    Returns:
        str: Description of hardware
    """
    os_type = hardware_profile.get("os", "unknown").upper()
    cpu_brand = (hardware_profile.get("cpu_brand") or "Unknown").upper()
    has_nvidia = hardware_profile.get("has_nvidia_gpu", False)
    has_amd = hardware_profile.get("has_amd_gpu", False)
    has_intel = hardware_profile.get("has_intel_gpu", False)

    parts = [f"OS: {os_type}", f"CPU: {cpu_brand}"]

    gpu_parts = []
    if has_nvidia:
        gpu_parts.append("NVIDIA GPU")
    if has_amd:
        gpu_parts.append("AMD GPU (ROCm)")
    if has_intel:
        gpu_parts.append("Intel GPU")

    if gpu_parts:
        parts.append(f"GPU: {', '.join(gpu_parts)}")
    else:
        parts.append("GPU: None detected")

    return " | ".join(parts)


def detect_installed_runtimes() -> Dict[str, bool]:
    """
    Detect which hardware acceleration runtimes are currently installed.

    Returns:
        dict: {'onnxruntime': bool, 'onnxruntime_directml': bool, 'onnxruntime_gpu': bool,
               'onnxruntime_rocm': bool, 'onnxruntime_metal': bool, 'openvino': bool}
    """
    runtimes = {
        "onnxruntime": False,
        "onnxruntime_directml": False,
        "onnxruntime_gpu": False,
        "onnxruntime_rocm": False,
        "onnxruntime_metal": False,
        "openvino": False,
    }

    # Try to import each runtime and check for availability
    try:
        import onnxruntime as ort

        runtimes["onnxruntime"] = True

        # Check for specific providers
        providers = ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            runtimes["onnxruntime_directml"] = True
        if "CUDAExecutionProvider" in providers:
            runtimes["onnxruntime_gpu"] = True
        if "ROCMExecutionProvider" in providers:
            runtimes["onnxruntime_rocm"] = True
        if "CoreMLExecutionProvider" in providers:
            runtimes["onnxruntime_metal"] = True
    except ImportError:
        pass

    try:
        import openvino as ov  # noqa: F401

        runtimes["openvino"] = True
    except ImportError:
        pass

    return runtimes


def get_available_runtimes_for_model(model_path: str) -> Dict[str, str]:
    """
    Get available runtimes for a specific model file.

    Args:
        model_path (str): Path to model file (.pt, .onnx, or openvino directory)

    Returns:
        dict: {'display_name': 'identifier'} of available runtimes for this model
    """
    installed = detect_installed_runtimes()
    model_path_obj = Path(model_path)
    model_ext = model_path_obj.suffix.lower()
    is_openvino_dir = model_path_obj.is_dir() and any(
        "openvino" in part.lower() for part in model_path_obj.parts
    )

    available = {}

    # For PT files, offer Ultralytics PyTorch runtime first
    if model_ext == ".pt":
        available["Ultralytics (PyTorch)"] = "ultralytics"

    # For ONNX/PT models, offer ONNX runtimes
    if model_ext in [".onnx", ".pt"] or is_openvino_dir:
        # Always offer CPU fallback
        if installed["onnxruntime"]:
            available["ONNX Runtime (CPU)"] = "onnxruntime"

        # Offer accelerated versions if available
        if installed["onnxruntime_directml"]:
            available["ONNX Runtime (DirectML - GPU)"] = "onnxruntime_directml"
        if installed["onnxruntime_gpu"]:
            available["ONNX Runtime (CUDA - NVIDIA GPU)"] = "onnxruntime_gpu"
        if installed["onnxruntime_rocm"]:
            available["ONNX Runtime (ROCm - AMD GPU)"] = "onnxruntime_rocm"
        if installed["onnxruntime_metal"]:
            available["ONNX Runtime (Metal - Apple GPU)"] = "onnxruntime_metal"

    # For OpenVINO models/pt files, offer OpenVINO runtime
    if is_openvino_dir or model_ext == ".pt":
        if installed["openvino"]:
            available["OpenVINO (Intel CPU)"] = "openvino"

    return available


if __name__ == "__main__":
    # For testing/inspection
    profile = get_hardware_profile()
    print("Hardware Profile:", profile)
    print("Description:", get_hardware_description(profile))

    onnx_pkgs, openvino_pkgs = get_optional_packages(profile)
    print(f"\nRecommended ONNX packages: {onnx_pkgs}")
    print(f"Recommended OpenVINO packages: {openvino_pkgs}")

    print("\n--- Installed Runtimes ---")
    runtimes = detect_installed_runtimes()
    for name, installed in runtimes.items():
        status = "✓ Installed" if installed else "✗ Not installed"
        print(f"  {name}: {status}")
