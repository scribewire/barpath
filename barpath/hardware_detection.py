"""
Hardware detection utility for barpath.

Detects OS and CPU type (Intel/AMD) to recommend appropriate
hardware-accelerated dependencies for ONNX Runtime and OpenVINO.
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
    Detect if an NVIDIA GPU is present.

    Returns:
        bool: True if NVIDIA GPU detected, False otherwise
    """
    try:
        # Try nvidia-smi command (works on Windows/Linux)
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # If nvidia-smi runs successfully, NVIDIA GPU is present
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        # nvidia-smi not found or failed
        pass

    # Fallback: Check system information
    try:
        if sys.platform == "win32":
            # Windows: Check for NVIDIA graphics adapters
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout.lower()
            if (
                "nvidia" in output
                or "geforce" in output
                or "rtx" in output
                or "gtx" in output
            ):
                return True
        elif sys.platform == "linux":
            # Linux: Check lspci for NVIDIA VGA controllers
            result = subprocess.run(
                ["lspci"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout.lower()
            if "nvidia" in output and ("vga" in output or "3d" in output):
                return True
        elif sys.platform == "darwin":
            # macOS: Use system_profiler to check graphics
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout.lower()
            if "nvidia" in output or "geforce" in output:
                return True
    except Exception:
        pass

    return False


def detect_intel_gpu() -> bool:
    """
    Detect if an Intel GPU (integrated or discrete) is present.

    Returns:
        bool: True if Intel GPU detected, False otherwise
    """
    try:
        if sys.platform == "win32":
            # Windows: Check for Intel graphics adapters
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout.lower()
            # Check for Intel integrated graphics (UHD, Iris, HD) or Arc discrete
            if any(term in output for term in ["intel", "arc", "iris", "uhd"]):
                return True
        elif sys.platform == "linux":
            # Linux: Check lspci for VGA/Display controllers
            result = subprocess.run(
                ["lspci"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout.lower()
            # Look for Intel VGA or display controller
            if "intel" in output and ("vga" in output or "display" in output):
                return True
        elif sys.platform == "darwin":
            # macOS: Use system_profiler to check graphics
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout.lower()
            if "intel" in output:
                return True
    except Exception:
        pass

    return False


def get_hardware_profile() -> Dict[str, Any]:
    """
    Get complete hardware profile.

    Returns:
        dict: Hardware profile with os, cpu_brand, intel_gpu, and nvidia_gpu
    """
    return {
        "os": detect_os(),
        "cpu_brand": detect_cpu_brand(),
        "intel_gpu": detect_intel_gpu(),
        "nvidia_gpu": detect_nvidia_gpu(),
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
    cpu_brand = hardware_profile.get("cpu_brand")
    has_nvidia_gpu = hardware_profile.get("nvidia_gpu", False)

    # ONNX Runtime - GPU version if NVIDIA GPU detected
    if has_nvidia_gpu:
        onnx_packages = ["onnxruntime-gpu"]
    else:
        onnx_packages = ["onnxruntime"]

    # OpenVINO options (Intel CPU only)
    openvino_packages = []
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
    has_nvidia_gpu = hardware_profile.get("nvidia_gpu", False)
    has_intel_gpu = hardware_profile.get("intel_gpu", False)

    parts = [f"OS: {os_type}", f"CPU: {cpu_brand}"]

    gpu_parts = []
    if has_nvidia_gpu:
        gpu_parts.append("NVIDIA GPU")
    if has_intel_gpu:
        gpu_parts.append("Intel GPU")

    if gpu_parts:
        parts.append(f"GPU: {', '.join(gpu_parts)}")

    return " | ".join(parts)


def detect_installed_runtimes() -> Dict[str, bool]:
    """
    Detect which hardware acceleration runtimes are currently installed.

    Returns:
        dict: {'onnxruntime': bool, 'openvino': bool, 'tensorrt': bool}
    """
    runtimes = {
        "onnxruntime": False,
        "openvino": False,
        "tensorrt": False,
    }

    # Try to import ONNX Runtime
    try:
        import onnxruntime as ort  # noqa: F401

        runtimes["onnxruntime"] = True
    except ImportError:
        pass

    # Try to import OpenVINO
    try:
        import openvino as ov  # noqa: F401

        runtimes["openvino"] = True
    except ImportError:
        pass

    # Try to import TensorRT (optional dependency)
    try:
        import tensorrt as trt  # type: ignore[attr-defined]  # noqa: F401

        runtimes["tensorrt"] = True
    except ImportError:
        pass

    return runtimes


def get_available_runtimes_for_model(model_path: str) -> Dict[str, str]:
    """
    Get available runtimes for a specific model file.

    Args:
        model_path (str): Path to model file (.pt, .onnx, .engine, or openvino directory)

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
        available["Ultralytics (PyTorch CPU)"] = "ultralytics"

    # For ONNX models, offer ONNX Runtime
    if model_ext == ".onnx":
        if installed["onnxruntime"]:
            available["ONNX Runtime (GPU/CPU)"] = "onnxruntime"

    # For TensorRT .engine models, offer TensorRT runtime
    if model_ext == ".engine":
        if installed["tensorrt"]:
            available["TensorRT (NVIDIA GPU)"] = "tensorrt"

    # For PT models (non-.engine), offer ONNX Runtime as alternative
    if model_ext == ".pt" and not is_openvino_dir:
        if installed["onnxruntime"]:
            available["ONNX Runtime (GPU/CPU)"] = "onnxruntime"

    # For OpenVINO models/pt files, offer OpenVINO runtime (Intel CPU only)
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

    print("\n--- GPU Detection ---")
    has_nvidia_gpu = detect_nvidia_gpu()
    has_intel_gpu = detect_intel_gpu()
    print(f"  NVIDIA GPU: {'✓ Detected' if has_nvidia_gpu else '✗ Not detected'}")
    print(f"  Intel GPU: {'✓ Detected' if has_intel_gpu else '✗ Not detected'}")
