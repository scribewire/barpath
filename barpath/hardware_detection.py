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


def get_hardware_profile() -> Dict[str, Any]:
    """
    Get complete hardware profile.

    Returns:
        dict: Hardware profile with os and cpu_brand
    """
    return {
        "os": detect_os(),
        "cpu_brand": detect_cpu_brand(),
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

    # ONNX Runtime - always use CPU version
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

    parts = [f"OS: {os_type}", f"CPU: {cpu_brand}"]

    return " | ".join(parts)


def detect_installed_runtimes() -> Dict[str, bool]:
    """
    Detect which hardware acceleration runtimes are currently installed.

    Returns:
        dict: {'onnxruntime': bool, 'openvino': bool}
    """
    runtimes = {
        "onnxruntime": False,
        "openvino": False,
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
        available["Ultralytics (PyTorch CPU)"] = "ultralytics"

    # For ONNX/PT models, offer ONNX Runtime (CPU only)
    if model_ext in [".onnx", ".pt"] or is_openvino_dir:
        if installed["onnxruntime"]:
            available["ONNX Runtime (CPU)"] = "onnxruntime"

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
