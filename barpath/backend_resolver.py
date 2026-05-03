"""Backend resolver for barpath — unified OpenVINO/PyTorch backend selection.

This module consolidates backend detection logic that was previously duplicated
across barpath_core.py, barpath_cli.py, and pipeline modules.

Exports:
    BACKEND_OPTIONS — tuple of valid backend names
    DEFAULT_BACKEND — default backend when no preference specified
    _is_openvino_model_dir — check if path looks like OpenVINO export
    resolve_backend — resolve which backend to use for a model
    validate_openvino_model_dir — validate OpenVINO directory structure
"""

from pathlib import Path
from typing import Union

BACKEND_OPTIONS = ("pytorch", "openvino", "auto")
DEFAULT_BACKEND = "auto"


def _is_openvino_model_dir(path_str: Union[str, Path]) -> bool:
    """Return True when the provided path looks like an OpenVINO export directory."""
    path = Path(path_str)
    if not path.is_dir():
        return False
    return any("openvino" in part.lower() for part in path.parts)


def resolve_backend(
    model_path: Union[str, Path], preference: str = "auto"
) -> tuple[str, str]:
    """Resolve which backend to use and return the model path string.

    Args:
        model_path: Path to the model file or OpenVINO export directory.
        preference: Backend preference — "auto", "openvino", or "pytorch".

    Returns:
        Tuple of (backend_name, resolved_model_path_str).

    Raises:
        ValueError: If preference is not in BACKEND_OPTIONS.
    """
    if preference not in BACKEND_OPTIONS:
        raise ValueError(
            f"Invalid backend preference '{preference}'. "
            f"Choose from: {', '.join(BACKEND_OPTIONS)}"
        )

    model_path_str = str(model_path)

    if preference == "auto":
        # Auto-detect: Intel GPU + importable openvino + not already an OV dir → suggest openvino
        try:
            from barpath.hardware_detection import detect_intel_gpu

            has_intel_gpu = detect_intel_gpu()
        except Exception:
            has_intel_gpu = False

        try:
            import openvino  # noqa: F401

            openvino_available = True
        except ImportError:
            openvino_available = False

        is_ov_dir = _is_openvino_model_dir(model_path)

        if has_intel_gpu and openvino_available and not is_ov_dir:
            return ("openvino", model_path_str)
        return ("pytorch", model_path_str)

    if preference == "openvino":
        try:
            import openvino  # noqa: F401

            return ("openvino", model_path_str)
        except ImportError:
            # OpenVINO not available — return pytorch so caller can warn+fallback
            return ("pytorch", model_path_str)

    # preference == "pytorch"
    return ("pytorch", model_path_str)


def validate_openvino_model_dir(model_path: Union[str, Path]) -> tuple[bool, str]:
    """Validate that a path is a proper OpenVINO model directory.

    Args:
        model_path: Path to check.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    path = Path(model_path)

    if not _is_openvino_model_dir(path):
        return (False, "Path is not an OpenVINO model directory")

    xml_files = list(path.glob("*.xml"))
    if not xml_files:
        return (
            False,
            f"OpenVINO directory '{path}' does not contain a .xml model file",
        )

    bin_files = list(path.glob("*.bin"))
    if not bin_files:
        return (
            False,
            f"OpenVINO directory '{path}' does not contain a .bin weights file. "
            f"OpenVINO models require both .xml (model definition) and .bin (weights) files.",
        )

    return (True, "")


__all__ = [
    "BACKEND_OPTIONS",
    "DEFAULT_BACKEND",
    "resolve_backend",
    "_is_openvino_model_dir",
    "validate_openvino_model_dir",
]
