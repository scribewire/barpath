from setuptools import find_packages, setup

from barpath.hardware_detection import get_hardware_profile, get_optional_packages

# Detect hardware to determine default/recommended packages
hardware_profile = get_hardware_profile()
onnx_packages, openvino_packages = get_optional_packages(hardware_profile)

# Core dependencies (hardware-agnostic)
core_requires = [
    "opencv-python>=4.8.0",
    "mediapipe>=0.10.0",
    "ultralytics>=8.0.0",
    "torchvision>=0.15.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "matplotlib>=3.7.0",
    "rich>=13.0.0",
    "pycairo>=1.17.0",
    "toga>=0.4.7",
]

# Optional extras for hardware acceleration
extras_dict = {
    # ONNX Runtime acceleration (platform/GPU specific)
    "onnx": onnx_packages,
    # OpenVINO (Intel CPU only)
    "openvino": openvino_packages,
    # Combined: All hardware acceleration
    "hardware": list(set(onnx_packages + openvino_packages)),
    # Development dependencies
    "dev": [
        "pytest>=7.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
    ],
}

setup(
    name="barpath",
    version="1.0.0",
    description="Offline Weightlifting Technique Analysis",
    author="Ethan Christian",
    packages=find_packages(),
    install_requires=core_requires,
    extras_require=extras_dict,
    entry_points={
        "console_scripts": [
            "barpath=barpath.barpath_cli:main",
            "barpath-gui=barpath.barpath_gui:main",
        ],
    },
    python_requires=">=3.8",
)
