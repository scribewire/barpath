"""Tests for realtime_processing package __init__.py re-exports."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_INIT = Path("barpath/pipeline/realtime_processing/__init__.py")


class TestRealtimeProcessingInit:
    """Test that __init__.py properly re-exports all public symbols."""

    def test_init_file_exists(self):
        """__init__.py exists in realtime_processing package."""
        assert PACKAGE_INIT.exists(), f"{PACKAGE_INIT} should exist"

    def test_init_valid_syntax(self):
        """__init__.py contains valid Python syntax."""
        content = PACKAGE_INIT.read_text()
        ast.parse(content)  # Raises SyntaxError if invalid

    def test_import_circular_frame_buffer(self):
        """__init__.py contains `from .live_buffer import CircularFrameBuffer, FrameData`."""
        content = PACKAGE_INIT.read_text()
        assert "from .live_buffer import CircularFrameBuffer, FrameData" in content

    def test_import_lift_detection_system(self):
        """__init__.py contains `from .live_detection_system import LiftDetectionSystem, DetectionState`."""
        content = PACKAGE_INIT.read_text()
        assert "from .live_detection_system import LiftDetectionSystem, DetectionState" in content

    def test_import_live_lift_recognizer(self):
        """__init__.py contains `from .live_lift_recognition import LiveLiftRecognizer, LiftState`."""
        content = PACKAGE_INIT.read_text()
        assert "from .live_lift_recognition import LiveLiftRecognizer, LiftState" in content

    def test_import_live_feature_extractor(self):
        """__init__.py contains `from .live_feature_extractor import LiveFeatureExtractor`."""
        content = PACKAGE_INIT.read_text()
        assert "from .live_feature_extractor import LiveFeatureExtractor" in content

    def test_all_exports_defined(self):
        """__all__ lists all public symbols."""
        content = PACKAGE_INIT.read_text()
        expected_symbols = [
            "CircularFrameBuffer",
            "FrameData",
            "LiftDetectionSystem",
            "DetectionState",
            "LiveFeatureExtractor",
            "LiveLiftRecognizer",
            "LiftState",
        ]
        for symbol in expected_symbols:
            assert symbol in content, f"__all__ should include {symbol}"

    def test_has_docstring(self):
        """__init__.py includes a package docstring."""
        content = PACKAGE_INIT.read_text()
        assert "Real-time lift detection" in content
