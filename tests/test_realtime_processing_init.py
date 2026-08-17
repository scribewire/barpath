"""Tests for realtime_processing package __init__.py re-exports."""

import ast
import re
from pathlib import Path

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
        """__init__.py re-exports CircularFrameBuffer and FrameData."""
        content = PACKAGE_INIT.read_text()
        assert "from .live_buffer import" in content
        assert "CircularFrameBuffer" in content
        assert "FrameData" in content

    def test_import_lift_detection_system(self):
        """__init__.py re-exports LiftDetectionSystem and DetectionState."""
        content = PACKAGE_INIT.read_text()
        assert "from .live_detection_system import" in content
        assert "LiftDetectionSystem" in content
        assert "DetectionState" in content

    def test_import_live_lift_recognizer(self):
        """__init__.py re-exports LiveLiftRecognizer and LiftState."""
        content = PACKAGE_INIT.read_text()
        assert "from .live_lift_recognition import" in content
        assert "LiveLiftRecognizer" in content
        assert "LiftState" in content

    def test_import_live_feature_extractor(self):
        """__init__.py re-exports LiveFeatureExtractor."""
        content = PACKAGE_INIT.read_text()
        assert "from .live_feature_extractor import" in content
        assert "LiveFeatureExtractor" in content

    def test_all_exports_defined(self):
        """__all__ lists all public symbols."""
        content = PACKAGE_INIT.read_text()
        match = re.search(r"__all__\s*=\s*(\[[^\]]*\])", content, re.DOTALL)
        assert match, "__all__ list not found"
        expected_symbols = {
            "CircularFrameBuffer",
            "FrameData",
            "LiftDetectionSystem",
            "DetectionState",
            "LiveFeatureExtractor",
            "LiveLiftRecognizer",
            "LiftState",
        }
        for symbol in expected_symbols:
            assert symbol in match.group(1), f"__all__ should include {symbol}"

    def test_has_docstring(self):
        """__init__.py includes a package docstring."""
        content = PACKAGE_INIT.read_text()
        assert "Real-time lift detection" in content
