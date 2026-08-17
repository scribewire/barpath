import cv2
import numpy as np

from barpath.pipeline.step5_helpers.overlay_metrics import OverlayMetrics
from barpath.pipeline.utils import draw_legend


def test_overlay_metrics_are_bounded_across_target_resolutions():
    assert OverlayMetrics.for_frame(640, 360).scale == OverlayMetrics.MIN_SCALE
    assert OverlayMetrics.for_frame(1920, 1080).scale == 1.0
    assert OverlayMetrics.for_frame(3840, 2160).scale == OverlayMetrics.MAX_SCALE


def test_legend_renders_at_target_resolutions():
    colors = {"Barbell Box": (255, 0, 0), "Pull": (0, 0, 255)}

    for width, height in ((640, 360), (1920, 1080), (3840, 2160)):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        draw_legend(frame, colors, OverlayMetrics.for_frame(width, height))

        assert frame.shape == (height, width, 3)
        assert cv2.countNonZero(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) > 0
