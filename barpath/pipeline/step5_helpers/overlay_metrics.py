"""Shared resolution-aware metrics for the rendered video overlay."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OverlayMetrics:
    """Pixel and font metrics derived from the output frame dimensions."""

    scale: float

    REFERENCE_WIDTH = 1920
    REFERENCE_HEIGHT = 1080
    MIN_SCALE = 0.4
    MAX_SCALE = 2.0

    @classmethod
    def for_frame(cls, frame_width: int, frame_height: int) -> "OverlayMetrics":
        scale = min(
            frame_width / cls.REFERENCE_WIDTH,
            frame_height / cls.REFERENCE_HEIGHT,
        )
        scale = max(cls.MIN_SCALE, min(cls.MAX_SCALE, scale))
        return cls(scale=scale)

    def px(self, value: float, minimum: int = 1) -> int:
        """Scale a reference pixel value while preserving visible primitives."""
        return max(minimum, round(value * self.scale))

    def font(self, value: float, minimum: float = 0.1) -> float:
        """Scale an OpenCV font scale without allowing it to disappear."""
        return max(minimum, value * self.scale)
