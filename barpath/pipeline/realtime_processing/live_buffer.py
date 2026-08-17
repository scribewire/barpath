"""
Circular frame buffer for live lift detection.
Time-indexed storage with sliding window extraction.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameData:
    """Single frame capture from live video stream."""

    timestamp_ms: float
    barbell_center: tuple[float, float] | None = None  # (x, y) in pixels
    barbell_box: tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2)
    landmarks: dict[int, tuple[float, float, float, float]] = field(
        default_factory=dict
    )  # MediaPipe idx -> (x, y, z, visibility)
    knee_y_avg: float = 0.0  # Average knee y position in pixels
    joint_angles: dict[str, float] = field(
        default_factory=dict
    )  # 'left_knee', 'right_knee', etc. in degrees


class CircularFrameBuffer:
    """Time-indexed circular buffer for live stream frames.

    Stores up to max_duration_ms of frame data, providing efficient
    window extraction for classification.
    """

    MIN_FRAMES_FOR_ANALYSIS = 30  # Minimum ~1 second at 30fps

    def __init__(self, max_duration_ms: float = 4000.0, fps: float = 30.0):
        """
        Args:
            max_duration_ms: Maximum buffer duration in milliseconds (default 4s)
            fps: Expected frames per second for estimation
        """
        self.max_duration_ms = max_duration_ms
        self.expected_fps = fps
        self._frames: deque[FrameData] = deque()
        self._start_time_ms: float | None = None
        self._last_timestamp_ms: float = 0.0

    def add_frame(self, frame: FrameData) -> None:
        """Add frame and evict old frames beyond max_duration_ms."""
        if self._start_time_ms is None:
            self._start_time_ms = frame.timestamp_ms

        self._last_timestamp_ms = frame.timestamp_ms

        # Add new frame
        self._frames.append(frame)

        # Evict old frames if we're over max duration
        while len(self._frames) > 1:
            current_span = self._frames[-1].timestamp_ms - self._frames[0].timestamp_ms
            if current_span <= self.max_duration_ms:
                break
            self._frames.popleft()

    def get_window(self, duration_ms: float, end_time_ms: float | None = None) -> list[FrameData]:
        """
        Extract last N milliseconds of frames.

        Args:
            duration_ms: Window duration
            end_time_ms: End time for window (default: most recent)

        Returns:
            List of FrameData within window
        """
        if len(self._frames) == 0:
            return []

        if end_time_ms is None:
            end_time_ms = self._last_timestamp_ms

        start_time = end_time_ms - duration_ms

        # Binary search for start index
        frames_in_window = []
        for frame in self._frames:
            if frame.timestamp_ms >= start_time:
                frames_in_window.append(frame)

        return frames_in_window

    def get_recent_frames(self, num_frames: int) -> list[FrameData]:
        """Get the most recent N frames."""
        if len(self._frames) <= num_frames:
            return list(self._frames)
        return list(self._frames)[-num_frames:]

    def get_barbell_positions(self, frames: list[FrameData] | None = None) -> np.ndarray:
        """
        Extract barbell (x, y) positions as Nx2 array.

        Args:
            frames: Frames to extract from (default: all in buffer)

        Returns:
            Nx2 array of (x, y) positions, or empty array if no data
        """
        if frames is None:
            frames = list(self._frames)

        positions = []
        for frame in frames:
            if frame.barbell_center is not None:
                positions.append(list(frame.barbell_center))
            elif len(positions) > 0:
                # Forward-fill missing positions
                positions.append(positions[-1])
            else:
                positions.append([0.0, 0.0])

        if not positions:
            return np.array([], dtype=np.float64).reshape(0, 2)

        return np.array(positions, dtype=np.float64)

    def get_barbell_y(self, frames: list[FrameData] | None = None) -> np.ndarray:
        """Extract just barbell y positions."""
        positions = self.get_barbell_positions(frames)
        if len(positions) == 0:
            return np.array([], dtype=np.float64)
        return positions[:, 1]

    def get_barbell_velocities(self, frames: list[FrameData] | None = None) -> np.ndarray:
        """Compute barbell vertical velocities from positions."""
        y_positions = self.get_barbell_y(frames)
        if len(y_positions) < 2:
            return np.array([], dtype=np.float64)

        # Get timestamps for dt calculation
        if frames is None:
            frames = list(self._frames)

        timestamps = np.array([f.timestamp_ms for f in frames], dtype=np.float64)
        dt = np.diff(timestamps) / 1000.0  # Convert to seconds

        # Avoid division by zero
        dt = np.where(dt == 0, 1.0 / self.expected_fps, dt)

        velocities = np.diff(y_positions) / dt
        return velocities

    def get_joint_angle_series(self, frames: list[FrameData], joint_name: str) -> np.ndarray:
        """Extract time series for specific joint angle."""
        angles = []
        for frame in frames:
            angle = frame.joint_angles.get(joint_name, 180.0)
            angles.append(angle)

        return np.array(angles, dtype=np.float64)

    def get_knee_angles(self, frames: list[FrameData] | None = None) -> np.ndarray:
        """Get average knee angle series (left + right)."""
        if frames is None:
            frames = list(self._frames)

        left_angles = self.get_joint_angle_series(frames, "left_knee")
        right_angles = self.get_joint_angle_series(frames, "right_knee")

        if len(left_angles) == 0 or len(right_angles) == 0:
            return np.array([], dtype=np.float64)

        return (left_angles + right_angles) / 2.0

    def get_elbow_angles(self, frames: list[FrameData] | None = None) -> np.ndarray:
        """Get average elbow angle series."""
        if frames is None:
            frames = list(self._frames)

        left_angles = self.get_joint_angle_series(frames, "left_elbow")
        right_angles = self.get_joint_angle_series(frames, "right_elbow")

        if len(left_angles) == 0 or len(right_angles) == 0:
            return np.array([], dtype=np.float64)

        return (left_angles + right_angles) / 2.0

    def get_timestamps(self, frames: list[FrameData] | None = None) -> np.ndarray:
        """Get timestamps in milliseconds."""
        if frames is None:
            frames = list(self._frames)

        return np.array([f.timestamp_ms for f in frames], dtype=np.float64)

    def get_landmarks_list(self, frames=None) -> list:
        """Get list of landmarks from frames."""
        if frames is None:
            frames = list(self._frames)
        return [f.landmarks for f in frames]

    @property
    def is_ready(self) -> bool:
        """Buffer has enough data for analysis."""
        return len(self._frames) >= self.MIN_FRAMES_FOR_ANALYSIS

    @property
    def num_frames(self) -> int:
        """Number of frames currently in buffer."""
        return len(self._frames)

    @property
    def duration_ms(self) -> float:
        """Actual duration of buffer in milliseconds."""
        if len(self._frames) < 2:
            return 0.0
        return self._frames[-1].timestamp_ms - self._frames[0].timestamp_ms

    @property
    def fps(self) -> float:
        """Estimated actual FPS from buffer."""
        if len(self._frames) < 10:
            return self.expected_fps

        timestamps = self.get_timestamps()
        duration_s = (timestamps[-1] - timestamps[0]) / 1000.0

        if duration_s <= 0:
            return self.expected_fps

        actual_fps = (len(timestamps) - 1) / duration_s
        return max(15.0, min(60.0, actual_fps))  # Clamp to reasonable range

    def clear(self) -> None:
        """Clear the buffer."""
        self._frames.clear()
        self._start_time_ms = None

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self):
        return iter(self._frames)

    def __getitem__(self, index):
        return self._frames[index]
