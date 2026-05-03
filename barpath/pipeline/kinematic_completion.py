"""
Kinematic rules to detect when a lift is biomechanically complete.
Uses velocity stability + position checks instead of simple thresholds.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from barpath.pipeline.realtime_processing.live_buffer import FrameData


class CompletionDetector:
    """Detects when a lift has reached its final position using kinematic rules.

    Instead of simple thresholds (bar passes knees), uses velocity stability
    and position checks to determine when a lift is complete.
    """

    # Configuration constants - tuned for real-world performance
    VELOCITY_STABILITY_THRESHOLD = (
        15.0  # pixels/frame - reduced from 5 for more detection
    )
    STABILITY_DURATION_MS = 200.0  # Must be stable for 200ms
    SNATCH_OVERHEAD_RATIO = 0.35  # Bar in top 35% of frame (increased from 30%)
    CLEAN_SHOULDER_TOLERANCE_PX = 80  # ±80px from shoulder height (increased from 50)
    MIN_VELOCITY_FRAMES = 5  # Minimum frames for velocity calculation

    def __init__(self, frame_height: int, frame_width: int):
        self.frame_height = frame_height
        self.frame_width = frame_width

    def is_complete(self, frames: List[FrameData], predicted_class: str) -> bool:
        """
        Check if lift is complete based on class-specific kinematic rules.

        Args:
            frames: Recent window of frames (typically last 500ms+)
            predicted_class: 'snatch', 'clean', or 'jerk'

        Returns:
            True if lift appears kinematically complete
        """
        if len(frames) < 10:  # Need at least 300ms at 30fps
            return False

        # Get barbell positions
        bar_positions = self._get_barbell_positions(frames)
        if len(bar_positions) < 10:
            return False

        # Check velocity stability first (must be stable before checking position)
        velocities = np.diff(bar_positions[:, 1])
        if len(velocities) < self.MIN_VELOCITY_FRAMES:
            return False

        # Check if velocity has been stable (near zero) for recent frames
        recent_velocities = velocities[-self.MIN_VELOCITY_FRAMES :]
        if not self._is_velocity_stable(recent_velocities):
            return False

        # Now check position based on lift type
        current_bar_y = bar_positions[-1, 1]

        if predicted_class == "snatch":
            return self._check_snatch_complete(current_bar_y, frames)
        elif predicted_class == "clean":
            return self._check_clean_complete(current_bar_y, frames)
        elif predicted_class == "jerk":
            return self._check_jerk_complete(current_bar_y, frames, velocities)

        return False

    def is_complete_any(self, frames: List[FrameData]) -> Tuple[bool, str]:
        """
        Check if any lift type appears complete.

        Returns:
            (is_complete, lift_class) - returns class that appears complete
        """
        for lift_class in ["snatch", "clean", "jerk"]:
            if self.is_complete(frames, lift_class):
                return True, lift_class
        return False, "none"

    def _is_velocity_stable(self, velocities: np.ndarray) -> bool:
        """Check if barbell velocity has been near zero for recent frames."""
        if len(velocities) == 0:
            return False
        # Use median instead of mean for robustness against outliers
        median_vel = np.median(np.abs(velocities))
        return median_vel < self.VELOCITY_STABILITY_THRESHOLD

    def _check_snatch_complete(self, bar_y: float, frames: List[FrameData]) -> bool:
        """Bar should be overhead (top 35% of frame) and stable."""
        overhead_threshold = self.frame_height * self.SNATCH_OVERHEAD_RATIO
        return bar_y < overhead_threshold

    def _check_clean_complete(self, bar_y: float, frames: List[FrameData]) -> bool:
        """Bar should be at shoulder height ±tolerance."""
        # Get shoulder height from landmarks
        shoulder_y = self._get_shoulder_height(frames)
        if shoulder_y is None:
            # Fallback: assume bar is at shoulder if in middle-upper portion of frame
            return self.frame_height * 0.3 < bar_y < self.frame_height * 0.7

        # Check if bar is near shoulder height
        return abs(bar_y - shoulder_y) < self.CLEAN_SHOULDER_TOLERANCE_PX

    def _check_jerk_complete(
        self, bar_y: float, frames: List[FrameData], velocities: np.ndarray
    ) -> bool:
        """Bar overhead after dip+drive motion detected."""
        # First check: bar is overhead
        if bar_y >= self.frame_height * self.SNATCH_OVERHEAD_RATIO:
            return False

        # Second check: verify dip+drive occurred earlier
        # Look for characteristic velocity pattern: down then up
        if len(velocities) < 15:
            return False

        # Check for dip (bar moving down = positive velocity in image coords)
        recent_vel = velocities[-15:-5]  # Earlier frames
        later_vel = velocities[-10:]  # More recent frames

        # Dip: velocity becomes positive then negative
        had_down_motion = np.any(recent_vel > self.VELOCITY_STABILITY_THRESHOLD)
        had_up_motion = np.any(later_vel < -self.VELOCITY_STABILITY_THRESHOLD)

        return bool(had_down_motion and had_up_motion)

    def _get_barbell_positions(self, frames: List[FrameData]) -> np.ndarray:
        """Extract barbell centers from frames."""
        positions = []
        last_valid = None

        for frame in frames:
            if frame.barbell_center is not None:
                positions.append(list(frame.barbell_center))
                last_valid = frame.barbell_center
            elif last_valid is not None:
                positions.append(list(last_valid))
            else:
                positions.append([0.0, 0.0])

        if not positions:
            return np.array([], dtype=np.float64).reshape(0, 2)

        return np.array(positions, dtype=np.float64)

    def _get_shoulder_height(self, frames: List[FrameData]) -> Optional[float]:
        """Get average shoulder height from landmarks."""
        shoulder_y_values = []

        for frame in frames:
            landmarks = frame.landmarks
            # MediaPipe indices: 11=left_shoulder, 12=right_shoulder
            left_shoulder = landmarks.get(11)  # (x, y, z, visibility)
            right_shoulder = landmarks.get(12)

            if left_shoulder and left_shoulder[3] > 0.3:
                shoulder_y_values.append(left_shoulder[1] * self.frame_height)
            if right_shoulder and right_shoulder[3] > 0.3:
                shoulder_y_values.append(right_shoulder[1] * self.frame_height)

        if not shoulder_y_values:
            return None

        return sum(shoulder_y_values) / len(shoulder_y_values)

    def _get_hip_height(self, frames: List[FrameData]) -> Optional[float]:
        """Get average hip height from landmarks."""
        hip_y_values = []

        for frame in frames:
            landmarks = frame.landmarks
            # MediaPipe indices: 23=left_hip, 24=right_hip
            left_hip = landmarks.get(23)
            right_hip = landmarks.get(24)

            if left_hip and left_hip[3] > 0.3:
                hip_y_values.append(left_hip[1] * self.frame_height)
            if right_hip and right_hip[3] > 0.3:
                hip_y_values.append(right_hip[1] * self.frame_height)

        if not hip_y_values:
            return None

        return sum(hip_y_values) / len(hip_y_values)


class DipDetector:
    """Detects jerk dip phase for completion validation.

    Analyzes velocity profile for characteristic dip pattern:
    1. Bar moves down slightly (positive velocity)
    2. Then bar moves up rapidly (negative velocity)
    """

    DIP_VELOCITY_THRESHOLD = 10.0  # pixels/frame - reduced for more sensitivity
    MIN_DIP_DURATION_MS = 150.0  # Minimum dip duration
    MAX_DIP_DURATION_MS = 800.0  # Maximum dip duration

    def detect_dip(self, frames: List[FrameData]) -> bool:
        """Detect if a dip occurred in recent frames."""
        if len(frames) < 20:  # Need ~600ms at 30fps
            return False

        y_positions = self._get_y_positions(frames)
        if len(y_positions) < 20:
            return False

        # Compute velocity
        velocities = np.diff(y_positions)

        # Find regions where velocity exceeds threshold (moving down in image coords)
        down_mask = velocities > self.DIP_VELOCITY_THRESHOLD
        up_mask = velocities < -self.DIP_VELOCITY_THRESHOLD

        if not np.any(down_mask):
            return False

        # Find the dip: down motion followed by up motion
        down_indices = np.where(down_mask)[0]
        up_indices = np.where(up_mask)[0]

        if len(down_indices) == 0 or len(up_indices) == 0:
            return False

        # Check if there's up motion after the last down motion
        last_down = down_indices[-1]
        has_recovery = np.any(up_indices > last_down)

        if has_recovery:
            # Verify dip duration is reasonable
            timestamps = np.array([f.timestamp_ms for f in frames])
            dip_duration = timestamps[last_down + 1] - timestamps[0]

            if self.MIN_DIP_DURATION_MS <= dip_duration <= self.MAX_DIP_DURATION_MS:
                return True

        return False

    def detect_dip_start(self, frames: List[FrameData]) -> Optional[int]:
        """Find the frame index where dip starts, if any."""
        if len(frames) < 15:
            return None

        y_positions = self._get_y_positions(frames)
        if len(y_positions) < 15:
            return None

        velocities = np.diff(y_positions)

        # Find first significant down motion
        for i, vel in enumerate(velocities[:-5]):
            if vel > self.DIP_VELOCITY_THRESHOLD:
                # Check there's a recovery (up motion) after
                if np.any(velocities[i + 5 :] < -self.DIP_VELOCITY_THRESHOLD):
                    return i

        return None

    def _get_y_positions(self, frames: List[FrameData]) -> np.ndarray:
        """Extract y positions from frames."""
        positions = []
        last_y = None

        for frame in frames:
            if frame.barbell_center is not None:
                positions.append(frame.barbell_center[1])
                last_y = frame.barbell_center[1]
            elif last_y is not None:
                positions.append(last_y)
            else:
                positions.append(0.0)

        return np.array(positions, dtype=np.float64)


class VelocityProfileAnalyzer:
    """Analyzes velocity profiles to detect lift phases and patterns."""

    @staticmethod
    def analyze_velocity_profile(velocities: np.ndarray) -> Dict:
        """Extract features from velocity profile."""
        features = {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "is_s_curve": False,
            "peak_count": 0,
            "direction_changes": 0,
        }

        if len(velocities) < 5:
            return features

        features["mean"] = float(np.mean(velocities))
        features["std"] = float(np.std(velocities))
        features["min"] = float(np.min(velocities))
        features["max"] = float(np.max(velocities))

        # Count direction changes
        sign_changes = np.diff(np.sign(velocities))
        features["direction_changes"] = int(np.sum(sign_changes != 0))

        # Detect S-curve pattern (sign changes: negative -> positive -> negative)
        # This is characteristic of a complete pull + recovery
        if features["direction_changes"] >= 2:
            features["is_s_curve"] = True

        # Count peaks (local extrema)
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(velocities, distance=5)
        features["peak_count"] = len(peaks)

        return features
