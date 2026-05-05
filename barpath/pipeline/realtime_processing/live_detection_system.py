"""
Main orchestrator for live lift detection.
State machine integrating buffer, classifier, completion detector.
"""

import time
from enum import Enum, auto
from typing import Callable, Dict, Optional

from barpath.pipeline.kinematic_completion import CompletionDetector
from barpath.pipeline.lift_classifier import LiveLiftClassifier, find_model_path

from .live_buffer import CircularFrameBuffer, FrameData
from .live_feature_extractor import LiveFeatureExtractor
from .coaching_tip import compute_coaching_tip


class DetectionState(Enum):
    """States for the detection state machine."""

    IDLE = auto()
    DETECTING = auto()
    COMPLETE = auto()
    JERK_WATCH = auto()
    DISPLAYING = auto()


class LiftDetectionSystem:
    """
    Main orchestrator for real-time lift detection.

    Usage:
        system = LiftDetectionSystem(frame_height=720, frame_width=1280)
        for frame in video_stream:
            result = system.process_frame(frame)
            if result:
                display(result['class'], result['confidence'])
    """

    # Configuration
    WINDOW_DURATION_MS = 2000.0  # 2 second classification window
    BUFFER_DURATION_MS = 4000.0  # 4 second circular buffer
    CLASSIFICATION_INTERVAL_FRAMES = 5  # Classify every 5 frames (~167ms at 30fps)
    CONFIDENCE_THRESHOLD = 0.40  # Below this triggers confirmation
    JERK_WATCH_DURATION_MS = 4000.0  # 4 seconds to detect jerk after clean
    DISPLAY_DURATION_MS = 3000.0  # Show result for 3 seconds
    MAX_DETECTION_FRAMES = 300  # Force complete after 10 seconds at 30fps

    def __init__(
        self,
        frame_height: int = 720,
        frame_width: int = 1280,
        fps: float = 30.0,
        model_path: Optional[str] = None,
    ):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.fps = fps

        # Components
        self.buffer = CircularFrameBuffer(
            max_duration_ms=self.BUFFER_DURATION_MS, fps=fps
        )
        self.extractor = LiveFeatureExtractor(frame_width, frame_height, fps)
        self.classifier = LiveLiftClassifier(model_path or find_model_path())
        self.completion_detector = CompletionDetector(frame_height, frame_width)

        # State machine
        self.state = DetectionState.IDLE
        self.frame_count = 0
        self.detection_frame_start = 0

        # Results
        self.current_result: Optional[Dict] = None
        self.pending_clean_result: Optional[Dict] = None
        self.display_start_time: float = 0.0
        self.jerk_watch_start_time: float = 0.0

        # Coaching tip
        self.coaching_tip: Optional[str] = None
        self.tip_display_start: float = 0.0

        # Callbacks (set by GUI)
        self.on_detection: Optional[Callable] = None
        self.on_confirmation_needed: Optional[Callable] = None

    def process_frame(self, frame_data: FrameData) -> Optional[Dict]:
        """
        Process a single frame from the live stream.

        Args:
            frame_data: FrameData with barbell position, landmarks, etc.

        Returns:
            Detection result dict if a lift state changed, else None
        """
        self.buffer.add_frame(frame_data)
        self.frame_count += 1

        # Handle state
        if self.state == DetectionState.IDLE:
            return self._handle_idle()
        elif self.state == DetectionState.DETECTING:
            return self._handle_detecting()
        elif self.state == DetectionState.COMPLETE:
            return self._handle_complete()
        elif self.state == DetectionState.JERK_WATCH:
            return self._handle_jerk_watch()
        elif self.state == DetectionState.DISPLAYING:
            return self._handle_displaying()

        return None

    def _handle_idle(self) -> Optional[Dict]:
        """Check for lift start using classifier on sliding window."""
        if not self.buffer.is_ready:
            return None

        # Only classify every N frames for performance
        if self.frame_count % self.CLASSIFICATION_INTERVAL_FRAMES != 0:
            return None

        # Skip if no barbell detected
        if not self._has_barbell_data():
            return None

        # Get window and classify
        window = self.buffer.get_window(self.WINDOW_DURATION_MS)
        if len(window) < 20:
            return None

        try:
            features = self.extractor.window_to_features(window)
            if features is None or len(features) < 37:
                return None

            prediction = self.classifier.predict_live(features, apply_smoothing=True)

            # Check if any lift class has reasonable confidence
            if prediction["class"] != "none" and prediction["confidence"] > 0.50:
                self.state = DetectionState.DETECTING
                self.detection_frame_start = self.frame_count
                self.current_result = prediction.copy()
                # Also store as detecting result
                return {
                    "state": "detecting",
                    "class": prediction["class"],
                    "confidence": prediction["confidence"],
                }

        except Exception:
            pass  # Silently ignore classification errors

        return None

    def _handle_detecting(self) -> Optional[Dict]:
        """Monitor for kinematic completion."""
        # Check if we've been detecting too long (force complete after 10s)
        if self.frame_count - self.detection_frame_start > self.MAX_DETECTION_FRAMES:
            return self._force_complete()

        # Not every frame - check periodically
        if self.frame_count % 3 != 0:
            return None

        # Get recent frames for completion check
        recent = self.buffer.get_recent_frames(30)  # Last ~1 second
        if len(recent) < 15:
            return None

        predicted_class = self.current_result.get("class", "none") if self.current_result else "none"

        # Check kinematic completion
        try:
            if self.completion_detector.is_complete(recent, predicted_class):
                return self._finalize_detection()
        except Exception:
            pass

        return None

    def _handle_complete(self) -> Optional[Dict]:
        """Handle completed lift: display or confirm."""
        if self.current_result and self.current_result["confidence"] < self.CONFIDENCE_THRESHOLD:
            # Low confidence - signal for confirmation UI
            if self.on_confirmation_needed:
                self.on_confirmation_needed(self.current_result)
            # Still show it but mark as low confidence
            result = self.current_result.copy() if self.current_result else {}
            result["state"] = "complete"
            result["needs_confirmation"] = True
            self.state = DetectionState.DISPLAYING
            self.display_start_time = time.time()
            return result

        # High confidence - display directly
        return self._display_current_result()

    def _handle_jerk_watch(self) -> Optional[Dict]:
        """Watch for jerk after clean detection."""
        elapsed = (time.time() - self.jerk_watch_start_time) * 1000

        # Check timeout
        if elapsed >= self.JERK_WATCH_DURATION_MS:
            # No jerk detected - finalize as clean
            self.state = DetectionState.DISPLAYING
            self.display_start_time = time.time()
            return self.current_result

        # Not every frame
        if self.frame_count % 3 != 0:
            return None

        # Check for jerk
        try:
            window = self.buffer.get_window(1000)  # Last 1 second
            if len(window) >= 15:
                features = self.extractor.window_to_features(window)
                prediction = self.classifier.predict(features)

                # Check if this looks like a jerk
                if prediction["class"] == "jerk" and prediction["confidence"] > 0.50:
                    # Found jerk! Merge to clean+jerk
                    merged_class = "clean_jerk"
                    merged_conf = min(
                        self.pending_clean_result["confidence"] if self.pending_clean_result else 0.0,
                        prediction["confidence"],
                    )

                    self.current_result = {
                        "class": merged_class,
                        "confidence": merged_conf,
                        "state": "complete",
                    }
                    self.state = DetectionState.DISPLAYING
                    self.display_start_time = time.time()
                    return self.current_result
        except Exception:
            pass

        return None

    def _handle_displaying(self) -> Optional[Dict]:
        """Wait for display duration to expire."""
        elapsed = (time.time() - self.display_start_time) * 1000

        if elapsed >= self.DISPLAY_DURATION_MS:
            self._reset()

        return None

    def _finalize_detection(self) -> Dict:
        """Prepare final detection result."""
        predicted_class = self.current_result.get("class", "none") if self.current_result else "none"

        # If this was a clean, enter jerk watch instead of displaying immediately
        if predicted_class == "clean":
            self.pending_clean_result = self.current_result.copy() if self.current_result else {} if self.current_result else {}
            self.state = DetectionState.JERK_WATCH
            self.jerk_watch_start_time = time.time()

            # Return clean result for immediate display
            result = self.current_result.copy() if self.current_result else {}
            result["state"] = "complete"
            return result

        # Otherwise display immediately
        return self._display_current_result()

    def _display_current_result(self) -> Dict:
        """Display current result."""
        self.state = DetectionState.DISPLAYING
        self.display_start_time = time.time()

        # Compute coaching tip on display entry
        detected_class = self.current_result.get("class", "none") if self.current_result else "none"
        buffer_frames = self.buffer.get_recent_frames(self.buffer.num_frames)
        self.coaching_tip = compute_coaching_tip(buffer_frames, detected_class)
        self.tip_display_start = time.time()

        result = self.current_result.copy() if self.current_result else {}
        result["state"] = "complete"
        return result

    @property
    def current_tip(self) -> Optional[str]:
        """Return coaching tip if within display duration, else None."""
        from barpath.pipeline.config import COACHING_TIP_DURATION_S
        if self.coaching_tip and self.state == DetectionState.DISPLAYING:
            elapsed = time.time() - self.tip_display_start
            if elapsed < COACHING_TIP_DURATION_S:
                return self.coaching_tip
        return None

    def _force_complete(self) -> Dict:
        """Force completion after max duration."""
        return self._display_current_result()

    def _reset(self) -> None:
        """Reset to IDLE state."""
        self.state = DetectionState.IDLE
        self.current_result = None
        self.pending_clean_result = None
        self.buffer.clear()
        self.classifier.reset_smoothing()

    def _has_barbell_data(self) -> bool:
        """Check if buffer has valid barbell data."""
        recent = self.buffer.get_recent_frames(10)
        for frame in recent:
            if frame.barbell_center is not None:
                return True
        return False

    def user_confirmed(self, confirmed_class: str) -> None:
        """
        Called by UI when user manually confirms lift type.

        Args:
            confirmed_class: User-selected class name
        """
        self.current_result = {
            "class": confirmed_class,
            "confidence": 1.0,  # User confirmed = 100%
            "state": "complete",
            "user_confirmed": True,
        }

        # Store for retraining
        self._save_for_retraining(confirmed_class)

        self.state = DetectionState.DISPLAYING
        self.display_start_time = time.time()

    def _save_for_retraining(self, confirmed_class: str) -> None:
        """Save current window data for future retraining."""
        # TODO: Save to outputs/uncertain_lifts/ with confirmed label
        pass

    @property
    def is_ready(self) -> bool:
        """Check if system is ready for detection."""
        return self.buffer.is_ready and self.classifier.is_loaded

    @property
    def is_detecting(self) -> bool:
        """Currently in DETECTING state."""
        return self.state == DetectionState.DETECTING

    @property
    def is_displaying(self) -> bool:
        """Currently showing a result."""
        return self.state == DetectionState.DISPLAYING

    @property
    def status_text(self) -> str:
        """Human-readable status for UI."""
        if self.state == DetectionState.IDLE:
            return "Waiting for lift..."
        elif self.state == DetectionState.DETECTING:
            cls = self.current_result.get("class", "unknown") if self.current_result else "unknown"
            conf = self.current_result.get("confidence", 0) if self.current_result else 0
            return f"Detecting {cls}... ({conf:.0%})"
        elif self.state == DetectionState.JERK_WATCH:
            return "Clean detected! Watching for jerk..."
        elif self.state == DetectionState.COMPLETE:
            return "Lift complete!"
        elif self.state == DetectionState.DISPLAYING:
            cls = self.current_result.get("class", "unknown") if self.current_result else "unknown"
            return f"{cls.upper()} ({self.current_result.get('confidence', 0) if self.current_result else 0:.0%})"
        return ""

    @property
    def current_class(self) -> str:
        """Get current detected class."""
        if self.current_result:
            return self.current_result.get("class", "none")
        return "none"

    @property
    def current_confidence(self) -> float:
        """Get current confidence."""
        if self.current_result:
            return self.current_result.get("confidence", 0.0)
        return 0.0

    def get_buffer(self) -> CircularFrameBuffer:
        """Get the frame buffer for drawing."""
        return self.buffer


def create_detection_system(
    frame_height: int = 720,
    frame_width: int = 1280,
    fps: float = 30.0,
    model_path: Optional[str] = None,
) -> LiftDetectionSystem:
    """Create detection system with defaults."""
    return LiftDetectionSystem(
        frame_height=frame_height,
        frame_width=frame_width,
        fps=fps,
        model_path=model_path,
    )
