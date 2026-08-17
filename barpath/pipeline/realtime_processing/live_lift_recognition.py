"""Live Lift Recognition Module.

State machine for real-time lift detection, bar path drawing,
and lift type classification during the webcam preview.

States:
    IDLE -> TRIGGERED -> RECORDING -> CLASSIFYING -> DISPLAYING -> IDLE
"""

from __future__ import annotations

import time
from collections import deque
from enum import Enum, auto
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pandas as pd

from barpath.pipeline.lift_detection_features import (
    build_lift_dataframe,
    detect_clean_jerk_split_point,
    load_lift_detection_model,
    predict_lift_type,
)
from barpath.pipeline.step4_helpers.compiled_analyzer import (
    CompiledAnalyzer,
    load_baselines_from_json,
)
from barpath.pipeline.step4_helpers.feature_extraction import extract_technique_features

from .live_window_features import extract_window_features

# ============================================================================
# Phase color scheme (matches step 5 PHASE_COLOR_SCHEMES)
# ============================================================================

PHASE_COLORS_BGR = {
    0: (32, 32, 224),  # Red - Pull / Dip
    1: (0, 120, 240),  # Orange - Pull-under / Drive
    2: (32, 160, 24),  # Green - Recovery
}

PHASE_COLORS_6_BGR = {
    0: (32, 32, 224),  # Red - Clean Pull
    1: (0, 120, 240),  # Orange - Clean Pull-under
    2: (32, 160, 24),  # Green - Clean Recovery
    3: (224, 96, 32),  # Blue - Jerk Dip
    4: (240, 32, 150),  # Purple - Jerk Drive
    5: (240, 200, 32),  # Cyan - Jerk Recovery
}


class LiftState(Enum):
    """State machine states for live lift recognition."""

    IDLE = auto()
    TRIGGERED = auto()
    RECORDING = auto()
    CLASSIFYING = auto()
    DISPLAYING = auto()
    SHOULDER_WAIT = auto()  # After clean, waiting for jerk


class FrameData:
    """Data captured from a single frame."""

    __slots__ = (
        "barbell_box",
        "barbell_center",
        "knee_y_avg",
        "landmarks",
        "timestamp_ms",
    )

    def __init__(
        self,
        barbell_center: tuple[float, float] | None,
        barbell_box: tuple[int, int, int, int] | None,
        landmarks: dict[int, tuple[float, float, float, float]],
        knee_y_avg: float,
        timestamp_ms: float,
    ):
        self.barbell_center = barbell_center
        self.barbell_box = barbell_box
        self.landmarks = landmarks
        self.knee_y_avg = knee_y_avg
        self.timestamp_ms = timestamp_ms


class LiveLiftRecognizer:
    """Real-time lift recognition state machine.

    Manages a circular buffer of frame data, detects lift triggers
    (barbell passing knees), records complete lifts, classifies them,
    and provides bar path drawing + label overlay.
    """

    def __init__(
        self,
        model_path: str,
        fps: float = 30.0,
        buffer_seconds: float = 1.0,
        display_seconds: float = 3.0,
        trigger_stability_frames: int = 3,
        max_recording_seconds: float = 8.0,
        lifter: str = "generic",
        gender: str = "male",
    ):
        """
        Args:
            model_path: Path to lift_detection_model.pkl
            fps: Camera frame rate
            buffer_seconds: Seconds of pre-trigger data to keep
            display_seconds: Seconds to show result after classification
            trigger_stability_frames: Consecutive frames for stable trigger
            max_recording_seconds: Max seconds before force-stopping a recording
            lifter: Lifter name for baseline selection
            gender: Gender for baseline selection
        """
        self.fps = fps
        self.display_seconds = display_seconds
        self.trigger_stability_frames = trigger_stability_frames
        self.max_recording_frames = int(max_recording_seconds * fps)

        # Circular buffer: stores recent frame data (1 second)
        self._buffer: deque[FrameData] = deque(maxlen=int(buffer_seconds * fps))

        # State
        self.state = LiftState.IDLE
        self._trigger_count = 0

        # Lift recording
        self._lift_frames: list[FrameData] = []
        self._recording_frame_count = 0

        # Classification result
        self._predicted_class: str | None = None
        self._predicted_confidence: float = 0.0
        self._is_clean_jerk: bool = False
        self._display_start_time: float = 0.0

        # Display stack for clean + jerk sequence
        self._display_stack: list[str] = []

        # Live preview window classification
        self._live_model_data: dict[str, Any] | None = None
        self._live_model_loaded: bool = False
        self._live_model_path = model_path.replace("lift_detection_model", "live_lift_model")
        self._class_prob_history: deque[dict[str, float]] = deque(maxlen=20)
        self._classification_interval = 5  # classify every 5 frames
        self._frame_counter = 0

        # Shoulder wait state
        self._shoulder_wait_start_time: float = 0.0
        self._shoulder_wait_timeout: float = 3.0  # seconds to wait for jerk
        self._shoulder_reference_y: float = 0.0  # y position when entering shoulder wait

        # Post-peak stabilization detection
        self._peak_detected: bool = False
        self._peak_frame_count: int = 0
        self._stabilization_frames: int = int(fps)  # 1 second of stabilization
        self._peak_y: float = float("inf")
        self._shoulder_y_estimate: float = 0.0  # Average shoulder y during lift

        # Real-time shoulder stabilization tracking
        self._shoulder_stable_count: int = 0  # frames bar has been stable at shoulder
        self._shoulder_stable_threshold: int = int(fps * 0.5)  # 0.5s of stability

        # Jerk expectation flag (set when coming from SHOULDER_WAIT)
        self._expecting_jerk: bool = False

        # Path drawing state
        self._path_points: list[tuple[int, int]] = []
        self._path_phases: list[int] = []
        self._trimmed_path_points: list[tuple[int, int]] = []
        self._trimmed_path_phases: list[int] = []

        # Analysis result
        self._top_fault: dict[str, Any] | None = None
        self._tip_display_start: float = 0.0

        # Model (legacy fallback)
        self._model_data: dict[str, Any] | None = None
        self._model_loaded: bool = False
        self._model_path = model_path

        # Baseline loading (lazy)
        self._lifter = lifter
        self._gender = gender
        self._baselines: dict[str, dict[str, dict[str, float]]] | None = None
        self._baselines_loaded: bool = False

    def _load_baselines(self) -> None:
        """Lazy-load baselines from per-lifter or pooled report."""
        if self._baselines_loaded:
            return
        models_base = Path(__file__).parent.parent / "models" / "analysis"
        if self._lifter and self._lifter != "generic":
            lifter_json = models_base / f"pro_baseline_report_{self._lifter}.json"
            if lifter_json.exists():
                self._baselines = load_baselines_from_json(lifter_json)
                self._baselines_loaded = True
                return
        pooled_json = models_base / "pro_baseline_report.json"
        self._baselines = load_baselines_from_json(pooled_json)
        self._baselines_loaded = True

    def _ensure_model_loaded(self) -> bool:
        """Lazy-load the classification model."""
        if self._model_loaded:
            return self._model_data is not None

        self._model_data = load_lift_detection_model(self._model_path)
        self._model_loaded = True
        return self._model_data is not None

    def _ensure_live_model_loaded(self) -> bool:
        """Lazy-load the live window classification model."""
        if self._live_model_loaded:
            return self._live_model_data is not None

        try:
            import pickle

            with open(self._live_model_path, "rb") as f:
                self._live_model_data = pickle.load(f)
        except (FileNotFoundError, Exception):
            self._live_model_data = None
        self._live_model_loaded = True
        return self._live_model_data is not None

    def _classify_current_window(self, frame_width: int, frame_height: int) -> None:
        """Classify the current partial window using the live model."""
        if not self._lift_frames or len(self._lift_frames) < 15:
            return

        # Build current window DataFrame
        barbell_y = []
        barbell_x = []
        timestamps = []
        landmarks_list = []

        for f in self._lift_frames:
            if f.barbell_center is not None:
                barbell_y.append(f.barbell_center[1])
                barbell_x.append(f.barbell_center[0])
            elif barbell_y:
                barbell_y.append(barbell_y[-1])
                barbell_x.append(barbell_x[-1])
            else:
                continue
            timestamps.append(f.timestamp_ms)
            landmarks_list.append(f.landmarks)

        # Instant jerk detection: if bar starts at shoulder height
        if len(barbell_y) >= 10:
            start_y_norm = barbell_y[0] / frame_height
            if start_y_norm < 0.5:
                self._predicted_class = "JERK"
                self._predicted_confidence = 0.95
                self._class_prob_history.clear()
                self._class_prob_history.append({"snatch": 0.0, "clean": 0.0, "jerk": 1.0})
                return

        df = build_lift_dataframe(
            barbell_y=barbell_y,
            barbell_x=barbell_x,
            timestamps_ms=timestamps,
            landmarks_list=landmarks_list,
            frame_width=frame_width,
            frame_height=frame_height,
            fps=self.fps,
        )

        if df.empty or not self._ensure_live_model_loaded():
            return

        features = extract_window_features(df)
        if not features:
            return

        model = self._live_model_data
        if model is None:
            return

        X_df = pd.DataFrame(
            [[features.get(name, 0.0) for name in model["feature_names"]]],
            columns=model["feature_names"],
        )
        X_scaled = model["scaler"].transform(X_df)
        probs = model["classifier"].predict_proba(X_scaled)[0]
        classes = [str(c) for c in model["classifier"].classes_]

        prob_dict = {c: float(p) for c, p in zip(classes, probs, strict=False)}
        self._class_prob_history.append(prob_dict)
        self._update_smoothed_prediction()

    def _update_smoothed_prediction(self) -> None:
        """Compute smoothed prediction from probability history."""
        if not self._class_prob_history:
            return

        # Simple average of recent probabilities
        avg_probs: dict[str, float] = {}
        for key in self._class_prob_history[0]:
            values = [p[key] for p in self._class_prob_history if key in p]
            avg_probs[key] = float(np.mean(values)) if values else 0.0

        best_class = max(avg_probs, key=lambda k: avg_probs[k])
        best_prob = avg_probs[best_class]

        self._predicted_class = self._format_lift_name(best_class)
        self._predicted_confidence = best_prob

    def update(
        self,
        barbell_center: tuple[float, float] | None,
        barbell_box: tuple[int, int, int, int] | None,
        landmarks: dict[int, tuple[float, float, float, float]],
        timestamp_ms: float,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Feed a new frame into the state machine.

        Args:
            barbell_center: (x, y) pixel center of barbell bbox, or None
            barbell_box: (x1, y1, x2, y2) bbox, or None
            landmarks: MediaPipe landmarks {idx: (x_norm, y_norm, z, vis)}
            timestamp_ms: Frame timestamp in milliseconds
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        # Compute knee y-coordinate
        knee_y_avg = self._compute_knee_y(landmarks, frame_height)

        frame_data = FrameData(
            barbell_center=barbell_center,
            barbell_box=barbell_box,
            landmarks=landmarks,
            knee_y_avg=knee_y_avg,
            timestamp_ms=timestamp_ms,
        )

        if self.state == LiftState.IDLE:
            self._handle_idle(frame_data, frame_height)
        elif self.state == LiftState.TRIGGERED:
            self._handle_triggered(frame_data, frame_height)
        elif self.state == LiftState.RECORDING:
            self._handle_recording(frame_data, frame_height)
            # Real-time window classification during recording
            self._frame_counter += 1
            if self._frame_counter % self._classification_interval == 0:
                self._classify_current_window(frame_width, frame_height)
        elif self.state == LiftState.CLASSIFYING:
            self._handle_classifying(frame_width, frame_height)
        elif self.state == LiftState.DISPLAYING:
            self._handle_displaying()
        elif self.state == LiftState.SHOULDER_WAIT:
            self._handle_shoulder_wait(frame_data, frame_height)

    def _compute_knee_y(
        self,
        landmarks: dict[int, tuple[float, float, float, float]],
        frame_height: int,
    ) -> float:
        """Compute average knee y-coordinate in pixels."""
        left_knee = landmarks.get(25)
        right_knee = landmarks.get(26)

        y_values = []
        if left_knee and left_knee[3] > 0.1:
            y_values.append(left_knee[1] * frame_height)
        if right_knee and right_knee[3] > 0.1:
            y_values.append(right_knee[1] * frame_height)

        if y_values:
            return sum(y_values) / len(y_values)
        return float(frame_height) / 2  # fallback: middle of frame

    def _handle_idle(self, frame_data: FrameData, frame_height: int) -> None:
        """IDLE state: fill buffer, check for trigger."""
        self._buffer.append(frame_data)

        # ENHANCED TRIGGER: Use multiple signals for more reliable detection
        # 1. Classic: barbell passes knees going up
        # 2. NEW: Bar velocity indicates lift start (bar moving up rapidly)
        # 3. NEW: Combined confidence from ML classifier (periodic check)

        trigger_signal = self._check_enhanced_trigger(frame_data, frame_height)

        if trigger_signal:
            self._trigger_count += 1
            if self._trigger_count >= self.trigger_stability_frames:
                # Transition to TRIGGERED
                self.state = LiftState.TRIGGERED
                self._trigger_count = 0
        else:
            self._trigger_count = 0

    def _check_enhanced_trigger(self, frame_data: FrameData, frame_height: int) -> bool:
        """
        Enhanced trigger detection using multiple signals.

        Returns True if any trigger condition is met.
        """
        if frame_data.barbell_center is None:
            return False

        bar_y = frame_data.barbell_center[1]
        buf_max = self._buffer.maxlen

        # Signal 1: Classic - bar passes knees going up
        if (
            frame_data.knee_y_avg is not None
            and buf_max is not None
            and bar_y <= frame_data.knee_y_avg
            and len(self._buffer) >= buf_max
        ):
            return True

        # Signal 2: Velocity-based trigger
        # Check if bar is moving upward rapidly (velocity < negative threshold in image coords)
        if len(self._buffer) >= 10:
            velocities = self._compute_recent_velocities()
            if len(velocities) >= 5:
                # Moving up rapidly (negative velocity in image coords where Y increases downward)
                avg_vel = sum(velocities[-5:]) / 5
                if avg_vel < -25:  # Strong upward movement
                    return True

        # Signal 3: Isolated jerk trigger
        # Bar at shoulder height (0.2-0.45 normalized) and starting to dip
        bar_y_norm = bar_y / frame_height
        if 0.2 <= bar_y_norm <= 0.45 and len(self._buffer) >= 10:
            velocities = self._compute_recent_velocities()
            if len(velocities) >= 3:
                # Recent downward movement (dip) at shoulder height
                recent_vel = sum(velocities[-3:]) / 3
                if recent_vel > 3:  # Moving down slowly (dip)
                    return True

        return False

    def _compute_recent_velocities(self) -> list[float]:
        """Compute recent barbell velocities from buffer."""
        if len(self._buffer) < 2:
            return []

        velocities = []
        frames = list(self._buffer)

        for i in range(1, len(frames)):
            prev = frames[i - 1].barbell_center
            curr = frames[i].barbell_center
            if prev is not None and curr is not None:
                # Vertical velocity (px per frame)
                vel = curr[1] - prev[1]
                velocities.append(vel)

        return velocities

    def _handle_triggered(self, frame_data: FrameData, frame_height: int) -> None:
        """TRIGGERED state: copy buffer, transition to RECORDING."""
        # Copy buffer contents as pre-trigger data
        self._lift_frames = list(self._buffer)
        self._lift_frames.append(frame_data)
        self._recording_frame_count = 1

        # Initialize path drawing
        self._update_path_points(frame_height)

        self.state = LiftState.RECORDING

    def _handle_recording(self, frame_data: FrameData, frame_height: int) -> None:
        """RECORDING state: accumulate frames until stabilization complete."""
        self._lift_frames.append(frame_data)
        self._recording_frame_count += 1

        # Update shoulder estimate from landmarks
        self._update_shoulder_estimate(frame_data, frame_height)

        # Update path drawing in real-time
        self._update_path_points(frame_height)

        # REAL-TIME shoulder stabilization detection
        # If bar stabilizes at shoulder height, it's a CLEAN immediately
        if self._recording_frame_count > self.fps * 0.5:
            if self._check_shoulder_stabilization(frame_data, frame_height):
                self._shoulder_stable_count += 1
                if self._shoulder_stable_count >= self._shoulder_stable_threshold:
                    self._predicted_class = "CLEAN"
                    self._predicted_confidence = 0.98
                    self._display_stack = ["CLEAN"]
                    self._trim_path_to_peak()
                    self._shoulder_wait_start_time = time.time()
                    self._shoulder_reference_y = (
                        frame_data.barbell_center[1]
                        if frame_data.barbell_center
                        else frame_height * 0.3
                    )
                    self._lift_frames = []
                    self._recording_frame_count = 0
                    self._frame_counter = 0
                    self._class_prob_history.clear()
                    self._path_points = []
                    self._path_phases = []
                    self._peak_detected = False
                    self._peak_frame_count = 0
                    self._peak_y = float("inf")
                    self.state = LiftState.SHOULDER_WAIT
                    return
            else:
                self._shoulder_stable_count = 0

        # If we know this is a jerk (from SHOULDER_WAIT), show JERK immediately
        # and skip live model classification
        if self._expecting_jerk:
            self._predicted_class = "JERK"
            self._predicted_confidence = 0.98
            self._display_stack = ["CLEAN", "JERK"]

        # Check for peak detection
        if not self._peak_detected and self._recording_frame_count > 5:
            if self._detect_peak():
                self._peak_detected = True
                self._peak_frame_count = 0

        # After peak: record for stabilization period (1 second)
        if self._peak_detected:
            self._peak_frame_count += 1
            if self._peak_frame_count >= self._stabilization_frames:
                # Stabilization complete - classify
                self._trim_path_to_peak()
                self.state = LiftState.CLASSIFYING
                return

        # Also stop if bar drops significantly after peak
        if self._peak_detected and self._recording_frame_count > self.fps:
            if self._check_post_peak_drop(frame_data, frame_height):
                self._trim_path_to_peak()
                self.state = LiftState.CLASSIFYING
                return

        # Force stop if recording too long
        if self._recording_frame_count >= self.max_recording_frames:
            self._trim_path_to_peak()
            self.state = LiftState.CLASSIFYING

    def _update_shoulder_estimate(self, frame_data: FrameData, frame_height: int) -> None:
        """Update running estimate of shoulder height from landmarks."""
        lm = frame_data.landmarks
        left_sh = lm.get(11)
        right_sh = lm.get(12)

        sh_ys = []
        if left_sh and left_sh[3] > 0.1:
            sh_ys.append(left_sh[1] * frame_height)
        if right_sh and right_sh[3] > 0.1:
            sh_ys.append(right_sh[1] * frame_height)

        if sh_ys:
            avg_sh = sum(sh_ys) / len(sh_ys)
            if self._shoulder_y_estimate == 0.0:
                self._shoulder_y_estimate = avg_sh
            else:
                # Running average
                self._shoulder_y_estimate = 0.9 * self._shoulder_y_estimate + 0.1 * avg_sh

    def _check_shoulder_stabilization(self, frame_data: FrameData, frame_height: int) -> bool:
        """Check if bar has stabilized at shoulder height (low velocity near shoulder)."""
        if frame_data.barbell_center is None:
            return False

        bar_y = frame_data.barbell_center[1]
        shoulder_y = self._shoulder_y_estimate

        # No shoulder estimate yet, use fallback
        if shoulder_y == 0.0:
            shoulder_y = frame_height * 0.3

        # Check bar is near shoulder
        tolerance = frame_height * 0.12
        if abs(bar_y - shoulder_y) > tolerance:
            return False

        # Check bar velocity is low (stable)
        if len(self._lift_frames) < 5:
            return False

        recent_ys = []
        for f in self._lift_frames[-5:]:
            if f.barbell_center is not None:
                recent_ys.append(f.barbell_center[1])
        if len(recent_ys) < 5:
            return False

        velocity = abs(recent_ys[-1] - recent_ys[0])
        # Bar should be moving less than 1% of frame height over 5 frames
        return velocity < frame_height * 0.01

    def _detect_peak(self) -> bool:
        """Detect if bar has reached its peak (minimum y) and started descending."""
        if len(self._lift_frames) < 10:
            return False

        barbell_ys = [
            f.barbell_center[1] for f in self._lift_frames if f.barbell_center is not None
        ]
        if len(barbell_ys) < 10:
            return False

        # Find minimum y in last 10 frames
        recent_ys = barbell_ys[-10:]
        min_y = min(recent_ys)
        min_idx = recent_ys.index(min_y)

        # Peak is detected if minimum is not at the very end
        # (i.e., bar has started coming down)
        if min_idx < len(recent_ys) - 3:
            self._peak_y = min_y
            return True

        # Also detect peak if we've been recording a while and bar stopped going up
        if len(barbell_ys) > 20:
            recent_vel = barbell_ys[-1] - barbell_ys[-5]
            if recent_vel > -2:  # Nearly stopped or moving down
                self._peak_y = min(barbell_ys)
                return True

        return False

    def _check_post_peak_drop(self, current_frame: FrameData, frame_height: int) -> bool:
        """Check if bar has dropped significantly after peak."""
        if current_frame.barbell_center is None:
            return False

        bar_y = current_frame.barbell_center[1]

        # If bar dropped more than 15% of frame from peak, stop
        if self._peak_y < float("inf"):
            drop = bar_y - self._peak_y
            if drop > frame_height * 0.15:
                return True

        return False

    def _trim_path_to_peak(self) -> None:
        """Trim the path to remove descent after peak bar height."""
        if not self._lift_frames:
            return

        # Find peak frame (minimum barbell y)
        peak_idx = 0
        peak_y = float("inf")
        for i, f in enumerate(self._lift_frames):
            if f.barbell_center is not None and f.barbell_center[1] < peak_y:
                peak_y = f.barbell_center[1]
                peak_idx = i

        # Trim path points to peak
        if peak_idx > 0 and peak_idx < len(self._path_points):
            self._trimmed_path_points = self._path_points[: peak_idx + 1]
            self._trimmed_path_phases = self._path_phases[: peak_idx + 1]
        else:
            self._trimmed_path_points = list(self._path_points)
            self._trimmed_path_phases = list(self._path_phases)

    def _update_path_points(self, frame_height: int) -> None:
        """Recompute path points and phases from current lift frames."""
        self._path_points = []
        self._path_phases = []

        # Collect barbell centers and hip y-coordinates
        centers = []
        hip_ys = []
        for f in self._lift_frames:
            if f.barbell_center is not None:
                centers.append((int(f.barbell_center[0]), int(f.barbell_center[1])))
            elif centers:
                centers.append(centers[-1])
            else:
                continue

            # Compute hip y from landmarks (MediaPipe indices 23=left_hip, 24=right_hip)
            lm = f.landmarks
            left_hip = lm.get(23)
            right_hip = lm.get(24)
            hip_vals = []
            if left_hip and left_hip[3] > 0.1:
                hip_vals.append(left_hip[1] * frame_height)
            if right_hip and right_hip[3] > 0.1:
                hip_vals.append(right_hip[1] * frame_height)
            if hip_vals:
                hip_ys.append(sum(hip_vals) / len(hip_vals))
            elif hip_ys:
                hip_ys.append(hip_ys[-1])
            else:
                hip_ys.append(float(frame_height) / 2)

        if len(centers) < 2:
            return

        # Smooth path using simple moving average
        smoothed = self._smooth_path(centers)
        self._path_points = smoothed

        # Detect phases using hip data when available
        hip_arr = np.array(hip_ys[: len(smoothed)], dtype=np.float64)
        self._path_phases = self._detect_path_phases(smoothed, frame_height, hip_arr)

    def _smooth_path(self, points: list[tuple[int, int]], window: int = 5) -> list[tuple[int, int]]:
        """Apply simple moving average smoothing to path points."""
        if len(points) < window:
            return points

        smoothed = []
        half_w = window // 2
        for i in range(len(points)):
            start = max(0, i - half_w)
            end = min(len(points), i + half_w + 1)
            chunk_x = [p[0] for p in points[start:end]]
            chunk_y = [p[1] for p in points[start:end]]
            smoothed.append(
                (
                    int(sum(chunk_x) / len(chunk_x)),
                    int(sum(chunk_y) / len(chunk_y)),
                )
            )
        return smoothed

    def _detect_path_phases(
        self,
        points: list[tuple[int, int]],
        frame_height: int,
        hip_y: np.ndarray | None = None,
    ) -> list[int]:
        """Detect phases for path coloring.

        Uses the same logic as the normal post-process pipeline:
        - When hip_y is available: detect pull-under via hip velocity
          (mirrors step2_helpers/kinematics.detect_three_phases)
        - When hip_y is not available: fall back to velocity-only detection

        Phase 0 (Pull): bar accelerating upward, hips not yet dropping
        Phase 1 (Pull-under): hips actively descending under the bar
        Phase 2 (Recovery): hips stopped descending
        """
        n = len(points)
        if n < 4:
            return [0] * n

        ys = np.array([p[1] for p in points], dtype=np.float64)
        vel = np.gradient(ys)

        # Smooth velocity
        vel_smooth = vel.copy()
        if n >= 5:
            win = min(5, n if n % 2 == 1 else n - 1)
            if win >= 3:
                from scipy.signal import savgol_filter

                vel_smooth = savgol_filter(vel, window_length=win, polyorder=2)

        vel_arr = cast(np.ndarray, vel_smooth)

        # Use hip-based detection when available
        if hip_y is not None and len(hip_y) >= n:
            hip = hip_y[:n]
            # Check for valid hip data (nonzero)
            if np.any(hip > 0):
                return self._detect_phases_hip(vel_arr, hip, n)

        # Fallback: velocity-only
        return self._detect_phases_velocity_only(vel_arr, ys, n)

    @staticmethod
    def _detect_phases_hip(vel_smooth: np.ndarray, hip_y: np.ndarray, n: int) -> list[int]:
        """Detect phases using hip velocity (matches normal pipeline).

        In image coordinates: positive hip velocity = hips moving DOWN
        (athlete squatting under bar) = pull-under phase.
        """
        from scipy.signal import savgol_filter

        # Find pull start: bar moving up (positive vel in pixel coords
        # where y increases downward — wait, in pixel coords y increases
        # downward, so negative vel = bar moving up. But here vel is
        # np.gradient(ys) where ys are pixel y-values. So negative vel
        # means bar going up. We want the most negative velocity.
        vel_max = float(np.max(np.abs(vel_smooth)))
        vel_threshold = max(2.0, vel_max * 0.05)
        bar_moving_up = vel_smooth < -vel_threshold  # negative = upward in pixels

        if not np.any(bar_moving_up):
            return [0] * n

        pull_start = int(np.argmax(bar_moving_up))

        # Hip velocity after pull starts
        hip_after = hip_y[pull_start:]
        if len(hip_after) < 5:
            return [0] * n

        # Smooth hip
        hw = min(9, len(hip_after) if len(hip_after) % 2 == 1 else len(hip_after) - 1)
        hip_sm = savgol_filter(hip_after, window_length=hw, polyorder=3) if hw >= 3 else hip_after
        hip_sm = cast(np.ndarray, hip_sm)
        hip_vel = np.gradient(hip_sm)

        hip_std = float(np.std(hip_sm))
        hip_drop_thresh = hip_std * 0.1 if hip_std > 0 else 0.5

        # In pixel coords: positive hip_vel = hips dropping DOWN = pull-under
        hips_dropping = hip_vel > hip_drop_thresh

        pull_under_start: int | None = None
        if np.any(hips_dropping):
            pull_under_start = pull_start + int(np.argmax(hips_dropping))

        recovery_start: int | None = None
        if pull_under_start is not None:
            hip_after_pu = hip_y[pull_under_start:]
            if len(hip_after_pu) >= 5:
                hw2 = min(
                    9,
                    len(hip_after_pu) if len(hip_after_pu) % 2 == 1 else len(hip_after_pu) - 1,
                )
                if hw2 >= 3:
                    hip_sm2 = savgol_filter(hip_after_pu, window_length=hw2, polyorder=3)
                else:
                    hip_sm2 = hip_after_pu
                hip_sm2 = cast(np.ndarray, hip_sm2)
                hip_vel2 = np.gradient(hip_sm2)
                hips_stopped = hip_vel2 <= hip_drop_thresh * 0.5
                if np.any(hips_stopped):
                    recovery_start = pull_under_start + int(np.argmax(hips_stopped))

        phases = [0] * n
        if pull_under_start is not None:
            for i in range(pull_under_start, n):
                phases[i] = 1
        if recovery_start is not None:
            for i in range(recovery_start, n):
                phases[i] = 2
        return phases

    @staticmethod
    def _detect_phases_velocity_only(vel_smooth: np.ndarray, ys: np.ndarray, n: int) -> list[int]:
        """Fallback: detect phases from velocity and position only."""
        peak_vel_idx = int(np.argmin(vel_smooth))
        peak_height_idx = int(np.argmin(ys))

        if peak_height_idx <= peak_vel_idx:
            search_start = peak_vel_idx + 1
            if search_start < n:
                peak_height_idx = search_start + int(np.argmin(ys[search_start:]))
            else:
                peak_height_idx = min(peak_vel_idx + 1, n - 1)

        if peak_height_idx - peak_vel_idx < 2:
            peak_height_idx = min(peak_vel_idx + 2, n - 1)

        phases = [0] * n
        for i in range(n):
            if i < peak_vel_idx:
                phases[i] = 0
            elif i < peak_height_idx:
                phases[i] = 1
            else:
                phases[i] = 2
        return phases

    def _handle_classifying(self, frame_width: int, frame_height: int) -> None:
        """CLASSIFYING state: finalize classification using stabilization."""
        # If we were expecting a jerk (from SHOULDER_WAIT), force JERK
        if self._expecting_jerk:
            self._predicted_class = "JERK"
            self._predicted_confidence = 0.98
            # Ensure stack shows CLEAN + JERK without duplicates
            if "CLEAN" not in self._display_stack:
                self._display_stack.append("CLEAN")
            if "JERK" not in self._display_stack:
                self._display_stack.append("JERK")
            self._expecting_jerk = False
            self._display_start_time = time.time()
            self._tip_display_start = time.time()
            self.state = LiftState.DISPLAYING
            return

        barbell_ys = [
            f.barbell_center[1] for f in self._lift_frames if f.barbell_center is not None
        ]
        if not barbell_ys:
            self._predicted_class = "Unknown"
            self._predicted_confidence = 0.0
            self._display_start_time = time.time()
            self._tip_display_start = time.time()
            self.state = LiftState.DISPLAYING
            return

        # Get final stabilized position (average of last 10 frames)
        final_y = float(np.mean(barbell_ys[-10:])) if len(barbell_ys) >= 10 else barbell_ys[-1]
        final_y_norm = final_y / frame_height
        start_y_norm = barbell_ys[0] / frame_height

        # Get shoulder reference
        shoulder_y = self._shoulder_y_estimate
        if shoulder_y == 0.0:
            # Fallback: estimate from frame ratio
            shoulder_y = frame_height * 0.3

        # STABILIZATION-BASED CLASSIFICATION
        # If bar ends at shoulder level, it's a CLEAN - nothing else it could be
        shoulder_tolerance = frame_height * 0.08  # 8% of frame height tolerance
        is_at_shoulder = abs(final_y - shoulder_y) < shoulder_tolerance

        if is_at_shoulder:
            # Bar stabilized at shoulder = CLEAN
            self._predicted_class = "CLEAN"
            self._predicted_confidence = 0.98
        elif final_y_norm < 0.15:
            # Bar well overhead = SNATCH or JERK
            if start_y_norm < 0.5:
                self._predicted_class = "JERK"
                self._predicted_confidence = 0.95
            else:
                self._predicted_class = "SNATCH"
                self._predicted_confidence = 0.95
        else:
            # Ambiguous - use model prediction if confident
            if self._predicted_class is not None and self._predicted_confidence > 0.6:
                pass  # Keep live prediction
            else:
                # Fallback: if it started from floor and peaked low, it's clean
                if start_y_norm > 0.5 and final_y_norm > 0.2:
                    self._predicted_class = "CLEAN"
                    self._predicted_confidence = 0.85
                else:
                    self._predicted_class = "Unknown"
                    self._predicted_confidence = 0.0

        # Add to display stack
        if self._predicted_class != "Unknown":
            self._display_stack.append(self._predicted_class)

        # If CLEAN detected, go to SHOULDER_WAIT to detect subsequent jerk
        if self._predicted_class == "CLEAN":
            self._shoulder_wait_start_time = time.time()
            self._shoulder_reference_y = final_y
            self._lift_frames = []
            self._recording_frame_count = 0
            self._frame_counter = 0
            self._class_prob_history.clear()
            self._path_points = []
            self._path_phases = []
            self._peak_detected = False
            self._peak_frame_count = 0
            self._peak_y = float("inf")
            self._shoulder_y_estimate = 0.0
            self.state = LiftState.SHOULDER_WAIT
            return

        self._display_start_time = time.time()
        self._tip_display_start = time.time()
        self.state = LiftState.DISPLAYING

    def _classify_with_legacy_model(self, frame_width: int, frame_height: int) -> None:
        """Fallback classification using the legacy full-trajectory model."""
        barbell_y = []
        barbell_x = []
        timestamps = []
        landmarks_list = []

        for f in self._lift_frames:
            if f.barbell_center is not None:
                barbell_y.append(f.barbell_center[1])
                barbell_x.append(f.barbell_center[0])
            elif barbell_y:
                barbell_y.append(barbell_y[-1])
                barbell_x.append(barbell_x[-1])
            else:
                continue
            timestamps.append(f.timestamp_ms)
            landmarks_list.append(f.landmarks)

        if len(barbell_y) < 10:
            self._predicted_class = "Unknown"
            self._predicted_confidence = 0.0
            self._is_clean_jerk = False
            return

        df = build_lift_dataframe(
            barbell_y=barbell_y,
            barbell_x=barbell_x,
            timestamps_ms=timestamps,
            landmarks_list=landmarks_list,
            frame_width=frame_width,
            frame_height=frame_height,
            fps=self.fps,
        )

        result = predict_lift_type(df, self._model_data) if self._model_data is not None else None

        if result:
            raw_class = result["predicted_class"]
            self._predicted_class = self._format_lift_name(raw_class)
            self._predicted_confidence = result["confidence"]
            self._is_clean_jerk = result["is_clean_jerk"]

            if raw_class in ("clean", "jerk", "clean_jerk"):
                split_point = detect_clean_jerk_split_point(df)
                if split_point is not None:
                    self._is_clean_jerk = True
        else:
            self._predicted_class = "Unknown"
            self._predicted_confidence = 0.0
            self._is_clean_jerk = False

        # Run technique analysis using CompiledAnalyzer
        self._run_technique_analysis(df)

        self._display_start_time = time.time()
        self._tip_display_start = time.time()
        self.state = LiftState.DISPLAYING

    def _run_technique_analysis(self, df: pd.DataFrame) -> None:
        """Run CompiledAnalyzer on the recorded trajectory and store top fault."""
        try:
            self._load_baselines()
            if not self._baselines:
                return

            # Determine lift type for analysis
            analysis_lift_type = self._predicted_class or "clean"
            if self._is_clean_jerk:
                # For live preview, analyze both segments and show worst fault
                split_idx = detect_clean_jerk_split_point(df)
                if split_idx is not None and split_idx < len(df):
                    df_clean = df.iloc[:split_idx]
                    df_jerk = df.iloc[split_idx:]
                    clean_features = extract_technique_features(df_clean, "clean")
                    jerk_features = extract_technique_features(df_jerk, "jerk")
                    clean_analyzer = CompiledAnalyzer("clean", self._gender, self._baselines)
                    jerk_analyzer = CompiledAnalyzer("jerk", self._gender, self._baselines)
                    clean_faults = clean_analyzer.analyze(clean_features, df_clean)
                    jerk_faults = jerk_analyzer.analyze(jerk_features, df_jerk)
                    all_faults = clean_faults + jerk_faults
                    if all_faults:
                        all_faults.sort(key=lambda f: f.get("confidence", 0), reverse=True)
                        self._top_fault = all_faults[0]
                    return
                else:
                    analysis_lift_type = "clean"

            features = extract_technique_features(df, analysis_lift_type)
            analyzer = CompiledAnalyzer(analysis_lift_type, self._gender, self._baselines)
            faults = analyzer.analyze(features, df)
            if faults:
                faults.sort(key=lambda f: f.get("confidence", 0), reverse=True)
                self._top_fault = faults[0]
        except Exception:
            # Silently skip fault overlay on failure
            self._top_fault = None

    def _handle_displaying(self) -> None:
        """DISPLAYING state: wait for display timer to expire."""
        elapsed = time.time() - self._display_start_time
        if elapsed >= self.display_seconds:
            self._reset()

    @property
    def current_tip(self) -> str | None:
        """Return coaching tip if within display duration, else None."""
        from barpath.pipeline.config import COACHING_TIP_DURATION_S

        if self.state == LiftState.DISPLAYING:
            elapsed = time.time() - self._tip_display_start
            if elapsed < COACHING_TIP_DURATION_S:
                if self._top_fault and self._top_fault.get("confidence", 0) > 0.6:
                    return self._top_fault.get("name", "Unknown Fault")
                return "Lift looks good"
        return None

    def _handle_shoulder_wait(self, frame_data: FrameData, frame_height: int) -> None:
        """SHOULDER_WAIT state: after clean, wait for jerk or new lift."""
        # Check timeout
        elapsed = time.time() - self._shoulder_wait_start_time
        if elapsed >= self._shoulder_wait_timeout:
            self._reset()
            return

        if frame_data.barbell_center is None:
            return

        bar_y = frame_data.barbell_center[1]
        bar_y_norm = bar_y / frame_height

        # Check for jerk trigger: bar dips from shoulder then drives up
        # Store recent bar positions to detect dip pattern
        self._buffer.append(frame_data)

        # If bar is near shoulder and starts moving down, transition to
        # triggered for jerk
        shoulder_zone = abs(bar_y - self._shoulder_reference_y) < frame_height * 0.1

        if shoulder_zone and len(self._buffer) >= 10:
            velocities = self._compute_recent_velocities()
            if len(velocities) >= 5:
                # Detect jerk: either dipping OR driving up from shoulder
                recent_vel = sum(velocities[-5:]) / 5
                # recent_vel > 0 = moving down (dip phase)
                # recent_vel < -10 = moving up rapidly (drive phase)
                if recent_vel > 2 or recent_vel < -10:
                    # This looks like a jerk - trigger recording
                    self._lift_frames = list(self._buffer)[-10:]
                    self._lift_frames.append(frame_data)
                    self._recording_frame_count = 1
                    self._frame_counter = 0
                    self._class_prob_history.clear()
                    self._path_points = []
                    self._path_phases = []
                    self._expecting_jerk = True
                    self._update_path_points(frame_height)
                    self.state = LiftState.RECORDING
                    return

        # Check for new floor lift: bar dropped to floor
        if bar_y_norm > 0.7:
            # Bar is at floor - ready for new lift
            self._reset()
            return

    def _reset(self) -> None:
        """Reset state machine to IDLE."""
        self.state = LiftState.IDLE
        self._lift_frames = []
        self._recording_frame_count = 0
        self._frame_counter = 0
        self._path_points = []
        self._path_phases = []
        self._trimmed_path_points = []
        self._trimmed_path_phases = []
        self._predicted_class = None
        self._predicted_confidence = 0.0
        self._is_clean_jerk = False
        self._top_fault = None
        self._trigger_count = 0
        self._class_prob_history.clear()
        self._display_stack = []
        self._shoulder_wait_start_time = 0.0
        self._shoulder_reference_y = 0.0
        self._shoulder_stable_count = 0
        self._peak_detected = False
        self._peak_frame_count = 0
        self._peak_y = float("inf")
        self._shoulder_y_estimate = 0.0
        self._expecting_jerk = False

    @staticmethod
    def _format_lift_name(raw_class: str) -> str:
        """Format model class name for display.

        "clean"  -> "CLEAN"
        "jerk"   -> "JERK"
        "snatch" -> "SNATCH"
        "clean_jerk" -> "CLEAN+JERK"
        """
        if raw_class == "clean_jerk":
            return "CLEAN+JERK"
        return raw_class.upper()

    # ========================================================================
    # Drawing
    # ========================================================================

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw bar path and lift label on the frame.

        Args:
            frame: BGR video frame to draw on (modified in place)

        Returns:
            The modified frame.
        """
        if self.state in (LiftState.RECORDING, LiftState.CLASSIFYING):
            self._draw_path(frame, self._path_points, self._path_phases)
            # Show real-time prediction during recording
            if self._predicted_class is not None:
                self._draw_live_prediction(frame)
        elif self.state in (LiftState.DISPLAYING, LiftState.SHOULDER_WAIT):
            # Draw trimmed path
            self._draw_path(frame, self._trimmed_path_points, self._trimmed_path_phases)
            # Draw stacked label
            self._draw_label(frame)

        return frame

    def _draw_path(
        self,
        frame: np.ndarray,
        points: list[tuple[int, int]],
        phases: list[int],
    ) -> None:
        """Draw phase-colored bar path on frame."""
        if len(points) < 2:
            return

        colors = PHASE_COLORS_6_BGR if self._is_clean_jerk else PHASE_COLORS_BGR
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            phase = phases[i] if i < len(phases) else 0
            color = colors.get(phase, (255, 255, 255))
            cv2.line(frame, p1, p2, color, 3)

    def _draw_live_prediction(self, frame: np.ndarray) -> None:
        """Draw small real-time prediction badge during recording."""
        if self._predicted_class is None:
            return

        _h, w = frame.shape[:2]
        text = f"{self._predicted_class} ({self._predicted_confidence:.0%})"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        padding = 6
        text_x = w - text_size[0] - padding - 10
        text_y = text_size[1] + padding + 10

        # Semi-transparent background
        overlay = frame.copy()
        bg_tl = (text_x - padding, text_y - text_size[1] - padding)
        bg_br = (text_x + text_size[0] + padding, text_y + padding)
        cv2.rectangle(overlay, bg_tl, bg_br, (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw text with outline
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _draw_label(self, frame: np.ndarray) -> None:
        """Draw lift type label and top fault above the bar path."""
        if not self._display_stack and self._predicted_class is None:
            return

        _h, w = frame.shape[:2]

        # Use display stack for stacked labels (e.g., "CLEAN + JERK")
        if len(self._display_stack) > 1:
            display_text = " + ".join(self._display_stack)
        elif self._display_stack:
            display_text = self._display_stack[0]
        else:
            display_text = self._predicted_class or "Unknown"

        # Add confidence
        conf_pct = int(self._predicted_confidence * 100)
        lines = [f"[{display_text}] {conf_pct}%"]

        # Show top 1 fault if available
        if self._top_fault:
            fault_name = self._top_fault.get("name", "Unknown Fault")
            fault_conf = self._top_fault.get("confidence", 0)
            lines.append(f"{fault_name} ({fault_conf}%)")

        # Position: center above path's highest point
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # Compute size of multi-line text block
        max_width = 0
        total_height = 0
        line_heights = []
        for line in lines:
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            max_width = max(max_width, text_size[0])
            total_height += text_size[1] + 8
            line_heights.append(text_size[1])

        if self._trimmed_path_points:
            min_y = min(p[1] for p in self._trimmed_path_points)
            center_x = w // 2
            text_x = max(10, center_x - max_width // 2)
            text_y = max(total_height + 10, min_y - 20)
        else:
            text_x = w // 2 - max_width // 2
            text_y = 60

        # Draw background rectangle
        padding = 8
        bg_tl = (text_x - padding, text_y - total_height - padding)
        bg_br = (text_x + max_width + padding, text_y + padding)
        overlay = frame.copy()
        cv2.rectangle(overlay, bg_tl, bg_br, (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw each line
        current_y = text_y - total_height + line_heights[0]
        for line in lines:
            # Draw text with outline for readability
            cv2.putText(
                frame,
                line,
                (text_x, current_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line,
                (text_x, current_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            current_y += line_heights[lines.index(line)] + 8

    @property
    def is_recording(self) -> bool:
        """True if currently recording a lift."""
        return self.state == LiftState.RECORDING

    @property
    def is_displaying(self) -> bool:
        """True if showing classification result."""
        return self.state == LiftState.DISPLAYING

    @property
    def status_text(self) -> str:
        """Human-readable status for the log."""
        if self.state == LiftState.IDLE:
            return "Waiting for lift..."
        elif self.state == LiftState.TRIGGERED:
            return "Lift detected! Recording..."
        elif self.state == LiftState.RECORDING:
            if self._predicted_class is not None:
                return f"Recording... {self._predicted_class} ({self._predicted_confidence:.0%})"
            return f"Recording... ({self._recording_frame_count} frames)"
        elif self.state == LiftState.CLASSIFYING:
            return "Classifying lift..."
        elif self.state == LiftState.DISPLAYING:
            if self._display_stack:
                return " + ".join(self._display_stack)
            return f"{self._predicted_class} ({self._predicted_confidence:.0%})"
        elif self.state == LiftState.SHOULDER_WAIT:
            if self._display_stack:
                return " + ".join(self._display_stack) + " (waiting for jerk...)"
            return "Waiting for jerk..."
        return ""
