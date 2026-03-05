import argparse
import pickle
import queue
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from hardware_detection import detect_intel_gpu, detect_nvidia_gpu
from step1_helpers import (
    StabilizationParams,
    create_background_mask,
    detect_features,
    estimate_motion,
    get_ankle_positions,
    get_landmark_enums,
    process_pose_results,
    track_features,
    update_features,
)
from ultralytics import YOLO  # type: ignore[attr-defined]
from utils import LANDMARK_NAMES

# --- Constants ---

# Use LANDMARK_NAMES from utils as base set
LANDMARKS_TO_TRACK = LANDMARK_NAMES

# Convert string names to MediaPipe PoseLandmark objects
LANDMARK_ENUMS = get_landmark_enums(LANDMARKS_TO_TRACK)

# Producer-consumer queue sentinel
_QUEUE_DONE = object()

# Queue size: buffer up to N decoded frames ahead of inference
_DECODE_QUEUE_SIZE = 8


def _get_model_path(model_path: Path) -> tuple[str, bool]:
    """
    Return the appropriate model path string for YOLO and whether it's OpenVINO.

    If model_path is a directory containing .xml files (OpenVINO format),
    returns the path to the directory. Otherwise returns the path as-is.

    Notes:
    - TensorRT `.engine` models are treated as regular model files (not OpenVINO).
      Ultralytics will select the TensorRT backend automatically when given an `.engine` path.
    - YOLO26 uses an NMS-free end-to-end architecture; no `iou` threshold is needed
      during inference since NMS is baked into the model head.

    Returns:
        tuple: (model_path_str, is_openvino)

    Raises ValueError if the path is invalid.
    """
    if model_path.is_dir():
        # Check for OpenVINO model files
        xml_files = list(model_path.glob("*.xml"))
        bin_files = list(model_path.glob("*.bin"))

        if xml_files and bin_files:
            # OpenVINO model directory
            print(f"Detected OpenVINO model in: {model_path}")
            return str(model_path), True
        elif xml_files:
            raise ValueError(
                f"OpenVINO directory missing .bin weights file: {model_path}"
            )
        elif bin_files:
            raise ValueError(
                f"OpenVINO directory missing .xml model file: {model_path}"
            )
        else:
            raise ValueError(f"Directory does not contain a valid model: {model_path}")
    elif model_path.is_file():
        # Regular model file (.pt, .onnx, .engine, etc.)
        return str(model_path), False
    else:
        raise ValueError(f"Model path does not exist: {model_path}")


def _is_yolo26_model(model_path_str: str) -> bool:
    """
    Heuristic check: return True if the model path looks like a YOLO26 model.

    YOLO26 uses an NMS-free architecture (end-to-end), so we should NOT pass
    `iou` threshold arguments during inference (the parameter is ignored/unsupported).
    We detect YOLO26 by checking for 'yolo26' or 'yolov26' in the path string
    (case-insensitive). If the model is already loaded, callers can also check
    `model.task` or the architecture name.
    """
    lower = model_path_str.lower()
    return "yolo26" in lower or "yolov26" in lower


# ---------------------------------------------------------------------------
# Producer: reads frames from disk and puts (frame_count, frame) onto a queue
# ---------------------------------------------------------------------------


def _frame_producer(
    video_path: str,
    out_queue: "queue.Queue[object]",
    total_frames: int,
) -> None:
    """
    Decode frames from *video_path* and push them onto *out_queue*.

    Each item is a ``(frame_count, frame_bgr)`` tuple.  Pushes the sentinel
    ``_QUEUE_DONE`` when the video is exhausted or an error occurs.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        for frame_count in range(total_frames):
            success, frame = cap.read()
            if not success:
                break
            out_queue.put((frame_count, frame))
    finally:
        cap.release()
        out_queue.put(_QUEUE_DONE)


# ---------------------------------------------------------------------------
# Step 1 main function
# ---------------------------------------------------------------------------


def step_1_collect_data(
    video_path,
    model_path,
    output_path,
    lift_type="none",
):
    print("--- Step 1: Collecting Raw Data ---")

    # -----------------------------------------------------------------------
    # Open video once to read metadata, then close it.  The actual frame
    # decoding is done by the background producer thread.
    # -----------------------------------------------------------------------
    _probe = cv2.VideoCapture(video_path)
    if not _probe.isOpened():
        raise FileNotFoundError(f"Could not open video file {video_path}")

    frame_width = int(_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = _probe.get(cv2.CAP_PROP_FPS)
    total_frames = int(_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    _probe.release()

    if total_frames == 0:
        raise ValueError(f"Video file {video_path} has no frames.")

    # -----------------------------------------------------------------------
    # Initialize MediaPipe Pose
    # -----------------------------------------------------------------------
    mp_pose_solution = mp.solutions.pose  # type: ignore
    pose = None
    if lift_type != "none":
        pose = mp_pose_solution.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=True,
            model_complexity=2,
        )

    # Initialize stabilization parameters
    stab_params = StabilizationParams()

    # -----------------------------------------------------------------------
    # Initialize YOLO Model (YOLO26 NMS-free architecture)
    # -----------------------------------------------------------------------
    try:
        model_path_obj = Path(model_path)
        model_path_str, is_openvino = _get_model_path(model_path_obj)
        print(f"Loading model: {model_path_str}")

        is_tensorrt_engine = (
            model_path_obj.is_file() and model_path_obj.suffix.lower() == ".engine"
        )
        if is_tensorrt_engine:
            print(
                "Detected TensorRT engine model (.engine). "
                "Ultralytics will use the TensorRT backend automatically."
            )

        # YOLO26 is NMS-free: the model head outputs final detections directly.
        # We do NOT pass iou= during inference; conf= is still valid.
        yolo_model = YOLO(model_path_str, task="detect")

        # Detect whether this is a YOLO26 model so we can adjust inference call.
        nms_free = _is_yolo26_model(model_path_str)
        if nms_free:
            print(
                "YOLO26 NMS-free architecture detected. "
                "Skipping iou threshold argument during inference."
            )

        # Determine inference device
        if is_tensorrt_engine:
            yolo_device = None
        elif is_openvino and detect_intel_gpu():
            yolo_device = "intel:gpu"
            print("Intel GPU detected - using GPU acceleration for OpenVINO")
        elif detect_nvidia_gpu() and torch.cuda.is_available():
            yolo_device = "cuda"
            print("NVIDIA GPU detected with CUDA support - using GPU acceleration")
        else:
            yolo_device = "cpu"
            if is_openvino:
                print("No GPU acceleration available - using CPU for OpenVINO")
            else:
                print("Using CPU for inference")
    except Exception as e:
        if pose:
            pose.close()
        raise RuntimeError(f"Failed to load model: {e}")

    # -----------------------------------------------------------------------
    # Start producer thread (I/O-bound: frame decoding runs on a background thread
    # so the main thread is never idle waiting for disk reads)
    # -----------------------------------------------------------------------
    frame_queue: "queue.Queue[object]" = queue.Queue(maxsize=_DECODE_QUEUE_SIZE)
    producer_thread = threading.Thread(
        target=_frame_producer,
        args=(video_path, frame_queue, total_frames),
        daemon=True,
    )
    producer_thread.start()

    # -----------------------------------------------------------------------
    # Stabilization state
    # -----------------------------------------------------------------------
    prev_gray = None
    prev_background_features = None

    raw_data_list: list[dict] = []

    # Tracking state
    last_known_barbell_center = None

    # Performance tracking
    last_iter_timestamp = time.perf_counter()
    smoothed_fps: float | None = None
    fps_smoothing = 0.2

    # -----------------------------------------------------------------------
    # Main processing loop — consumer side of the producer-consumer pipeline.
    # The main thread performs all CPU-bound work: YOLO inference, MediaPipe
    # pose estimation, optical-flow stabilization, and data aggregation.
    # -----------------------------------------------------------------------
    frames_processed = 0

    while True:
        # Block until a frame is available (the producer does the disk I/O)
        item = frame_queue.get()

        if item is _QUEUE_DONE:
            break

        frame_count, frame = item  # type: ignore[misc]

        # Initialize all keys for this frame with None / zero
        frame_data: dict = {
            "frame": frame_count,
            "landmarks": None,
            "world_landmarks": None,
            "barbell_center": None,
            "barbell_box": None,
            "shake_dx": 0.0,
            "shake_dy": 0.0,
        }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- MediaPipe Pose ---
        results_pose = None
        if pose:
            results_pose = pose.process(frame_rgb)

        # --- YOLO Inference (YOLO26 NMS-free) ---
        # YOLO26's end-to-end head already suppresses overlapping boxes, so we
        # only pass `conf` (minimum confidence).  The `iou` parameter used by
        # traditional NMS-based YOLO models is intentionally omitted here.
        _infer_kwargs: dict = {"verbose": False, "conf": 0.25}
        if yolo_device is not None:
            _infer_kwargs["device"] = yolo_device

        results_yolo = yolo_model(frame, **_infer_kwargs)

        # --- Process MediaPipe ---
        landmarks_data, world_landmarks_data, segmentation_mask = None, None, None
        if results_pose is not None:
            landmarks_data, world_landmarks_data, segmentation_mask = (
                process_pose_results(results_pose, LANDMARK_ENUMS)
            )
            frame_data["landmarks"] = landmarks_data
            frame_data["world_landmarks"] = world_landmarks_data

        # --- Process YOLO detections ---
        best_endcap = None
        detected_endcaps: list[dict] = []

        if results_yolo:
            for r in results_yolo:
                # YOLO26 NMS-free: r.boxes contains the final post-processed detections.
                # The boxes API is identical to previous YOLO versions, so the rest of
                # the detection pipeline is unchanged.
                for box in r.boxes:
                    cls_id = int(box.cls[0])

                    # Class 0 = barbell endcap (single-class dataset convention)
                    if cls_id == 0:
                        coords = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = coords

                        x1 = float(max(0, min(x1, frame_width - 1)))
                        x2 = float(max(0, min(x2, frame_width - 1)))
                        y1 = float(max(0, min(y1, frame_height - 1)))
                        y2 = float(max(0, min(y2, frame_height - 1)))

                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        detected_endcaps.append(
                            {"center": center, "box": (x1, y1, x2, y2)}
                        )

        if detected_endcaps:
            if last_known_barbell_center is None:
                # --- Initial detection: anchor to feet or frame centre ---
                feet_pos_px = None
                if results_pose and results_pose.pose_landmarks:
                    feet_pos_px = get_ankle_positions(
                        results_pose.pose_landmarks,
                        mp_pose_solution,
                        frame_width,
                        frame_height,
                    )

                if feet_pos_px is not None:
                    best_endcap = min(
                        detected_endcaps,
                        key=lambda e: np.linalg.norm(
                            np.array(e["center"]) - feet_pos_px
                        ),
                    )
                    print(
                        f"[Info] Barbell initially detected at frame {frame_count} (near feet)."
                    )
                else:
                    best_endcap = min(
                        detected_endcaps,
                        key=lambda e: abs(e["center"][0] - (frame_width / 2)),
                    )
                    print(
                        f"[Info] Barbell initially detected at frame {frame_count} "
                        f"(near center). No feet visible."
                    )
            else:
                # --- Tracking by proximity to last known position ---
                best_endcap = min(
                    detected_endcaps,
                    key=lambda e: np.linalg.norm(
                        np.array(e["center"]) - last_known_barbell_center
                    ),
                )

            last_known_barbell_center = np.array(best_endcap["center"])
            frame_data["barbell_center"] = best_endcap["center"]
            frame_data["barbell_box"] = best_endcap["box"]

        # --- Stabilization via Lucas-Kanade optical flow on background features ---
        shake_dx, shake_dy = 0.0, 0.0

        background_mask = None
        if segmentation_mask is not None:
            background_mask = create_background_mask(segmentation_mask)

        curr_background_features = None

        if prev_gray is not None and prev_background_features is not None:
            curr_features, status, _err = track_features(
                prev_gray, gray, prev_background_features, stab_params
            )
            if curr_features is not None and status is not None:
                shake_dx, shake_dy, curr_background_features = estimate_motion(
                    prev_background_features, curr_features, status, stab_params
                )

        if curr_background_features is None or len(curr_background_features) < 50:
            new_features = detect_features(gray, background_mask, stab_params)
            if new_features is not None:
                curr_background_features = update_features(
                    curr_background_features, new_features, min_features=50
                )

        frame_data["shake_dx"] = shake_dx
        frame_data["shake_dy"] = shake_dy

        # Update state for next iteration
        prev_background_features = curr_background_features
        prev_gray = gray

        raw_data_list.append(frame_data)
        frames_processed += 1

        # --- Yield progress ---
        now_ts = time.perf_counter()
        frame_duration = max(now_ts - last_iter_timestamp, 1e-6)
        inst_fps = 1.0 / frame_duration
        if smoothed_fps is None:
            smoothed_fps = inst_fps
        else:
            smoothed_fps = (fps_smoothing * inst_fps) + (
                (1 - fps_smoothing) * smoothed_fps
            )
        last_iter_timestamp = now_ts

        progress_fraction = (frames_processed) / total_frames
        yield (
            "step1",
            progress_fraction,
            f"Collecting data: frame {frames_processed}/{total_frames} ({smoothed_fps:.1f} FPS)",
        )

    # Wait for producer to finish (should already be done at this point)
    producer_thread.join(timeout=5)

    if pose:
        pose.close()

    # --- Save to pickle ---
    output_data = {
        "metadata": {
            "frame_width": frame_width,
            "frame_height": frame_height,
            "fps": fps,
            "lift_type": lift_type,
            "total_frames_processed": len(raw_data_list),
        },
        "data": raw_data_list,
    }

    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)

    print(f"\nStep 1 Complete. Processed {len(raw_data_list)} frames.")
    print(f"Raw data saved to '{output_path}'")


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Collect raw motion data from video."
    )
    parser.add_argument(
        "--input", required=True, help="Path to the source video file (e.g., video.mp4)"
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Path to the trained YOLO model "
            "(e.g., best.pt, best.onnx, yolo26n.pt, or an OpenVINO export directory)"
        ),
    )
    parser.add_argument(
        "--output",
        default="raw_data.pkl",
        help="Path to save the raw data pickle file.",
    )
    parser.add_argument(
        "--lift_type",
        default="none",
        help=(
            "Type of lift (e.g., 'clean', 'snatch', 'none'). "
            "If 'none', pose estimation is skipped."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)

    if not input_path.exists():
        print(f"Error: Input file not found at {args.input}")
        return

    if not model_path.exists():
        print(f"Error: Model path not found at {args.model}")
        return

    try:
        _get_model_path(model_path)
    except ValueError as e:
        print(f"Error: {e}")
        return

    for _ in step_1_collect_data(
        args.input,
        args.model,
        args.output,
        args.lift_type,
    ):
        pass


if __name__ == "__main__":
    main()
