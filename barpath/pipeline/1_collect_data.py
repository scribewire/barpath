"""
Step 1: Collect raw data from video.

This step processes video frames to extract:
- Barbell position (via YOLO detection)
- Pose landmarks (via MediaPipe)
- Camera stabilization (via optical flow)

Updated for mediapipe 0.10.x Tasks API.
"""

import argparse
import pickle
import queue
import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from config import (  # type: ignore[import-untyped]
    DECODE_QUEUE_SIZE,
    MEDIAPIPE_DETECTION_CONFIDENCE,
    MEDIAPIPE_TRACKING_CONFIDENCE,
    STAB_MIN_FEATURES,
    YOLO_CONFIDENCE_THRESHOLD,
)
from hardware_detection import detect_intel_gpu, detect_nvidia_gpu  # type: ignore[import-untyped]
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from step1_helpers import (  # type: ignore[import-untyped]
    StabilizationParams,
    create_background_mask,
    detect_features,
    estimate_motion,
    get_ankle_positions,
    get_landmark_enums,
    get_pose_landmarker_model_path,
    process_pose_results,
    track_features,
    update_features,
)
from ultralytics import YOLO  # type: ignore
from utils import LANDMARK_NAMES  # type: ignore[import-untyped]

LANDMARKS_TO_TRACK = LANDMARK_NAMES
LANDMARK_ENUMS = get_landmark_enums(LANDMARKS_TO_TRACK)
_QUEUE_DONE = object()


def _get_model_path(model_path: Path) -> tuple[str, bool]:
    """
    Return the appropriate model path string for YOLO and whether it's OpenVINO.
    """
    if model_path.is_dir():
        xml_files = list(model_path.glob("*.xml"))
        bin_files = list(model_path.glob("*.bin"))

        if xml_files and bin_files:
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
        return str(model_path), False
    else:
        raise ValueError(f"Model path does not exist: {model_path}")


def _is_yolo26_model(model_path_str: str) -> bool:
    """Check if the model path looks like a YOLO26 model."""
    lower = model_path_str.lower()
    return "yolo26" in lower or "yolov26" in lower


def _frame_producer(
    video_path: str,
    out_queue: "queue.Queue[object]",
    total_frames: int,
) -> None:
    """
    Decode frames from video and push them onto the queue.
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


def step_1_collect_data(
    video_path,
    model_path,
    output_path,
    lift_type="none",
):
    print("--- Step 1: Collecting Raw Data ---")

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

    pose_landmarker = None
    if lift_type != "none":
        pose_model_path = get_pose_landmarker_model_path()
        base_options = mp_python.BaseOptions(model_asset_path=str(pose_model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=MEDIAPIPE_TRACKING_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_TRACKING_CONFIDENCE,
            output_segmentation_masks=True,
        )
        pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    stab_params = StabilizationParams()

    try:
        model_path_obj = Path(model_path)
        model_path_str, is_openvino = _get_model_path(model_path_obj)
        print(f"Loading model: {model_path_str}")

        is_tensorrt_engine = (
            model_path_obj.is_file() and model_path_obj.suffix.lower() == ".engine"
        )
        if is_tensorrt_engine:
            print("Detected TensorRT engine model (.engine).")

        yolo_model = YOLO(model_path_str, task="detect")

        nms_free = _is_yolo26_model(model_path_str)
        if nms_free:
            print("YOLO26 NMS-free architecture detected.")

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
        if pose_landmarker:
            pose_landmarker.close()
        raise RuntimeError(f"Failed to load model: {e}")

    frame_queue: "queue.Queue[object]" = queue.Queue(maxsize=DECODE_QUEUE_SIZE)
    producer_thread = threading.Thread(
        target=_frame_producer,
        args=(video_path, frame_queue, total_frames),
        daemon=True,
    )
    producer_thread.start()

    prev_gray = None
    prev_background_features = None

    raw_data_list: list[dict] = []
    last_known_barbell_center = None

    frames_processed = 0

    while True:
        item = frame_queue.get()

        if item is _QUEUE_DONE:
            break

        frame_count, frame = item  # type: ignore

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

        results_pose = None
        if pose_landmarker:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((frame_count / fps) * 1000)
            results_pose = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        _infer_kwargs: dict = {"verbose": False, "conf": YOLO_CONFIDENCE_THRESHOLD}
        if yolo_device is not None:
            _infer_kwargs["device"] = yolo_device

        results_yolo = yolo_model(frame, **_infer_kwargs)

        landmarks_data, world_landmarks_data, segmentation_mask = None, None, None
        if results_pose is not None:
            landmarks_data, world_landmarks_data, segmentation_mask = (
                process_pose_results(results_pose, LANDMARK_ENUMS)
            )
            frame_data["landmarks"] = landmarks_data
            frame_data["world_landmarks"] = world_landmarks_data

        best_endcap = None
        detected_endcaps: list[dict] = []

        if results_yolo:
            for r in results_yolo:
                for box in r.boxes:
                    cls_id = int(box.cls[0])

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
                feet_pos_px = None
                if results_pose and results_pose.pose_landmarks:
                    pose_landmarks_list = results_pose.pose_landmarks
                    if len(pose_landmarks_list) > 0:
                        feet_pos_px = get_ankle_positions(
                            pose_landmarks_list[0],
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
                best_endcap = min(
                    detected_endcaps,
                    key=lambda e: np.linalg.norm(
                        np.array(e["center"]) - last_known_barbell_center
                    ),
                )

            last_known_barbell_center = np.array(best_endcap["center"])
            frame_data["barbell_center"] = best_endcap["center"]
            frame_data["barbell_box"] = best_endcap["box"]

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

        if (
            curr_background_features is None
            or len(curr_background_features) < STAB_MIN_FEATURES
        ):
            new_features = detect_features(gray, background_mask, stab_params)
            if new_features is not None:
                curr_background_features = update_features(
                    curr_background_features,
                    new_features,
                    min_features=STAB_MIN_FEATURES,
                )

        frame_data["shake_dx"] = shake_dx
        frame_data["shake_dy"] = shake_dy

        prev_background_features = curr_background_features
        prev_gray = gray

        raw_data_list.append(frame_data)
        frames_processed += 1

        progress_fraction = frames_processed / total_frames
        yield (
            "step1",
            progress_fraction,
            f"Collecting data: frame {frames_processed}/{total_frames}",
        )

    producer_thread.join(timeout=5)

    if pose_landmarker:
        pose_landmarker.close()

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


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Collect raw motion data from video."
    )
    parser.add_argument("--input", required=True, help="Path to the source video file")
    parser.add_argument("--model", required=True, help="Path to the trained YOLO model")
    parser.add_argument(
        "--output",
        default="raw_data.pkl",
        help="Path to save the raw data pickle file.",
    )
    parser.add_argument(
        "--lift_type",
        default="none",
        help="Type of lift (e.g., 'clean', 'snatch', 'none').",
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

    for _ in step_1_collect_data(args.input, args.model, args.output, args.lift_type):
        pass


if __name__ == "__main__":
    main()
