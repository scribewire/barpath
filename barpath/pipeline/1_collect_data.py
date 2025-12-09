import argparse
import pickle
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from hardware_detection import detect_intel_gpu
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


def _get_model_path(model_path: Path) -> tuple[str, bool]:
    """
    Return the appropriate model path string for YOLO and whether it's OpenVINO.

    If model_path is a directory containing .xml files (OpenVINO format),
    returns the path to the directory. Otherwise returns the path as-is.

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
        # Regular model file (.pt, .onnx, etc.)
        return str(model_path), False
    else:
        raise ValueError(f"Model path does not exist: {model_path}")


# --- Step 1: Data Collection Function ---
def step_1_collect_data(
    video_path,
    model_path,
    output_path,
    lift_type="none",
):
    print("--- Step 1: Collecting Raw Data ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video file {video_path} has no frames.")

    # Initialize MediaPipe Pose
    mp_pose_solution = mp.solutions.pose  # type: ignore
    pose = None
    if lift_type != "none":
        pose = mp_pose_solution.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=True,  # Enable segmentation for stabilization
            model_complexity=1,  # Balance between accuracy and performance
        )

    # Initialize stabilization parameters
    stab_params = StabilizationParams()

    # Initialize YOLO Model
    try:
        model_path_obj = Path(model_path)
        model_path_str, is_openvino = _get_model_path(model_path_obj)
        print(f"Loading model: {model_path_str}")
        yolo_model = YOLO(model_path_str, task="detect")

        # Determine device for inference
        if is_openvino and detect_intel_gpu():
            yolo_device = "intel:gpu"
            print("Intel GPU detected - using GPU acceleration for OpenVINO")
        else:
            yolo_device = "cpu"
            if is_openvino:
                print("No Intel GPU detected - using CPU for OpenVINO")
    except Exception as e:
        cap.release()
        if pose:
            pose.close()
        raise RuntimeError(f"Failed to load model: {e}")

    # Stabilization state
    prev_gray = None
    prev_background_features = None  # Features from previous frame

    raw_data_list = []

    # State variable for tracking-by-proximity
    last_known_barbell_center = None
    # Performance tracking for FPS display
    last_iter_timestamp = time.perf_counter()
    smoothed_fps = None
    fps_smoothing = 0.2

    # Loop through frames and yield progress
    for frame_count in range(total_frames):
        success, frame = cap.read()
        if not success:
            break

        # Initialize all keys for this frame with None
        frame_data = {
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

        # Run MediaPipe and YOLO
        results_pose = None
        if pose:
            results_pose = pose.process(frame_rgb)

        # Run YOLO inference
        results_yolo = yolo_model(frame, verbose=False, conf=0.25, device=yolo_device)

        # 1. Process MediaPipe Data
        landmarks_data, world_landmarks_data, segmentation_mask = None, None, None
        if results_pose is not None:
            # Use helper to extract both landmark types and segmentation mask
            landmarks_data, world_landmarks_data, segmentation_mask = (
                process_pose_results(results_pose, LANDMARK_ENUMS)
            )

            frame_data["landmarks"] = landmarks_data
            frame_data["world_landmarks"] = world_landmarks_data

        # 2. Process YOLO Data
        best_endcap = None
        detected_endcaps = []

        if results_yolo:
            for r in results_yolo:
                for box in r.boxes:
                    cls_id = int(box.cls[0])

                    # YOLO convention: class 0 is the target object (barbell endcap)
                    # For models trained on single-class datasets, all detections will be class 0
                    is_match = cls_id == 0

                    if is_match:
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
                # --- INITIAL DETECTION ---
                feet_pos_px = None
                if results_pose and results_pose.pose_landmarks:
                    feet_pos_px = get_ankle_positions(
                        results_pose.pose_landmarks,
                        mp_pose_solution,
                        frame_width,
                        frame_height,
                    )

                if feet_pos_px is not None:
                    # Logic 1: Use feet position
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
                    # Logic 2: Fallback to center of frame
                    best_endcap = min(
                        detected_endcaps,
                        key=lambda e: abs(e["center"][0] - (frame_width / 2)),
                    )
                    print(
                        f"[Info] Barbell initially detected at frame {frame_count} (near center). No feet visible."
                    )

            else:
                # --- TRACKING ---
                best_endcap = min(
                    detected_endcaps,
                    key=lambda e: np.linalg.norm(
                        np.array(e["center"]) - last_known_barbell_center
                    ),
                )

            last_known_barbell_center = np.array(best_endcap["center"])
            frame_data["barbell_center"] = best_endcap["center"]
            frame_data["barbell_box"] = best_endcap["box"]

        # 3. Process Stabilization Data with Global Motion Model
        shake_dx, shake_dy = 0.0, 0.0

        # Create background mask for feature detection (exclude person and bar)
        background_mask = None
        if segmentation_mask is not None:
            background_mask = create_background_mask(segmentation_mask)

        # Detect or track background features
        curr_background_features = None

        if prev_gray is not None and prev_background_features is not None:
            # Track features from previous frame using Lucas-Kanade
            curr_features, status, err = track_features(
                prev_gray, gray, prev_background_features, stab_params
            )

            if curr_features is not None and status is not None:
                # Estimate motion from tracked features
                shake_dx, shake_dy, curr_background_features = estimate_motion(
                    prev_background_features, curr_features, status, stab_params
                )

        # Detect new features if we don't have enough or this is the first frame
        if curr_background_features is None or len(curr_background_features) < 50:
            # Detect new features in background regions
            new_features = detect_features(gray, background_mask, stab_params)
            if new_features is not None:
                curr_background_features = update_features(
                    curr_background_features, new_features, min_features=50
                )

        # Store stabilization data
        frame_data["shake_dx"] = shake_dx
        frame_data["shake_dy"] = shake_dy

        # Update state for next iteration
        prev_background_features = curr_background_features

        raw_data_list.append(frame_data)
        prev_gray = gray

        # Yield progress update with FPS measurement
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

        progress_fraction = (frame_count + 1) / total_frames
        yield (
            "step1",
            progress_fraction,
            f"Collecting data: frame {frame_count + 1}/{total_frames} ({smoothed_fps:.1f} FPS)",
        )

    cap.release()
    if pose:
        pose.close()

    # --- Save data to pickle file ---
    output_data = {
        "metadata": {
            "frame_width": frame_width,
            "frame_height": frame_height,
            "fps": fps,
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
        help="Path to the trained YOLO model (e.g., best.pt, best.onnx, or an OpenVINO export directory)",
    )
    parser.add_argument(
        "--output",
        default="raw_data.pkl",
        help="Path to save the raw data pickle file.",
    )
    parser.add_argument(
        "--lift_type",
        default="none",
        help="Type of lift (e.g., 'clean', 'snatch', 'none'). If 'none', pose estimation is skipped.",
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

    # Validate model path
    try:
        _get_model_path(model_path)  # Returns tuple but we just validate here
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Consume the generator to run the function
    for _ in step_1_collect_data(
        args.input,
        args.model,
        args.output,
        args.lift_type,
    ):
        pass  # Progress updates ignored when run standalone


if __name__ == "__main__":
    main()
