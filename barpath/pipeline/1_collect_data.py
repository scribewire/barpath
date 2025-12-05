import argparse
import pickle
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO  # type: ignore[attr-defined]
from utils import LANDMARK_NAMES

# --- Constants ---

# Use LANDMARK_NAMES from utils as base set
LANDMARKS_TO_TRACK = LANDMARK_NAMES

# Convert string names to MediaPipe PoseLandmark objects
LANDMARK_ENUMS = {
    name: mp.solutions.pose.PoseLandmark[name.upper()]  # type: ignore[attr-defined]
    for name in LANDMARKS_TO_TRACK
}


def _looks_like_openvino_dir(path: Path) -> bool:
    """Return True when the provided path appears to be an OpenVINO export directory."""
    return path.is_dir() and any("openvino" in part.lower() for part in path.parts)


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

    # Initialize YOLO Model
    model_path_str = str(model_path)
    yolo_device = "cpu"

    print(f"Loading YOLO model: {model_path_str}")

    try:
        yolo_model = YOLO(model_path_str, task="detect")
    except Exception as e:
        cap.release()
        if pose:
            pose.close()
        raise RuntimeError(f"Error loading YOLO model from {model_path}: {e}")

    # Stabilization parameters for global motion model
    prev_gray = None
    prev_background_features = None  # Features from previous frame

    # Feature detection parameters
    feature_max_corners = 200
    feature_quality_level = 0.01
    feature_min_distance = 10
    feature_block_size = 7

    # Lucas-Kanade optical flow parameters (for feature tracking)
    lk_win_size = (21, 21)
    lk_max_level = 3
    lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    # Minimum number of matched features for motion estimation
    min_inliers = 10

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

        # Run YOLO inference on CPU
        results_yolo = yolo_model(frame, verbose=False, conf=0.25, device=yolo_device)

        # 1. Process MediaPipe Data
        segmentation_mask = None
        if results_pose and results_pose.pose_landmarks:
            landmarks_data = {}
            for name, enum in LANDMARK_ENUMS.items():
                lm = results_pose.pose_landmarks.landmark[enum]
                landmarks_data[name] = (lm.x, lm.y, lm.z, lm.visibility)
            frame_data["landmarks"] = landmarks_data

            if results_pose.segmentation_mask is not None:
                # Create a binary mask (1 for person, 0 for background)
                segmentation_mask = (results_pose.segmentation_mask > 0.5).astype(
                    np.uint8
                )

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
                    l_ankle = results_pose.pose_landmarks.landmark[
                        mp_pose_solution.PoseLandmark.LEFT_ANKLE
                    ]
                    r_ankle = results_pose.pose_landmarks.landmark[
                        mp_pose_solution.PoseLandmark.RIGHT_ANKLE
                    ]

                    l_visible = l_ankle.visibility > 0.3
                    r_visible = r_ankle.visibility > 0.3

                    l_pos = (
                        np.array([l_ankle.x * frame_width, l_ankle.y * frame_height])
                        if l_visible
                        else None
                    )
                    r_pos = (
                        np.array([r_ankle.x * frame_width, r_ankle.y * frame_height])
                        if r_visible
                        else None
                    )

                    if (
                        l_visible
                        and r_visible
                        and l_pos is not None
                        and r_pos is not None
                    ):
                        feet_pos_px = (l_pos + r_pos) / 2
                    elif l_visible and l_pos is not None:
                        feet_pos_px = l_pos
                    elif r_visible and r_pos is not None:
                        feet_pos_px = r_pos

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
            # Invert segmentation mask: 1 for background, 0 for foreground
            background_mask = (1 - segmentation_mask) * 255
            background_mask = background_mask.astype(np.uint8)

            # Optionally dilate the foreground mask to create a safety margin
            kernel = np.ones((15, 15), np.uint8)
            background_mask = cv2.erode(background_mask, kernel, iterations=1)

        # Detect or track background features
        curr_background_features = None

        if prev_gray is not None and prev_background_features is not None:
            # Track features from previous frame using Lucas-Kanade
            curr_features, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                prev_background_features,
                None,  # type: ignore[arg-type]
                winSize=lk_win_size,
                maxLevel=lk_max_level,
                criteria=lk_criteria,
            )

            if curr_features is not None and status is not None:
                # Select good matches
                good_new = curr_features[status.flatten() == 1]
                good_old = prev_background_features[status.flatten() == 1]

                # Estimate translation-only motion using median displacement
                if len(good_new) >= min_inliers:
                    try:
                        displacements = good_new - good_old
                        median_dx = float(np.median(displacements[:, 0, 0]))
                        median_dy = float(np.median(displacements[:, 0, 1]))

                        shake_dx = median_dx
                        shake_dy = median_dy

                        curr_background_features = good_new
                    except (cv2.error, ValueError, IndexError):
                        curr_background_features = None
                else:
                    # Not enough matched features
                    curr_background_features = None

        # Detect new features if we don't have enough or this is the first frame
        if curr_background_features is None or len(curr_background_features) < 50:
            # Detect new features in background regions
            if background_mask is not None:
                new_features = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=feature_max_corners,
                    qualityLevel=feature_quality_level,
                    minDistance=feature_min_distance,
                    mask=background_mask,
                    blockSize=feature_block_size,
                )
            else:
                # Fallback: detect features in entire frame if no segmentation mask
                new_features = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=feature_max_corners,
                    qualityLevel=feature_quality_level,
                    minDistance=feature_min_distance,
                    mask=None,
                    blockSize=feature_block_size,
                )

            if new_features is not None:
                if curr_background_features is not None:
                    # Merge with existing features
                    curr_background_features = np.vstack(
                        (curr_background_features, new_features)
                    )
                else:
                    curr_background_features = new_features

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

    is_openvino_dir = _looks_like_openvino_dir(model_path)

    if not model_path.exists():
        print(f"Error: Model path not found at {args.model}")
        return

    if model_path.is_dir() and not is_openvino_dir:
        print(
            "Error: Model directory must include 'openvino' in its name to be treated as an OpenVINO export."
        )
        return

    if is_openvino_dir and not any(model_path.glob("*.xml")):
        print(
            f"Error: OpenVINO directory '{args.model}' does not contain a .xml model definition."
        )
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
