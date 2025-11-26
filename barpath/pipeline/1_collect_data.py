import sys
import gc
import time
from pathlib import Path
try:
    import cv2
except ImportError:
    print("Missing dependency: opencv-python (cv2). Install with: pip install opencv-python")
    sys.exit(1)
import argparse
try:
    import numpy as np
except ImportError:
    print("Missing dependency: numpy. Install with: pip install numpy")
    sys.exit(1)
try:
    import mediapipe as mp
except ImportError:
    print("Missing dependency: mediapipe. Install with: pip install mediapipe")
    sys.exit(1)
try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    print("Missing dependency: ultralytics. Install with: pip install ultralytics")
    sys.exit(1)
import pickle
from utils import LANDMARK_NAMES

# --- Constants ---

# Use LANDMARK_NAMES from utils as base set
LANDMARKS_TO_TRACK = LANDMARK_NAMES

# Convert string names to MediaPipe PoseLandmark objects
LANDMARK_ENUMS = {name: mp.solutions.pose.PoseLandmark[name.upper()] for name in LANDMARKS_TO_TRACK} # type: ignore


def _looks_like_openvino_dir(path: Path) -> bool:
    """Return True when the provided path appears to be an OpenVINO export directory."""
    return path.is_dir() and any('openvino' in part.lower() for part in path.parts)


def _get_yolo_device() -> str:
    """
    Determine the best device for YOLO inference based on available hardware-accelerated packages.
    
    Returns:
        str: Device string for Ultralytics YOLO ('cpu' or 'cuda').
    
    Note: For ONNX models, both CPU and GPU acceleration work through ONNX Runtime's
    execution providers (DirectML, CoreML, etc.), so we use 'cpu' as the device parameter
    and let ONNX handle the actual hardware acceleration transparently.
    """
    device = 'cpu'  # Default - works with all setups
    
    # Try to import and detect available accelerators
    try:
        import onnxruntime as ort
        # Check for hardware-accelerated providers
        providers = ort.get_available_providers()
        
        # Only use 'cuda' device if we have actual CUDA support
        if 'CUDAExecutionProvider' in providers:
            device = 'cuda'  # NVIDIA GPU (CUDA)
            print("  ✓ CUDA detected - using GPU acceleration")
        elif 'ROCMExecutionProvider' in providers:
            device = 'cuda'  # AMD GPU (ROCm works with CUDA device string in YOLO)
            print("  ✓ ROCm detected - using AMD GPU acceleration")
        elif 'TensorrtExecutionProvider' in providers:
            device = 'cuda'
            print("  ✓ TensorRT detected - using GPU acceleration")
        elif 'DmlExecutionProvider' in providers:
            # DirectML is handled automatically by ONNX Runtime at the CPU device level
            print("  ✓ DirectML detected - using GPU acceleration via ONNX Runtime")
        elif 'CoreMLExecutionProvider' in providers:
            # Core ML is handled automatically by ONNX Runtime at the CPU device level
            print("  ✓ Core ML detected - using Metal acceleration via ONNX Runtime")
        elif 'TFLiteExecutionProvider' in providers:
            print("  ℹ TFLite provider available - using CPU for YOLO")
        else:
            print("  ℹ Using CPU for inference")
    except ImportError:
        pass
    
    # OpenVINO is only supported for native PyTorch models, not for ONNX models in YOLO
    # so we don't check for it here - stick with ONNX Runtime's backend optimization
    
    return device


# --- Step 1: Data Collection Function ---
def step_1_collect_data(video_path, model_path, output_path, lift_type='none', selected_runtime='onnxruntime'):
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
    mp_pose_solution = mp.solutions.pose # type: ignore
    pose = None
    if lift_type != 'none':
        pose = mp_pose_solution.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=True  # Enable segmentation for stabilization
        )
    
    # Initialize YOLO Model
    model_path_obj = Path(model_path)
    model_path_str = str(model_path_obj)
    is_openvino_dir = _looks_like_openvino_dir(model_path_obj)
    is_onnx = model_path_obj.suffix.lower() == '.onnx'

    # Determine device to use based on selected_runtime
    # For ONNX models, the device parameter needs to be 'cpu' to allow
    # ONNX Runtime to manage the actual hardware acceleration
    cuda_like_runtimes = ['onnxruntime_gpu', 'onnxruntime_rocm']  # These use YOLO's 'cuda' device
    
    if selected_runtime in cuda_like_runtimes:
        # NVIDIA CUDA or AMD ROCm - these use YOLO's CUDA device string
        yolo_device = 'cuda'
        print(f"  Using {selected_runtime} GPU acceleration")
    else:
        # For DirectML, Metal, or plain CPU, use 'cpu' and let ONNX Runtime manage the backend
        yolo_device = 'cpu'
        if selected_runtime == 'onnxruntime_directml':
            print(f"  Using DirectML GPU acceleration via ONNX Runtime")
        elif selected_runtime == 'onnxruntime_metal':
            print(f"  Using Metal GPU acceleration via ONNX Runtime")
        elif selected_runtime == 'openvino':
            print(f"  Using OpenVINO CPU optimization")
        else:
            print(f"  Using CPU inference")
    
    print(f"  Selected runtime: {selected_runtime} -> YOLO device: {yolo_device}")

    try:
        if is_openvino_dir:
            print(f"Loading OpenVINO model directory: {model_path_str}")
            yolo_model = YOLO(model_path_str, task='detect')
        elif is_onnx:
            print(f"Loading ONNX model: {model_path_str}")
            yolo_model = YOLO(model_path_str, task='detect')
        else:
            print(f"Loading YOLO model: {model_path_str}")
            yolo_model = YOLO(model_path_str)
    except Exception as e:
        cap.release()
        if pose:
            pose.close()
        raise RuntimeError(f"Error loading YOLO model from {model_path}: {e}")
    
    # Stabilization parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    prev_gray = None
    background_features = None
    
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
            'frame': frame_count,
            'landmarks': None,
            'barbell_center': None,
            'barbell_box': None,
            'shake_dx': 0.0,
            'shake_dy': 0.0
        }
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run MediaPipe and YOLO
        results_pose = None
        if pose:
            results_pose = pose.process(frame_rgb)
        
        # Run YOLO inference with hardware acceleration device if available
        results_yolo = yolo_model(frame, verbose=False, conf=0.25, device=yolo_device)
        
        # 1. Process MediaPipe Data
        segmentation_mask = None
        if results_pose and results_pose.pose_landmarks:
            landmarks_data = {}
            for name, enum in LANDMARK_ENUMS.items():
                lm = results_pose.pose_landmarks.landmark[enum]
                landmarks_data[name] = (lm.x, lm.y, lm.z, lm.visibility)
            frame_data['landmarks'] = landmarks_data
            
            if results_pose.segmentation_mask is not None:
                # Create a binary mask (1 for person, 0 for background)
                segmentation_mask = (results_pose.segmentation_mask > 0.5).astype(np.uint8)
        
        # 2. Process YOLO Data
        best_endcap = None
        detected_endcaps = []

        if results_yolo:
            for r in results_yolo:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    
                    # YOLO convention: class 0 is the target object (barbell endcap)
                    # For models trained on single-class datasets, all detections will be class 0
                    is_match = (cls_id == 0)

                    if is_match:
                        coords = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = coords
                        
                        x1 = float(max(0, min(x1, frame_width - 1)))
                        x2 = float(max(0, min(x2, frame_width - 1)))
                        y1 = float(max(0, min(y1, frame_height - 1)))
                        y2 = float(max(0, min(y2, frame_height - 1)))
                        
                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        detected_endcaps.append({'center': center, 'box': (x1, y1, x2, y2)})
        
        if detected_endcaps:
            if last_known_barbell_center is None:
                # --- INITIAL DETECTION ---
                feet_pos_px = None
                if results_pose and results_pose.pose_landmarks:
                    l_ankle = results_pose.pose_landmarks.landmark[mp_pose_solution.PoseLandmark.LEFT_ANKLE]
                    r_ankle = results_pose.pose_landmarks.landmark[mp_pose_solution.PoseLandmark.RIGHT_ANKLE]
                    
                    l_visible = l_ankle.visibility > 0.3
                    r_visible = r_ankle.visibility > 0.3
                    
                    l_pos = np.array([l_ankle.x * frame_width, l_ankle.y * frame_height]) if l_visible else None
                    r_pos = np.array([r_ankle.x * frame_width, r_ankle.y * frame_height]) if r_visible else None

                    if l_visible and r_visible and l_pos is not None and r_pos is not None:
                        feet_pos_px = (l_pos + r_pos) / 2
                    elif l_visible and l_pos is not None:
                        feet_pos_px = l_pos
                    elif r_visible and r_pos is not None:
                        feet_pos_px = r_pos
                
                if feet_pos_px is not None:
                    # Logic 1: Use feet position
                    best_endcap = min(detected_endcaps, 
                                      key=lambda e: np.linalg.norm(np.array(e['center']) - feet_pos_px))
                    # --- CHANGED: 'tqdm.write' to 'print' ---
                    print(f"[Info] Barbell initially detected at frame {frame_count} (near feet).")
                else:
                    # Logic 2: Fallback to center of frame
                    best_endcap = min(detected_endcaps, 
                                      key=lambda e: abs(e['center'][0] - (frame_width / 2)))
                    # --- CHANGED: 'tqdm.write' to 'print' ---
                    print(f"[Info] Barbell initially detected at frame {frame_count} (near center). No feet visible.")

            else:
                # --- TRACKING ---
                best_endcap = min(detected_endcaps, 
                                  key=lambda e: np.linalg.norm(np.array(e['center']) - last_known_barbell_center))
            
            last_known_barbell_center = np.array(best_endcap['center'])
            frame_data['barbell_center'] = best_endcap['center']
            frame_data['barbell_box'] = best_endcap['box']
            
        # 3. Process Stabilization Data
        shake_dx, shake_dy = 0.0, 0.0
        if prev_gray is not None:
            if background_features is not None and len(background_features) > 0:
                next_features, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, background_features, None, **lk_params) # type: ignore
                good_new = next_features[status == 1]
                good_old = background_features[status == 1]
                
                if len(good_new) > 5:
                    deltas_x = good_new[:, 0] - good_old[:, 0]
                    deltas_y = good_new[:, 1] - good_old[:, 1]
                    shake_dx = float(np.median(deltas_x))
                    shake_dy = float(np.median(deltas_y))
                
                background_features = good_new.reshape(-1, 1, 2)
            else:
                background_features = None
        
        background_mask = None
        if (background_features is None and segmentation_mask is not None):
            background_mask = 1 - segmentation_mask
            new_features = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10, mask=background_mask)
            if new_features is not None:
                background_features = new_features
        
        frame_data['shake_dx'] = shake_dx
        frame_data['shake_dy'] = shake_dy
        
        raw_data_list.append(frame_data)
        prev_gray = gray
        
        # Yield progress update with FPS measurement
        now_ts = time.perf_counter()
        frame_duration = max(now_ts - last_iter_timestamp, 1e-6)
        inst_fps = 1.0 / frame_duration
        if smoothed_fps is None:
            smoothed_fps = inst_fps
        else:
            smoothed_fps = (fps_smoothing * inst_fps) + ((1 - fps_smoothing) * smoothed_fps)
        last_iter_timestamp = now_ts

        progress_fraction = (frame_count + 1) / total_frames
        yield (
            'step1',
            progress_fraction,
            f'Collecting data: frame {frame_count + 1}/{total_frames} ({smoothed_fps:.1f} FPS)'
        )

        # --- Memory Management ---
        # Explicitly delete large objects to help GC
        del frame, frame_rgb, results_pose, results_yolo
        if segmentation_mask is not None: del segmentation_mask
        if background_mask is not None: del background_mask
        
        # Periodically force garbage collection to prevent memory ballooning
        if frame_count % 50 == 0:
            gc.collect()

    cap.release()
    if pose:
        pose.close()
    
    # --- Save data to pickle file ---
    output_data = {
        "metadata": {
            "frame_width": frame_width,
            "frame_height": frame_height,
            "fps": fps,
            "total_frames_processed": len(raw_data_list)
        },
        "data": raw_data_list
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\nStep 1 Complete. Processed {len(raw_data_list)} frames.")
    print(f"Raw data saved to '{output_path}'")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Step 1: Collect raw motion data from video.")
    parser.add_argument("--input", required=True, help="Path to the source video file (e.g., video.mp4)")
    parser.add_argument("--model", required=True, help="Path to the trained YOLO model (e.g., best.pt, best.onnx, or an OpenVINO export directory)")
    parser.add_argument("--output", default="raw_data.pkl", help="Path to save the raw data pickle file.")
    # NEW: Added class_name argument
    parser.add_argument("--class_name", default='endcap', 
                        help="The exact class name for the barbell endcap.")
    parser.add_argument("--lift_type", default='none',
                        help="Type of lift (e.g., 'clean', 'snatch', 'none'). If 'none', pose estimation is skipped.")
    
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
        print("Error: Model directory must include 'openvino' in its name to be treated as an OpenVINO export.")
        return

    if is_openvino_dir and not any(model_path.glob("*.xml")):
        print(f"Error: OpenVINO directory '{args.model}' does not contain a .xml model definition.")
        return

    # NEW: Pass class_name to the main function
    # Consume the generator to run the function
    for _ in step_1_collect_data(args.input, args.model, args.output, args.class_name, args.lift_type):
        pass  # Progress updates ignored when run standalone

if __name__ == '__main__':
    main()