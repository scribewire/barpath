import sys
import gc
try:
    import cv2
except ImportError:
    print("Missing dependency: opencv-python (cv2). Install with: pip install opencv-python")
    sys.exit(1)
import os
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


# --- Step 1: Data Collection Function ---
# NEW: Added class_name parameter
def step_1_collect_data(video_path, model_path, output_path, class_name):
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
    pose = mp_pose_solution.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        enable_segmentation=True  # Enable segmentation for stabilization
    )
    
    # Initialize YOLO Model
    try:
        # Check if model is ONNX
        is_onnx = model_path.lower().endswith('.onnx')
        if is_onnx:
            print(f"Loading ONNX model: {model_path}")
            # For ONNX, we need to specify the task explicitly if not auto-detected
            yolo_model = YOLO(model_path, task='detect')
        else:
            yolo_model = YOLO(model_path)
    except Exception as e:
        cap.release()
        pose.close()
        raise RuntimeError(f"Error loading YOLO model from {model_path}: {e}")
    
    # --- NEW: Validate class name ---
    target_class_name = class_name
    
    # ONNX models loaded via Ultralytics might not have 'names' populated correctly immediately
    # or might have default names. We'll try to access it, but be robust.
    if hasattr(yolo_model, 'names') and yolo_model.names:
        if target_class_name not in yolo_model.names.values():
            print(f"\n[Warning] Class name '{target_class_name}' not found in model.")
            print(f"  Available classes: {list(yolo_model.names.values())}")
            # Fallback logic
            if len(yolo_model.names) > 0:
                target_class_name = yolo_model.names[0] # Get name of class ID 0
                print(f"  Falling back to class ID 0: '{target_class_name}'\n")
        else:
            print(f"  Target class name '{target_class_name}' found in model.")
    else:
        print(f"\n[Warning] Could not validate class names for model (likely ONNX). Assuming class ID 0.")
        # For ONNX without metadata, we might just have to assume class 0 is what we want
        # or rely on the user providing the correct class name if the model outputs it.
        # However, raw ONNX output from YOLO usually includes class indices.
        # If we can't map name->id, we might need to rely on the user knowing the ID or just take ID 0.
        # Let's assume the user wants the first class if we can't verify.
        pass 
    # --- End new block ---
    
    # Stabilization parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    prev_gray = None
    background_features = None
    
    raw_data_list = []
    
    # State variable for tracking-by-proximity
    last_known_barbell_center = None
    
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
        results_pose = pose.process(frame_rgb)
        
        # Explicitly set confidence threshold
        results_yolo = yolo_model(frame, verbose=False, conf=0.25)
        
        # 1. Process MediaPipe Data
        segmentation_mask = None
        if results_pose.pose_landmarks:
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
                    
                    # Check if we can map ID to name
                    if hasattr(yolo_model, 'names') and cls_id in yolo_model.names:
                        detected_name = yolo_model.names[cls_id]
                        is_match = (detected_name == target_class_name)
                    else:
                        # If no names metadata (common in bare ONNX), assume class 0 is the target
                        # or match strictly on class ID 0 if that's the convention
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
                if results_pose.pose_landmarks:
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
        
        # Yield progress update
        progress_fraction = (frame_count + 1) / total_frames
        yield ('step1', progress_fraction, f'Collecting data: frame {frame_count + 1}/{total_frames}')

        # --- Memory Management ---
        # Explicitly delete large objects to help GC
        del frame, frame_rgb, results_pose, results_yolo
        if segmentation_mask is not None: del segmentation_mask
        if background_mask is not None: del background_mask
        
        # Periodically force garbage collection to prevent memory ballooning
        if frame_count % 50 == 0:
            gc.collect()

    cap.release()
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
    parser.add_argument("--model", required=True, help="Path to the trained YOLO model file (e.g., best.pt)")
    parser.add_argument("--output", default="raw_data.pkl", help="Path to save the raw data pickle file.")
    # NEW: Added class_name argument
    parser.add_argument("--class_name", default='endcap', 
                        help="The exact class name for the barbell endcap.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        return
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return

    # NEW: Pass class_name to the main function
    # Consume the generator to run the function
    for _ in step_1_collect_data(args.input, args.model, args.output, args.class_name):
        pass  # Progress updates ignored when run standalone

if __name__ == '__main__':
    main()