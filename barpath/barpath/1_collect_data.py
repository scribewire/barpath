import cv2
import os
import argparse
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from tqdm import tqdm
import pickle

# --- Constants ---

LANDMARKS_TO_TRACK = {
    'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
}

# Convert string names to MediaPipe PoseLandmark objects
LANDMARK_ENUMS = {name: mp.solutions.pose.PoseLandmark[name.upper()] for name in LANDMARKS_TO_TRACK}


# --- Step 1: Data Collection Function ---
# NEW: Added class_name parameter
def step_1_collect_data(video_path, model_path, output_path, class_name):
    print("--- Step 1: Collecting Raw Data ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"Error: Video file {video_path} has no frames.")
        cap.release()
        return

    # Initialize MediaPipe Pose
    mp_pose_solution = mp.solutions.pose
    pose = mp_pose_solution.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        enable_segmentation=True  # Enable segmentation for stabilization
    )
    
    # Initialize YOLO Model
    try:
        yolo_model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading YOLO model from {model_path}: {e}")
        cap.release()
        pose.close()
        return
    
    # --- NEW: Validate class name ---
    target_class_name = class_name
    if target_class_name not in yolo_model.names.values():
        print(f"\n[Warning] Class name '{target_class_name}' not found in model.")
        print(f"  Available classes: {list(yolo_model.names.values())}")
        target_class_name = yolo_model.names[0] # Get name of class ID 0
        print(f"  Falling back to class ID 0: '{target_class_name}'\n")
    else:
        print(f"  Target class name '{target_class_name}' found in model.")
    # --- End new block ---
    
    # Stabilization parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    prev_gray = None
    background_features = None
    
    raw_data_list = []
    
    # State variable for tracking-by-proximity
    last_known_barbell_center = None
    
    for frame_count in tqdm(range(total_frames), desc="Pass 1: Collecting Data"):
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
                    # NEW: Use the validated target_class_name
                    if yolo_model.names[cls_id] == target_class_name:
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

                    if l_visible and r_visible:
                        feet_pos_px = (l_pos + r_pos) / 2
                    elif l_visible:
                        feet_pos_px = l_pos
                    elif r_visible:
                        feet_pos_px = r_pos
                
                if feet_pos_px is not None:
                    # Logic 1: Use feet position
                    best_endcap = min(detected_endcaps, 
                                      key=lambda e: np.linalg.norm(np.array(e['center']) - feet_pos_px))
                    tqdm.write(f"\n[Info] Barbell initially detected at frame {frame_count} (near feet).")
                else:
                    # Logic 2: Fallback to center of frame
                    best_endcap = min(detected_endcaps, 
                                      key=lambda e: abs(e['center'][0] - (frame_width / 2)))
                    tqdm.write(f"\n[Info] Barbell initially detected at frame {frame_count} (near center). No feet visible.")

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
                next_features, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, background_features, None, **lk_params)
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
        
        if (background_features is None and segmentation_mask is not None):
            background_mask = 1 - segmentation_mask
            new_features = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10, mask=background_mask)
            if new_features is not None:
                background_features = new_features
        
        frame_data['shake_dx'] = shake_dx
        frame_data['shake_dy'] = shake_dy
        
        raw_data_list.append(frame_data)
        prev_gray = gray

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
    step_1_collect_data(args.input, args.model, args.output, args.class_name)

if __name__ == '__main__':
    main()