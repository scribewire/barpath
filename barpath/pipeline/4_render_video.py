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
    import pandas as pd
except ImportError:
    print("Missing dependency: pandas. Install with: pip install pandas")
    sys.exit(1)
from utils import (
    draw_legend, get_connection_color, parse_landmarks_from_string,
    parse_barbell_box, COLOR_SCHEME
)

# --- Constants ---
# NEW: Updated legend colors (using COLOR_SCHEME from utils)
LEGEND_COLORS = {
    "Torso": COLOR_SCHEME["Torso"],
    "Left Arm": COLOR_SCHEME["Left Arm"],
    "Right Arm": COLOR_SCHEME["Right Arm"],
    "Left Leg": COLOR_SCHEME["Left Leg"],
    "Right Leg": COLOR_SCHEME["Right Leg"],
    "Barbell Box": COLOR_SCHEME["Barbell Box"],
    "Path (Up)": (0, 0, 255),       # Red
    "Path (Down)": (0, 165, 255),   # Orange
    "Path (Up 2)": (0, 255, 0),     # Green
}

# BGR color map for phases
PHASE_COLORS_BGR = {
    0: (0, 0, 255),   # Phase 0: Red
    1: (0, 165, 255), # Phase 1: Orange
    2: (0, 255, 0)    # Phase 2: Green
}

# Skeleton connections to draw
SKELETON_CONNECTIONS = [
    # Torso
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    # Left arm
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    # Right arm
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    # Left leg
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    # Right leg
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
]

def step_4_render_video(df, video_path, output_video_path):
    """
    Takes the final analysis data and the original video, and renders
    the full visualization video.
    """
    print("--- Step 4: Rendering Final Video ---")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file {video_path}")
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # --- NEW: Get all stabilized path points WITH phase data ---
    path_cols = ['barbell_x_stable', 'barbell_y_stable', 'bar_phase']
    if not all(col in df.columns for col in path_cols):
        cap.release()
        raise ValueError("Missing required path or phase columns in CSV. Please re-run Step 2.")
        
    path_df = df[path_cols].dropna()
    # This maps frame numbers to indices in the path_points array
    path_indices = path_df.index.values 
    # Array of (x, y) coordinates
    path_points = path_df[['barbell_x_stable', 'barbell_y_stable']].values
    # Array of phase numbers
    path_phases = path_df['bar_phase'].values
    
        # --- CHANGED: Store frame count in a variable ---
    # Render frames up to the end of analysis + 1 second of raw footage
    # This gives a buffer after the lift/analysis finishes before cutting the video
    last_analyzed_frame = int(df.index.max()) if not df.empty else 0
    extra_frames = int(fps)
    frames_to_render = min(last_analyzed_frame + extra_frames, total_frames)
    print(f"Rendering {frames_to_render} frames (Analysis end: {last_analyzed_frame})...")
    
    # Initialize persistent state for "extra frames" rendering
    last_shake_x = 0.0
    last_shake_y = 0.0
    last_lifter_angle = np.nan

    # Loop through frames and yield progress
    for frame_count in range(frames_to_render):
        points_to_draw = None
        success, frame = cap.read()
        if not success:
            print(f"Warning: Could not read frame {frame_count}")
            break
        
        # Determine drawing parameters
        if frame_count in df.index:
            row = df.loc[frame_count]
            
            # Update persistent state
            if not pd.isna(row.get('total_shake_x')):
                last_shake_x = row['total_shake_x']
                last_shake_y = row['total_shake_y']
            if not pd.isna(row.get('lifter_angle_deg')):
                last_lifter_angle = row['lifter_angle_deg']
            
            current_shake_x = last_shake_x
            current_shake_y = last_shake_y
            
            max_path_index = np.searchsorted(path_indices, frame_count, side='right')
            
            draw_skeleton = True
            draw_box = True
            
            lifter_angle = row.get('lifter_angle_deg', np.nan)
            time_s = row.get('time_s', frame_count / fps)
            
            # Strings for parsing
            landmarks_str = row.get('landmarks_str', '{}')
            barbell_box_str = row.get('barbell_box_str', '')
            
        else:
            # Extra frames: use last known state
            current_shake_x = last_shake_x
            current_shake_y = last_shake_y
            
            max_path_index = len(path_points) # Show full path
            
            draw_skeleton = False
            draw_box = False
            
            lifter_angle = last_lifter_angle
            time_s = frame_count / fps
            
            landmarks_str = '{}'
            barbell_box_str = ''

        # --- Draw Stabilized Bar Path with Colors ---
        # Always draw path if we have points (using current_shake)
        if max_path_index >= 2:
            # Get all points and phases up to this moment
            points_to_draw = path_points[:max_path_index].copy()
            phases_to_draw = path_phases[:max_path_index]
            
            # Apply current shake to all historical points
            points_to_draw[:, 0] += current_shake_x
            points_to_draw[:, 1] += current_shake_y
            points_to_draw = points_to_draw.astype(np.int32)
            
            # Draw segment by segment to apply colors
            for i in range(len(points_to_draw) - 1):
                p1 = (points_to_draw[i, 0], points_to_draw[i, 1])
                p2 = (points_to_draw[i+1, 0], points_to_draw[i+1, 1])
                
                # Get phase, cycle through colors using modulo
                phase_index = int(phases_to_draw[i]) % len(PHASE_COLORS_BGR)
                color = PHASE_COLORS_BGR.get(phase_index, (255, 255, 255)) # Default white
                
                cv2.line(frame, p1, p2, color, 3)

        # --- Draw Barbell Box ---
        if draw_box:
            barbell_box = parse_barbell_box(barbell_box_str)
            if barbell_box:
                x1, y1, x2, y2 = barbell_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), LEGEND_COLORS["Barbell Box"], 2)
            
        # --- Draw Skeleton ---
        if draw_skeleton:
            landmarks = parse_landmarks_from_string(landmarks_str)
            
            if landmarks:
                landmark_pixels = {}
                for name, (x, y, z, vis) in landmarks.items():
                    if vis > 0.1:  # Only use visible landmarks
                        px = int(x * frame_width)
                        py = int(y * frame_height)
                        landmark_pixels[name] = (px, py)
                
                for lm1_name, lm2_name in SKELETON_CONNECTIONS:
                    if lm1_name in landmark_pixels and lm2_name in landmark_pixels:
                        p1 = landmark_pixels[lm1_name]
                        p2 = landmark_pixels[lm2_name]
                        color = get_connection_color(lm1_name, lm2_name, LEGEND_COLORS)
                        cv2.line(frame, p1, p2, color, 3)
                
                for name, (px, py) in landmark_pixels.items():
                    cv2.circle(frame, (px, py), 5, (255, 255, 255), -1)

        # --- Draw Legend and Info Text ---
        last_y = draw_legend(frame, LEGEND_COLORS)
        
        if not pd.isna(lifter_angle):
            angle_text = f"Lifter Angle: {lifter_angle:.1f} deg"
            cv2.putText(frame, angle_text, (15, last_y + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        time_text = f"Time: {time_s:.2f}s"
        cv2.putText(frame, time_text, (15, last_y + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        frame_text = f"Frame: {frame_count}"
        cv2.putText(frame, frame_text, (15, last_y + 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
        
        # Yield progress update
        progress_fraction = (frame_count + 1) / frames_to_render
        yield ('step4', progress_fraction, f'Rendering video: frame {frame_count + 1}/{frames_to_render}')

        # --- Memory Management ---
        del frame
        if points_to_draw is not None: del points_to_draw
        
        if frame_count % 50 == 0:
            gc.collect()
            
    cap.release()
    out.release()
    print(f"Step 4 Complete. Final video saved to '{output_video_path}'")

def main():
    parser = argparse.ArgumentParser(description="Step 4: Render final analysis video.")
    parser.add_argument("--input_video", required=True, help="Path to the original source video file.")
    parser.add_argument("--input_csv", default="final_analysis.csv", help="Path to the final analysis CSV from Step 2.")
    parser.add_argument("--output_video", required=True, help="Path to save the final visualized video.")
    args = parser.parse_args()

    if not os.path.exists(args.input_video):
        print(f"Error: Input video not found at {args.input_video}")
        return
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV not found at {args.input_csv}")
        return
        
    try:
        df = pd.read_csv(args.input_csv)
        if 'frame' in df.columns:
            df = df.set_index('frame')
        print(f"Loaded CSV with {len(df)} frames and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading CSV file {args.input_csv}: {e}")
        return
        
    # Consume the generator to run the function
    for _ in step_4_render_video(df, args.input_video, args.output_video):
        pass  # Progress updates ignored when run standalone

if __name__ == '__main__':
    main()