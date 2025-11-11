import cv2
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import ast

# --- Constants ---
# NEW: Updated legend colors
LEGEND_COLORS = {
    "Torso": (255, 255, 0), "Left Arm": (0, 165, 255), "Right Arm": (0, 255, 255),
    "Left Leg": (255, 0, 128), "Right Leg": (0, 255, 0), "Barbell Box": (255, 0, 255),
    "Path (Up)": (0, 0, 255),       # Red
    "Path (Down)": (0, 165, 255),  # Orange
    "Path (Up 2)": (0, 255, 0),    # Green
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

# --- Drawing Utilities ---
def draw_legend(image, colors):
    """Draws a color legend on the image."""
    y_offset = 30
    for i, (name, color) in enumerate(colors.items()):
        cv2.rectangle(image, (15, 10 + i * y_offset), (35, 30 + i * y_offset), color, -1)
        cv2.putText(image, name, (45, 25 + i * y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return 15 + len(colors) * y_offset

def get_connection_color(lm1_name, lm2_name):
    """Determines the color for a skeleton connection based on body part."""
    # Check for torso connections
    if ('shoulder' in lm1_name or 'hip' in lm1_name) and ('shoulder' in lm2_name or 'hip' in lm2_name):
        return LEGEND_COLORS["Torso"]
    
    # Check for left side
    if 'left' in lm1_name and 'left' in lm2_name:
        if any(part in lm1_name for part in ['shoulder', 'elbow', 'wrist']):
            return LEGEND_COLORS["Left Arm"]
        if any(part in lm1_name for part in ['hip', 'knee', 'ankle']):
            return LEGEND_COLORS["Left Leg"]
    
    # Check for right side
    if 'right' in lm1_name and 'right' in lm2_name:
        if any(part in lm1_name for part in ['shoulder', 'elbow', 'wrist']):
            return LEGEND_COLORS["Right Arm"]
        if any(part in lm1_name for part in ['hip', 'knee', 'ankle']):
            return LEGEND_COLORS["Right Leg"]
    
    return (255, 255, 255)

def parse_landmarks_from_string(landmarks_str):
    """Safely parses the landmark dictionary string from the CSV."""
    try:
        if pd.isna(landmarks_str) or landmarks_str == '{}':
            return {}
        return ast.literal_eval(landmarks_str)
    except Exception as e:
        return {}

def parse_barbell_box(box_str):
    """Parses the barbell box string from CSV."""
    try:
        if pd.isna(box_str) or box_str == '':
            return None
        values = [float(v.strip()) for v in str(box_str).split(',')]
        if len(values) == 4:
            return tuple(map(int, values))
    except Exception as e:
        pass
    return None

def step_4_render_video(df, video_path, output_video_path):
    """
    Takes the final analysis data and the original video, and renders
    the full visualization video.
    """
    print("--- Step 4: Rendering Final Video ---")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # --- NEW: Get all stabilized path points WITH phase data ---
    path_cols = ['barbell_x_stable', 'barbell_y_stable', 'bar_phase']
    if not all(col in df.columns for col in path_cols):
        print("Error: Missing required path or phase columns in CSV.")
        print("Please re-run Step 2.")
        return
        
    path_df = df[path_cols].dropna()
    # This maps frame numbers to indices in the path_points array
    path_indices = path_df.index.values 
    # Array of (x, y) coordinates
    path_points = path_df[['barbell_x_stable', 'barbell_y_stable']].values
    # Array of phase numbers
    path_phases = path_df['bar_phase'].values
    
    print(f"Rendering {min(len(df), total_frames)} frames...")
    
    for frame_count in tqdm(range(min(len(df), total_frames)), desc="Rendering Video"):
        success, frame = cap.read()
        if not success:
            print(f"Warning: Could not read frame {frame_count}")
            break
        
        if frame_count not in df.index:
            out.write(frame)
            continue
            
        row = df.loc[frame_count]
        
        # --- NEW: Draw Stabilized Bar Path with Colors ---
        if not pd.isna(row['total_shake_x']):
            current_shake_x = row['total_shake_x']
            current_shake_y = row['total_shake_y']
            
            # Find all path points that have occurred up to this frame
            # np.searchsorted finds the insertion point to keep the array sorted
            max_path_index = np.searchsorted(path_indices, frame_count, side='right')
            
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
        barbell_box = parse_barbell_box(row.get('barbell_box_str', ''))
        if barbell_box:
            x1, y1, x2, y2 = barbell_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), LEGEND_COLORS["Barbell Box"], 2)
            
        # --- Draw Skeleton ---
        landmarks = parse_landmarks_from_string(row.get('landmarks_str', '{}'))
        
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
                    color = get_connection_color(lm1_name, lm2_name)
                    cv2.line(frame, p1, p2, color, 3)
            
            for name, (px, py) in landmark_pixels.items():
                cv2.circle(frame, (px, py), 5, (255, 255, 255), -1)

        # --- Draw Legend and Info Text ---
        last_y = draw_legend(frame, LEGEND_COLORS)
        
        lifter_angle = row.get('lifter_angle_deg', np.nan)
        if not pd.isna(lifter_angle):
            angle_text = f"Lifter Angle: {lifter_angle:.1f} deg"
            cv2.putText(frame, angle_text, (15, last_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        time_s = row.get('time_s', frame_count / fps)
        time_text = f"Time: {time_s:.2f}s"
        cv2.putText(frame, time_text, (15, last_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        frame_text = f"Frame: {frame_count}"
        cv2.putText(frame, frame_text, (15, last_y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
        
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
        
    step_4_render_video(df, args.input_video, args.output_video)

if __name__ == '__main__':
    main()