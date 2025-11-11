import pandas as pd
import numpy as np
import argparse
import os
import pickle
from scipy.signal import savgol_filter

# --- Helper Functions ---

def calculate_angle(p1, p2, p3):
    """Calculates the angle (in degrees) between three 2D points (p1, p2, p3)."""
    # p2 is the vertex of the angle
    
    # Check for NaN values, which can happen if landmarks were not visible
    if np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any():
        return np.nan
        
    v1 = p1 - p2
    v2 = p3 - p2
    
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    # Handle numerical instability or zero-length vectors
    if norm == 0:
        return np.nan
        
    # Clamp the cosine value to [-1, 1] to avoid domain errors
    cosine_angle = np.clip(dot / norm, -1.0, 1.0)
    
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_lifter_angle(landmarks):
    """Calculates the lifter's orientation angle using (x, z) coordinates."""
    try:
        l_shoulder = landmarks.get('left_shoulder')
        r_shoulder = landmarks.get('right_shoulder')
        
        if l_shoulder is None or r_shoulder is None:
            return np.nan
            
        # Use (x, z) coordinates (indices 0 and 2)
        delta_x = l_shoulder[0] - r_shoulder[0]
        delta_z = l_shoulder[2] - r_shoulder[2]
        
        angle_rad = np.arctan2(delta_z, delta_x)
        angle_deg = 90 - abs(np.degrees(angle_rad))
        return angle_deg
    except Exception:
        return np.nan

# --- Step 2: Data Analysis Function ---
def step_2_analyze_data(input_data, output_path):
    print("--- Step 2: Analyzing Data ---")
    
    # Unpack the input data
    metadata = input_data.get("metadata", {})
    df_list = input_data.get("data", [])
    
    if not df_list:
        print("Error: No data found in pickle file.")
        return
        
    df = pd.DataFrame(df_list)
    
    if 'frame' not in df.columns:
        print("Error: No 'frame' column in data.")
        return
    
    df = df.set_index('frame').sort_index()
    
    frame_gaps = df.index.to_series().diff()
    if (frame_gaps > 1).any():
        print(f"Warning: Detected {(frame_gaps > 1).sum()} gaps in frame sequence.")
        print("Time calculations will be adjusted for missing frames.")
    
    # --- Metadata ---
    frame_width = metadata.get('frame_width', 1920)
    frame_height = metadata.get('frame_height', 1080)
    fps = metadata.get('fps', 30.0)
    
    df['frame_width'] = frame_width
    df['frame_height'] = frame_height

    # --- Unpack Landmark Data ---
    LANDMARKS_TO_TRACK = {
        'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    }
    
    for name in LANDMARKS_TO_TRACK:
        df[name] = df['landmarks'].apply(lambda x: x.get(name) if isinstance(x, dict) else None)
        
        df[f'{name}_x'] = df[name].apply(
            lambda x: x[0] if (x is not None and len(x) >= 4 and x[3] > 0.1) else np.nan
        )
        df[f'{name}_y'] = df[name].apply(
            lambda x: x[1] if (x is not None and len(x) >= 4 and x[3] > 0.1) else np.nan
        )
        df[f'{name}_z'] = df[name].apply(
            lambda x: x[2] if (x is not None and len(x) >= 4 and x[3] > 0.1) else np.nan
        )
        df[f'{name}_vis'] = df[name].apply(
            lambda x: x[3] if (x is not None and len(x) >= 4) else np.nan
        )

    # --- Calculate Angles ---
    df['lifter_angle_deg'] = df['landmarks'].apply(calculate_lifter_angle)
    
    def get_pixel_pos(row, name):
        x_norm = row.get(f'{name}_x')
        y_norm = row.get(f'{name}_y')
        if pd.isna(x_norm) or pd.isna(y_norm):
            return np.array([np.nan, np.nan])
        return np.array([x_norm * frame_width, y_norm * frame_height])

    df['left_knee_angle'] = df.apply(lambda row: calculate_angle(
        get_pixel_pos(row, 'left_hip'),
        get_pixel_pos(row, 'left_knee'),
        get_pixel_pos(row, 'left_ankle')
    ), axis=1)
    
    df['right_knee_angle'] = df.apply(lambda row: calculate_angle(
        get_pixel_pos(row, 'right_hip'),
        get_pixel_pos(row, 'right_knee'),
        get_pixel_pos(row, 'right_ankle')
    ), axis=1)
    
    df['left_elbow_angle'] = df.apply(lambda row: calculate_angle(
        get_pixel_pos(row, 'left_shoulder'),
        get_pixel_pos(row, 'left_elbow'),
        get_pixel_pos(row, 'left_wrist')
    ), axis=1)

    df['right_elbow_angle'] = df.apply(lambda row: calculate_angle(
        get_pixel_pos(row, 'right_shoulder'),
        get_pixel_pos(row, 'right_elbow'),
        get_pixel_pos(row, 'right_wrist')
    ), axis=1)
    
    df['hip_y_avg'] = df[['left_hip_y', 'right_hip_y']].mean(axis=1) * frame_height

    # --- Calculate Stabilized Coordinates ---
    df['total_shake_x'] = df['shake_dx'].cumsum()
    df['total_shake_y'] = df['shake_dy'].cumsum()
    
    if 'barbell_center' in df.columns:
        df['barbell_x_raw'] = df['barbell_center'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) else np.nan)
        df['barbell_y_raw'] = df['barbell_center'].apply(lambda x: x[1] if isinstance(x, (list, tuple)) else np.nan)
    else:
        print("Warning: 'barbell_center' column not found. No barbell data will be processed.")
        df['barbell_x_raw'] = np.nan
        df['barbell_y_raw'] = np.nan
    
    df['barbell_x_stable'] = df['barbell_x_raw'] - df['total_shake_x']
    df['barbell_y_stable'] = df['barbell_y_raw'] - df['total_shake_y']
    
    # --- Calculate Kinematics ---
    if df.index.is_monotonic_increasing:
        df['time_s'] = (df.index - df.index[0]) / fps
    else:
        print("Warning: Frame indices are not monotonic. Using sequential time.")
        df['time_s'] = np.arange(len(df)) / fps
    
    df['dt'] = df['time_s'].diff()
    df['dt'] = df['dt'].fillna(1/fps)
    
    df['vel_y_px_s'] = (df['barbell_y_stable'].diff() / df['dt']) * -1
    
    # --- NEW: Calculate Bar Path Phases ---
    # 1. Interpolate and fill NaNs to create a continuous velocity signal for smoothing
    vel_filled = df['vel_y_px_s'].interpolate(method='linear').fillna(0)
    
    # 2. Smooth the velocity to remove noise. Window must be odd and less than data length.
    window_length = min(15, len(vel_filled) // 2 * 2 + 1) # Must be odd
    if window_length >= 5:
        print(f"Applying Savitzky-Golay smoothing with window {window_length}...")
        df['vel_y_smooth'] = savgol_filter(vel_filled, window_length, 3)
    else:
        print("Warning: Not enough data to smooth velocity. Phases may be noisy.")
        df['vel_y_smooth'] = vel_filled
        
    # 3. Define a velocity threshold to ignore minor jitters (5% of peak, or 10px/s)
    vel_threshold = max(10, df['vel_y_smooth'].abs().max() * 0.05)
    print(f"Using velocity threshold of {vel_threshold:.2f} px/s for phase change.")

    # 4. Determine direction state (1=Up, -1=Down)
    df['direction_state'] = 0
    df.loc[df['vel_y_smooth'] > vel_threshold, 'direction_state'] = 1
    df.loc[df['vel_y_smooth'] < -vel_threshold, 'direction_state'] = -1
    
    # 5. Fill gaps (0s) with the previous valid state
    df['direction_state'] = df['direction_state'].replace(0, method='ffill').fillna(1) # Default to 1 (Up) at start
    
    # 6. Find where the state *changes*
    df['phase_change'] = df['direction_state'].diff().ne(0)
    
    # 7. Create the phase number by taking a cumulative sum of the changes
    df['bar_phase'] = df['phase_change'].cumsum()
    # --- End new block ---

    # Y-Acceleration (px/s^2)
    df['accel_y_px_s2'] = df['vel_y_px_s'].diff() / df['dt']
    
    # Y-Jerk (px/s^3)
    df['jerk_y_px_s3'] = df['accel_y_px_s2'].diff() / df['dt']
    
    # "Specific Power" (Power-to-Mass ratio, proxy)
    df['specific_power_y'] = df['accel_y_px_s2'] * df['vel_y_px_s']
    
    # --- Preserve landmarks as string for video rendering ---
    df['landmarks_str'] = df['landmarks'].apply(lambda x: str(x) if isinstance(x, dict) else '{}')
    
    def box_to_str(x):
        """Convert box coordinates to clean string format."""
        if isinstance(x, (list, tuple)):
            values = []
            for v in x:
                if hasattr(v, 'item'):  # It's a tensor
                    values.append(v.item())
                else:
                    values.append(float(v))
            return ','.join(f'{v:.2f}' for v in values)
        return ''
    
    if 'barbell_box' in df.columns:
        df['barbell_box_str'] = df['barbell_box'].apply(box_to_str)
    else:
        df['barbell_box_str'] = ''
    
    # --- Clean up and Save ---
    # Drop raw data columns that are no longer needed
    cols_to_drop = ['landmarks', 'shake_dx', 'shake_dy'] + list(LANDMARKS_TO_TRACK)
    if 'barbell_center' in df.columns:
        cols_to_drop.append('barbell_center')
    if 'barbell_box' in df.columns:
        cols_to_drop.append('barbell_box')

    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    df.to_csv(output_path)
    print(f"Analysis complete. Enriched data saved to '{output_path}'")
    print(f"Saved {len(df)} frames with {len(df.columns)} columns")
    
    barbell_tracked = df['barbell_y_stable'].notna().sum()
    print(f"Barbell tracked in {barbell_tracked}/{len(df)} frames ({100*barbell_tracked/len(df):.1f}%)")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Step 2: Analyze raw data and save to CSV.")
    parser.add_argument("--input", default="raw_data.pkl", help="Path to the raw data pickle file from Step 1.")
    parser.add_argument("--output", default="final_analysis.csv", help="Path to save the final analysis CSV file.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        return
        
    try:
        with open(args.input, 'rb') as f:
            input_data = pickle.load(f)
    except ImportError:
        print("\nError: `scipy` is required for smoothing. Please install it:")
        print("  pip install scipy")
        return
    except Exception as e:
        print(f"Error loading pickle file {args.input}: {e}")
        return

    step_2_analyze_data(input_data, args.output)

if __name__ == '__main__':
    main()