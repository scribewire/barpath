import pandas as pd
import numpy as np
import argparse
import os

def analyze_phases(df, lift_type):
    """
    Identifies lift phases and returns indices.
    """
    # Check columns
    required = ['barbell_y_stable', 'hip_y_avg', 'left_knee_y', 'right_knee_y', 
                'left_shoulder_y', 'right_shoulder_y', 'frame_height', 'time_s']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Error: Missing columns {', '.join(missing)}")
        return None

    # Helper for Y coordinates (Y=0 is top)
    
    # 1. Identify Start of Lift (T0)
    start_search_limit = min(30, len(df))
    baseline_y = df['barbell_y_stable'].iloc[:start_search_limit].mean()
    frame_height = df['frame_height'].iloc[0]
    threshold = frame_height * 0.02 
    
    mask_started = df['barbell_y_stable'] < (baseline_y - threshold)
    if not mask_started.any():
        return None
    
    t0_frame = mask_started.idxmax()
    
    # 2. Identify End of First Pull (T1) - Bar at Knees
    df_post_t0 = df.loc[t0_frame:]
    knee_y_avg = (df_post_t0['left_knee_y'] + df_post_t0['right_knee_y']) / 2 * frame_height
    mask_at_knees = df_post_t0['barbell_y_stable'] <= knee_y_avg
    
    if not mask_at_knees.any():
        return None
        
    t1_frame = mask_at_knees.idxmax()
    
    # 3. Identify End of Second Pull (T2) - Hip Turnover
    df_post_t1 = df.loc[t1_frame:]
    if df_post_t1.empty: return None

    bar_peak_frame = df_post_t1['barbell_y_stable'].idxmin()
    search_window = df.loc[t1_frame:bar_peak_frame]
    if search_window.empty: search_window = df_post_t1.iloc[:10] 
         
    t2_frame = search_window['hip_y_avg'].idxmin()
    
    # 4. Identify End of Third Pull (T3) - Bottom of Catch
    df_post_t2 = df.loc[t2_frame:]
    t3_frame = df_post_t2['hip_y_avg'].idxmax()
    
    # 5. Identify End of Recovery (T4) - Max Bar Height
    t4_frame = df['barbell_y_stable'].idxmin()
    
    return {
        't0': t0_frame,
        't1': t1_frame,
        't2': t2_frame,
        't3': t3_frame,
        't4': t4_frame
    }

def check_clean_faults(df, phases):
    critiques = []
    
    t0, t1, t2, t3, t4 = phases['t0'], phases['t1'], phases['t2'], phases['t3'], phases['t4']
    frame_height = df['frame_height'].iloc[0]
    
    # Helper to get slice
    def get_phase_df(start, end):
        return df.loc[start:end]

    # --- First Pull (T0 -> T1) ---
    p1 = get_phase_df(t0, t1)
    if not p1.empty:
        # Check 1: Positive acceleration
        if 'accel_y_smooth' in p1.columns:
            # Check if mean acceleration is non-positive
            if p1['accel_y_smooth'].mean() <= 0:
                 critiques.append("First Pull: You are slowing down in the first pull")
        
        # Check 2: Vertical path
        if 'barbell_x_stable' in p1.columns:
            x_start = p1['barbell_x_stable'].iloc[0]
            max_dev = (p1['barbell_x_stable'] - x_start).abs().max()
            # Threshold: 5% of frame height (proxy for scale)
            if max_dev > (frame_height * 0.05):
                critiques.append("First Pull: The bar is being kicked out/pulled back too far")

    # --- Second Pull (T1 -> T2) ---
    p2 = get_phase_df(t1, t2)
    if not p2.empty:
        # Check 1: Positive, increasing acceleration
        if 'accel_y_smooth' in p2.columns:
            # Check if acceleration is generally increasing (second half > first half)
            # and generally positive
            mid = len(p2) // 2
            if mid > 0:
                first_half_mean = p2['accel_y_smooth'].iloc[:mid].mean()
                second_half_mean = p2['accel_y_smooth'].iloc[mid:].mean()
                
                # If slowing down (decreasing accel) or not positive
                if second_half_mean < first_half_mean or p2['accel_y_smooth'].mean() <= 0:
                     critiques.append("Second Pull: You are hitching in the second pull")

        # Check 2: Flat feet at power position (end of phase)
        if 'left_ankle_y' in p2.columns:
            start_y = p2[['left_ankle_y', 'right_ankle_y']].mean(axis=1).iloc[0]
            end_y = p2[['left_ankle_y', 'right_ankle_y']].mean(axis=1).iloc[-1]
            # If ankles rose significantly (> 2% frame height)
            if (start_y - end_y) > (frame_height * 0.02):
                critiques.append("Second Pull: You are jumping too soon")

        # Check 3: Straight arms
        if 'left_elbow_angle' in p2.columns:
            min_elbow = p2[['left_elbow_angle', 'right_elbow_angle']].min().min()
            if min_elbow < 160:
                critiques.append("Second Pull: You are bending your arms too early")

    # --- Third Pull (T2 -> T3) ---
    p3 = get_phase_df(t2, t3)
    if not p3.empty:
        # Check 1: Hips descend within 0.1 seconds
        hip_peak_y = df.loc[t2, 'hip_y_avg']
        # Define "descend" as dropping 5% of frame height
        threshold_drop = frame_height * 0.05
        
        drop_mask = p3['hip_y_avg'] > (hip_peak_y + threshold_drop)
        if drop_mask.any():
            drop_frame = drop_mask.idxmax()
            time_taken = df.loc[drop_frame, 'time_s'] - df.loc[t2, 'time_s']
            if time_taken > 0.1:
                critiques.append("Third Pull: You are getting stuck in the transition")
        else:
            # If hips never dropped that much, check total duration
            duration = df.loc[t3, 'time_s'] - df.loc[t2, 'time_s']
            if duration > 0.2: # Fallback check if drop is small but slow
                 critiques.append("Third Pull: You are getting stuck in the transition")

    # --- Recovery (T3 -> T4) ---
    p4 = get_phase_df(t3, t4)
    if not p4.empty:
        # Check 1: Continuous upward movement
        if 'vel_y_smooth' in p4.columns:
            # If velocity dips below 0 (downward movement) significantly
            # vel > 0 is UP. vel < 0 is DOWN.
            if (p4['vel_y_smooth'] < -10).any(): # Tolerance of 10px/s
                critiques.append("Recovery: Your recovery is too tiring")

    return critiques

def write_analysis_md(critiques, phases, df, lift_type):
    try:
        with open("analysis.md", "w") as f:
            f.write(f"# Analysis Report: {lift_type.capitalize()}\n\n")
            
            f.write("## Phase Timing\n")
            if phases:
                def get_duration(start_idx, end_idx):
                    return df.loc[end_idx, 'time_s'] - df.loc[start_idx, 'time_s']
                
                f.write(f"- **First Pull:**  {get_duration(phases['t0'], phases['t1']):.2f}s\n")
                f.write(f"- **Second Pull:** {get_duration(phases['t1'], phases['t2']):.2f}s\n")
                f.write(f"- **Third Pull:**  {get_duration(phases['t2'], phases['t3']):.2f}s\n")
                f.write(f"- **Recovery:**    {get_duration(phases['t3'], phases['t4']):.2f}s\n")
                f.write(f"- **Total Time:**  {get_duration(phases['t0'], phases['t4']):.2f}s\n")
            else:
                f.write("Could not identify phases.\n")
            
            f.write("\n## Critique\n")
            if not critiques:
                f.write("No major faults detected based on configured checks.\n")
            else:
                for c in critiques:
                    f.write(f"- {c}\n")
        print("Analysis report saved to 'analysis.md'")
    except Exception as e:
        print(f"Error writing analysis.md: {e}")

def critique_lift(df, lift_type='clean'):
    phases = analyze_phases(df, lift_type)
    
    critiques = []
    if phases:
        if lift_type == 'clean':
            critiques = check_clean_faults(df, phases)
        # Add snatch checks here if needed
        
        write_analysis_md(critiques, phases, df, lift_type)
        
        # Return formatted strings for CLI output
        results = []
        results.append(f"Phases identified. See analysis.md for details.")
        if critiques:
            results.extend(critiques)
        else:
            results.append("No faults detected.")
        return results
    else:
        return ["Could not identify lift phases."]

def main():
    parser = argparse.ArgumentParser(description="Step 5: Identify lift phases.")
    parser.add_argument("--input", default="final_analysis.csv", help="Path to analysis CSV.")
    parser.add_argument("--lift_type", required=True, choices=['clean', 'snatch', 'none'])
    args = parser.parse_args()

    if args.lift_type == 'none': return

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        return
        
    try:
        df = pd.read_csv(args.input)
        if 'frame' in df.columns: df = df.set_index('frame')
        
        results = critique_lift(df, args.lift_type)
        for r in results:
            print(r)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()