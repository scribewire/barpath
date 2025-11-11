import pandas as pd
import numpy as np
import argparse
import os

def critique_clean(df):
    """
    Analyzes the DataFrame for a 'clean' and returns a list of critiques.
    Note: Assumes Y=0 is TOP of frame, Y=MAX is BOTTOM of frame.
    """
    critiques = []
    
    try:
        # Validate we have the necessary data
        required_cols = ['hip_y_avg', 'barbell_y_stable', 'vel_y_px_s']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return [f"Missing required data columns: {', '.join(missing_cols)}"]
        
        # --- Find Key Frames ---
        # 1. Start of pull (first frame with valid data)
        valid_hip_data = df['hip_y_avg'].dropna()
        valid_bar_data = df['barbell_y_stable'].dropna()
        
        if valid_hip_data.empty or valid_bar_data.empty:
            return ["Insufficient tracking data to analyze lift. Ensure lifter and barbell are visible."]
        
        start_frame = valid_hip_data.index[0]
        
        # 2. Bar at peak height (min stable_y value - remember lower Y = higher in frame)
        bar_peak_frame = df['barbell_y_stable'].idxmin()
        if pd.isna(bar_peak_frame):
            return ["Could not detect barbell movement peak."]
        
        # 3. Hip at highest point (min Y value)
        hip_peak_frame = df['hip_y_avg'].idxmin()
        if pd.isna(hip_peak_frame):
            return ["Could not detect hip movement in the lift."]
        
        # --- Define Lift Phases ---
        # Pulling phase is from start to hip peak
        try:
            pull_df = df.loc[start_frame:hip_peak_frame].copy()
        except KeyError:
            return ["Error defining pull phase - data inconsistency."]
        
        # Catch phase is from hip peak to end
        try:
            catch_df = df.loc[hip_peak_frame:].copy()
        except KeyError:
            return ["Error defining catch phase - data inconsistency."]
        
        if pull_df.empty:
            return ["Could not detect upward pull phase of the lift."]

        # --- Rule 1: Early Arm Bend ---
        if 'left_elbow_angle' in df.columns and 'right_elbow_angle' in df.columns:
            elbow_angle = df[['left_elbow_angle', 'right_elbow_angle']].mean(axis=1)
            
            if 'left_knee_angle' in df.columns and 'right_knee_angle' in df.columns:
                knee_angle = df[['left_knee_angle', 'right_knee_angle']].mean(axis=1)
                
                # Find first frame of arm bend during the pull
                arm_bend_frames = pull_df[elbow_angle < 170].dropna()
                if not arm_bend_frames.empty:
                    arm_bend_frame = arm_bend_frames.index[0]
                    
                    # Find frame of max knee extension during the pull
                    knee_angles_in_pull = pull_df[['left_knee_angle', 'right_knee_angle']].mean(axis=1).dropna()
                    if not knee_angles_in_pull.empty:
                        max_knee_ext_frame = knee_angles_in_pull.idxmax()
                        
                        if arm_bend_frame < max_knee_ext_frame:
                            critiques.append("Arms are bending early, before full knee/hip extension is reached.")

        # --- Rule 2: Incomplete Extension ---
        if 'left_knee_angle' in df.columns and 'right_knee_angle' in df.columns:
            knee_angle = df[['left_knee_angle', 'right_knee_angle']].mean(axis=1)
            
            if hip_peak_frame in knee_angle.index:
                knee_angle_at_hip_peak = knee_angle.loc[hip_peak_frame]
                if not pd.isna(knee_angle_at_hip_peak) and knee_angle_at_hip_peak < 170:
                    critiques.append("Incomplete second pull: Knees are not fully extended at the peak of the pull.")

        # --- Rule 3: Poor Timing ---
        # Bar speed peaks after hips start dropping
        if 'vel_y_px_s' in df.columns:
            valid_vel = df['vel_y_px_s'].dropna()
            if not valid_vel.empty:
                bar_speed_peak_frame = valid_vel.idxmax()
                
                if not pd.isna(bar_speed_peak_frame) and bar_speed_peak_frame > hip_peak_frame:
                    critiques.append("Poor timing: Bar speed is peaking after you have already started to drop under it.")
            
        # --- Rule 4: Feet Striking Too Soon ---
        if 'left_ankle_y' in df.columns and 'right_ankle_y' in df.columns:
            ankle_y = df[['left_ankle_y', 'right_ankle_y']].mean(axis=1)
            
            if not ankle_y.empty and 'frame_height' in df.columns:
                start_ankle_y = ankle_y.iloc[0]
                frame_height = df['frame_height'].iloc[0]
                
                # Check if feet moved significantly (> 5% of frame height)
                ankle_displacement = (ankle_y - start_ankle_y).abs()
                max_displacement = ankle_displacement.max()
                
                if not pd.isna(max_displacement) and max_displacement > (frame_height * 0.05):
                    # Find when feet return to approximately starting position during catch
                    feet_threshold = start_ankle_y + (frame_height * 0.02)  # 2% tolerance
                    feet_returned = catch_df[ankle_y >= feet_threshold]
                    
                    if not feet_returned.empty:
                        feet_return_frame = feet_returned.index[0]
                        
                        # Find when hips pass knee level during catch
                        if 'left_knee_y' in df.columns and 'right_knee_y' in df.columns:
                            knee_y_at_catch = catch_df[['left_knee_y', 'right_knee_y']].mean(axis=1)
                            hip_y_at_catch = catch_df['hip_y_avg']
                            
                            hip_below_knee = catch_df[hip_y_at_catch > knee_y_at_catch]
                            
                            if not hip_below_knee.empty:
                                hip_pass_knee_frame = hip_below_knee.index[0]
                                
                                if feet_return_frame < hip_pass_knee_frame:
                                    critiques.append("Feet are striking the ground too soon, before the catch is complete (hips below knees).")

        # --- Rule 5: Catch Mechanics (Shoulders) ---
        if not catch_df.empty and 'left_shoulder_y' in df.columns and 'right_shoulder_y' in df.columns:
            # Find the frame where hips are deepest in the catch
            deepest_catch_frame = catch_df['hip_y_avg'].idxmax()
            
            if not pd.isna(deepest_catch_frame) and deepest_catch_frame in catch_df.index:
                shoulder_y_avg = catch_df.loc[deepest_catch_frame, ['left_shoulder_y', 'right_shoulder_y']].mean()
                bar_y_catch = catch_df.loc[deepest_catch_frame, 'barbell_y_stable']
                
                # Convert normalized shoulder Y to pixel coordinates for comparison
                if 'frame_height' in df.columns:
                    shoulder_y_pixels = shoulder_y_avg * df['frame_height'].iloc[0]
                    
                    # Bar Y is higher (lower pixel value) means bar is above shoulders (good)
                    # Bar Y is lower (higher pixel value) means bar is below shoulders (bad)
                    if not pd.isna(bar_y_catch) and not pd.isna(shoulder_y_pixels):
                        if bar_y_catch > shoulder_y_pixels + 20:  # 20px tolerance
                            critiques.append("Poor rack position: The bar is being caught too low, below the shoulders.")

        # --- Additional Analysis: Triple Extension ---
        # Check if athlete achieves proper triple extension (ankle, knee, hip)
        if all(col in df.columns for col in ['left_knee_angle', 'right_knee_angle', 'hip_y_avg']):
            # At hip peak, check knee extension
            max_knee_angle = pull_df[['left_knee_angle', 'right_knee_angle']].max().max()
            if not pd.isna(max_knee_angle) and max_knee_angle < 165:
                critiques.append("Limited triple extension: Consider focusing on fully extending knees and hips together.")

    except Exception as e:
        print(f"Error during critique analysis: {e}")
        import traceback
        traceback.print_exc()
        critiques.append(f"Analysis encountered an error: {str(e)}")

    return critiques


def main():
    parser = argparse.ArgumentParser(description="Step 5: Provide heuristic critique for a specific lift.")
    parser.add_argument("--input", default="final_analysis.csv", help="Path to the final analysis CSV file from Step 2.")
    parser.add_argument("--lift_type", required=True, choices=['clean', 'none'], 
                       help="The lift to analyze. Use 'none' to skip critique.")
    args = parser.parse_args()

    if args.lift_type == 'none':
        print("Skipping lift critique (lift_type set to 'none').")
        return

    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        return
        
    try:
        df = pd.read_csv(args.input)
        if 'frame' in df.columns:
            df = df.set_index('frame')
        print(f"Loaded analysis data: {len(df)} frames")
    except Exception as e:
        print(f"Error loading CSV file {args.input}: {e}")
        return

    print(f"\n--- Critiquing Lift: {args.lift_type.capitalize()} ---\n")
    
    critiques = []
    if args.lift_type == 'clean':
        critiques = critique_clean(df)
    # Add more lift types here as needed
    # elif args.lift_type == 'snatch':
    #     critiques = critique_snatch(df)

    if not critiques:
        print("✓ No major technical issues detected based on the current rules.")
        print("  This indicates good timing and proper sequencing of the pull and catch.")
    else:
        print("The following technical concerns were identified:\n")
        for i, concern in enumerate(critiques, 1):
            print(f"  {i}. {concern}")
    
    print("\n" + "="*70)
    print("Note: This is an automated heuristic analysis. Consider consulting")
    print("a qualified coach for detailed technique feedback.")
    print("="*70)

if __name__ == '__main__':
    main()