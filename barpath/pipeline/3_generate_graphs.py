import sys
try:
    import pandas as pd
except ImportError:
    print("Missing dependency: pandas. Install with: pip install pandas")
    sys.exit(1)
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
except Exception:
    print("Missing dependency: matplotlib. Install with: pip install matplotlib")
    sys.exit(1)
import os
import argparse
try:
    import numpy as np
except ImportError:
    print("Missing dependency: numpy. Install with: pip install numpy")
    sys.exit(1)

def step_3_generate_graphs(df, output_dir):
    """
    Takes the final analysis DataFrame and generates all kinematic graphs.
    """
    print("--- Step 3: Generating Kinematic Graphs ---")
    if df.empty:
        print("Error: No data in DataFrame.")
        return
    
    if 'time_s' not in df.columns:
        print("Error: 'time_s' column not found in data.")
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # --- NEW: Truncate data at maximum height ---
    # Ensure graphs only show the lift up to the peak height
    if 'barbell_y_stable' in df.columns and df['barbell_y_stable'].notna().any():
        # Find the index where the bar reaches its highest point (min Y)
        peak_height_idx = df['barbell_y_stable'].idxmin()
        print(f"Truncating graphs data at peak height (index {peak_height_idx}).")
        df = df.loc[:peak_height_idx]
        
    # Define the kinematics to plot
    kinematics = {
        'Smoothed Vertical Velocity (px/s)': 'vel_y_smooth',
        'Smoothed Vertical Acceleration (px/s^2)': 'accel_y_smooth',
        'Smoothed Vertical Specific Power': 'specific_power_y_smooth'
    }
    
    graph_files = []
    skipped = []
    
    for title, column in kinematics.items():
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in data. Skipping graph.")
            skipped.append(title)
            continue
        
        valid_data = df[['time_s', column]].dropna()
        
        if len(valid_data) < 2:
            print(f"Warning: Insufficient data for '{title}' ({len(valid_data)} points). Skipping graph.")
            skipped.append(title)
            continue
        
        plt.figure(figsize=(12, 6))
        plt.plot(valid_data['time_s'], valid_data[column], linewidth=1.5)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel(title, fontsize=12)
        plt.grid(True, alpha=0.3)
        
        y_min, y_max = plt.ylim()
        if y_min < 0 < y_max:
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        
        graph_path = os.path.join(output_dir, f"{column}_graph.png")
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close()
        graph_files.append(graph_path)
        print(f"  ✓ Generated: {graph_path}")
    
    # --- NEW: Add special Bar Path X-Y Graph with Phases ---
    path_cols = ['barbell_x_stable', 'barbell_y_stable', 'bar_phase']
    if all(col in df.columns for col in path_cols):
        path_data_df = df[path_cols].dropna()
        
        if len(path_data_df) > 2:
            path_data = path_data_df.values
            
            plt.figure(figsize=(8, 10)) # Taller than wide
            
            # Define colors and labels
            colors = ['red', 'orange', 'green']
            labels = ['Phase (Up)', 'Phase (Down)', 'Phase (Up)']
            plotted_labels = set()
            
            current_phase = int(path_data[0, 2])
            start_index = 0
            
            # Plot segment by segment to change colors
            for i in range(1, len(path_data)):
                new_phase = int(path_data[i, 2])
                # Plot if phase changes or if it's the last point
                if new_phase != current_phase or i == len(path_data) - 1:
                    segment = path_data[start_index : i+1] # Get (x,y)
                    
                    color_index = current_phase % len(colors)
                    color = colors[color_index]
                    label = labels[color_index]
                    
                    # Add label only once per color
                    if label not in plotted_labels:
                        plt.plot(segment[:, 0], segment[:, 1], color=color, linewidth=2, label=label)
                        plotted_labels.add(label)
                    else:
                        plt.plot(segment[:, 0], segment[:, 1], color=color, linewidth=2)
                        
                    start_index = i
                    current_phase = new_phase

            # Mark start (green circle) and end (red 'x')
            plt.plot(path_data[0, 0], path_data[0, 1], 'go', markersize=10, label='Start') # Start point
            plt.plot(path_data[-1, 0], path_data[-1, 1], 'rx', markersize=10, mew=3, label='End') # End point
            
            plt.title("Stabilized Bar Path by Phase", fontsize=16, fontweight='bold')
            plt.xlabel("Horizontal Position (px)", fontsize=12)
            plt.ylabel("Vertical Position (px)", fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.gca().invert_yaxis()
            plt.axis('equal') 
            plt.legend()
            
            graph_path = os.path.join(output_dir, "barbell_xy_stable_path.png")
            plt.savefig(graph_path, dpi=150, bbox_inches='tight')
            plt.close()
            graph_files.append(graph_path)
            print(f"  ✓ Generated: {graph_path}")
        else:
            print(f"Warning: Insufficient data for 'Stabilized Bar Path' ({len(path_data_df)} points). Skipping graph.")
            skipped.append("Stabilized Bar Path")
    else:
        print("Warning: Columns for 'barbell_xy_stable_path' not found. Skipping Bar Path graph.")
        skipped.append("Stabilized Bar Path")

    # Summary
    print(f"\nStep 3 Complete.")
    print(f"  Generated: {len(graph_files)} graphs in '{output_dir}'")
    if skipped:
        print(f"  Skipped: {len(skipped)} graphs due to missing/insufficient data")
        for title in skipped:
            print(f"    - {title}")
            
    # Ensure all figures are closed to free memory
    plt.close('all')

def main():
    parser = argparse.ArgumentParser(description="Step 3: Generate kinematic graphs from analysis CSV.")
    parser.add_argument("--input", default="final_analysis.csv", help="Path to the final analysis CSV file from Step 2.")
    parser.add_argument("--output_dir", default="graphs", help="Directory to save the generated graph PNGs.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        return
        
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded data: {len(df)} frames, {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading CSV file {args.input}: {e}")
        return
        
    step_3_generate_graphs(df, args.output_dir)

if __name__ == '__main__':
    main()