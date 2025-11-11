import subprocess
import argparse
import os
import sys
import shlex

def run_step(step_name, cmd_list, stop_on_failure=True):
    """Helper function to run a command, print it, and check for errors."""
    
    # Use shlex.join to create a shell-safe, printable version of the command
    printable_cmd = shlex.join(cmd_list)
    print(f"\n--- Running Command ---\n{printable_cmd}\n-------------------------")
    
    try:
        # By setting capture_output=False (the default), the subprocess
        # streams its stdout/stderr directly to this process's console
        # in real-time. This is what allows tqdm progress bars to work.
        result = subprocess.run(cmd_list, check=True, text=True)
            
    except subprocess.CalledProcessError as e:
        # The error from the subprocess was already streamed to the console.
        print(f"Error: Command failed with exit code {e.returncode}", file=sys.stderr)
        if stop_on_failure:
            print("Pipeline stopped.", file=sys.stderr)
            sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Command '{cmd_list[0]}' not found.", file=sys.stderr)
        print("Please ensure Python scripts are in the same directory.", file=sys.stderr)
        sys.exit(1)


def main():
    # --- Formatter for the help text ---
    class CustomFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description="barpath: Offline Weightlifting Technique Analysis Pipeline",
        epilog="""
Sample Commands:
  1. Quick analysis (graphs + critique, no video):
     python %(prog)s --input_video "my_lift.mp4" --model "best.pt" --lift_type clean --no-video --class_name barbell-endcap

  2. Full analysis (all steps):
     python %(prog)s --input_video "my_lift.mp4" --model "yolo11.pt" --lift_type clean --output_video "final.mp4" --class_name endcap

Usage Notes:
  - 'barpath' is an alpha-stage tool.
  - For best results, record video on a stable tripod/surface.
  - The lifter's full body and the nearest barbell endcap must be visible.
  - Optimal camera angle is from a side view (90-deg) to a 20-degree offset.
""",
        formatter_class=CustomFormatter
    )
    
    # --- Main Arguments ---
    parser.add_argument("--input_video", required=True, 
                       help="Path to the source video file (e.g., 'videos/my_clean.mp4')")
    parser.add_argument("--model", required=True, 
                       help="Path to the trained YOLO model file (e.g., 'models/best.pt')")
    parser.add_argument("--output_video", required=False, default="output.mp4",
                       help="Path to save the final visualized video (e.g., 'renders/final.mp4')")

    # --- Pipeline Control Arguments ---
    parser.add_argument("--lift_type", choices=['clean', 'none'], default='none',
                        help="The type of lift to critique. Select 'none' to skip critique.")
    parser.add_argument("--no-video", action='store_true',
                        help="If set, skips Step 4 (video rendering), which is computationally expensive.")
    # NEW: Added class_name argument
    parser.add_argument("--class_name", default='endcap',
                       help="The exact class name of the barbell endcap in your YOLO model (e.g., 'endcap').")

    args = parser.parse_args()

    # Validate inputs before starting pipeline
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found: {args.input_video}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    
    if not args.no_video and not args.output_video:
        print("Error: --output_video required when rendering video (not using --no-video)", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    if not args.no_video:
        output_dir = os.path.dirname(args.output_video)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created output directory: {output_dir}")
            except Exception as e:
                print(f"Error creating output directory: {e}", file=sys.stderr)
                sys.exit(1)

    # --- Define file paths ---
    raw_data_path = "raw_data.pkl"
    analysis_csv_path = "final_analysis.csv"
    graphs_dir = "graphs"

    # FIXED: Use sys.executable to ensure we use the same Python interpreter
    python_exe = sys.executable

    # --- Print startup message ---
    print("="*60)
    print("   barpath: Weightlifting Technique Analysis Pipeline")
    print("="*60)
    print(f"  Input Video:  {args.input_video}")
    print(f"  Model File:   {args.model}")
    print(f"  Class Name:   {args.class_name}") # NEW: Show class name
    if not args.no_video:
        print(f"  Output Video: {args.output_video}")
    else:
        print(f"  Output Video: [SKIPPED - using --no-video]")
    print(f"  Lift Type:    {args.lift_type}")
    print("="*60)

    # --- STEP 1: Collect Data ---
    print("\n>>> STEP 1: Collecting Raw Data...")
    cmd_step1 = [
        python_exe, "1_collect_data.py",
        "--input", args.input_video,
        "--model", args.model,
        "--output", raw_data_path,
        "--class_name", args.class_name  # NEW: Pass class_name
    ]
    run_step("Step 1", cmd_step1)
    print(">>> Step 1 Complete.")

    # --- STEP 2: Analyze Data ---
    print("\n>>> STEP 2: Analyzing Data...")
    cmd_step2 = [
        python_exe, "2_analyze_data.py",
        "--input", raw_data_path,
        "--output", analysis_csv_path
    ]
    run_step("Step 2", cmd_step2)
    print(">>> Step 2 Complete.")

    # --- STEP 3: Generate Graphs ---
    print("\n>>> STEP 3: Generating Graphs...")
    cmd_step3 = [
        python_exe, "3_generate_graphs.py",
        "--input", analysis_csv_path,
        "--output_dir", graphs_dir
    ]
    run_step("Step 3", cmd_step3)
    print(">>> Step 3 Complete.")

    # --- STEP 4: Render Video ---
    if args.no_video:
        print("\n>>> STEP 4: Skipping video rendering due to --no-video flag.")
    else:
        print("\n>>> STEP 4: Rendering Final Video...")
        cmd_step4 = [
            python_exe, "4_render_video.py",
            "--input_csv", analysis_csv_path,
            "--input_video", args.input_video,
            "--output_video", args.output_video
        ]
        run_step("Step 4", cmd_step4)
        print(">>> Step 4 Complete.")

    # --- STEP 5: Critique Lift ---
    if args.lift_type == 'none':
        print("\n>>> STEP 5: Skipping lift critique (lift_type set to 'none').")
    else:
        print("\n>>> STEP 5: Running Lift Critique...")
        cmd_step5 = [
            python_exe, "5_critique_lift.py",
            "--input", analysis_csv_path,
            "--lift_type", args.lift_type
        ]
        run_step("Step 5", cmd_step5)
        print(">>> Step 5 Complete.")

    # --- Final Summary ---
    print("\n" + "="*60)
    print("   PIPELINE COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print(f"  - Raw data:        {raw_data_path}")
    print(f"  - Analysis CSV:    {analysis_csv_path}")
    print(f"  - Graphs:          {graphs_dir}/")
    if not args.no_video:
        print(f"  - Output video:    {args.output_video}")
    print("\nPipeline finished successfully!")

if __name__ == "__main__":
    main()