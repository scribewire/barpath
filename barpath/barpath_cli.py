#!/usr/bin/env python3
"""
Command-line interface for barpath analysis.

This CLI provides a rich terminal interface with progress bars for running
the barpath weightlifting analysis pipeline.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

# Suppress Google protobuf deprecation warnings (can't be fixed in our code)
warnings.filterwarnings("ignore", message=".*google._upb._message.*", category=DeprecationWarning)

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)
from rich.panel import Panel
from rich.text import Text

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import the core pipeline runner
from barpath_core import run_pipeline


def main():
    """Main CLI entry point."""
    
    # Set up argument parser
    class CustomFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description="barpath: Offline Weightlifting Technique Analysis Pipeline",
        epilog="""
Sample Commands:
  1. Quick analysis (graphs + critique, no video):
     python %(prog)s --input_video "my_lift.mp4" --model "best.pt" --lift_type clean --no-video --class_name barbell-endcap --graphs_dir "my_graphs"

  2. Full analysis (all steps):
     python %(prog)s --input_video "my_lift.mp4" --model "yolo11.pt" --lift_type clean --output_video "final.mp4" --class_name endcap --graphs_dir "graphs"

Usage Notes:
  - 'barpath' is an alpha-stage tool.
  - For best results, record video on a stable tripod/surface.
  - The lifter's full body and the nearest barbell endcap must be visible.
  - Optimal camera angle is from a side view (90-deg) to a 20-degree offset.
""",
        formatter_class=CustomFormatter
    )
    
    # Main Arguments
    parser.add_argument("--input_video", required=True, 
                       help="Path to the source video file (e.g., 'videos/my_clean.mp4')")
    parser.add_argument("--model", required=True, 
                       help="Path to the trained YOLO model file (e.g., 'models/best.pt')")
    parser.add_argument("--output_video", required=False, default="output.mp4",
                       help="Path to save the final visualized video (e.g., 'renders/final.mp4')")

    # Pipeline Control Arguments
    parser.add_argument("--lift_type", choices=['clean', 'none'], default='none',
                        help="The type of lift to critique. Select 'none' to skip critique.")
    parser.add_argument("--no-video", action='store_true',
                        help="If set, skips Step 4 (video rendering), which is computationally expensive.")
    parser.add_argument("--class_name", default='endcap',
                       help="The exact class name of the barbell endcap in your YOLO model (e.g., 'endcap').")
    parser.add_argument("--graphs_dir", default='graphs',
                       help="Directory to save generated graphs (e.g., 'graphs').")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found: {args.input_video}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    
    if not args.no_video and not args.output_video:
        print("Error: --output_video required when rendering video (not using --no-video)", file=sys.stderr)
        sys.exit(1)
    
    # Set up rich console
    console = Console()
    
    # Print startup banner
    console.print(Panel.fit(
        "[bold cyan]barpath: Weightlifting Technique Analysis Pipeline[/bold cyan]",
        border_style="cyan"
    ))
    
    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Input Video:  [cyan]{args.input_video}[/cyan]")
    console.print(f"  Model File:   [cyan]{args.model}[/cyan]")
    console.print(f"  Class Name:   [cyan]{args.class_name}[/cyan]")
    if not args.no_video:
        console.print(f"  Output Video: [cyan]{args.output_video}[/cyan]")
    else:
        console.print(f"  Output Video: [yellow][SKIPPED - using --no-video][/yellow]")
    console.print(f"  Lift Type:    [cyan]{args.lift_type}[/cyan]")
    console.print(f"  Graphs Dir:   [cyan]{args.graphs_dir}[/cyan]\n")
    
    # Set up progress bar with rich
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        # Map step names to task IDs (created dynamically)
        task_map = {}
        
        try:
            # Run the pipeline and consume progress updates
            for step_name, prog_value, message in run_pipeline(
                input_video=args.input_video,
                model_path=args.model,
                output_video=args.output_video if not args.no_video else None,
                lift_type=args.lift_type,
                class_name=args.class_name,
                graphs_dir=args.graphs_dir,
                encode_video=not args.no_video,
                technique_analysis=(args.lift_type != 'none')
            ):
                # Create task on first encounter of each step
                if step_name not in task_map and step_name != 'complete':
                    if step_name == 'step1':
                        task_map[step_name] = progress.add_task("[cyan]Step 1: Collecting data...", total=100)
                    elif step_name == 'step2':
                        task_map[step_name] = progress.add_task("[cyan]Step 2: Analyzing data...", total=None)
                    elif step_name == 'step3':
                        task_map[step_name] = progress.add_task("[cyan]Step 3: Generating graphs...", total=None)
                    elif step_name == 'step4':
                        task_map[step_name] = progress.add_task("[cyan]Step 4: Rendering video...", total=100 if not args.no_video else None)
                    elif step_name == 'step5':
                        task_map[step_name] = progress.add_task("[cyan]Step 5: Critiquing lift...", total=None)
                
                # Update the corresponding task
                if step_name in task_map:
                    task_id = task_map[step_name]
                    
                    if prog_value is not None:
                        # Update progress bar
                        progress.update(task_id, completed=prog_value * 100, description=f"[cyan]{message}")
                    else:
                        # Just update the description for steps without progress
                        progress.update(task_id, description=f"[green]✓[/green] {message}")
                        progress.stop_task(task_id)
                elif step_name == 'complete':
                    # Pipeline complete
                    pass
            
            # Final summary
            console.print("\n[bold green]✓ Pipeline Complete![/bold green]")
            console.print("\n[bold]Generated files:[/bold]")
            console.print(f"  • Raw data:        [cyan]raw_data.pkl[/cyan]")
            console.print(f"  • Analysis CSV:    [cyan]final_analysis.csv[/cyan]")
            console.print(f"  • Graphs:          [cyan]{args.graphs_dir}/[/cyan]")
            if not args.no_video:
                console.print(f"  • Output video:    [cyan]{args.output_video}[/cyan]")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
            sys.exit(130)
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
