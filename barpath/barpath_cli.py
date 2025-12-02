#!/usr/bin/env python3
"""
Command-line interface for barpath analysis.

This CLI provides a rich terminal interface with progress bars for running
the barpath weightlifting analysis pipeline.
"""

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import the core pipeline runner and hardware detection
from barpath_core import run_pipeline
from hardware_detection import (
    get_available_runtimes_for_model,
)


def _is_openvino_model_dir(path_str: str) -> bool:
    """Return True when the provided path looks like an OpenVINO export directory."""
    path = Path(path_str)
    if not path.is_dir():
        return False
    return any("openvino" in part.lower() for part in path.parts)


def print_rich_help(console, parser):
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]barpath: Weightlifting Technique Analysis Pipeline[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print(f"  {parser.description}\n")

    # Arguments Table
    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 2),
        expand=True,
    )
    table.add_column("Option", style="cyan", ratio=1)
    table.add_column("Type", style="dim", ratio=1)
    table.add_column("Description", ratio=3)

    # Add Help manually since we disabled it
    table.add_row("-h, --help", "Flag", "Show this help message and exit.")

    for action in parser._actions:
        if action.dest == "help":
            continue

        opts = ", ".join(action.option_strings)

        # Determine type/requirement
        if action.required:
            type_info = "[bold red]REQUIRED[/bold red]"
        elif action.const is not None:  # boolean flag usually
            type_info = "Flag"
        else:
            default_val = action.default
            if default_val == argparse.SUPPRESS:
                default_val = None
            type_info = f"[yellow]Default: {default_val}[/yellow]"

        # Help text
        help_text = action.help or ""
        if action.choices:
            help_text += f"\n[dim]Choices: {', '.join(map(str, action.choices))}[/dim]"

        table.add_row(opts, type_info, help_text)

    console.print("[bold]Arguments:[/bold]")
    console.print(table)
    console.print()

    # Examples
    console.print("[bold]Examples:[/bold]")
    example_text = """
[dim]# 1. Quick analysis with clean lift (CPU, no video)[/dim]
python barpath/barpath_cli.py --input_video lift.mp4 --model yolo.pt --lift_type clean --no-video

[dim]# 2. Full analysis with snatch (CPU, output video)[/dim]
python barpath/barpath_cli.py --input_video lift.mp4 --model yolo.pt --lift_type snatch --output_video out.mp4

[dim]# 3. OpenVINO model (Intel CPU optimization)[/dim]
python barpath/barpath_cli.py --input_video lift.mp4 --model models/yolo_openvino_export --lift_type none --no-video

[dim]# 4. Custom output directory[/dim]
python barpath/barpath_cli.py --input_video lift.mp4 --model yolo.pt --output_dir my_results/
"""
    console.print(
        Panel(example_text.strip(), title="Sample Commands", border_style="green")
    )
    console.print()


def main():
    """Main CLI entry point."""

    # Set up rich console
    console = Console()

    # Set up argument parser
    class CustomFormatter(
        argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        description="barpath: Offline Weightlifting Technique Analysis Pipeline",
        add_help=False,
        formatter_class=CustomFormatter,
    )

    # Main Arguments
    parser.add_argument(
        "--input_video",
        required=True,
        help="Path to the source video file (e.g., 'videos/my_clean.mp4')",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained YOLO model (e.g., 'models/best.pt', 'models/best.onnx', or an OpenVINO export directory)",
    )
    parser.add_argument(
        "--output_video",
        required=False,
        default="outputs/output.mp4",
        help="Path to save the final visualized video",
    )

    # Pipeline Control Arguments
    parser.add_argument(
        "--lift_type",
        choices=["clean", "snatch", "none"],
        default="none",
        help="The type of lift to critique. Select 'none' to skip critique.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="If set, skips Step 4 (video rendering), which is computationally expensive.",
    )

    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory to save outputs (graphs, analysis, video).",
    )

    # Check for help flag manually
    if "-h" in sys.argv or "--help" in sys.argv:
        print_rich_help(console, parser)
        sys.exit(0)

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        print_rich_help(console, parser)
        sys.exit(1)
    except SystemExit:
        # Argparse exits on error, we want to show help if possible or just let it exit
        # But since we disabled help, it only exits on error
        # We can catch it to show our help?
        # Actually, argparse prints usage to stderr on error.
        # Let's just let it be for errors, but we handled -h above.
        # However, required args missing will trigger SystemExit.
        # We can try to catch it but argparse prints to stderr directly.
        # Let's just proceed.
        raise

    # Validate inputs
    input_video_path = Path(args.input_video)
    model_path = Path(args.model)

    if not input_video_path.exists():
        print(f"Error: Input video file not found: {args.input_video}", file=sys.stderr)
        sys.exit(1)

    is_openvino_dir = _is_openvino_model_dir(args.model)

    if not model_path.exists():
        print(f"Error: Model path not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    if model_path.is_dir() and not is_openvino_dir:
        print(
            "Error: Model directory paths must include 'openvino' in the name to be treated as OpenVINO exports.",
            file=sys.stderr,
        )
        sys.exit(1)

    if is_openvino_dir and not any(model_path.glob("*.xml")):
        print(
            f"Error: OpenVINO directory '{args.model}' does not contain a .xml model definition.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Set default output video path if not provided
    if not args.output_video and not args.no_video:
        args.output_video = os.path.join(args.output_dir, "output.mp4")

    if not args.no_video and not args.output_video:
        print(
            "Error: --output_video required when rendering video (not using --no-video)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine runtime to use based on model type and available runtimes
    model_path_obj = Path(args.model)
    model_ext = model_path_obj.suffix.lower()
    is_openvino_dir = model_path_obj.is_dir() and any(
        "openvino" in part.lower() for part in model_path_obj.parts
    )

    # Get available runtimes for this model
    available_runtimes = get_available_runtimes_for_model(args.model)

    # Default runtime selection logic
    if is_openvino_dir:
        # OpenVINO directory - use openvino runtime if available
        selected_runtime = (
            "openvino" if "openvino" in available_runtimes.values() else "onnxruntime"
        )
    elif model_ext == ".onnx":
        # ONNX model - use onnxruntime
        selected_runtime = "onnxruntime"
    elif model_ext == ".pt":
        # PyTorch model - prefer ultralytics, fallback to onnxruntime
        selected_runtime = (
            "ultralytics"
            if "ultralytics" in available_runtimes.values()
            else "onnxruntime"
        )
    else:
        # Default fallback
        selected_runtime = "onnxruntime"

    # Set up rich console
    # console = Console() # Already initialized in main

    # Print startup banner
    console.print()
    console.print("[bold green]═══ Barpath Pipeline ═══[/bold green]")
    console.print()

    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  Input Video:  [cyan]{args.input_video}[/cyan]")
    console.print(f"  Model Source: [cyan]{args.model}[/cyan]")

    # Display selected runtime
    runtime_display_map = {
        "onnxruntime": "ONNX Runtime (CPU)",
        "openvino": "OpenVINO (Intel CPU)",
        "ultralytics": "Ultralytics PyTorch (CPU)",
    }
    runtime_display = runtime_display_map.get(selected_runtime, selected_runtime)
    console.print(f"  Runtime:      [cyan]{runtime_display}[/cyan]")
    if not args.no_video:
        console.print(f"  Output Video: [cyan]{args.output_video}[/cyan]")
    else:
        console.print("  Output Video: [yellow][SKIPPED - using --no-video][/yellow]")
    console.print(f"  Lift Type:    [cyan]{args.lift_type}[/cyan]")
    console.print(f"  Output Dir:   [cyan]{args.output_dir}[/cyan]\n")

    # Set up progress bar with rich
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
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
                output_dir=args.output_dir,
                encode_video=not args.no_video,
                technique_analysis=(args.lift_type != "none"),
                selected_runtime=selected_runtime,
            ):
                # Create task on first encounter of each step
                if step_name not in task_map and step_name != "complete":
                    if step_name == "step1":
                        task_map[step_name] = progress.add_task(
                            "[cyan]Step 1: Collecting data...", total=100
                        )
                    elif step_name == "step2":
                        task_map[step_name] = progress.add_task(
                            "[cyan]Step 2: Analyzing data...", total=None
                        )
                    elif step_name == "step3":
                        task_map[step_name] = progress.add_task(
                            "[cyan]Step 3: Generating graphs...", total=None
                        )
                    elif step_name == "step4":
                        task_map[step_name] = progress.add_task(
                            "[cyan]Step 4: Rendering video...",
                            total=100 if not args.no_video else None,
                        )
                    elif step_name == "step5":
                        task_map[step_name] = progress.add_task(
                            "[cyan]Step 5: Critiquing lift...", total=None
                        )

                # Update the corresponding task
                if step_name in task_map:
                    task_id = task_map[step_name]

                    if prog_value is not None:
                        # Update progress bar
                        progress.update(
                            task_id,
                            completed=prog_value * 100,
                            description=f"[cyan]{message}",
                        )
                    else:
                        # Just update the description for steps without progress
                        progress.update(
                            task_id, description=f"[green]✓[/green] {message}"
                        )
                        progress.stop_task(task_id)
                elif step_name == "complete":
                    # Pipeline complete
                    pass

            # Final summary
            console.print("\n[bold green]✓ Pipeline Complete![/bold green]")
            console.print("\n[bold]Generated files:[/bold]")
            console.print(f"  • Output Dir:      [cyan]{args.output_dir}/[/cyan]")
            console.print(
                f"  • Raw data:        [cyan]{os.path.join(args.output_dir, 'raw_data.pkl')}[/cyan]"
            )
            console.print(
                f"  • Analysis CSV:    [cyan]{os.path.join(args.output_dir, 'final_analysis.csv')}[/cyan]"
            )
            if not args.no_video:
                console.print(f"  • Output video:    [cyan]{args.output_video}[/cyan]")

            # Display Analysis Report if available
            analysis_path = os.path.join(args.output_dir, "analysis.md")
            if os.path.exists(analysis_path) and args.lift_type != "none":
                console.print()
                try:
                    with open(analysis_path, "r") as f:
                        md_content = f.read()

                    # Render markdown inside a styled panel
                    console.print(
                        Panel(
                            Markdown(md_content),
                            title="[bold cyan]Detailed Analysis Report[/bold cyan]",
                            subtitle=f"[dim]Generated from {analysis_path}[/dim]",
                            border_style="cyan",
                            padding=(1, 2),
                        )
                    )
                    console.print()
                except Exception as e:
                    console.print(f"[yellow]Could not read analysis.md: {e}[/yellow]")

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
