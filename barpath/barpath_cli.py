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

# Add repo root to path so `barpath` is importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the core pipeline runner
from barpath.barpath_core import run_pipeline


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
    [dim]# 1. Auto-detect lift type (default)[/dim]
    python barpath/barpath_cli.py --input_video lift.mp4 --model barpath/models/std_nano.pt --lift_type auto --no-video

    [dim]# 2. Full clean analysis with video output (YOLO26)[/dim]
    python barpath/barpath_cli.py --input_video lift.mp4 --model barpath/models/std_nano.pt --lift_type clean --output_video out.mp4

    [dim]# 3. Snatch analysis — reports Pull / Pull-under / Recovery phases[/dim]
    python barpath/barpath_cli.py --input_video lift.mp4 --model barpath/models/std_nano.pt --lift_type snatch --output_video out.mp4

    [dim]# 4. Jerk analysis — reports Dip / Drive / Recovery phases[/dim]
    python barpath/barpath_cli.py --input_video jerk.mp4 --model barpath/models/std_nano.pt --lift_type jerk --output_video out.mp4

    [dim]# 5. Clean & Jerk analysis — splits into two segments[/dim]
    python barpath/barpath_cli.py --input_video clean_jerk.mp4 --model barpath/models/std_nano.pt --lift_type clean_jerk --output_video out.mp4

    [dim]# 6. OpenVINO model (Intel CPU optimization, YOLO26 export)[/dim]
    python barpath/barpath_cli.py --input_video lift.mp4 --model barpath/models/std_nano_openvino_model --lift_type none --no-video

    [dim]# 7. Batch processing multiple videos[/dim]
    python barpath/barpath_cli.py --input_video vid1.mp4 vid2.mp4 vid3.mp4 --model barpath/models/std_nano.pt --lift_type auto --no-video

    [dim]# 8. Custom output directory[/dim]
    python barpath/barpath_cli.py --input_video lift.mp4 --model barpath/models/std_nano.pt --output_dir my_results/

    [dim]# 9. Skip already processed videos[/dim]
    python barpath/barpath_cli.py --input_video vid1.mp4 vid2.mp4 --model barpath/models/std_nano.pt --skip-existing

    [dim]# 10. Force reprocess all videos (overwrite existing)[/dim]
    python barpath/barpath_cli.py --input_video vid1.mp4 vid2.mp4 --model barpath/models/std_nano.pt --force
    """
    console.print(Panel(example_text.strip(), title="Sample Commands", border_style="green"))
    console.print()


def main():
    """Main CLI entry point."""

    # Set up rich console
    console = Console()

    # Set up argument parser
    class CustomFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description="barpath: Offline Weightlifting Technique Analysis Pipeline",
        add_help=False,
        formatter_class=CustomFormatter,
    )

    # Main Arguments
    _ = parser.add_argument(
        "--input_video",
        required=True,
        nargs="+",  # KEY: Accept one or more arguments
        help="Path to the source video file(s) (e.g., 'videos/my_clean.mp4' or multiple files for batch processing)",
    )
    _ = parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained YOLO model (e.g., 'barpath/models/std_nano.pt', 'models/best.onnx', 'models/best.engine', or an OpenVINO export directory like 'barpath/models/std_nano_openvino_model'). YOLO26 NMS-free models are fully supported.",
    )
    _ = parser.add_argument(
        "--output_video",
        required=False,
        default="outputs/output.mp4",
        help="Path to save the final visualized video",
    )

    # Pipeline Control Arguments
    _ = parser.add_argument(
        "--lift_type",
        choices=["auto", "clean", "snatch", "jerk", "clean_jerk", "none"],
        default="auto",
        help="The type of lift to critique. 'auto' detects the lift type automatically after Step 2. 'none' skips technique analysis.",
    )
    _ = parser.add_argument(
        "--no-video",
        action="store_true",
        help="If set, skips Step 5 (video rendering), which is computationally expensive.",
    )

    _ = parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory to save outputs (graphs, analysis, video).",
    )

    _ = parser.add_argument(
        "--lifter",
        default="generic",
        help=(
            "Lifter name for baseline selection (e.g., 'liao_hui', 'lu_xiaojun', 'generic'). "
            "Determines which pro baseline to compare against for Technique Analysis. "
            "Falls back to pooled report if lifter-specific baselines are not found."
        ),
    )

    _ = parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of videos even if output folders already exist (skip confirmation).",
    )

    _ = parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Automatically skip videos that have already been processed (skip confirmation).",
    )

    # HUD Element Toggles
    _ = parser.add_argument(
        "--no-skeleton",
        action="store_true",
        help="Hide skeleton overlay on rendered video",
    )
    _ = parser.add_argument(
        "--no-sparkline",
        action="store_true",
        help="Hide velocity sparkline HUD element",
    )
    _ = parser.add_argument(
        "--no-power-zones",
        action="store_true",
        help="Hide power zone band HUD element",
    )
    _ = parser.add_argument(
        "--no-error-markers",
        action="store_true",
        help="Hide fault error markers on bar path",
    )

    # Check for help flag manually
    if "-h" in sys.argv or "--help" in sys.argv:
        print_rich_help(console, parser)
        sys.exit(0)

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        console.print(f"[bold red]Error:[/bold red] {e!s}")
        print_rich_help(console, parser)
        sys.exit(1)
    except SystemExit:
        raise

    # Supported video extensions
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    # Process input videos - can be multiple files
    input_videos = []
    for video_arg in args.input_video:
        video_path = Path(video_arg)
        # Check if it's a valid video file
        if video_path.suffix.lower() in video_extensions:
            if video_path.exists():
                input_videos.append(video_path)
            else:
                print(f"Error: Input video file not found: {video_arg}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Warning: Skipping non-video file: {video_arg}", file=sys.stderr)

    if not input_videos:
        print("Error: No valid video files provided", file=sys.stderr)
        sys.exit(1)

    # Determine if batch processing
    is_batch = len(input_videos) > 1

    # Validate model
    model_path = Path(args.model)

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

    if is_openvino_dir:
        if not any(model_path.glob("*.xml")):
            print(
                f"Error: OpenVINO directory '{args.model}' does not contain a .xml model definition.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not any(model_path.glob("*.bin")):
            print(
                f"Error: OpenVINO directory '{args.model}' does not contain a .bin weights file.",
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

    # Print startup banner
    console.print()
    console.print("[bold green]═══ Barpath Pipeline ═══[/bold green]")
    console.print()

    console.print("\n[bold]Configuration:[/bold]")
    if is_batch:
        console.print(f"  Input Videos: [cyan]{len(input_videos)} videos to process[/cyan]")
        for i, vid in enumerate(input_videos, 1):
            console.print(f"    {i}. {vid.name}")
    else:
        console.print(f"  Input Video:  [cyan]{input_videos[0]}[/cyan]")
    console.print(f"  Model Source: [cyan]{args.model}[/cyan]")
    if not args.no_video:
        console.print(f"  Output Video: [cyan]{args.output_video}[/cyan]")
    else:
        console.print("  Output Video: [yellow][SKIPPED - using --no-video][/yellow]")
    console.print(f"  Lift Type:    [cyan]{args.lift_type}[/cyan]")
    console.print(f"  Lifter:       [cyan]{args.lifter}[/cyan]")
    console.print(f"  Output Dir:   [cyan]{args.output_dir}[/cyan]")
    console.print()

    # Pre-check all videos for existing outputs
    videos_to_process: list[Path] = []
    videos_to_skip: list[tuple[Path, Path]] = []  # (video, output_dir)

    if not args.force:
        console.print("[bold]Checking for existing outputs...[/bold]")
        for video in input_videos:
            if is_batch:
                video_output_dir = Path(args.output_dir) / video.stem
            else:
                video_output_dir = Path(args.output_dir)

            has_output = video_output_dir.exists() and (
                (video_output_dir / "final_analysis.csv").exists()
                or (video_output_dir / "raw_data.pkl").exists()
            )

            if has_output:
                videos_to_skip.append((video, video_output_dir))
            else:
                videos_to_process.append(video)

        if videos_to_skip and not args.skip_existing:
            console.print()
            console.print(
                f"[yellow]Found {len(videos_to_skip)} video(s) with existing output:[/yellow]"
            )
            for video, out_dir in videos_to_skip:
                console.print(f"  {video.name} -> {out_dir}")
            console.print()
            console.print("  [1] Skip all existing (process only new videos)")
            console.print("  [2] Rerun all existing (overwrite outputs)")
            console.print("  [3] Cancel")
            console.print()

            while True:
                try:
                    choice = input("  Enter choice [1-3]: ").strip()
                    if choice == "1":
                        console.print(
                            f"[yellow]Skipping {len(videos_to_skip)} existing video(s)[/yellow]"
                        )
                        console.print(f"[cyan]Processing {len(videos_to_process)} video(s)[/cyan]")
                        break
                    elif choice == "2":
                        videos_to_process = input_videos  # Process all
                        videos_to_skip = []
                        console.print(f"[cyan]Reprocessing all {len(input_videos)} video(s)[/cyan]")
                        break
                    elif choice == "3":
                        console.print("[yellow]Cancelled.[/yellow]")
                        sys.exit(0)
                    else:
                        console.print("[red]Invalid choice. Enter 1-3.[/red]")
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[yellow]Cancelled.[/yellow]")
                    sys.exit(0)
        elif videos_to_skip and args.skip_existing:
            console.print(f"[yellow]Skipping {len(videos_to_skip)} existing video(s)[/yellow]")
            console.print(f"[cyan]Processing {len(videos_to_process)} video(s)[/cyan]")
        else:
            videos_to_process = input_videos
    else:
        videos_to_process = input_videos
        console.print("[cyan]Force mode: processing all videos[/cyan]")

    if not videos_to_process:
        console.print("[yellow]No videos to process.[/yellow]")
        sys.exit(0)

    console.print()

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
        # Track videos skipped due to insufficient data
        skipped_insufficient: list[tuple[Path, str]] = []

        try:
            # Process each video (already filtered for existing outputs)
            for video_idx, input_video in enumerate(videos_to_process, 1):
                # Determine output directory for this video
                if is_batch:
                    video_output_dir = Path(args.output_dir) / input_video.stem
                else:
                    video_output_dir = Path(args.output_dir)

                # Create output directory
                video_output_dir.mkdir(parents=True, exist_ok=True)

                console.print(
                    f"\n[bold cyan]Processing video {video_idx}/{len(videos_to_process)}: {input_video.name}[/bold cyan]"
                )

                # Determine output video path
                if not args.no_video:
                    if is_batch:
                        video_output_path = str(video_output_dir / "output.mp4")
                    else:
                        video_output_path = args.output_video
                else:
                    video_output_path = None

                # Map step names to task IDs (created dynamically per video)
                task_map = {}

                # Run the pipeline and consume progress updates
                try:
                    hud_options = {
                        "show_skeleton": not args.no_skeleton,
                        "show_sparkline": not args.no_sparkline,
                        "show_power_zones": not args.no_power_zones,
                        "show_error_markers": not args.no_error_markers,
                    }

                    for step_name, prog_value, message in run_pipeline(
                        input_video=str(input_video),
                        model_path=args.model,
                        output_video=video_output_path,
                        lift_type=args.lift_type,
                        output_dir=str(video_output_dir),
                        encode_video=not args.no_video,
                        technique_analysis=(args.lift_type != "none"),
                        lifter=args.lifter,
                        hud_options=hud_options,
                    ):
                        # Check for insufficient data signal
                        if step_name == "_insufficient_data_":
                            console.print(
                                f"[yellow]Skipping {input_video.name}: {message}[/yellow]"
                            )
                            skipped_insufficient.append((input_video, message))
                            # Clean up empty output directory
                            try:
                                if video_output_dir.exists() and not any(
                                    video_output_dir.iterdir()
                                ):
                                    video_output_dir.rmdir()
                            except Exception:
                                pass
                            break  # Exit the for loop for this video

                        # Create task on first encounter of each step
                        if step_name not in task_map and step_name != "complete":
                            if step_name == "step1":
                                task_map[step_name] = progress.add_task(
                                    f"[cyan][{video_idx}/{len(videos_to_process)}] Step 1: Collecting data...",
                                    total=100,
                                )
                            elif step_name == "step2":
                                task_map[step_name] = progress.add_task(
                                    f"[cyan][{video_idx}/{len(videos_to_process)}] Step 2: Analyzing data...",
                                    total=None,
                                )
                            elif step_name == "step3":
                                task_map[step_name] = progress.add_task(
                                    f"[cyan][{video_idx}/{len(videos_to_process)}] Step 3: Generating graphs...",
                                    total=None,
                                )
                            elif step_name == "step4":
                                task_map[step_name] = progress.add_task(
                                    f"[cyan][{video_idx}/{len(videos_to_process)}] Step 4: Analyzing technique...",
                                    total=None,
                                )
                            elif step_name == "step5":
                                task_map[step_name] = progress.add_task(
                                    f"[cyan][{video_idx}/{len(videos_to_process)}] Step 5: Rendering video...",
                                    total=100 if not args.no_video else None,
                                )

                        # Update the corresponding task
                        if step_name in task_map:
                            task_id = task_map[step_name]

                            if prog_value is not None:
                                # Update progress bar
                                progress.update(
                                    task_id,
                                    completed=prog_value * 100,
                                    description=f"[cyan][{video_idx}/{len(videos_to_process)}] {message}",
                                )
                            else:
                                # Just update the description for steps without progress
                                progress.update(
                                    task_id,
                                    description=f"[green]:heavy_check_mark:[/green] [{video_idx}/{len(videos_to_process)}] {message}",
                                )
                                progress.stop_task(task_id)
                        elif step_name == "complete":
                            # Pipeline complete
                            pass
                    else:
                        # Only mark as completed if we didn't break due to insufficient data
                        pass  # Video completed successfully

                except Exception as e:
                    # Catch any other exceptions during pipeline execution
                    console.print(f"[red]Error processing {input_video.name}: {e!s}[/red]")
                    continue

            # Final summary
            console.print("\n[bold green]:heavy_check_mark: All Videos Processed![/bold green]")

            # Report skipped videos due to insufficient data
            if skipped_insufficient:
                console.print(
                    f"\n[yellow]Skipped {len(skipped_insufficient)} video(s) due to insufficient data:[/yellow]"
                )
                for video, reason in skipped_insufficient:
                    console.print(f"  - {video.name}: {reason}")

            console.print("\n[bold]Generated files:[/bold]")
            if is_batch:
                console.print(f"  - Output Dir:      [cyan]{args.output_dir}/[/cyan]")
                console.print("  - [cyan]Results saved in subfolders for each video[/cyan]")
                for vid in videos_to_process:
                    subfolder = os.path.join(args.output_dir, vid.stem)
                    console.print(f"    - {vid.name} -> {subfolder}/")
            else:
                console.print(f"  - Output Dir:      [cyan]{args.output_dir}/[/cyan]")
                console.print(
                    f"  - Raw data:        [cyan]{os.path.join(args.output_dir, 'raw_data.pkl')}[/cyan]"
                )
                console.print(
                    f"  - Analysis CSV:    [cyan]{os.path.join(args.output_dir, 'final_analysis.csv')}[/cyan]"
                )
                # Dynamic phase display based on lift type
                if args.lift_type == "jerk":
                    phase_text = "Dip -> Drive -> Recovery"
                elif args.lift_type in ("clean", "snatch"):
                    phase_text = "Pull -> Pull-under -> Recovery"
                elif args.lift_type == "clean_jerk":
                    phase_text = "Clean: Pull->Pull-under->Recovery | Jerk: Dip->Drive->Recovery"
                else:
                    phase_text = "N/A"
                console.print(f"  - Phases:          [cyan]{phase_text}[/cyan]")
                if not args.no_video:
                    console.print(f"  - Output video:    [cyan]{args.output_video}[/cyan]")

            # Display Analysis Report if available (for last video in batch mode)
            if is_batch and input_videos:
                last_video = input_videos[-1]
                analysis_path = os.path.join(args.output_dir, last_video.stem, "analysis.md")
            else:
                analysis_path = os.path.join(args.output_dir, "analysis.md")

            if os.path.exists(analysis_path) and args.lift_type != "none":
                console.print()
                try:
                    with open(analysis_path) as f:
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
            console.print(f"\n[bold red]Error:[/bold red] {e!s}")
            import traceback

            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
