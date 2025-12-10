#!/usr/bin/env python3
"""Toga-based GUI frontend for the barpath pipeline.

This provides a cross-platform GUI using Toga that imports and uses
the barpath_core runner directly, displaying progress via Toga's
add_background_task mechanism.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, List, Optional

import toga
from toga.style import Pack

# Prepare for lazy import of the pipeline runner
sys.path.insert(0, str(Path(__file__).parent))
_RUN_PIPELINE = None


def _get_run_pipeline():
    """Lazy-load barpath_core.run_pipeline so the GUI starts faster."""
    global _RUN_PIPELINE
    if _RUN_PIPELINE is None:
        from barpath_core import run_pipeline  # Local import keeps startup lightweight

        _RUN_PIPELINE = run_pipeline
    return _RUN_PIPELINE


class BarpathTogaApp(toga.App):
    """Main application class for the Barpath GUI."""

    def startup(self) -> None:  # type: ignore[override]
        """Construct the main window and widgets."""

        # --- State ---
        self.model_dir: Optional[Path] = None
        self.model_files: List[Path] = []
        self.selected_model: Optional[Path] = None
        self.input_video: Optional[Path] = None
        self.input_videos: List[Path] = []  # List of videos for batch processing
        self.output_video: Optional[Path] = None
        self.output_dir: Path = Path("outputs")
        self.lift_type: str = "none"
        self.encode_video: bool = True
        self.technique_analysis: bool = True
        self._is_running: bool = False
        self._pipeline_task: Optional[asyncio.Task[Any]] = None
        self._cancel_event = threading.Event()

        # Supported video extensions
        self.video_extensions = [
            "mp4",
            "MP4",
            "avi",
            "mov",
            "MOV",
            "mkv",
            "MKV",
            "webm",
            "WEBM",
        ]

        # --- Main window ---
        self.main_window = toga.MainWindow(
            title="Barpath - Weightlifting Analysis Tool", size=(750, 550)
        )

        # Root layout
        root_box = toga.Box(style=Pack(direction="column", margin=10))

        # --- Top horizontal layout: Left (videos) and Right (configuration) ---
        top_horizontal_box = toga.Box(style=Pack(direction="row", margin_bottom=10))

        # --- Left box: Video management ---
        left_box = toga.Box(
            style=Pack(direction="column", margin=6, margin_right=10, flex=0.65)
        )

        video_label = toga.Label(
            "Input Videos", style=Pack(font_weight="bold", margin_bottom=6)
        )
        left_box.add(video_label)

        # Row: Add/Clear buttons
        button_row = toga.Box(
            style=Pack(direction="row", margin_bottom=6, align_items="center")
        )
        button_row.add(
            toga.Button(
                "Add Videos",
                on_press=self.on_browse_video,
                style=Pack(flex=1, margin_right=6),
            )
        )
        self.clear_videos_button = toga.Button(
            "Clear Videos",
            on_press=self.on_clear_videos,
            enabled=False,
            style=Pack(flex=1),
        )
        button_row.add(self.clear_videos_button)
        left_box.add(button_row)

        # Video queue list - scrollable container with unique background
        self.video_list_container = toga.ScrollContainer(
            horizontal=True,
            vertical=True,
            style=Pack(
                flex=1,
                margin=5,
            ),
        )
        self.video_list_box = toga.Box(style=Pack(direction="column"))
        self.video_list_container.content = self.video_list_box
        left_box.add(self.video_list_container)

        top_horizontal_box.add(left_box)

        # --- Right box: Configuration panel ---
        config_box = toga.Box(style=Pack(direction="column", margin=6, flex=0.35))

        config_label = toga.Label(
            "Configuration", style=Pack(font_weight="bold", margin_bottom=6)
        )
        config_box.add(config_label)

        # Row: Select model dropdown
        model_row = toga.Box(
            style=Pack(direction="row", margin_bottom=6, align_items="center")
        )
        model_row.add(toga.Label("Select Model:", style=Pack(width=100)))
        self.model_select = toga.Selection(
            items=["(Select directory first)"],
            style=Pack(flex=1),
        )
        self.model_select.enabled = False
        model_row.add(self.model_select)
        config_box.add(model_row)

        # Lift type dropdown
        lift_row = toga.Box(
            style=Pack(direction="row", margin_bottom=6, align_items="center")
        )
        lift_row.add(toga.Label("Lift Type:", style=Pack(width=100)))
        self.lift_select = toga.Selection(
            items=["none", "clean", "snatch"], style=Pack(flex=1)
        )
        self.lift_select.value = "none"
        lift_row.add(self.lift_select)
        config_box.add(lift_row)

        top_horizontal_box.add(config_box)
        root_box.add(top_horizontal_box)

        # Row: Output directory
        output_dir_row = toga.Box(
            style=Pack(direction="row", margin_bottom=6, align_items="center")
        )
        output_dir_row.add(toga.Label("Output Directory:", style=Pack(width=120)))
        self.output_dir_input = toga.TextInput(
            value="outputs",
            placeholder="outputs",
            style=Pack(flex=1, background_color="#E4E5F1"),
        )
        output_dir_row.add(self.output_dir_input)
        output_dir_row.add(
            toga.Button(
                "Open",
                on_press=self.on_open_output_dir,
                style=Pack(width=90, margin_left=6),
            )
        )
        root_box.add(output_dir_row)

        # --- Progress section ---
        progress_label = toga.Label(
            "Progress", style=Pack(font_weight="bold", margin=(10, 0, 6, 0))
        )
        root_box.add(progress_label)

        self.progress_bar = toga.ProgressBar(max=100, style=Pack(margin=6, flex=1))
        root_box.add(self.progress_bar)

        self.progress_label = toga.Label(
            "Ready to start analysis", style=Pack(margin=(0, 6, 6, 6))
        )
        root_box.add(self.progress_label)

        # --- Log/Output area ---
        log_label = toga.Label(
            "Output Log",
            style=Pack(font_weight="bold", margin=(10, 0, 6, 0)),
        )
        root_box.add(log_label)

        self.log_output = toga.MultilineTextInput(
            readonly=True,
            placeholder="Pipeline output will appear here...",
            style=Pack(flex=1, margin=6, height=150, background_color="#E4E5F1"),
        )
        root_box.add(self.log_output)

        # --- Action buttons ---
        button_box = toga.Box(style=Pack(direction="row", margin=6))
        self.run_button = toga.Button(
            "Run Analysis", on_press=self.on_run_analysis, style=Pack(margin_right=6)
        )
        button_box.add(self.run_button)

        self.view_analysis_button = toga.Button(
            "View Analysis",
            on_press=self.on_view_analysis,
            enabled=False,
            style=Pack(margin_right=6),
        )
        button_box.add(self.view_analysis_button)

        self.cancel_button = toga.Button(
            "Cancel", on_press=self.on_cancel_analysis, enabled=False
        )
        button_box.add(self.cancel_button)

        root_box.add(button_box)

        # Set main content
        self.main_window.content = root_box  # type: ignore
        self.main_window.show()  # type: ignore

        # Try to populate default model directory
        self._populate_default_model_dir()

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def append_log(self, text: str) -> None:
        """Append text to the output log."""
        current = self.log_output.value or ""
        self.log_output.value = current + text + "\n"
        # Auto-scroll to bottom (Toga doesn't have direct scroll control, but this helps)

    def _populate_default_model_dir(self) -> None:
        """Try to find and populate the default models directory."""
        # Look for barpath/models relative to this file
        models_dir = Path(__file__).parent / "models"
        if models_dir.exists() and models_dir.is_dir():
            self._populate_model_files(models_dir)

    def _populate_model_files(self, directory: Path) -> None:
        """Populate the model selection dropdown with supported model files and OpenVINO exports."""
        self.model_dir = directory

        # Find all .pt and .onnx files
        pt_files = list(directory.glob("*.pt"))
        onnx_files = list(directory.glob("*.onnx"))
        openvino_dirs = [
            p
            for p in directory.iterdir()
            if p.is_dir() and "openvino" in p.name.lower()
        ]
        candidates = pt_files + onnx_files + openvino_dirs
        self.model_files = sorted(candidates, key=lambda p: p.name.lower())

        if self.model_files:
            model_names = [f.name for f in self.model_files]
            self.model_select.items = model_names
            self.model_select.value = model_names[0]
            self.model_select.enabled = True
            self.append_log(f"Found {len(model_names)} model source(s) in {directory}")
        else:
            self.model_select.items = ["(No supported models found)"]
            self.model_select.enabled = False
            self.append_log(f"No supported model sources found in {directory}")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_open_output_dir(self, widget: toga.Widget) -> None:
        """Open the currently configured output directory using the OS file browser."""
        target_value = self.output_dir_input.value or "outputs"
        target_path = Path(target_value).expanduser()
        if not target_path.is_absolute():
            target_path = Path.cwd() / target_path
        try:
            target_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.append_log(f"[ERROR] Could not create output directory: {e}")
            return

        try:
            if sys.platform.startswith("win"):
                os.startfile(str(target_path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(target_path)], check=False)
            else:
                subprocess.run(["xdg-open", str(target_path)], check=False)
        except Exception as e:
            self.append_log(f"[ERROR] Could not open output directory: {e}")
        else:
            self.append_log(f"[INFO] Opened output directory: {target_path}")

    async def on_browse_video(self, widget: toga.Widget) -> None:
        """Browse for input video files (supports multiple selection)."""
        try:
            path = await self.main_window.dialog(  # type: ignore
                toga.OpenFileDialog(
                    title="Select Video File(s)",
                    file_types=self.video_extensions,
                    multiple_select=True,  # KEY: Enable multiple selection
                )
            )
            if path:
                # Handle both single file and multiple files
                paths = path if isinstance(path, list) else [path]

                for p in paths:
                    video_path = Path(p)
                    if video_path not in self.input_videos:
                        self.input_videos.append(video_path)
                        self._add_video_row(video_path)  # Add to UI
                        self.append_log(f"Added video: {p}")
                # Enable clear button after adding videos
                self.clear_videos_button.enabled = len(self.input_videos) > 0
        except Exception as e:
            await self.main_window.error_dialog("Error", f"Could not select file: {e}")  # type: ignore

    def on_remove_video(self, widget: toga.Widget, video_path: Path) -> None:
        """Remove a video from the queue."""
        if video_path in self.input_videos:
            self.input_videos.remove(video_path)
            # Remove the corresponding row from UI
            # We need to rebuild the entire list
            self.video_list_box.clear()
            for vp in self.input_videos:
                self._add_video_row(vp)
            self.append_log(f"Removed video: {video_path}")
            # Update clear button state
            self.clear_videos_button.enabled = len(self.input_videos) > 0

    def on_clear_videos(self, widget: toga.Widget) -> None:
        """Clear all videos from the queue."""
        self.input_videos.clear()
        self.video_list_box.clear()
        self.append_log("Cleared all videos")
        # Disable clear button when no videos
        self.clear_videos_button.enabled = False

    def _add_video_row(self, video_path: Path) -> None:
        """Add a video row with a remove button to the list."""
        row = toga.Box(
            style=Pack(
                direction="row",
                margin_bottom=3,
                margin=5,
                background_color="#E4E5F1",
            )
        )

        # Remove button with more prominent styling
        remove_btn = toga.Button(
            "Remove",
            on_press=lambda widget, vp=video_path: self.on_remove_video(widget, vp),
            style=Pack(width=80, margin_left=5),
        )
        row.add(remove_btn)

        # Video name label
        label = toga.Label(
            str(video_path),
            style=Pack(flex=1, margin_left=10, margin_right=5),
        )
        row.add(label)

        self.video_list_box.add(row)
        # Enable clear button when videos are added
        self.clear_videos_button.enabled = True

    def _resolve_selected_model(self) -> Optional[Path]:
        """Get the full path of the currently selected model."""
        if not self.model_dir or not self.model_select.value:
            return None
        # Type guard: ensure value is a string
        selected_value = str(self.model_select.value)
        if selected_value.startswith("("):
            return None
        return self.model_dir / selected_value

    def on_run_analysis(self, widget: toga.Widget) -> None:
        """Start the analysis pipeline."""
        # Validate inputs
        if not self.input_videos:
            self.append_log("[ERROR] Please add at least one video file")
            return

        selected_model = self._resolve_selected_model()
        if not selected_model:
            self.append_log("[ERROR] Please select a valid model")
            return

        # Get parameters
        self.lift_type = (
            str(self.lift_select.value) if self.lift_select.value else "none"
        )
        self.output_dir = Path(
            str(self.output_dir_input.value)
            if self.output_dir_input.value
            else "outputs"
        )
        self.encode_video = True
        self.output_video = self.output_dir / "output.mp4"

        # Clear log
        self.log_output.value = ""
        self.append_log("=== Starting Barpath Analysis ===")
        if len(self.input_videos) > 1:
            self.append_log(f"Batch Mode: Processing {len(self.input_videos)} videos")
            for idx, vid in enumerate(self.input_videos, 1):
                self.append_log(f"  {idx}. {vid.name}")
        else:
            self.append_log(f"Input Video: {self.input_videos[0]}")
        self.append_log(f"Model: {selected_model}")
        self.append_log(f"Lift Type: {self.lift_type}")

        self.append_log(f"Output Dir: {self.output_dir}")
        self.append_log("")

        # Update UI state
        self._is_running = True
        self.run_button.enabled = False
        self.cancel_button.enabled = True
        self.progress_bar.value = 0
        self.progress_label.text = "Starting pipeline..."
        self._cancel_event.clear()

        # Run pipeline in background using asyncio directly (add_background_task is deprecated)
        self._pipeline_task = asyncio.create_task(self._run_pipeline_async())

    async def _run_pipeline_async(self) -> None:
        """Background task that runs the pipeline and updates progress."""
        try:
            run_pipeline = _get_run_pipeline()
            selected_model = self._resolve_selected_model()

            # Determine if batch processing (multiple videos)
            is_batch = len(self.input_videos) > 1
            total_videos = len(self.input_videos)

            # Process each video
            for video_idx, input_video in enumerate(self.input_videos, 1):
                if self._cancel_event.is_set():
                    break

                self.append_log(
                    f"\n=== Processing video {video_idx}/{total_videos}: {input_video.name} ==="
                )

                # Determine output directory for this video
                if is_batch:
                    # Create subfolder for each video
                    video_output_dir = self.output_dir / input_video.stem
                    video_output_dir.mkdir(parents=True, exist_ok=True)
                else:
                    # Single video: use main output directory
                    video_output_dir = self.output_dir

                # Determine output video path
                if self.encode_video:
                    if is_batch:
                        video_output_path = video_output_dir / "output.mp4"
                    else:
                        video_output_path = self.output_video
                else:
                    video_output_path = None

                # Run the pipeline generator
                for step_name, progress_value, message in run_pipeline(
                    input_video=str(input_video),
                    model_path=str(selected_model),
                    output_video=str(video_output_path) if video_output_path else None,
                    lift_type=self.lift_type,
                    output_dir=str(video_output_dir),
                    encode_video=self.encode_video,
                    technique_analysis=(self.lift_type != "none"),
                    cancel_event=self._cancel_event,
                ):
                    # Update UI
                    # Only log if it's not a frame update to avoid freezing/OOM
                    if "frame" not in message.lower() or progress_value is None:
                        self.append_log(f"[{step_name}] {message}")

                    if progress_value is not None:
                        # Calculate overall progress including video index
                        video_progress = (video_idx - 1) / total_videos
                        step_progress = progress_value / total_videos
                        overall_progress = video_progress + step_progress

                        self.progress_bar.value = int(overall_progress * 100)
                        self.progress_label.text = (
                            f"[{video_idx}/{total_videos}] {message}"
                        )
                    else:
                        self.progress_label.text = (
                            f"[{video_idx}/{total_videos}] ✓ {message}"
                        )

                    # Allow UI to update
                    await asyncio.sleep(0.01)

                self.append_log(
                    f"✓ Completed video {video_idx}/{total_videos}: {input_video.name}"
                )

            # Success!
            self.append_log("\n=== All Videos Complete! ===")
            self.progress_bar.value = 100
            self.progress_label.text = "Analysis complete!"

            # Enable view analysis button for the last processed video
            if is_batch:
                last_video = self.input_videos[-1]
                analysis_path = self.output_dir / last_video.stem / "analysis.md"
            else:
                analysis_path = self.output_dir / "analysis.md"

            if analysis_path.exists():
                self.view_analysis_button.enabled = True

        except InterruptedError:
            self.append_log("\n[CANCELLED] Pipeline stopped by user.")
            self.progress_label.text = "Cancelled"
            self.progress_bar.value = 0

        except Exception as e:
            self.append_log(f"\n[ERROR] Pipeline failed: {e}")
            import traceback

            self.append_log(traceback.format_exc())
            self.progress_label.text = f"Error: {e}"

        finally:
            # Reset UI state
            self._is_running = False
            self.run_button.enabled = True
            self.cancel_button.enabled = False
            self._pipeline_task = None

    def on_cancel_analysis(self, widget: toga.Widget) -> None:
        """Cancel the running analysis."""
        if self._is_running:
            self.append_log("\n[INFO] Cancellation requested...")
            self._cancel_event.set()
            self.cancel_button.enabled = False  # Prevent double clicks

    def on_view_analysis(self, widget: toga.Widget) -> None:
        """Open a dialog to view the analysis report."""
        # Check if batch mode
        is_batch = len(self.input_videos) > 1

        if is_batch and self.input_videos:
            # For batch mode, show the last video's analysis
            last_video = self.input_videos[-1]
            analysis_path = self.output_dir / last_video.stem / "analysis.md"
        else:
            analysis_path = self.output_dir / "analysis.md"

        if not analysis_path.exists():
            self.main_window.info_dialog(  # type: ignore[attr-defined]
                "Info", f"No analysis report found at {analysis_path}"
            )
            return

        try:
            with open(analysis_path, "r") as f:
                content = f.read()
        except Exception as e:
            self.main_window.error_dialog("Error", f"Could not read analysis file: {e}")  # type: ignore
            return

        # Create a new window to show results
        self.analysis_window = toga.Window(title="Analysis Report", size=(600, 500))

        # Scroll container for content
        scroll = toga.ScrollContainer(horizontal=False)
        content_box = toga.Box(style=Pack(direction="column", margin=15))

        # Simple Markdown Parser for Toga
        # This converts the specific structure of analysis.md into Toga widgets
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("# "):
                # H1
                label = toga.Label(
                    line[2:],
                    style=Pack(
                        font_weight="bold",
                        font_size=18,
                        margin_bottom=10,
                        margin_top=5,
                        color="#2c3e50",
                    ),
                )
                content_box.add(label)
            elif line.startswith("## "):
                # H2
                label = toga.Label(
                    line[3:],
                    style=Pack(
                        font_weight="bold",
                        font_size=14,
                        margin_bottom=5,
                        margin_top=10,
                        color="#34495e",
                    ),
                )
                content_box.add(label)
            elif line.startswith("- "):
                # List item
                text = line[2:].replace("**", "")  # Remove bold markers
                label = toga.Label(
                    f"• {text}",
                    style=Pack(margin_bottom=3, margin_left=15, font_size=10),
                )
                content_box.add(label)
            else:
                # Normal text
                label = toga.Label(line, style=Pack(margin_bottom=2))
                content_box.add(label)

        scroll.content = content_box
        self.analysis_window.content = scroll
        self.analysis_window.show()


def main() -> None:
    """Main entry point."""
    # Path to the app icon
    icon_path = Path(__file__).resolve().parent / "assets" / "barpath_icon.png"

    if icon_path.exists():
        app = BarpathTogaApp(
            "Barpath",
            "org.barpath.app",
            icon=str(icon_path),
            description="Weightlifting Technique Analysis Tool",
            version="1.0.0",
            author="Barpath Team",
            home_page="https://github.com/scribewire/barpath",
        )
    else:
        app = BarpathTogaApp(
            "Barpath",
            "org.barpath.app",
            description="Weightlifting Technique Analysis Tool",
            version="1.0.0",
            author="Barpath Team",
            home_page="https://github.com/scribewire/barpath",
        )

    return app.main_loop()


if __name__ == "__main__":
    main()
