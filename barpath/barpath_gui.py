#!/usr/bin/env python3
"""Toga-based GUI frontend for the barpath pipeline.

This provides a cross-platform GUI using Toga that imports and uses
the barpath_core runner directly, displaying progress via Toga's
add_background_task mechanism.
"""

from __future__ import annotations

import asyncio
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, List, Any

# Suppress Google protobuf deprecation warnings (can't be fixed in our code)
warnings.filterwarnings("ignore", message=".*google._upb._message.*", category=DeprecationWarning)

import toga
from toga.style import Pack

# Import the core pipeline runner
sys.path.insert(0, str(Path(__file__).parent))
from barpath_core import run_pipeline


class BarpathTogaApp(toga.App):
    """Main application class for the Barpath GUI."""
    
    def startup(self) -> None:  # type: ignore[override]
        """Construct the main window and widgets."""
        
        # --- State ---
        self.model_dir: Optional[Path] = None
        self.model_files: List[Path] = []
        self.selected_model: Optional[Path] = None
        self.input_video: Optional[Path] = None
        self.output_video: Optional[Path] = None
        self.output_dir: Path = Path("outputs")
        self.lift_type: str = "none"
        self.class_name: str = "endcap"
        self.encode_video: bool = True
        self.technique_analysis: bool = True
        self._is_running: bool = False
        self._pipeline_task: Optional[asyncio.Task[Any]] = None
        
        # --- Main window ---
        self.main_window = toga.MainWindow(title="Barpath - Weightlifting Analysis Tool")
        
        # Root layout
        root_box = toga.Box(style=Pack(direction="column", margin=10))
        
        # --- Configuration panel ---
        config_label = toga.Label("Configuration", style=Pack(font_weight="bold", margin_bottom=6))
        root_box.add(config_label)
        
        config_box = toga.Box(style=Pack(direction="column", margin=6, margin_bottom=10))
        root_box.add(config_box)
        
        # Row: YOLO models directory
        models_row = toga.Box(style=Pack(direction="row", margin_bottom=6, align_items="center"))
        models_row.add(toga.Label("YOLO Models Directory:", style=Pack(width=170)))
        self.model_dir_input = toga.TextInput(
            readonly=True,
            placeholder="Select directory containing YOLO model files...",
            style=Pack(flex=1, margin_right=6),
        )
        models_row.add(self.model_dir_input)
        models_row.add(toga.Button("Browse", on_press=self.on_browse_models_dir, style=Pack(width=90)))
        config_box.add(models_row)
        
        # Row: Input video file
        video_row = toga.Box(style=Pack(direction="row", margin_bottom=6, align_items="center"))
        video_row.add(toga.Label("Input Video File:", style=Pack(width=170)))
        self.video_input = toga.TextInput(
            readonly=True,
            placeholder="Select video file to analyze...",
            style=Pack(flex=1, margin_right=6),
        )
        video_row.add(self.video_input)
        video_row.add(toga.Button("Browse", on_press=self.on_browse_video, style=Pack(width=90)))
        config_box.add(video_row)
        
        # Row: Select model dropdown
        model_row = toga.Box(style=Pack(direction="row", margin_bottom=6, align_items="center"))
        model_row.add(toga.Label("Select Model:", style=Pack(width=170)))
        self.model_select = toga.Selection(items=["(Select directory first)"], style=Pack(flex=1))
        self.model_select.enabled = False
        model_row.add(self.model_select)
        config_box.add(model_row)
        
        # Row: Lift type dropdown
        lift_row = toga.Box(style=Pack(direction="row", margin_bottom=6, align_items="center"))
        lift_row.add(toga.Label("Lift Type:", style=Pack(width=170)))
        self.lift_select = toga.Selection(items=["none", "clean", "snatch"], style=Pack(width=160))
        self.lift_select.value = "none"
        lift_row.add(self.lift_select)
        config_box.add(lift_row)
        
        # Row: Encode video toggle
        encode_row = toga.Box(style=Pack(direction="row", margin_bottom=6, align_items="center"))
        self.encode_switch = toga.Switch("Generate output video", value=True, on_change=self.on_encode_toggle, style=Pack(flex=1))
        encode_row.add(self.encode_switch)
        config_box.add(encode_row)
        
        # Row: Output video path (conditionally visible)
        self.output_video_row = toga.Box(style=Pack(direction="row", margin_bottom=6, align_items="center"))
        self.output_video_row.add(toga.Label("Output Video:", style=Pack(width=170)))
        self.output_video_input = toga.TextInput(
            value="",
            placeholder="Leave empty for default (outputs/output.mp4)",
            style=Pack(flex=1),
        )
        self.output_video_row.add(self.output_video_input)
        config_box.add(self.output_video_row)
        
        # Row: Output directory
        output_dir_row = toga.Box(style=Pack(direction="row", margin_bottom=6, align_items="center"))
        output_dir_row.add(toga.Label("Output Directory:", style=Pack(width=170)))
        self.output_dir_input = toga.TextInput(
            value="outputs",
            placeholder="outputs",
            style=Pack(flex=1),
        )
        output_dir_row.add(self.output_dir_input)
        config_box.add(output_dir_row)
        
        # Row: Class name
        class_row = toga.Box(style=Pack(direction="row", margin_bottom=6, align_items="center"))
        class_row.add(toga.Label("YOLO Class Name:", style=Pack(width=170)))
        self.class_name_input = toga.TextInput(
            value="endcap",
            placeholder="endcap",
            style=Pack(flex=1),
        )
        class_row.add(self.class_name_input)
        config_box.add(class_row)
        
        # --- Progress section ---
        progress_label = toga.Label("Progress", style=Pack(font_weight="bold", margin=(10, 0, 6, 0)))
        root_box.add(progress_label)
        
        self.progress_bar = toga.ProgressBar(max=100, style=Pack(margin=6, flex=1))
        root_box.add(self.progress_bar)
        
        self.progress_label = toga.Label("Ready to start analysis", style=Pack(margin=(0, 6, 6, 6)))
        root_box.add(self.progress_label)
        
        # --- Log/Output area ---
        log_label = toga.Label("Log Output", style=Pack(font_weight="bold", margin=(10, 0, 6, 0)))
        root_box.add(log_label)
        
        self.log_output = toga.MultilineTextInput(
            readonly=True,
            placeholder="Pipeline output will appear here...",
            style=Pack(flex=1, margin=6, height=200),
        )
        root_box.add(self.log_output)
        
        # --- Action buttons ---
        button_box = toga.Box(style=Pack(direction="row", margin=6))
        self.run_button = toga.Button("Run Analysis", on_press=self.on_run_analysis, style=Pack(margin_right=6))
        button_box.add(self.run_button)
        
        self.view_analysis_button = toga.Button("View Analysis", on_press=self.on_view_analysis, enabled=False, style=Pack(margin_right=6))
        button_box.add(self.view_analysis_button)
        
        self.cancel_button = toga.Button("Cancel", on_press=self.on_cancel_analysis, enabled=False)
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
        """Append text to the log output."""
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
        """Populate the model selection dropdown with .pt files from the directory."""
        self.model_dir = directory
        self.model_dir_input.value = str(directory)
        
        # Find all .pt files
        self.model_files = sorted(directory.glob("*.pt"))
        
        if self.model_files:
            model_names = [f.name for f in self.model_files]
            self.model_select.items = model_names
            self.model_select.value = model_names[0]
            self.model_select.enabled = True
            self.append_log(f"Found {len(model_names)} model(s) in {directory}")
        else:
            self.model_select.items = ["(No .pt files found)"]
            self.model_select.enabled = False
            self.append_log(f"No .pt model files found in {directory}")
    
    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    
    def on_encode_toggle(self, widget: toga.Widget) -> None:
        """Handle encode video toggle."""
        self.encode_video = self.encode_switch.value
        # Could show/hide output_video_row here if needed
    
    async def on_browse_models_dir(self, widget: toga.Widget) -> None:
        """Browse for models directory."""
        try:
            path = await self.main_window.dialog(  # type: ignore
                toga.SelectFolderDialog(title="Select Models Directory")
            )
            if path:
                self._populate_model_files(Path(path))
        except Exception as e:
            await self.main_window.error_dialog("Error", f"Could not select directory: {e}")  # type: ignore
    
    async def on_browse_video(self, widget: toga.Widget) -> None:
        """Browse for input video file."""
        try:
            path = await self.main_window.dialog(  # type: ignore
                toga.OpenFileDialog(
                    title="Select Video File",
                    file_types=["mp4", "MP4", "avi", "mov", "MOV", "mkv", "MKV", "webm", "WEBM"]
                )
            )
            if path:
                self.input_video = Path(path)
                self.video_input.value = str(path)
                self.append_log(f"Selected video: {path}")
        except Exception as e:
            await self.main_window.error_dialog("Error", f"Could not select file: {e}")  # type: ignore
    
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
        if not self.input_video:
            self.append_log("[ERROR] Please select an input video file")
            return
        
        selected_model = self._resolve_selected_model()
        if not selected_model:
            self.append_log("[ERROR] Please select a valid model")
            return
        
        # Get parameters
        self.lift_type = str(self.lift_select.value) if self.lift_select.value else "none"
        self.class_name = str(self.class_name_input.value) if self.class_name_input.value else "endcap"
        self.output_dir = Path(str(self.output_dir_input.value) if self.output_dir_input.value else "outputs")
        self.encode_video = bool(self.encode_switch.value)
        
        if self.encode_video:
            output_value = str(self.output_video_input.value) if self.output_video_input.value else ""
            if output_value:
                self.output_video = Path(output_value)
            else:
                self.output_video = self.output_dir / "output.mp4"
        else:
            self.output_video = None
        
        # Clear log
        self.log_output.value = ""
        self.append_log("=== Starting Barpath Analysis ===")
        self.append_log(f"Input Video: {self.input_video}")
        self.append_log(f"Model: {selected_model}")
        self.append_log(f"Class Name: {self.class_name}")
        self.append_log(f"Lift Type: {self.lift_type}")
        self.append_log(f"Encode Video: {self.encode_video}")
        self.append_log(f"Output Dir: {self.output_dir}")
        self.append_log("")
        
        # Update UI state
        self._is_running = True
        self.run_button.enabled = False
        self.cancel_button.enabled = True
        self.progress_bar.value = 0
        self.progress_label.text = "Starting pipeline..."
        
        # Run pipeline in background using asyncio directly (add_background_task is deprecated)
        self._pipeline_task = asyncio.create_task(self._run_pipeline_async())
    
    async def _run_pipeline_async(self) -> None:
        """Background task that runs the pipeline and updates progress."""
        try:
            selected_model = self._resolve_selected_model()
            
            # Run the pipeline generator
            for step_name, progress_value, message in run_pipeline(
                input_video=str(self.input_video),
                model_path=str(selected_model),
                output_video=str(self.output_video) if self.output_video else None,
                lift_type=self.lift_type,
                class_name=self.class_name,
                output_dir=str(self.output_dir),
                encode_video=self.encode_video,
                technique_analysis=(self.lift_type != "none")
            ):
                # Update UI
                # Only log if it's not a frame update to avoid freezing/OOM
                if "frame" not in message.lower() or progress_value is None:
                    self.append_log(f"[{step_name}] {message}")
                
                if progress_value is not None:
                    self.progress_bar.value = int(progress_value * 100)
                    self.progress_label.text = message
                else:
                    self.progress_label.text = f"✓ {message}"
                
                # Allow UI to update
                await asyncio.sleep(0.01)
            
            # Success!
            self.append_log("\n=== Pipeline Complete! ===")
            self.progress_bar.value = 100
            self.progress_label.text = "Analysis complete!"
            
            if (self.output_dir / "analysis.md").exists():
                self.view_analysis_button.enabled = True
            
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
        # Note: Cancellation is tricky with generators - would need to be implemented
        # in the core pipeline with threading/multiprocessing support
        self.append_log("\n[WARNING] Cancel not yet implemented for generator-based pipeline")
        self.append_log("Please close the application to stop the analysis.")

    def on_view_analysis(self, widget: toga.Widget) -> None:
        """Open a dialog to view the analysis report."""
        analysis_path = self.output_dir / "analysis.md"
        if not analysis_path.exists():
            self.main_window.info_dialog("Info", f"No analysis report found at {analysis_path}")  # type: ignore
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
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("# "):
                # H1
                label = toga.Label(
                    line[2:], 
                    style=Pack(font_weight="bold", font_size=18, margin_bottom=10, margin_top=5, color="#2c3e50")
                )
                content_box.add(label)
            elif line.startswith("## "):
                # H2
                label = toga.Label(
                    line[3:], 
                    style=Pack(font_weight="bold", font_size=14, margin_bottom=5, margin_top=10, color="#34495e")
                )
                content_box.add(label)
            elif line.startswith("- "):
                # List item
                text = line[2:].replace("**", "") # Remove bold markers
                label = toga.Label(
                    f"• {text}", 
                    style=Pack(margin_bottom=3, margin_left=15, font_size=10)
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
    # Prefer local project asset for app icon and about dialog
    repo_assets = Path(__file__).resolve().parent / "assets" / "assets"
    icon_candidate = repo_assets / "barpath.png"
    
    # Fallback paths
    if not icon_candidate.exists():
        repo_assets = Path(__file__).resolve().parent / "gui" / "assets"
        icon_candidate = repo_assets / "barpath.png"
    
    if not icon_candidate.exists():
        icon_candidate = Path(__file__).resolve().parent / "assets" / "barpath.png"
    
    if icon_candidate.exists():
        app = BarpathTogaApp(
            "Barpath",
            "org.barpath.app",
            icon=str(icon_candidate),
            description="Weightlifting Technique Analysis Tool",
            version="1.0.0",
            author="Barpath Team",
            home_page="https://github.com/scribewire/barpath"
        )
    else:
        app = BarpathTogaApp(
            "Barpath",
            "org.barpath.app",
            description="Weightlifting Technique Analysis Tool",
            version="1.0.0",
            author="Barpath Team",
            home_page="https://github.com/scribewire/barpath"
        )
    
    return app.main_loop()


if __name__ == "__main__":
    main()
