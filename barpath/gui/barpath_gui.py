#!/usr/bin/env python3
"""Toga-based GUI frontend for the barpath pipeline.

This provides a cross-platform GUI using Toga, inspired by
`barpath_gui_test.py` (Tkinter) and wired into `barpath_cli.py`.

Features implemented in this first pass:
- Configuration panel for YOLO model directory, model file, input video
- Lift type selection, encode-video and technique-analysis toggles
- Output video path and graphs directory configuration
- Simple log/output area showing pipeline stdout/stderr
- Run/Cancel buttons that launch `barpath_cli.py` in a background thread

Video preview & rich timeline controls are stubbed for now with a
placeholder image view and slider; these can be expanded later.
"""

from __future__ import annotations

import os
import sys
import threading
import subprocess
from pathlib import Path
from typing import List, Optional

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW


class BarpathTogaApp(toga.App):
	def startup(self) -> None:  # type: ignore[override]
		"""Construct the main window and widgets."""

		# --- State ---
		self.model_dir: Optional[Path] = None
		self.model_files: List[Path] = []
		self.selected_model: Optional[Path] = None
		self.input_video: Optional[Path] = None
		self.output_video: Path = Path("output.mp4")
		self.graphs_dir: Path = Path("graphs")
		self.lift_type: str = "none"
		self.class_name: str = "endcap"
		self.encode_video: bool = False
		self.technique_analysis: bool = True

		# Subprocess tracking
		self._proc: Optional[subprocess.Popen[str]] = None
		self._reader_thread: Optional[threading.Thread] = None
		self._stop_event = threading.Event()

		# --- Main window ---
		self.main_window = toga.MainWindow(title="Barpath - Weightlifting Analysis Tool")

		# Root layout: left (config+log) and right (video preview), bottom actions
		root_box = toga.Box(style=Pack(direction=COLUMN, margin=10))

		top_box = toga.Box(style=Pack(direction=ROW, margin_bottom=10))
		root_box.add(top_box)

		# Left: configuration + log
		left_box = toga.Box(style=Pack(direction=COLUMN, flex=1, margin_right=10))
		top_box.add(left_box)

		# Right: video preview (stub for now)
		right_box = toga.Box(style=Pack(direction=COLUMN, flex=1))
		top_box.add(right_box)

		# --- Configuration panel ---
		config_label = toga.Label("Configuration", style=Pack(font_weight="bold", margin_bottom=6))
		left_box.add(config_label)

		config_box = toga.Box(style=Pack(direction=COLUMN, margin=6, margin_bottom=10))
		left_box.add(config_box)

		# Row: YOLO models directory
		models_row = toga.Box(style=Pack(direction=ROW, margin_bottom=6, align_items="center"))
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
		video_row = toga.Box(style=Pack(direction=ROW, margin_bottom=6, align_items="center"))
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
		model_row = toga.Box(style=Pack(direction=ROW, margin_bottom=6, align_items="center"))
		model_row.add(toga.Label("Select Model:", style=Pack(width=170)))
		self.model_select = toga.Selection(items=["(Select directory first)"], style=Pack(flex=1))
		self.model_select.enabled = False
		model_row.add(self.model_select)
		config_box.add(model_row)

		# Row: Lift type dropdown
		lift_row = toga.Box(style=Pack(direction=ROW, margin_bottom=6, align_items="center"))
		lift_row.add(toga.Label("Lift Type:", style=Pack(width=170)))
		self.lift_select = toga.Selection(items=["none", "clean"], style=Pack(width=160))
		self.lift_select.value = "none"
		lift_row.add(self.lift_select)
		config_box.add(lift_row)

		# Row: Encode video toggle + optional output name
		encode_row = toga.Box(style=Pack(direction=ROW, margin_bottom=6, align_items="center"))
		self.encode_video_switch = toga.Switch("Generate Output Video", value=False, on_change=self.on_encode_toggle)
		encode_row.add(self.encode_video_switch)
		self.output_video_input = toga.TextInput(
			value=str(self.output_video),
			placeholder="output filename (e.g. output.mp4)",
			style=Pack(flex=1, margin_left=12),
		)
		encode_row.add(self.output_video_input)
		config_box.add(encode_row)

		# Hide output name box until encoding enabled
		self.output_video_input.visible = bool(self.encode_video_switch.value)
		self.output_video_input.enabled = bool(self.encode_video_switch.value)

		# Row: Technique analysis checkbox & class name
		extra_row = toga.Box(style=Pack(direction=ROW, margin_bottom=6, align_items="center"))
		self.technique_switch = toga.Switch("Technique Analysis", value=True)
		extra_row.add(self.technique_switch)

		extra_row.add(toga.Label("Class Name:", style=Pack(margin_left=12)))
		self.class_name_input = toga.TextInput(value="endcap", style=Pack(width=140))
		extra_row.add(self.class_name_input)
		config_box.add(extra_row)


		# Row: Graphs dir
		graphs_row = toga.Box(style=Pack(direction=ROW, margin_bottom=6, align_items="center"))
		graphs_row.add(toga.Label("Graphs Directory:", style=Pack(width=170)))
		self.graphs_input = toga.TextInput(value=str(self.graphs_dir), style=Pack(flex=1))
		graphs_row.add(self.graphs_input)
		config_box.add(graphs_row)

		# --- Log/output panel ---
		log_label = toga.Label("Analysis Log", style=Pack(font_weight="bold", margin_top=6, margin_bottom=4))
		left_box.add(log_label)

		self.log_output = toga.MultilineTextInput(readonly=True, style=Pack(flex=1, margin_top=4))
		left_box.add(self.log_output)

		# --- Video preview panel (stub) ---
		preview_label = toga.Label("Video Preview", style=Pack(font_weight="bold", margin_bottom=6))
		right_box.add(preview_label)

		self.preview_placeholder = toga.Label("No video loaded", style=Pack(margin=10))
		right_box.add(self.preview_placeholder)

		# Simple scrub slider stub
		timeline_label = toga.Label("Video Timeline", style=Pack(margin_top=10, margin_bottom=4))
		right_box.add(timeline_label)
		self.timeline_slider = toga.Slider(min=0, max=100, value=0, style=Pack(flex=1))
		right_box.add(self.timeline_slider)

		# Time display stub
		times_row = toga.Box(style=Pack(direction=ROW, margin_top=4))
		self.start_time_label = toga.Label("Start: 00:00", style=Pack(flex=1))
		self.current_time_label = toga.Label("Current: 00:00", style=Pack(flex=1))
		self.end_time_label = toga.Label("End: 00:00", style=Pack(flex=1))
		times_row.add(self.start_time_label)
		times_row.add(self.current_time_label)
		times_row.add(self.end_time_label)
		right_box.add(times_row)

		# --- Bottom action buttons ---
		button_row = toga.Box(style=Pack(direction=ROW, margin_top=10))
		self.run_button = toga.Button("Run Analysis", on_press=self.on_run_analysis, style=Pack(margin_right=6))
		self.cancel_button = toga.Button("Cancel", on_press=self.on_cancel_analysis, enabled=False)
		button_row.add(self.run_button)
		button_row.add(self.cancel_button)
		root_box.add(button_row)

		self.main_window.content = root_box
		self.main_window.size = (1200, 720)
		self.main_window.show()

		# Attempt to auto-populate default model directory similar to Tk GUI
		self._populate_default_model_dir()

	# ------------------------------------------------------------------
	# Helpers & callbacks
	# ------------------------------------------------------------------

	def append_log(self, text: str) -> None:
		"""Append a line of text to the log output."""

		existing = self.log_output.value or ""
		new_text = f"{existing}{text}\n" if existing else f"{text}\n"
		self.log_output.value = new_text
		# MultilineTextInput doesn't have scroll-to-end, but this keeps content.

	def _populate_default_model_dir(self) -> None:
		"""Try to find and set a default model directory (like Tk GUI)."""

		candidates = [
			Path(__file__).resolve().parent.parent.parent / "models",  # repo_root/models
			Path(__file__).resolve().parent.parent / "models",  # sibling models
			Path(__file__).resolve().parent / "models",  # local models
		]

		chosen = None
		for cand in candidates:
			if cand.exists() and cand.is_dir():
				chosen = cand
				break

		if chosen is not None:
			self.model_dir = chosen
			self.model_dir_input.value = str(chosen)
			self._populate_model_files(chosen)
			self.append_log(f"✓ Found models directory: {chosen}")
		else:
			self.model_dir = None
			self.model_dir_input.value = ""
			self.model_select.items = ["(Select directory first)"]
			self.model_select.enabled = False
			self.append_log("⚠ No models directory found. Please select one.")

	def _populate_model_files(self, directory: Path) -> None:
		"""Populate model dropdown with .pt files in directory."""

		pt_files = sorted(directory.glob("*.pt"))
		self.model_files = pt_files

		if pt_files:
			items = [p.name for p in pt_files]
			self.model_select.items = items
			self.model_select.enabled = True
			self.model_select.value = items[0]
			self.selected_model = pt_files[0]
			self.append_log(f"✓ Found {len(pt_files)} model file(s)")
		else:
			self.model_select.items = ["(No .pt files found)"]
			self.model_select.enabled = False
			self.selected_model = None
			self.append_log("⚠ No .pt model files found in directory")

	# --- Event handlers ------------------------------------------------

	def on_encode_toggle(self, widget: toga.Widget) -> None:  # pragma: no cover - UI callback
		show_box = bool(self.encode_video_switch.value)
		self.output_video_input.visible = show_box
		self.output_video_input.enabled = show_box

	async def on_browse_models_dir(self, widget: toga.Widget) -> None:  # pragma: no cover - UI callback
		dialog = toga.SelectFolderDialog("Select YOLO Models Directory")
		selection = await self.main_window.dialog(dialog)
		if selection:
			path = Path(selection)
			self.model_dir = path
			self.model_dir_input.value = str(path)
			self._populate_model_files(path)

	async def on_browse_video(self, widget: toga.Widget) -> None:  # pragma: no cover - UI callback
		try:
			dialog = toga.OpenFileDialog(
				title="Select Input Video",
				file_types=["mp4", "avi", "mov", "mkv", "webm"],
			)
			selection = await self.main_window.dialog(dialog)
		except Exception as exc:  # pragma: no cover - dialog failure
			self.append_log(f"❌ Unable to open file picker: {exc}")
			return

		if selection:
			path = Path(selection)
			self.input_video = path
			self.video_input.value = str(path)
			self.preview_placeholder.text = f"Loaded: {path.name}"

	def _resolve_selected_model(self) -> Optional[Path]:
		if not self.model_files or not self.model_select.enabled:
			return None
		name = self.model_select.value
		if not name:
			return None
		for p in self.model_files:
			if p.name == name:
				return p
		return None

	def on_run_analysis(self, widget: toga.Widget) -> None:  # pragma: no cover - UI callback
		"""Validate inputs and launch barpath_cli.py as a subprocess."""

		self._stop_event.clear()

		self.lift_type = self.lift_select.value or "none"
		self.encode_video = bool(self.encode_video_switch.value)
		self.technique_analysis = bool(self.technique_switch.is_on)
		self.class_name = self.class_name_input.value or "endcap"
		self.output_video = Path(self.output_video_input.value or "output.mp4")
		self.graphs_dir = Path(self.graphs_input.value or "graphs")

		# Validate input video
		if not self.input_video or not self.input_video.exists():
			self.append_log("❌ Please select a valid input video file.")
			return

		# Validate model
		selected_model = self._resolve_selected_model()
		if not selected_model or not selected_model.exists():
			self.append_log("❌ Please select a valid YOLO model file.")
			return

		# Build CLI command similar to Tk GUI & barpath_cli
		repo_root = Path(__file__).resolve().parent.parent.parent
		cli_script = repo_root / "barpath" / "cli" / "barpath_cli.py"

		if not cli_script.exists():
			self.append_log(f"❌ CLI script not found at: {cli_script}")
			return

		cmd = [
			sys.executable,
			str(cli_script),
			"--input_video",
			str(self.input_video),
			"--output_video",
			str(self.output_video),
			"--graphs_dir",
			str(self.graphs_dir),
			"--model",
			str(selected_model),
			"--class_name",
			self.class_name,
		]

		# Encode video / no-video flag
		if self.encode_video:
			cmd.append("--encode_video")
		else:
			cmd.append("--no-video")

		# Technique analysis / lift type (if disabled, send none)
		lift_type = self.lift_type if self.technique_analysis else "none"
		cmd.extend(["--lift_type", lift_type])

		self.log_output.value = ""
		self.append_log("=" * 60)
		self.append_log("Starting Barpath Analysis Pipeline")
		self.append_log("=" * 60)
		self.append_log(f"Video: {self.input_video.name}")
		self.append_log(f"Model: {selected_model.name}")
		self.append_log(f"Lift Type: {lift_type}")
		self.append_log(f"Technique Analysis: {'Yes' if self.technique_analysis else 'No'}")
		self.append_log("=" * 60)

		env = dict(os.environ)
		env["PYTHONUNBUFFERED"] = "1"
		env.setdefault("TERM", "dumb")

		def run_pipeline() -> None:
			try:
				self._proc = subprocess.Popen(
					cmd,
					stdout=subprocess.PIPE,
					stderr=subprocess.STDOUT,
					text=True,
					bufsize=1,
					env=env,
				)
			except Exception as e:  # pragma: no cover - subprocess failure
				self.app.loop.call_soon(self.append_log, f"❌ Failed to start pipeline: {e}")
				self._on_process_finished(1)
				return

			try:
				assert self._proc is not None
				assert self._proc.stdout is not None
				for line in self._proc.stdout:
					if self._stop_event.is_set():
						break
					if not line:
						continue
					# Strip trailing newline and send to UI thread
					text = line.rstrip("\n")
					self.app.loop.call_soon(self.append_log, text)
			except Exception as e:  # pragma: no cover - stream failure
				self.app.loop.call_soon(self.append_log, f"❌ Error reading output: {e}")

			rc = self._proc.wait() if self._proc is not None else 1
			self._on_process_finished(rc)

		# Disable/enable buttons
		self.run_button.enabled = False
		self.cancel_button.enabled = True

		self._reader_thread = threading.Thread(target=run_pipeline, daemon=True)
		self._reader_thread.start()

	def on_cancel_analysis(self, widget: toga.Widget) -> None:  # pragma: no cover - UI callback
		self.append_log("\n⚠ Cancelling analysis...")
		self._stop_event.set()
		if self._proc is not None and self._proc.poll() is None:
			try:
				self._proc.terminate()
			except Exception:
				try:
					self._proc.kill()
				except Exception:
					pass

	def _on_process_finished(self, returncode: int) -> None:
		def ui_update() -> None:
			if returncode == 0:
				self.append_log("\n" + "=" * 60)
				self.append_log("✓ Pipeline completed successfully!")
				self.append_log("=" * 60)
			else:
				self.append_log("\n" + "=" * 60)
				self.append_log(f"❌ Pipeline failed with exit code {returncode}")
				self.append_log("=" * 60)

			self.run_button.enabled = True
			self.cancel_button.enabled = False

		# Schedule UI update in main loop
		self.app.loop.call_soon(ui_update)


def main() -> None:
	return BarpathTogaApp("Barpath", "org.barpath.gui").main_loop()


if __name__ == "__main__":
	main()

