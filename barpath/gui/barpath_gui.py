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

import asyncio
import os
import queue
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, cast

import toga
from toga.style import Pack


def strip_ansi(text: str) -> str:
	"""Strip ANSI escape codes from text."""
	ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
	return ansi_escape.sub('', text)


class BarpathTogaApp(toga.App):
	STEP_LABELS = {
		"step1": "Step 1: Collecting raw data",
		"step2": "Step 2: Analyzing data",
		"step3": "Step 3: Generating graphs",
		"step4": "Step 4: Rendering video",
		"step5": "Step 5: Critiquing lift",
	}

	STEP_START_MARKERS = {
		"step1": ">>> step 1:",
		"step2": ">>> step 2:",
		"step3": ">>> step 3:",
		"step4": ">>> step 4:",
		"step5": ">>> step 5:",
	}

	STEP_COMPLETE_MARKERS = {
		"step1": ">>> step 1 complete",
		"step2": ">>> step 2 complete",
		"step3": ">>> step 3 complete",
		"step4": ">>> step 4 complete",
		"step5": ">>> step 5 complete",
	}

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
		self.encode_video: bool = True
		self.technique_analysis: bool = True
		self.progress_queue: queue.Queue[dict[str, object]] = queue.Queue()
		self._planned_steps: List[str] = []
		self._completed_steps: Set[str] = set()
		self._progress_total: int = 0
		self._progress_task: Optional[asyncio.Task[None]] = None

		# Subprocess tracking
		self._proc: Optional[subprocess.Popen[str]] = None
		self._reader_thread: Optional[threading.Thread] = None
		self._stop_event = threading.Event()

		# --- Main window ---
		self.main_window = toga.MainWindow(title="Barpath - Weightlifting Analysis Tool")

		# Root layout: left (config+log) and right (video preview), bottom actions
		root_box = toga.Box(style=Pack(direction="column", margin=10))

		top_box = toga.Box(style=Pack(direction="row", flex=1, margin_bottom=10))
		root_box.add(top_box)

		# Left: configuration + log
		left_box = toga.Box(style=Pack(direction="column", flex=1, margin_right=10))
		top_box.add(left_box)

		# (Video preview/side column removed) — left column will fill window

		# --- Configuration panel ---
		config_label = toga.Label("Configuration", style=Pack(font_weight="bold", margin_bottom=6))
		left_box.add(config_label)

		config_box = toga.Box(style=Pack(direction="column", margin=6, margin_bottom=10))
		left_box.add(config_box)

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
		self.lift_select = toga.Selection(items=["none", "clean"], style=Pack(width=160))
		self.lift_select.value = "none"
		lift_row.add(self.lift_select)
		config_box.add(lift_row)

		# Row: Encode video toggle + optional output name
		encode_row = toga.Box(style=Pack(direction="row", margin_bottom=6, align_items="center"))
		self.encode_video_switch = toga.Switch("Generate Output Video", value=True, on_change=self.on_encode_toggle)
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
		extra_row = toga.Box(style=Pack(direction="row", margin_bottom=6, align_items="center"))
		self.technique_switch = toga.Switch("Technique Analysis", value=True)
		extra_row.add(self.technique_switch)

		extra_row.add(toga.Label("Class Name:", style=Pack(margin_left=12)))
		self.class_name_input = toga.TextInput(value="endcap", style=Pack(width=140))
		extra_row.add(self.class_name_input)
		config_box.add(extra_row)


		# Row: Graphs dir
		graphs_row = toga.Box(style=Pack(direction="row", margin_bottom=6, align_items="center"))
		graphs_row.add(toga.Label("Graphs Directory:", style=Pack(width=170)))
		self.graphs_input = toga.TextInput(value=str(self.graphs_dir), style=Pack(flex=1))
		graphs_row.add(self.graphs_input)
		config_box.add(graphs_row)

		# --- Progress panel ---
		progress_container = toga.Box(style=Pack(direction="column", margin_top=6, margin_bottom=6))
		progress_header = toga.Label(
			"Pipeline Progress",
			style=Pack(font_weight="bold", margin_bottom=4),
		)
		progress_container.add(progress_header)
		self.progress_status = toga.Label("Idle", style=Pack(margin_bottom=4))
		progress_container.add(self.progress_status)
		self.progress_bar = toga.ProgressBar(max=1.0, value=0.0, style=Pack(height=16))
		progress_container.add(self.progress_bar)
		left_box.add(progress_container)

		# --- Log/output panel ---
		log_label = toga.Label("Analysis Log", style=Pack(font_weight="bold", margin_top=6, margin_bottom=4))
		left_box.add(log_label)

		self.log_output = toga.MultilineTextInput(readonly=True, style=Pack(flex=1, margin_top=4))
		self.log_scroll = toga.ScrollContainer(content=self.log_output, style=Pack(flex=1))
		left_box.add(self.log_scroll)

		# --- Bottom action buttons ---
		button_row = toga.Box(style=Pack(direction="row", margin_top=10))
		self.run_button = toga.Button("Run Analysis", on_press=self.on_run_analysis, style=Pack(margin_right=6))
		self.cancel_button = toga.Button("Cancel", on_press=self.on_cancel_analysis, enabled=False)
		button_row.add(self.run_button)
		button_row.add(self.cancel_button)
		root_box.add(button_row)

		window = cast(toga.MainWindow, self.main_window)
		window.content = root_box
		window.size = (800, 720)
		window.show()

		# Attempt to auto-populate default model directory similar to Tk GUI
		self._populate_default_model_dir()

	# ------------------------------------------------------------------
	# Helpers & callbacks
	# ------------------------------------------------------------------

	def append_log(self, text: str) -> None:
		"""Append a line of text to the log output."""

		text = strip_ansi(text)
		existing = self.log_output.value or ""
		new_text = f"{existing}{text}\n" if existing else f"{text}\n"
		self.log_output.value = new_text

	def _schedule_ui(self, callback: Callable[..., None], *args: object) -> None:
		"""Invoke a callback on the Toga app loop when available."""

		loop = getattr(self.app, "loop", None)
		if loop is None:
			callback(*args)
		else:
			loop.call_soon(callback, *args)

	def _terminate_active_process(self, graceful_timeout: float = 2.0) -> None:
		"""Attempt to stop the running CLI subprocess and its children."""

		proc = self._proc
		if proc is None or proc.poll() is not None:
			return

		pgid: Optional[int] = None
		if os.name != "nt" and hasattr(os, "getpgid"):
			try:
				pgid = os.getpgid(proc.pid)
			except Exception:
				pgid = None

		def send(sig: int) -> None:
			try:
				if pgid is not None and hasattr(os, "killpg"):
					os.killpg(pgid, sig)
				else:
					proc.send_signal(sig)
			except Exception:
				pass

		def close_stdout() -> None:
			try:
				if proc.stdout is not None:
					proc.stdout.close()
			except Exception:
				pass

		send(signal.SIGINT)
		try:
			proc.wait(timeout=graceful_timeout)
			close_stdout()
			return
		except subprocess.TimeoutExpired:
			pass

		term_sig = getattr(signal, "SIGTERM", signal.SIGINT)
		send(term_sig)
		try:
			proc.wait(timeout=1)
			close_stdout()
			return
		except subprocess.TimeoutExpired:
			pass

		if os.name != "nt" and hasattr(signal, "SIGKILL"):
			send(signal.SIGKILL)
		else:
			try:
				proc.kill()
			except Exception:
				pass

		try:
			proc.wait(timeout=1)
		except Exception:
			pass

		close_stdout()

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

	def _clear_pending_progress_updates(self) -> None:
		"""Drain any queued progress updates to avoid stale UI changes."""

		try:
			while True:
				self.progress_queue.get_nowait()
		except queue.Empty:
			return

	def _prepare_progress_tracking(self, effective_lift_type: str) -> None:
		"""Compute expected steps and reset progress bar state."""

		self._clear_pending_progress_updates()
		self._completed_steps.clear()
		self._planned_steps = ["step1", "step2", "step3"]
		if self.encode_video:
			self._planned_steps.append("step4")
		if effective_lift_type != "none":
			self._planned_steps.append("step5")
		self._progress_total = len(self._planned_steps)
		status = "Pipeline queued..." if self._progress_total else "Idle"
		self._send_progress_update(value=0.0, text=status)

	def _progress_fraction(self) -> float:
		if not self._progress_total:
			return 0.0
		return len(self._completed_steps) / self._progress_total

	def _send_progress_update(self, *, value: Optional[float] = None, text: Optional[str] = None) -> None:
		payload: dict[str, object] = {}
		if value is not None:
			payload["value"] = max(0.0, min(1.0, value))
		if text is not None:
			payload["text"] = text
		if payload:
			self.progress_queue.put(payload)

	def _announce_step_start(self, step_id: str) -> None:
		if not self._planned_steps or step_id not in self._planned_steps:
			return
		label = self.STEP_LABELS.get(step_id)
		if label:
			self._send_progress_update(text=label)

	def _mark_step_complete(self, step_id: str) -> None:
		if not self._planned_steps or step_id not in self._planned_steps:
			return
		if step_id in self._completed_steps:
			return
		self._completed_steps.add(step_id)
		label = self.STEP_LABELS.get(step_id)
		fraction = self._progress_fraction()
		if label:
			self._send_progress_update(value=fraction, text=f"{label} ✓")
		else:
			self._send_progress_update(value=fraction)

	def _handle_progress_line(self, text: str) -> None:
		if not self._planned_steps:
			return
		lower = text.lower()
		for step_id, marker in self.STEP_START_MARKERS.items():
			if marker in lower:
				self._announce_step_start(step_id)
				if "skipping" in lower:
					self._mark_step_complete(step_id)
				return
		for step_id, marker in self.STEP_COMPLETE_MARKERS.items():
			if marker in lower:
				self._mark_step_complete(step_id)
				return

	def _finalize_progress(self, success: bool, cancelled: bool = False) -> None:
		if not self._planned_steps:
			return
		if success:
			self._completed_steps = set(self._planned_steps)
			self._send_progress_update(value=1.0, text="Pipeline complete ✓")
		elif cancelled:
			self._send_progress_update(text="Pipeline cancelled.")
		else:
			self._send_progress_update(text="Pipeline failed - see log.")
		self._planned_steps = []
		self._completed_steps.clear()
		self._progress_total = 0

	async def _drain_progress_queue(self) -> None:  # pragma: no cover - UI loop helper
		while True:
			try:
				update = self.progress_queue.get_nowait()
			except queue.Empty:
				await asyncio.sleep(0.1)
				continue
			value = update.get("value") if isinstance(update, dict) else None
			text = update.get("text") if isinstance(update, dict) else None
			if isinstance(value, (int, float)) and self.progress_bar is not None:
				self.progress_bar.value = float(value)
			if text is not None and self.progress_status is not None:
				self.progress_status.text = str(text)

	async def on_running(self) -> None:  # pragma: no cover - lifecycle hook
		if self._progress_task is None or self._progress_task.done():
			self._progress_task = asyncio.create_task(self._drain_progress_queue())

	def on_exit(self) -> bool:  # pragma: no cover - lifecycle hook
		task = self._progress_task
		if task is not None and not task.done():
			task.cancel()
		self._progress_task = None
		return True

	# --- Event handlers ------------------------------------------------

	def on_encode_toggle(self, widget: toga.Widget) -> None:  # pragma: no cover - UI callback
		show_box = bool(self.encode_video_switch.value)
		self.output_video_input.visible = show_box
		self.output_video_input.enabled = show_box

	async def on_browse_models_dir(self, widget: toga.Widget) -> None:  # pragma: no cover - UI callback
		dialog = toga.SelectFolderDialog("Select YOLO Models Directory")
		window = cast(toga.MainWindow, self.main_window)
		selection = await window.dialog(dialog)
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
			window = cast(toga.MainWindow, self.main_window)
			selection = await window.dialog(dialog)
		except Exception as exc:  # pragma: no cover - dialog failure
			self.append_log(f"❌ Unable to open file picker: {exc}")
			return

		if selection:
			path = Path(selection)
			self.input_video = path
			self.video_input.value = str(path)

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

		selected_lift = cast(Optional[str], self.lift_select.value)
		self.lift_type = selected_lift or "none"
		self.encode_video = bool(self.encode_video_switch.value)
		self.technique_analysis = bool(self.technique_switch.value)
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

		lift_type = self.lift_type if self.technique_analysis else "none"
		self._prepare_progress_tracking(lift_type)

		cmd = [
			sys.executable,
			str(cli_script),
			"--input_video",
			str(self.input_video),
			"--graphs_dir",
			str(self.graphs_dir),
			"--model",
			str(selected_model),
			"--class_name",
			self.class_name,
		]

		# Encode video / no-video flag
		if self.encode_video:
			cmd.extend(["--output_video", str(self.output_video)])
		else:
			cmd.append("--no-video")

		# Technique analysis / lift type (if disabled, send none)
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
		self._send_progress_update(text="Launching pipeline...")

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
					start_new_session=True,
					env=env,
				)
			except Exception as e:  # pragma: no cover - subprocess failure
				self._schedule_ui(self.append_log, f"❌ Failed to start pipeline: {e}")
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
					self._handle_progress_line(text)
					self._schedule_ui(self.append_log, text)
			except Exception as e:  # pragma: no cover - stream failure
				self._schedule_ui(self.append_log, f"❌ Error reading output: {e}")

			rc = self._proc.wait() if self._proc is not None else 1
			self._on_process_finished(rc)

		# Disable/enable buttons
		self.run_button.enabled = False
		self.cancel_button.enabled = True

		self._reader_thread = threading.Thread(target=run_pipeline, daemon=True)
		self._reader_thread.start()
		self._send_progress_update(text="Pipeline running...")

	def on_cancel_analysis(self, widget: toga.Widget) -> None:  # pragma: no cover - UI callback
		self.append_log("\n⚠ Cancelling analysis...")
		self._stop_event.set()
		self._send_progress_update(text="Cancelling pipeline...")
		self._terminate_active_process()

	def _on_process_finished(self, returncode: int) -> None:
		def ui_update() -> None:
			cancelled = self._stop_event.is_set()
			if returncode == 0 and not cancelled:
				self.append_log("\n" + "=" * 60)
				self.append_log("✓ Pipeline completed successfully!")
				self.append_log("=" * 60)
				self._finalize_progress(success=True)
			elif cancelled:
				self.append_log("\n" + "=" * 60)
				self.append_log("⚠ Pipeline cancelled by user.")
				self.append_log("=" * 60)
				self._finalize_progress(success=False, cancelled=True)
			else:
				self.append_log("\n" + "=" * 60)
				self.append_log(f"❌ Pipeline failed with exit code {returncode}")
				self.append_log("=" * 60)
				self._finalize_progress(success=False)

			self.run_button.enabled = True
			self.cancel_button.enabled = False
			self._stop_event.clear()

		# Schedule UI update in main loop
		self._schedule_ui(ui_update)


def main() -> None:
	# Prefer local project asset for about/app icon (project-level overrides default toga icon)
	repo_assets = Path(__file__).resolve().parent / "assets"
	icon_candidate = repo_assets / "barpath.png"
	if not icon_candidate.exists():
		icon_candidate = repo_assets / "barpath.svg"

	if icon_candidate.exists():
		return BarpathTogaApp("Barpath", "org.barpath.gui", icon=str(icon_candidate)).main_loop()
	else:
		return BarpathTogaApp("Barpath", "org.barpath.gui").main_loop()


if __name__ == "__main__":
	main()

