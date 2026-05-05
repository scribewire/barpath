#!/usr/bin/env python3
"""Toga-based GUI frontend for the barpath pipeline.

This GUI is organized with a left sidebar that acts as navigation for 3 sections:
- Files: manage input videos and choose an output directory (via system picker)
- Settings: configure model + lift type using single-select horizontal button groups
- Analyze: run/cancel analysis, show progress, and display richer logs as HTML (WebView)

Implementation notes:
- Toga/Travertino Pack layout: `padding*` is deprecated; this GUI uses `margin*` instead.
- Some backends don't support `display` reliably; to reduce flashing, pages remain mounted and
  we toggle visibility + height. To reset a style value, use `del widget.style.<prop>` (do not
  assign `None`).
- Log rendering: we convert a small subset of Rich markup (e.g. [bold], [cyan], [dim]) to HTML
  spans and render the log in a WebView so it looks more like an app panel than a terminal.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import toga
from gui_helpers.log_renderer import LogRenderer
from gui_helpers.markdown_renderer import MarkdownRenderer
from toga.style import Pack

# Prepare for lazy import of the pipeline runner
sys.path.insert(0, str(Path(__file__).parent.parent))
_run_pipeline = None
_run_batch_postprocess = None
_run_pipeline_from_folder = None


def _get_run_pipeline():
    """Lazy-load barpath_core.run_pipeline so the GUI starts faster."""
    global _run_pipeline
    if _run_pipeline is None:
        from barpath_core import run_pipeline  # Local import keeps startup lightweight

        _run_pipeline = run_pipeline
    return _run_pipeline


def _get_run_batch_postprocess():
    """Lazy-load barpath_core.run_batch_postprocess so the GUI starts faster."""
    global _run_batch_postprocess
    if _run_batch_postprocess is None:
        from barpath_core import run_batch_postprocess

        _run_batch_postprocess = run_batch_postprocess
    return _run_batch_postprocess


def _get_run_pipeline_from_folder():
    """Lazy-load barpath_core.run_pipeline_from_folder so the GUI starts faster."""
    global _run_pipeline_from_folder
    if _run_pipeline_from_folder is None:
        from barpath_core import run_pipeline_from_folder

        _run_pipeline_from_folder = run_pipeline_from_folder
    return _run_pipeline_from_folder


class BarpathTogaApp(toga.App):
    """Main application class for the Barpath GUI."""

    # ----------------------------
    # App lifecycle
    # ----------------------------

    def startup(self) -> None:  # type: ignore[override]
        # --- State ---
        self.model_dir: Path | None = None
        self.model_files: list[Path] = []
        self.selected_model: Path | None = None

        self.input_videos: list[Path] = []
        self.input_folders: list[Path] = []
        # "videos", "folders", or "" (nothing added yet)
        self.input_mode: str = ""
        self.output_dir: Path = Path("outputs")
        self.lift_type: str = "none"

        self.encode_video: bool = True
        self.technique_analysis: bool = True
        self.use_filenames_in_legend: bool = False

        self.lifter: str = "generic"

        # Skip/rerun decisions for batch processing
        self._skip_all_existing: bool = False
        self._rerun_all_existing: bool = False
        self._force_rerun: bool = False
        self._skip_existing: bool = False

        self._is_running: bool = False
        self._pipeline_task: asyncio.Task[Any] | None = None
        self._cancel_event = threading.Event()

        # Thread-safe queue used to ferry progress messages from the
        # background pipeline thread to the main (Toga/asyncio) thread.
        # Items are (step_name, progress_value, message) tuples, or the
        # sentinel string "_DONE_" / "_ERROR_:<msg>" to signal completion.
        self._progress_queue: queue.Queue[Any] = queue.Queue()

        # Executor that runs the blocking pipeline on a real OS thread so
        # the Toga event loop is never stalled by CPU / I/O work.
        self._thread_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="barpath-pipeline"
        )

        # Preview state: webcam live preview with YOLO + MediaPipe overlay
        self._preview_running: bool = False
        self._preview_thread: threading.Thread | None = None
        self._preview_stop_event = threading.Event()

        # Supported video extensions for OpenFileDialog (Toga expects list of extensions)
        self.video_extensions = [
            "mp4",
            "avi",
            "mov",
            "mkv",
            "webm",
            "MP4",
            "MOV",
            "MKV",
            "WEBM",
        ]

        # --- Main window ---
        self.main_window = toga.MainWindow(
            title="Barpath - Weightlifting Analysis Tool",
            size=(840, 600),
        )

        # Root: sidebar + content area
        root = toga.Box(style=Pack(direction="row", margin=10))

        # --- Sidebar (left): tab buttons + short tips ---
        self.sidebar = toga.Box(
            style=Pack(direction="column", width=220, margin_right=12)
        )

        # Wrap "BARPATH" label in a centered container with flexible spacers
        barpath_label = toga.Label(
            "BARPATH",
            style=Pack(font_weight="bold", font_size=24),
        )
        barpath_container = toga.Box(
            style=Pack(direction="row", margin_bottom=10, margin_top=10)
        )
        barpath_container.add(toga.Box(style=Pack(flex=1)))
        barpath_container.add(barpath_label)
        barpath_container.add(toga.Box(style=Pack(flex=1)))
        self.sidebar.add(barpath_container)

        def _tab_row(
            title: str, tip: str, tab_key: str
        ) -> tuple[toga.Button, toga.Label, toga.Box]:
            # More space-efficient: stack the tip under the button so it can wrap.
            btn = toga.Button(
                title,
                on_press=lambda w, k=tab_key: self._select_tab(k),
                style=Pack(flex=1, margin=(6, 10)),
            )
            tip_label = toga.Label(
                tip,
                style=Pack(
                    margin_top=4,
                    font_size=9,
                    color="#5B6472",
                ),
            )
            row = toga.Box(
                style=Pack(
                    direction="column",
                    margin_bottom=8,
                    margin=8,
                    background_color="#F2F3F7",
                )
            )
            row.add(btn)
            row.add(tip_label)
            return btn, tip_label, row

        self.tab_btn_files, self.tab_tip_files, files_row = _tab_row(
            "Files",
            "Add videos + choose output folder",
            "files",
        )
        self.tab_btn_settings, self.tab_tip_settings, settings_row = _tab_row(
            "Settings",
            "Pick model and lift type",
            "settings",
        )
        self.tab_btn_analyze, self.tab_tip_analyze, analyze_row = _tab_row(
            "Analyze",
            "Run pipeline and view logs",
            "analyze",
        )
        self.tab_btn_analysis, self.tab_tip_analysis, analysis_row = _tab_row(
            "Analysis",
            "View lift analysis results",
            "analysis",
        )

        self.sidebar.add(files_row)
        self.sidebar.add(settings_row)
        self.sidebar.add(analyze_row)
        self.sidebar.add(analysis_row)
        self.sidebar.add(toga.Box(style=Pack(flex=1)))

        # --- Content host (right) ---
        # Keep all pages mounted and toggle visibility to reduce repaints/flashing.
        self.page_host = toga.Box(
            style=Pack(
                direction="column", flex=1, margin=10, background_color="#FFFFFF"
            )
        )

        # Log renderer for Rich markup and HTML rendering
        self.log_renderer = LogRenderer()

        # Markdown renderer for analysis display
        self.markdown_renderer = MarkdownRenderer()

        # Build pages once (mounted for the lifetime of the app)
        self.page_files = self._build_files_page()
        self.page_settings = self._build_settings_page()
        self.page_analyze = self._build_analyze_page()
        self.page_analysis = self._build_analysis_page()

        # Track which tab is active (for debouncing redundant updates)
        self._current_tab: str = ""  # Start empty so _select_tab doesn't get debounced

        root.add(self.sidebar)
        root.add(self.page_host)

        self.main_window.content = root  # type: ignore

        # Start on Files tab BEFORE showing window (prevents initial redraw flash)
        self._select_tab("files")

        self.main_window.show()  # type: ignore

        # Populate model dir if available
        self._populate_default_model_dir()
        self._refresh_settings_buttons()

        # Seed output directory label/value
        self._set_output_dir_value(self.output_dir)

        # Initial log banner
        self._log_banner()

    # ----------------------------
    # Page builders
    # ----------------------------

    def _build_files_page(self) -> toga.Box:
        page = toga.Box(style=Pack(direction="column", flex=1))

        # Wrap content in ScrollContainer to prevent window resizing
        content = toga.Box(style=Pack(direction="column"))

        header = toga.Label(
            "📂 Files", style=Pack(font_weight="bold", font_size=18, margin_bottom=8)
        )
        content.add(header)

        # Input videos / folders section
        self.files_section_title = toga.Label(
            "Input Videos", style=Pack(font_weight="bold", margin=(10, 0, 6, 0))
        )
        content.add(self.files_section_title)

        button_row = toga.Box(style=Pack(direction="row", margin_bottom=6))
        self.btn_add_videos = toga.Button(
            "Add Videos",
            on_press=self.on_browse_video,
            style=Pack(margin_right=6, flex=1),
        )
        self.btn_add_folders = toga.Button(
            "Add Folders (Reanalyze)",
            on_press=self.on_browse_folders,
            style=Pack(margin_right=6, flex=1),
        )
        self.btn_clear_videos = toga.Button(
            "Clear",
            on_press=self.on_clear_videos,
            enabled=False,
            style=Pack(flex=1),
        )
        button_row.add(self.btn_add_videos)
        button_row.add(self.btn_add_folders)
        button_row.add(self.btn_clear_videos)
        content.add(button_row)

        # Mode hint label shown when a mode is locked in
        self.files_mode_hint = toga.Label(
            "",
            style=Pack(font_size=9, color="#5B6472", margin_bottom=4),
        )
        content.add(self.files_mode_hint)

        self.video_list_container = toga.ScrollContainer(
            horizontal=True,
            vertical=True,
            style=Pack(flex=1, height=220, margin=6),
        )
        self.video_list_box = toga.Box(style=Pack(direction="column"))
        self.video_list_container.content = self.video_list_box
        content.add(self.video_list_container)

        # Output directory section
        out_title = toga.Label(
            "Output Directory",
            style=Pack(font_weight="bold", margin=(14, 0, 6, 0)),
        )
        content.add(out_title)

        out_row = toga.Box(
            style=Pack(direction="row", align_items="center", margin_bottom=6)
        )

        # Show selected directory as read-only label (instead of text input).
        self.output_dir_label = toga.Label(
            "",
            style=Pack(flex=1, margin=(6, 8), background_color="#F2F3F7"),
        )

        self.btn_open_output_dir = toga.Button(
            "Open",
            on_press=self.on_open_output_dir,
            style=Pack(width=90, margin_left=6),
        )
        self.btn_select_output_dir = toga.Button(
            "Select",
            on_press=self.on_select_output_dir,
            style=Pack(width=90, margin_left=6),
        )

        out_row.add(self.output_dir_label)
        out_row.add(self.btn_select_output_dir)
        out_row.add(self.btn_open_output_dir)
        content.add(out_row)

        content.add(
            toga.Label(
                "Your analysis files (graphs, CSV, report, and optional video) will be saved here.",
                style=Pack(font_size=9, color="#5B6472", margin_top=6),
            )
        )

        scroll = toga.ScrollContainer(content=content, style=Pack(flex=1))
        page.add(scroll)

        return page

    def _build_settings_page(self) -> toga.Box:
        page = toga.Box(style=Pack(direction="column", flex=1))

        # Wrap content in ScrollContainer to prevent window resizing
        content = toga.Box(style=Pack(direction="column"))

        header = toga.Label(
            "🔧 Settings", style=Pack(font_weight="bold", font_size=18, margin_bottom=8)
        )
        content.add(header)

        config_title = toga.Label(
            "Configuration", style=Pack(font_weight="bold", margin=(10, 0, 6, 0))
        )
        content.add(config_title)

        # Model selector (dropdown)
        content.add(
            toga.Label(
                "Select Model", style=Pack(font_weight="bold", margin=(10, 0, 6, 0))
            )
        )

        self.model_dropdown = toga.Selection(
            items=[],
            on_change=self._on_model_dropdown_change,
            style=Pack(flex=1, margin_bottom=4),
        )
        content.add(self.model_dropdown)

        self.model_hint_label = toga.Label(
            "Models are loaded from barpath/models. Supports .pt, .onnx, .engine and OpenVINO directories.",
            style=Pack(font_size=9, color="#5B6472", margin_bottom=10),
        )
        content.add(self.model_hint_label)

        # Lift type selector (horizontal buttons)
        content.add(
            toga.Label(
                "Lift Type", style=Pack(font_weight="bold", margin=(10, 0, 6, 0))
            )
        )
        self.lift_button_row = toga.Box(style=Pack(direction="row", margin_bottom=6))
        content.add(self.lift_button_row)

        # Additional toggles could be added here later; kept minimal per request
        content.add(
            toga.Label(
                "Lift Type controls whether critique is generated (`none` disables technique critique).",
                style=Pack(font_size=9, color="#5B6472", margin_top=6),
            )
        )

        content.add(
            toga.Label(
                "Analysis Options",
                style=Pack(font_weight="bold", margin=(14, 0, 6, 0)),
            )
        )

        content.add(
            toga.Label(
                "Select Lifter (Baseline)",
                style=Pack(font_weight="bold", margin=(10, 0, 6, 0)),
            )
        )

        self.lifter_dropdown = toga.Selection(
            items=["generic"],
            on_change=self._on_lifter_dropdown_change,
            style=Pack(flex=1, margin_bottom=4),
        )
        content.add(self.lifter_dropdown)

        self.lifter_hint_label = toga.Label(
            "Lifter determines which pro baseline to compare against for Technique Analysis. "
            "Falls back to pooled report if lifter-specific baselines are not found.",
            style=Pack(font_size=9, color="#5B6472", margin_bottom=10),
        )
        content.add(self.lifter_hint_label)

        analysis_row = toga.Box(style=Pack(direction="column", margin_bottom=6))
        self.technique_analysis_switch = toga.Switch(
            "Technique Analysis",
            value=True,
            on_change=self._on_technique_analysis_change,
            style=Pack(margin_bottom=4),
        )
        analysis_row.add(self.technique_analysis_switch)
        content.add(analysis_row)

        content.add(
            toga.Label(
                "Technique Analysis detects faults using biomechanical rules and pro baselines.",
                style=Pack(font_size=9, color="#5B6472", margin_top=4, margin_bottom=6),
            )
        )

        # ------------------------------------------------------------------
        # Multi-video graph options
        # ------------------------------------------------------------------
        content.add(
            toga.Label(
                "Multi-Video Options",
                style=Pack(font_weight="bold", margin=(14, 0, 6, 0)),
            )
        )

        filenames_row = toga.Box(
            style=Pack(direction="row", align_items="center", margin_bottom=4)
        )
        self.use_filenames_switch = toga.Switch(
            "Use filenames in superimposed path legend",
            value=False,
            on_change=self._on_use_filenames_change,
            style=Pack(margin_right=8),
        )
        filenames_row.add(self.use_filenames_switch)
        content.add(filenames_row)

        content.add(
            toga.Label(
                "When unchecked (default), lifts are labelled 'Lift 1', 'Lift 2', etc. "
                "When checked, the video filename stem is used instead. "
                "Only affects the superimposed bar-path graph produced when multiple videos are analysed.",
                style=Pack(font_size=9, color="#5B6472", margin_top=4, margin_bottom=6),
            )
        )

        # ------------------------------------------------------------------
        # HUD overlay toggles
        # ------------------------------------------------------------------
        content.add(
            toga.Label(
                "HUD Overlay Options",
                style=Pack(font_weight="bold", margin=(14, 0, 6, 0)),
            )
        )

        hud_toggles = [
            ("show_skeleton", "Skeleton overlay", True),
            ("show_sparkline", "Velocity sparkline", True),
            ("show_power_zones", "Power zone band", True),
            ("show_error_markers", "Error markers", True),
        ]
        self.hud_switches: dict[str, toga.Switch] = {}
        for key, label, default in hud_toggles:
            row = toga.Box(style=Pack(direction="row", align_items="center", margin_bottom=4))
            sw = toga.Switch(label, value=default, style=Pack(margin_right=8))
            row.add(sw)
            self.hud_switches[key] = sw
            content.add(row)

        content.add(
            toga.Label(
                "Toggle individual HUD elements on/off for the output video overlay.",
                style=Pack(font_size=9, color="#5B6472", margin_top=4, margin_bottom=6),
            )
        )

        scroll = toga.ScrollContainer(content=content, style=Pack(flex=1))
        page.add(scroll)

        return page

    def _build_analyze_page(self) -> toga.Box:
        page = toga.Box(style=Pack(direction="column", flex=1))

        # Wrap content in ScrollContainer to prevent window resizing
        content = toga.Box(style=Pack(direction="column"))

        header = toga.Label(
            "📊 Analyze", style=Pack(font_weight="bold", font_size=18, margin_bottom=8)
        )
        content.add(header)

        # Run controls
        controls = toga.Box(
            style=Pack(direction="row", margin=(6, 0, 6, 0), align_items="center")
        )
        self.run_button = toga.Button(
            "Run Analysis", on_press=self.on_run_analysis, style=Pack(margin_right=6)
        )
        self.view_results_button = toga.Button(
            "View Results",
            on_press=self.on_open_output_dir,
            style=Pack(margin_right=6),
        )
        self.cancel_button = toga.Button(
            "Cancel",
            on_press=self.on_cancel_analysis,
            enabled=False,
            style=Pack(margin_right=6),
        )
        self.preview_button = toga.Button(
            "Preview (Alpha)",
            on_press=self.on_toggle_preview,
            enabled=True,
            style=Pack(margin_right=6),
        )

        controls.add(self.run_button)
        controls.add(self.view_results_button)
        controls.add(self.cancel_button)
        controls.add(self.preview_button)
        content.add(controls)

        # Progress
        content.add(
            toga.Label("Progress", style=Pack(font_weight="bold", margin=(10, 0, 6, 0)))
        )
        self.progress_bar = toga.ProgressBar(max=100, style=Pack(margin_bottom=6))
        self.progress_label = toga.Label("Ready", style=Pack(margin_bottom=10))
        content.add(self.progress_bar)
        content.add(self.progress_label)

        # Output log (HTML-rendered)
        content.add(
            toga.Label(
                "Output Log", style=Pack(font_weight="bold", margin=(10, 0, 6, 0))
            )
        )

        # Render logs as HTML so the output feels like an app panel, not a terminal.
        # We update the full HTML document as new lines arrive.
        self.log_webview = toga.WebView(style=Pack(flex=1, margin=8))
        content.add(self.log_webview)

        # Small helper row
        content.add(
            toga.Label(
                "Log is rendered as HTML (Rich-like markup is styled).",
                style=Pack(font_size=9, color="#5B6472", margin_top=6),
            )
        )

        scroll = toga.ScrollContainer(content=content, style=Pack(flex=1))
        page.add(scroll)

        # Initialize the log view with an empty document
        self._render_log_html()

        return page

    def _build_analysis_page(self) -> toga.Box:
        page = toga.Box(style=Pack(direction="column", flex=1))

        # Wrap content in ScrollContainer to prevent window resizing
        content = toga.Box(style=Pack(direction="column"))

        header = toga.Label(
            "📄 Analysis", style=Pack(font_weight="bold", font_size=18, margin_bottom=8)
        )
        content.add(header)

        # WebView for rendering the analysis markdown as HTML
        self.analysis_webview = toga.WebView(style=Pack(flex=1, margin=8))
        content.add(self.analysis_webview)

        # Load initial empty state
        self._render_analysis()

        scroll = toga.ScrollContainer(content=content, style=Pack(flex=1))
        page.add(scroll)

        return page

    # ----------------------------
    # Tab strip helpers (classic look)
    # ----------------------------

    def _apply_tab_styles(self, active: str) -> None:
        """Style the sidebar tab rows + buttons to indicate which section is active."""
        active_btn = dict(background_color="#2D6CDF", color="white", font_weight="bold")
        inactive_btn = dict(
            background_color="#FFFFFF", color="#222", font_weight="normal"
        )

        # Tip text: keep it muted regardless of selection state
        tip_style = dict(color="#5B6472")

        # Safe-guard: these may not exist if startup hasn't built the sidebar yet
        files_ok = hasattr(self, "tab_btn_files") and hasattr(self, "tab_tip_files")
        settings_ok = hasattr(self, "tab_btn_settings") and hasattr(
            self, "tab_tip_settings"
        )
        analyze_ok = hasattr(self, "tab_btn_analyze") and hasattr(
            self, "tab_tip_analyze"
        )
        analysis_ok = hasattr(self, "tab_btn_analysis") and hasattr(
            self, "tab_tip_analysis"
        )

        def _set(btn, tip, is_active: bool, force_disabled: bool = False):
            if force_disabled:
                # Greyed-out tab: don't apply the normal active/inactive style;
                # just ensure it looks muted regardless of selection state.
                btn.style.update(
                    background_color="#E8E8E8", color="#AAAAAA", font_weight="normal"
                )
                tip.style.update(color="#BBBBBB")
            else:
                btn.style.update(**(active_btn if is_active else inactive_btn))
                tip.style.update(**tip_style)

        if files_ok and settings_ok and analyze_ok and analysis_ok:
            settings_disabled = not self.tab_btn_settings.enabled
            _set(self.tab_btn_files, self.tab_tip_files, active == "files")
            _set(
                self.tab_btn_settings,
                self.tab_tip_settings,
                active == "settings",
                force_disabled=settings_disabled,
            )
            _set(self.tab_btn_analyze, self.tab_tip_analyze, active == "analyze")
            _set(self.tab_btn_analysis, self.tab_tip_analysis, active == "analysis")

    # ----------------------------
    # Tab navigation (swap active page)
    # ----------------------------

    # ----------------------------
    # Input-mode helpers
    # ----------------------------

    def _set_input_mode(self, mode: str) -> None:
        """
        Lock the UI into 'videos' or 'folders' mode, or reset to '' (empty).

        When locked:
        - The button for the other mode is disabled.
        - The Settings sidebar tab is greyed out when mode is 'folders'.
        - A hint label explains the current mode.
        """
        self.input_mode = mode

        if mode == "videos":
            self.btn_add_videos.enabled = True
            self.btn_add_folders.enabled = False
            self.files_section_title.text = "Input Videos"
            self.files_mode_hint.text = (
                "Videos mode — clear the list to switch to Reanalyze mode."
            )
            self._set_settings_tab_enabled(True)

        elif mode == "folders":
            self.btn_add_videos.enabled = False
            self.btn_add_folders.enabled = True
            self.files_section_title.text = "Output Folders (Reanalyze)"
            self.files_mode_hint.text = (
                "Reanalyze mode: steps 2–5 are re-run on existing output folders. "
                "Clear the list to switch back to Videos mode."
            )
            self._set_settings_tab_enabled(False)

        else:
            # No mode yet — both buttons active
            self.btn_add_videos.enabled = True
            self.btn_add_folders.enabled = True
            self.files_section_title.text = "Input Videos / Folders"
            self.files_mode_hint.text = ""
            self._set_settings_tab_enabled(True)

    def _set_settings_tab_enabled(self, enabled: bool) -> None:
        """Grey out or restore the Settings sidebar tab button."""
        self.tab_btn_settings.enabled = enabled
        self.tab_tip_settings.style.color = "#5B6472" if enabled else "#BBBBBB"

    def _select_tab(self, tab_key: str) -> None:
        """Select a tab by removing and adding pages (eliminates redraw flashing)."""
        if tab_key not in ("files", "settings", "analyze", "analysis"):
            tab_key = "files"

        # If Settings is disabled (folders mode) redirect to Files
        if tab_key == "settings" and not self.tab_btn_settings.enabled:
            tab_key = "files"

        # Debounce redundant selections (avoids unnecessary churn)
        if getattr(self, "_current_tab", None) == tab_key:
            return
        self._current_tab = tab_key

        # Clear the page host and add only the selected page
        self.page_host.clear()

        if tab_key == "files":
            self.page_host.add(self.page_files)
        elif tab_key == "settings":
            self.page_host.add(self.page_settings)
        elif tab_key == "analyze":
            self.page_host.add(self.page_analyze)
        else:  # analysis
            self.page_host.add(self.page_analysis)
            self._render_analysis()

        # Update visual state of tabs
        self._apply_tab_styles(tab_key)

    # ----------------------------
    # Logging (Rich-ish)
    # ----------------------------

    def _log(self, text: str) -> None:
        """Append a Rich-markup-ish line to the HTML log."""
        self.log_renderer.add_log_line(text)
        self._render_log_html()

    def _log_banner(self) -> None:
        self._log("[bold green]═══ Barpath Pipeline (GUI) ═══[/bold green]")
        self._log(
            "[dim]Choose inputs in Files, configure in Settings, then run in Analyze.[/dim]"
        )
        self._log("")

    def _log_config(self) -> None:
        self._log("[bold]Configuration:[/bold]")

        if self.input_mode == "folders":
            active_list = self.input_folders
            self._log("  Mode:         [cyan]Reanalyze (steps 2-5)[/cyan]")
            self._log(f"  Folders:      [cyan]{len(active_list)}[/cyan]")
            if len(active_list) <= 8:
                for i, item in enumerate(active_list, 1):
                    self._log(f"    {i}. [dim]{item.name}[/dim]")
            else:
                for i, item in enumerate(active_list[:5], 1):
                    self._log(f"    {i}. [dim]{item.name}[/dim]")
                self._log(f"    ... [dim]+{len(active_list) - 5} more[/dim]")
        else:
            active_list = self.input_videos
            model = self._resolve_selected_model()
            self._log("  Mode:         [cyan]Full pipeline (steps 1-5)[/cyan]")
            self._log(f"  Input Videos: [cyan]{len(active_list)}[/cyan]")
            if len(active_list) <= 8:
                for i, item in enumerate(active_list, 1):
                    self._log(f"    {i}. [dim]{item.name}[/dim]")
            else:
                for i, item in enumerate(active_list[:5], 1):
                    self._log(f"    {i}. [dim]{item.name}[/dim]")
                self._log(f"    ... [dim]+{len(active_list) - 5} more[/dim]")
            self._log(
                f"  Model:        [cyan]{model if model else '(not selected)'}[/cyan]"
            )

        self._log(f"  Lift Type:    [cyan]{self.lift_type}[/cyan]")
        self._log(f"  Lifter:       [cyan]{self.lifter}[/cyan]")
        self._log(
            f"  Technique Analysis: [cyan]{'enabled' if self.technique_analysis else 'disabled'}[/cyan]"
        )
        self._log(f"  Output Dir:   [cyan]{self._effective_output_dir()}[/cyan]")
        self._log("")

    # ----------------------------
    # Model discovery + selection UI
    # ----------------------------

    def _populate_default_model_dir(self) -> None:
        models_dir = Path(__file__).parent / "models"
        if models_dir.exists() and models_dir.is_dir():
            self._populate_model_files(models_dir)

        analysis_models_dir = models_dir / "analysis"
        if analysis_models_dir.exists() and analysis_models_dir.is_dir():
            self._populate_lifter_options(analysis_models_dir)

    def _populate_model_files(self, directory: Path) -> None:
        self.model_dir = directory

        pt_files = list(directory.glob("*.pt"))
        onnx_files = list(directory.glob("*.onnx"))
        engine_files = list(directory.glob("*.engine"))
        openvino_dirs = [
            p
            for p in directory.iterdir()
            if p.is_dir() and "openvino" in p.name.lower()
        ]
        candidates = pt_files + onnx_files + engine_files + openvino_dirs
        self.model_files = sorted(candidates, key=lambda p: p.name.lower())

        if self.model_files:
            if self.selected_model not in self.model_files:
                self.selected_model = self.model_files[0]
        else:
            self.selected_model = None

    def _populate_lifter_options(self, analysis_dir: Path) -> None:
        """Populate the lifter dropdown from available analysis models."""
        lifters = set(["generic"])

        if analysis_dir.exists():
            for item in analysis_dir.iterdir():
                if item.is_dir() and item.name not in ("generic",):
                    lifters.add(item.name)

        lifter_list = sorted(lifters)

        if hasattr(self, "lifter_dropdown"):
            self.lifter_dropdown.items = lifter_list
            if self.lifter in lifter_list:
                self.lifter_dropdown.value = self.lifter

    def _refresh_settings_buttons(self) -> None:
        """Rebuild the model dropdown items and sync lift button styles."""
        if not hasattr(self, "_lift_buttons"):
            self._lift_buttons = {}  # type: ignore[attr-defined]

        # --- Model dropdown ---
        # Detach the change handler while we touch .items so that Toga's
        # internal reset of the widget value (to index 0) does not fire
        # _on_model_dropdown_change and clobber self.selected_model.
        if not self.model_files:
            self.model_dropdown.on_change = None
            self.model_dropdown.items = ["(No models found)"]
            self.model_dropdown.on_change = self._on_model_dropdown_change
            self.model_dropdown.enabled = False
        else:
            names = [p.name for p in self.model_files]
            last_names = getattr(self, "_last_model_dropdown_names", None)
            if last_names != names:
                # Only reassign items when the list has actually changed, and
                # mute the handler for the duration to avoid the selection reset.
                self.model_dropdown.on_change = None
                self.model_dropdown.items = names
                self.model_dropdown.on_change = self._on_model_dropdown_change
                self._last_model_dropdown_names = names
            self.model_dropdown.enabled = True
            # Sync dropdown to currently selected model
            if self.selected_model is not None:
                try:
                    self.model_dropdown.value = self.selected_model.name
                except Exception:
                    pass

        # --- Lift buttons ---
        for lift in ("auto", "none", "clean", "snatch", "jerk", "clean_jerk"):
            if lift not in self._lift_buttons:  # type: ignore[attr-defined]
                label = "Clean & Jerk" if lift == "clean_jerk" else lift.capitalize()
                btn = toga.Button(
                    label,
                    on_press=lambda w, lt=lift: self._set_lift_type(lt),
                    style=self._pill_style(selected=False),
                )
                self._lift_buttons[lift] = btn  # type: ignore[attr-defined]
                self.lift_button_row.add(btn)

        for lift in ("auto", "none", "clean", "snatch", "jerk", "clean_jerk"):
            btn = self._lift_buttons.get(lift)  # type: ignore[attr-defined]
            if btn is not None:
                btn.style.update(
                    **self._pill_style_dict(selected=(self.lift_type == lift))
                )

    def _pill_style_dict(self, selected: bool) -> dict:
        """Return a dict of style keys so we can update styles in-place."""
        if selected:
            return dict(
                margin_right=6,
                margin=(6, 10),
                background_color="#2D6CDF",
                color="white",
                font_weight="bold",
            )
        return dict(
            margin_right=6,
            margin=(6, 10),
            background_color="#F2F3F7",
            color="#222",
            font_weight="normal",
        )

    def _pill_style(self, selected: bool) -> Pack:
        # Keep existing callers working
        return Pack(**self._pill_style_dict(selected))

    def _render_log_html(self) -> None:
        """Render the full HTML log document into the WebView with beautiful styling."""
        doc = self.log_renderer.render_html()

        try:
            self.log_webview.set_content(root_url="about:blank", content=doc)
        except Exception:
            try:
                self.log_webview.set_content(root_url="", content=doc)
            except Exception:
                pass

    def _render_analysis(self) -> None:
        """Render the analysis markdown as HTML in the WebView."""
        analysis_path = self._effective_output_dir() / "analysis.md"

        if analysis_path.exists():
            doc = self.markdown_renderer.render_markdown_file(analysis_path)
        else:
            doc = self.markdown_renderer.render_no_analysis()

        try:
            self.analysis_webview.set_content(root_url="about:blank", content=doc)
        except Exception:
            try:
                self.analysis_webview.set_content(root_url="", content=doc)
            except Exception:
                pass

    def _on_model_dropdown_change(self, widget: Any) -> None:
        """Called when the user picks an entry in the model dropdown."""
        selected_name = widget.value
        match = next((p for p in self.model_files if p.name == selected_name), None)
        if match is not None:
            self.selected_model = match
            self._log(f"[green]✓[/green] Selected model: [cyan]{match.name}[/cyan]")

    def _set_selected_model(self, model_path: Path) -> None:
        self.selected_model = model_path
        self._refresh_settings_buttons()
        self._log(f"[green]✓[/green] Selected model: [cyan]{model_path.name}[/cyan]")

    def _set_lift_type(self, lift_type: str) -> None:
        self.lift_type = lift_type
        self._refresh_settings_buttons()
        self._log(f"[green]✓[/green] Lift type: [cyan]{self.lift_type}[/cyan]")

    def _on_use_filenames_change(self, widget: Any) -> None:
        self.use_filenames_in_legend = bool(widget.value)

    def _on_lifter_dropdown_change(self, widget: Any) -> None:
        """Called when the user picks an entry in the lifter dropdown."""
        selected = widget.value
        if selected:
            self.lifter = selected
            self._log(f"[green]✓[/green] Selected lifter: [cyan]{selected}[/cyan]")

    def _on_technique_analysis_change(self, widget: Any) -> None:
        self.technique_analysis = bool(widget.value)
        status = "enabled" if self.technique_analysis else "disabled"
        self._log(f"[green]✓[/green] Technique Analysis {status}")

    def _resolve_selected_model(self) -> Path | None:
        if self.selected_model is None:
            return None
        return self.selected_model

    # ----------------------------
    # Output directory helpers
    # ----------------------------

    def _effective_output_dir(self) -> Path:
        # Always treat the output directory as user-chosen absolute/relative path.
        # Keep it as Path for pipeline invocation.
        return self.output_dir

    def _set_output_dir_value(self, directory: Path) -> None:
        self.output_dir = directory
        # Display as a nice absolute path for clarity
        self.output_dir_label.text = str(directory.expanduser().resolve())

    # ----------------------------
    # Event handlers: Files
    # ----------------------------

    async def on_browse_video(self, widget: toga.Widget) -> None:
        try:
            path = await self.main_window.dialog(  # type: ignore
                toga.OpenFileDialog(
                    title="Select Video File(s)",
                    file_types=self.video_extensions,
                    multiple_select=True,
                )
            )
            if not path:
                return

            paths = path if isinstance(path, list) else [path]
            added = 0
            for p in paths:
                vp = Path(p)
                if vp not in self.input_videos:
                    self.input_videos.append(vp)
                    self._add_video_row(vp)
                    added += 1

            if added:
                self._set_input_mode("videos")
                self.btn_clear_videos.enabled = True
                self._log(f"[green]✓[/green] Added [cyan]{added}[/cyan] video(s)")
        except Exception as e:
            await self.main_window.dialog(  # type: ignore[attr-defined]
                toga.ErrorDialog("Error", f"Could not select file(s): {e}")
            )

    async def on_browse_folders(self, widget: toga.Widget) -> None:
        """Let the user pick one or more existing output folders to reanalyze."""
        try:
            chosen = await self.main_window.dialog(  # type: ignore
                toga.SelectFolderDialog(
                    title="Select Output Folder to Reanalyze",
                )
            )
            if not chosen:
                return

            folder = Path(chosen)
            pkl_path = folder / "raw_data.pkl"
            if not pkl_path.exists():
                await self.main_window.dialog(  # type: ignore[attr-defined]
                    toga.ErrorDialog(
                        "Invalid Folder",
                        f"'{folder.name}' does not contain raw_data.pkl.\n"
                        "Only folders previously processed by Barpath can be reanalyzed.",
                    )
                )
                return

            if folder not in self.input_folders:
                self.input_folders.append(folder)
                self._add_video_row(folder)

            self._set_input_mode("folders")
            self.btn_clear_videos.enabled = True
            self._log(f"[green]✓[/green] Added folder: [cyan]{folder.name}[/cyan]")
        except Exception as e:
            await self.main_window.dialog(  # type: ignore[attr-defined]
                toga.ErrorDialog("Error", f"Could not select folder: {e}")
            )

    def on_clear_videos(self, widget: toga.Widget) -> None:
        self.input_videos.clear()
        self.input_folders.clear()
        self.video_list_box.clear()
        self.btn_clear_videos.enabled = False
        self._set_input_mode("")
        self._log("[yellow]![/yellow] Cleared all inputs")

    def _add_video_row(self, item_path: Path) -> None:
        row = toga.Box(
            style=Pack(
                direction="row", margin_bottom=4, margin=6, background_color="#F2F3F7"
            )
        )

        remove_btn = toga.Button(
            "Remove",
            on_press=lambda w, vp=item_path: self.on_remove_video(w, vp),
            style=Pack(width=80, margin_right=8),
        )
        row.add(remove_btn)

        row.add(toga.Label(str(item_path), style=Pack(flex=1, color="#222")))

        self.video_list_box.add(row)

    def on_remove_video(self, widget: toga.Widget, item_path: Path) -> None:
        if item_path in self.input_videos:
            self.input_videos.remove(item_path)
        if item_path in self.input_folders:
            self.input_folders.remove(item_path)

        self.video_list_box.clear()
        active_list = (
            self.input_folders if self.input_mode == "folders" else self.input_videos
        )
        for vp in active_list:
            self._add_video_row(vp)

        remaining = len(active_list)
        self.btn_clear_videos.enabled = remaining > 0
        if remaining == 0:
            self._set_input_mode("")
        self._log(f"[yellow]–[/yellow] Removed: [dim]{item_path.name}[/dim]")

    def on_open_output_dir(self, widget: toga.Widget) -> None:
        target_path = self._effective_output_dir().expanduser()
        if not target_path.is_absolute():
            target_path = Path.cwd() / target_path

        try:
            target_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self._log(
                f"[bold red]ERROR[/bold red] Could not create output directory: {e}"
            )
            return

        try:
            if sys.platform.startswith("win"):
                os.startfile(str(target_path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(target_path)], check=False)
            else:
                subprocess.run(["xdg-open", str(target_path)], check=False)
        except Exception as e:
            self._log(
                f"[bold red]ERROR[/bold red] Could not open output directory: {e}"
            )
        else:
            self._log(
                f"[green]✓[/green] Opened output directory: [cyan]{target_path}[/cyan]"
            )

    async def on_select_output_dir(self, widget: toga.Widget) -> None:
        """Select output directory using system picker."""
        try:
            chosen = await self.main_window.dialog(  # type: ignore
                toga.SelectFolderDialog(
                    title="Select Output Directory",
                    # Some backends support initial directory; keep minimal for compatibility.
                )
            )
            if not chosen:
                return

            out_dir = Path(chosen)
            self._set_output_dir_value(out_dir)
            self._log(
                f"[green]✓[/green] Output directory set to: [cyan]{self.output_dir_label.text}[/cyan]"
            )
        except Exception as e:
            await self.main_window.dialog(  # type: ignore[attr-defined]
                toga.ErrorDialog("Error", f"Could not select output directory: {e}")
            )

    async def _check_existing_outputs(
        self, input_videos: list[Path], output_base: Path, use_filenames: bool
    ) -> tuple[list[Path], list[Path]]:
        """
        Check for existing output folders for ALL videos upfront.

        Returns:
            Tuple of (videos_to_process, videos_to_skip)
        """
        existing_videos: list[tuple[Path, Path]] = []  # (video, output_dir)
        new_videos: list[Path] = []

        # Check all videos upfront
        for idx, video in enumerate(input_videos, 1):
            if len(input_videos) > 1:
                folder_name = video.stem if use_filenames else f"lift_{idx}"
                video_output_dir = output_base / folder_name
            else:
                video_output_dir = output_base

            # Check if output exists
            has_output = video_output_dir.exists() and (
                (video_output_dir / "final_analysis.csv").exists()
                or (video_output_dir / "raw_data.pkl").exists()
            )

            if has_output:
                existing_videos.append((video, video_output_dir))
            else:
                new_videos.append(video)

        if not existing_videos:
            return input_videos, []

        # Show single dialog for all existing videos
        if len(existing_videos) == 1:
            message = (
                f"Output already exists for:\n\n"
                f"  {existing_videos[0][0].name}\n\n"
                f"Skip this video or reprocess it?"
            )
        else:
            video_list = "\n".join([f"  • {v.name}" for v, _ in existing_videos[:5]])
            if len(existing_videos) > 5:
                video_list += f"\n  ... and {len(existing_videos) - 5} more"
            message = (
                f"Output already exists for {len(existing_videos)} video(s):\n\n"
                f"{video_list}\n\n"
                f"Skip all existing or reprocess all?"
            )

        try:
            # Ask: Skip or Reprocess?
            skip_all = await self.main_window.dialog(  # type: ignore
                toga.ConfirmDialog(
                    "Existing Outputs Found",
                    f"{message}\n\n"
                    f"Click 'Yes' to SKIP existing videos.\n"
                    f"Click 'No' to REPROCESS all videos.",
                )
            )

            if skip_all:
                # User chose to skip all existing
                self._log(
                    f"[yellow]Skipping {len(existing_videos)} already processed video(s)[/yellow]"
                )
                self._log(f"[cyan]Processing {len(new_videos)} video(s)[/cyan]")
                return new_videos, [v for v, _ in existing_videos]
            else:
                # User chose to reprocess all
                self._log(f"[cyan]Reprocessing all {len(input_videos)} video(s)[/cyan]")
                return input_videos, []

        except Exception as e:
            # On error, default to processing all
            self._log(f"[yellow]Warning: Could not show dialog: {e}[/yellow]")
            return input_videos, []

    # ----------------------------
    # Event handlers: Analyze
    # ----------------------------

    def on_run_analysis(self, widget: toga.Widget) -> None:
        """Button handler — validates inputs then kicks off the async watcher task."""
        if self._is_running:
            return

        using_folders = self.input_mode == "folders"

        if not using_folders and not self.input_videos:
            self._select_tab("files")
            self._log(
                "[bold red]ERROR[/bold red] Please add at least one video or output folder in Files."
            )
            return

        if using_folders and not self.input_folders:
            self._select_tab("files")
            self._log(
                "[bold red]ERROR[/bold red] Please add at least one output folder in Files."
            )
            return

        selected_model: Path | None = None
        if not using_folders:
            selected_model = self._resolve_selected_model()
            if not selected_model:
                self._select_tab("settings")
                self._log(
                    "[bold red]ERROR[/bold red] Please select a model in Settings."
                )
                return

        # Clear log and print configuration
        self._log_html_lines = []
        self._log_seq = 0
        self._render_log_html()
        self._log_banner()
        self._log_config()

        # Update UI state
        self._is_running = True
        self.run_button.enabled = False
        self.cancel_button.enabled = True
        self.progress_bar.value = 0
        self.progress_label.text = "Starting pipeline..."

        # Reset skip/rerun decisions
        self._skip_all_existing = False
        self._rerun_all_existing = False

        # For video processing, check for existing outputs asynchronously
        if not using_folders:
            # Create async task to check existing and then start pipeline
            asyncio.create_task(self._check_existing_and_run(selected_model))
        else:
            # For folder reprocessing, start directly
            self._start_pipeline(
                selected_model, using_folders, self.input_videos, self.input_folders
            )

    async def _check_existing_and_run(self, selected_model: Path | None) -> None:
        """Check for existing outputs and start pipeline after user decision."""
        try:
            output_base = self._effective_output_dir()
            use_filenames = self.use_filenames_in_legend

            # Check for existing outputs
            videos_to_process, videos_to_skip = await self._check_existing_outputs(
                self.input_videos, output_base, use_filenames
            )

            if videos_to_skip:
                self._log(
                    f"[yellow]Skipping {len(videos_to_skip)} already processed video(s)[/yellow]"
                )

            if not videos_to_process:
                self._log("[yellow]No videos to process.[/yellow]")
                self._is_running = False
                self.run_button.enabled = True
                self.cancel_button.enabled = False
                return

            # Start pipeline with filtered list
            self._start_pipeline(
                selected_model,
                False,  # using_folders
                videos_to_process,
                self.input_folders,
            )
        except Exception as e:
            self._log(
                f"[bold red]ERROR[/bold red] Failed to check existing outputs: {e}"
            )
            self._is_running = False
            self.run_button.enabled = True
            self.cancel_button.enabled = False

    def _start_pipeline(
        self,
        selected_model: Path | None,
        using_folders: bool,
        videos_to_process: list[Path],
        folders_to_process: list[Path],
    ) -> None:
        """Start the pipeline with the given inputs."""
        self._cancel_event.clear()

        # Drain any leftover messages from a previous run
        while not self._progress_queue.empty():
            try:
                self._progress_queue.get_nowait()
            except queue.Empty:
                break

        # Switch to Analyze tab
        self._select_tab("analyze")

        # 1. Submit the blocking pipeline work to the background thread pool.
        #    The worker function drains the run_pipeline generator and pushes
        #    every (step, progress, message) tuple onto _progress_queue.
        #    It never touches Toga widgets directly.
        input_videos_snapshot = list(videos_to_process)
        input_folders_snapshot = list(folders_to_process)
        selected_model_snapshot = selected_model
        encode_video_snapshot = self.encode_video
        lift_type_snapshot = self.lift_type
        use_filenames_snapshot = self.use_filenames_in_legend
        using_folders_snapshot = using_folders

        self._thread_executor.submit(
            self._pipeline_worker,
            input_videos_snapshot,
            selected_model_snapshot,
            encode_video_snapshot,
            lift_type_snapshot,
            use_filenames_snapshot,
            input_folders_snapshot,
            using_folders_snapshot,
        )

        # 2. Kick off the lightweight async watcher that reads _progress_queue
        #    and updates the Toga UI on the main thread.
        self._pipeline_task = asyncio.create_task(self._progress_watcher_async())

    # ------------------------------------------------------------------
    # Background pipeline worker (runs on a real OS thread — NOT the
    # Toga/asyncio main thread).  It must NOT call any Toga API.
    # ------------------------------------------------------------------

    def _pipeline_worker(
        self,
        input_videos: list[Path],
        selected_model: Path | None,
        encode_video: bool,
        lift_type: str,
        use_filenames_in_legend: bool = False,
        input_folders: list[Path] | None = None,
        using_folders: bool = False,
    ) -> None:
        """
        Execute the barpath pipeline for every queued video or output folder.

        When ``using_folders`` is True the worker re-runs steps 2-5 on each
        folder using ``run_pipeline_from_folder``; otherwise the full 5-step
        pipeline is run on each video file.

        Progress tuples are pushed onto ``self._progress_queue``.  When all
        items are finished (or on error/cancellation) a sentinel string is
        pushed to wake up the async watcher on the main thread.
        """
        if input_folders is None:
            input_folders = []

        if using_folders:
            self._pipeline_worker_folders(
                input_folders, encode_video, lift_type, use_filenames_in_legend
            )
        else:
            self._pipeline_worker_videos(
                input_videos,
                selected_model,  # type: ignore[arg-type]
                encode_video,
                lift_type,
                use_filenames_in_legend,
            )

    def _pipeline_worker_videos(
        self,
        input_videos: list[Path],
        selected_model: Path,
        encode_video: bool,
        lift_type: str,
        use_filenames_in_legend: bool,
    ) -> None:
        """Full 5-step pipeline for a list of raw video files."""
        run_pipeline = _get_run_pipeline()
        is_batch = len(input_videos) > 1
        total_videos = len(input_videos)

        completed_video_dirs: list[Path] = []
        completed_video_labels: list[str] = []
        skipped_insufficient: list[tuple[Path, str]] = []  # (video, reason)

        try:
            for video_idx, input_video in enumerate(input_videos, 1):
                if self._cancel_event.is_set():
                    break

                out_base = self._effective_output_dir()
                if is_batch:
                    if use_filenames_in_legend:
                        folder_name = input_video.stem
                    else:
                        folder_name = f"lift_{video_idx}"
                    video_output_dir = out_base / folder_name
                    video_output_dir.mkdir(parents=True, exist_ok=True)
                else:
                    video_output_dir = out_base
                    video_output_dir.mkdir(parents=True, exist_ok=True)

                output_video_path = (
                    video_output_dir / "output.mp4" if encode_video else None
                )

                self._progress_queue.put(
                    (
                        "_banner_",
                        None,
                        f"[bold cyan]Processing video {video_idx}/{total_videos}[/bold cyan]: "
                        f"[dim]{input_video.name}[/dim]",
                    )
                )

                try:
                    hud_options = {k: sw.value for k, sw in self.hud_switches.items()}
                    for step_name, progress_value, message in run_pipeline(
                        input_video=str(input_video),
                        model_path=str(selected_model),
                        output_video=(
                            str(output_video_path) if output_video_path else None
                        ),
                        lift_type=lift_type,
                        output_dir=str(video_output_dir),
                        encode_video=encode_video,
                        technique_analysis=(
                            lift_type != "none" and self.technique_analysis
                        ),
                        cancel_event=self._cancel_event,
                        lifter=self.lifter,
                        hud_options=hud_options,
                    ):
                        # Check for insufficient data signal
                        if step_name == "_insufficient_data_":
                            # Skip this video due to insufficient data
                            self._progress_queue.put(
                                (
                                    "_video_skipped_",
                                    None,
                                    f"[yellow]Skipped {input_video.name}: {message}[/yellow]",
                                )
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

                        self._progress_queue.put((step_name, progress_value, message))
                    else:
                        # Only mark as completed if we didn't break due to insufficient data
                        completed_video_dirs.append(video_output_dir)
                        if use_filenames_in_legend:
                            completed_video_labels.append(input_video.stem)
                        else:
                            completed_video_labels.append(f"Lift {video_idx}")

                        self._progress_queue.put(
                            (
                                "_video_done_",
                                None,
                                f"[green]✓[/green] Completed: [dim]{input_video.name}[/dim]",
                            )
                        )

                except Exception as e:
                    # Catch any other exceptions during pipeline execution
                    self._progress_queue.put(
                        (
                            "_video_error_",
                            None,
                            f"[red]Error processing {input_video.name}: {str(e)}[/red]",
                        )
                    )
                    continue

            # Report skipped videos at the end
            if skipped_insufficient:
                self._progress_queue.put(
                    (
                        "_banner_",
                        None,
                        f"[yellow]Skipped {len(skipped_insufficient)} video(s) due to insufficient data[/yellow]",
                    )
                )
                for video, reason in skipped_insufficient:
                    self._progress_queue.put(
                        (
                            "_info_",
                            None,
                            f"  • {video.name}: {reason}",
                        )
                    )

            if (
                not self._cancel_event.is_set()
                and is_batch
                and len(completed_video_dirs) > 1
            ):
                self._progress_queue.put(
                    (
                        "_banner_",
                        None,
                        "[bold cyan]Batch post-processing...[/bold cyan]",
                    )
                )
                run_batch_postprocess = _get_run_batch_postprocess()
                for step_name, progress_value, message in run_batch_postprocess(
                    video_output_dirs=completed_video_dirs,
                    video_labels=completed_video_labels,
                    batch_output_dir=self._effective_output_dir(),
                    use_filenames=use_filenames_in_legend,
                    cancel_event=self._cancel_event,
                ):
                    self._progress_queue.put((step_name, progress_value, message))

            if self._cancel_event.is_set():
                self._progress_queue.put("_CANCELLED_")
            else:
                self._progress_queue.put("_DONE_")

        except InterruptedError:
            self._progress_queue.put("_CANCELLED_")
        except Exception as exc:
            import traceback

            tb = traceback.format_exc()
            self._progress_queue.put(f"_ERROR_:{exc}\n{tb}")

    def _pipeline_worker_folders(
        self,
        input_folders: list[Path],
        encode_video: bool,
        lift_type: str,
        use_filenames_in_legend: bool,
    ) -> None:
        """Re-run steps 2-5 for a list of existing output folders."""
        run_pipeline_from_folder = _get_run_pipeline_from_folder()
        is_batch = len(input_folders) > 1
        total_folders = len(input_folders)

        completed_video_dirs: list[Path] = []
        completed_video_labels: list[str] = []

        try:
            for folder_idx, folder in enumerate(input_folders, 1):
                if self._cancel_event.is_set():
                    break

                self._progress_queue.put(
                    (
                        "_banner_",
                        None,
                        f"[bold cyan]Reanalyzing folder {folder_idx}/{total_folders}[/bold cyan]: "
                        f"[dim]{folder.name}[/dim]",
                    )
                )

                hud_options = {k: sw.value for k, sw in self.hud_switches.items()}
                for step_name, progress_value, message in run_pipeline_from_folder(
                    output_folder=folder,
                    lift_type="none",
                    encode_video=encode_video,
                    technique_analysis=True,
                    cancel_event=self._cancel_event,
                    hud_options=hud_options,
                ):
                    self._progress_queue.put((step_name, progress_value, message))

                completed_video_dirs.append(folder)
                if use_filenames_in_legend:
                    completed_video_labels.append(folder.name)
                else:
                    completed_video_labels.append(f"Lift {folder_idx}")

                self._progress_queue.put(
                    (
                        "_video_done_",
                        None,
                        f"[green]✓[/green] Completed: [dim]{folder.name}[/dim]",
                    )
                )

            if (
                not self._cancel_event.is_set()
                and is_batch
                and len(completed_video_dirs) > 1
            ):
                self._progress_queue.put(
                    (
                        "_banner_",
                        None,
                        "[bold cyan]Batch post-processing...[/bold cyan]",
                    )
                )
                run_batch_postprocess = _get_run_batch_postprocess()
                for step_name, progress_value, message in run_batch_postprocess(
                    video_output_dirs=completed_video_dirs,
                    video_labels=completed_video_labels,
                    batch_output_dir=(
                        completed_video_dirs[0].parent
                        if completed_video_dirs
                        else Path("outputs")
                    ),
                    use_filenames=use_filenames_in_legend,
                    cancel_event=self._cancel_event,
                ):
                    self._progress_queue.put((step_name, progress_value, message))

            if self._cancel_event.is_set():
                self._progress_queue.put("_CANCELLED_")
            else:
                self._progress_queue.put("_DONE_")

        except InterruptedError:
            self._progress_queue.put("_CANCELLED_")
        except Exception as exc:
            import traceback

            tb = traceback.format_exc()
            self._progress_queue.put(f"_ERROR_:{exc}\n{tb}")

    # ------------------------------------------------------------------
    # Async watcher (runs on the main Toga/asyncio thread).
    # It only reads from _progress_queue and updates UI widgets.
    # ------------------------------------------------------------------

    async def _progress_watcher_async(self) -> None:
        """
        Poll ``_progress_queue`` and update the Toga UI accordingly.

        This coroutine yields control back to the event loop between batches
        so that the GUI remains fully responsive (repaints, clicks, etc.)
        while the pipeline runs on a background thread.
        """
        active_list = (
            self.input_folders if self.input_mode == "folders" else self.input_videos
        )
        is_batch = len(active_list) > 1
        total_videos = len(active_list)
        # We track the video index here by counting _video_done_ sentinels
        videos_done = 0

        try:
            while True:
                # Drain all currently-available messages in one go, then yield.
                # This batches rapid frame-by-frame updates while still
                # keeping the GUI responsive.
                processed_any = False
                while True:
                    try:
                        item = self._progress_queue.get_nowait()
                    except queue.Empty:
                        break

                    processed_any = True

                    # --- Sentinel strings ---
                    if isinstance(item, str):
                        if item == "_DONE_":
                            await self._on_pipeline_done(is_batch)
                            return
                        elif item == "_CANCELLED_":
                            self._log(
                                "[yellow]![/yellow] Cancellation requested; stopping."
                            )
                            self.progress_label.text = "Cancelled"
                            self.progress_bar.value = 0
                            self._is_running = False
                            self.run_button.enabled = True
                            self.cancel_button.enabled = False
                            self._pipeline_task = None
                            return
                        elif item.startswith("_ERROR_:"):
                            error_body = item[len("_ERROR_:") :]
                            first_line = error_body.split("\n")[0]
                            self._log(
                                f"\n[bold red]ERROR[/bold red] Pipeline failed: {first_line}"
                            )
                            self._log(error_body)
                            self.progress_label.text = f"Error: {first_line}"
                            return
                        continue

                    # --- Progress tuples ---
                    step_name, progress_value, message = item

                    if step_name == "_banner_":
                        self._log(message)
                        continue

                    if step_name == "_video_done_":
                        self._log(message)
                        self._log("")
                        videos_done += 1
                        continue

                    if step_name == "batch":
                        self._log(f"[green]✓[/green] [dim]batch[/dim] {message}")
                        continue

                    # Throttle per-frame log spam (only log non-frame messages)
                    if "frame" not in str(message).lower() or progress_value is None:
                        if progress_value is not None:
                            self._log(f"[dim]{step_name}[/dim] {message}")
                        else:
                            self._log(
                                f"[green]✓[/green] [dim]{step_name}[/dim] {message}"
                            )

                    # Update progress bar with overall progress across all videos
                    if progress_value is not None:
                        video_progress = videos_done / max(total_videos, 1)
                        step_progress = float(progress_value) / max(total_videos, 1)
                        overall_progress = video_progress + step_progress
                        self.progress_bar.value = int(overall_progress * 100)
                        self.progress_label.text = (
                            f"[{videos_done + 1}/{total_videos}] {message}"
                        )
                    else:
                        self.progress_label.text = (
                            f"[{videos_done + 1}/{total_videos}] ✓ {message}"
                        )

                # Yield to the Toga event loop so the GUI can repaint / handle
                # user input.  Use a short sleep when there was work to do, a
                # slightly longer one when the queue was empty to avoid busy-
                # waiting while the background thread is doing heavy work.
                await asyncio.sleep(0.02 if processed_any else 0.1)

        except asyncio.CancelledError:
            # Task was cancelled externally (e.g. app shutdown)
            pass
        finally:
            self._is_running = False
            self.run_button.enabled = True
            self.cancel_button.enabled = False
            self._pipeline_task = None

    async def _on_pipeline_done(self, is_batch: bool) -> None:
        """Handle successful pipeline completion: update UI and show report."""
        label = "Folders" if self.input_mode == "folders" else "Videos"
        self._log(f"[bold green]✓ All {label} Complete![/bold green]")
        self.progress_bar.value = 100
        self.progress_label.text = "Analysis complete!"

        if self.input_mode == "folders" and self.input_folders:
            # For folders mode, look for the report in the last folder directly
            analysis_path = self.input_folders[-1] / "analysis.md"
        elif is_batch and self.input_videos:
            last_video = self.input_videos[-1]
            analysis_path = (
                self._effective_output_dir() / last_video.stem / "analysis.md"
            )
        else:
            analysis_path = self._effective_output_dir() / "analysis.md"

        if analysis_path.exists():
            self._log(f"[green]✓[/green] Found report: [cyan]{analysis_path}[/cyan]")
            self._render_analysis()
        else:
            self._log(
                f"[yellow]![/yellow] No analysis report found at: [dim]{analysis_path}[/dim]"
            )

    def on_cancel_analysis(self, widget: toga.Widget) -> None:
        if self._is_running:
            self._log("[yellow]![/yellow] Cancellation requested...")
            self._cancel_event.set()
            self.cancel_button.enabled = False

    # ----------------------------
    # Live webcam preview (YOLO + MediaPipe)
    # ----------------------------

    def on_toggle_preview(self, widget: toga.Widget) -> None:
        """Toggle the live webcam preview on/off."""
        if self._preview_running:
            self._stop_preview()
        else:
            self._start_preview()

    def _start_preview(self) -> None:
        """Start the live preview in a background thread."""
        model_path = self._resolve_selected_model()
        if model_path is None:
            self._log("[red]![/red] No model selected for preview")
            return

        self._preview_stop_event.clear()
        self._preview_running = True
        self.preview_button.text = "Stop Preview"
        self._log(
            "[cyan]Starting preview...[/cyan] Press 'q' in the preview window to stop."
        )

        self._preview_thread = threading.Thread(
            target=self._run_preview,
            args=(str(model_path),),
            daemon=True,
            name="barpath-preview",
        )
        self._preview_thread.start()

    def _stop_preview(self) -> None:
        """Signal the preview thread to stop."""
        self._preview_stop_event.set()
        self._preview_running = False
        self.preview_button.text = "Preview (Alpha)"
        self._log("[cyan]Preview stopped.[/cyan]")

    def _run_preview(self, model_path: str) -> None:
        """
        Background thread that captures webcam frames, runs YOLO + MediaPipe,
        and displays the annotated feed in an OpenCV window.

        Includes live lift recognition: bar path tracing, lift type detection,
        and classification overlay.

        Press 'q' in the preview window to stop.
        """
        import cv2
        import time

        import mediapipe as mp
        from ultralytics import YOLO  # type: ignore
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        from pipeline.step1_helpers.landmarks import get_pose_landmarker_model_path
        from barpath.pipeline.realtime_processing.live_lift_recognition import LiveLiftRecognizer

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._preview_stop_event.set()
            self._preview_running = False
            return

        pose_model_path = get_pose_landmarker_model_path()
        base_options = mp_python.BaseOptions(model_asset_path=str(pose_model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        yolo_model = YOLO(model_path, task="detect")

        # Initialize live lift recognizer with the lift detection model
        lift_model_path = str(
            Path(__file__).parent
            / "models"
            / "lift_detection"
            / "lift_detection_model.pkl"
        )
        recognizer = LiveLiftRecognizer(
            model_path=lift_model_path,
            fps=30.0,
            buffer_seconds=1.0,
            display_seconds=3.0,
        )

        cv2.namedWindow("Barpath Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Barpath Preview", 960, 540)

        frame_times = []
        frame_count = 0
        timestamp_ms = 0

        while not self._preview_stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            pose_results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            yolo_results = yolo_model(frame, verbose=False, conf=0.25)

            # Extract barbell data for recognizer
            barbell_center = None
            barbell_box = None
            if yolo_results and len(yolo_results) > 0:
                for result in yolo_results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            barbell_box = (x1, y1, x2, y2)
                            barbell_center = (
                                (x1 + x2) / 2.0,
                                (y1 + y2) / 2.0,
                            )
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                frame,
                                "Barbell",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )
                            break  # Use first detection only

            # Extract landmark data for recognizer
            landmarks_dict = {}
            if pose_results and pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks[0]
                landmark_pixels = {}
                for idx, lm in enumerate(landmarks):
                    landmarks_dict[idx] = (lm.x, lm.y, lm.z, lm.visibility)
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    vis = lm.visibility
                    if vis > 0.1:
                        landmark_pixels[idx] = (px, py)

                POSE_CONNECTIONS = [
                    (11, 12),
                    (11, 23),
                    (12, 24),
                    (11, 13),
                    (13, 15),
                    (12, 14),
                    (14, 16),
                    (23, 24),
                    (23, 25),
                    (25, 27),
                    (24, 26),
                    (26, 28),
                ]
                for i1, i2 in POSE_CONNECTIONS:
                    if i1 in landmark_pixels and i2 in landmark_pixels:
                        p1 = landmark_pixels[i1]
                        p2 = landmark_pixels[i2]
                        cv2.line(frame, p1, p2, (255, 255, 255), 3)

                for idx, (px, py) in landmark_pixels.items():
                    cv2.circle(frame, (px, py), 4, (255, 0, 0), -1)

            # Feed frame data to live lift recognizer
            recognizer.update(
                barbell_center=barbell_center,
                barbell_box=barbell_box,
                landmarks=landmarks_dict,
                timestamp_ms=float(timestamp_ms),
                frame_width=w,
                frame_height=h,
            )

            # Draw lift recognition overlay (bar path + label)
            recognizer.draw_overlay(frame)

            # Draw FPS counter
            current_time = time.time()
            frame_times.append(current_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            if len(frame_times) >= 2:
                fps = len(frame_times) / (frame_times[-1] - frame_times[0] + 0.001)
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                )

            # Draw recognizer status
            status = recognizer.status_text
            if status:
                cv2.putText(
                    frame,
                    status,
                    (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

            # Draw coaching tip (top-right, yellow, shown for 5s after lift)
            tip = recognizer.current_tip
            if tip:
                tip_font = cv2.FONT_HERSHEY_SIMPLEX
                tip_scale = 0.7
                tip_thickness = 2
                tip_size = cv2.getTextSize(tip, tip_font, tip_scale, tip_thickness)[0]
                tip_pad = 10
                tip_x = w - tip_size[0] - tip_pad - 15
                tip_y = tip_size[1] + tip_pad + 15
                overlay = frame.copy()
                cv2.rectangle(overlay, (tip_x - tip_pad, tip_y - tip_size[1] - tip_pad), (tip_x + tip_size[0] + tip_pad, tip_y + tip_pad), (30, 30, 30), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                cv2.putText(
                    frame,
                    tip,
                    (tip_x, tip_y),
                    tip_font,
                    tip_scale,
                    (0, 255, 255),
                    tip_thickness,
                    cv2.LINE_AA,
                )

            cv2.imshow("Barpath Preview", frame)
            timestamp_ms += 33
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyWindow("Barpath Preview")
        pose_landmarker.close()

        self._preview_stop_event.set()
        self._preview_running = False

    # ----------------------------
    # View analysis (unchanged logic, lightly styled)
    # ----------------------------
    # Utility
    # ----------------------------

    def _validate_environment(self) -> tuple[bool, str]:
        if not self.input_videos:
            return False, "No input videos selected"
        if not self._resolve_selected_model():
            return False, "No model selected"
        return True, "OK"


def main() -> None:
    """Main entry point."""
    icon_path = Path(__file__).resolve().parent / "assets" / "barpath_icon.png"

    kwargs: dict[str, Any] = dict(
        formal_name="Barpath",
        app_id="org.barpath.app",
        description="Weightlifting Technique Analysis Tool",
        version="1.0.0",
        author="Barpath Team",
        home_page="https://github.com/scribewire/barpath",
    )

    if icon_path.exists():
        kwargs["icon"] = str(icon_path)

    app = BarpathTogaApp(**kwargs)
    return app.main_loop()


if __name__ == "__main__":
    main()
