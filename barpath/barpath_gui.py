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
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple

import toga
from gui_helpers.log_renderer import LogRenderer
from gui_helpers.markdown_renderer import MarkdownRenderer
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

    # ----------------------------
    # App lifecycle
    # ----------------------------

    def startup(self) -> None:  # type: ignore[override]
        # --- State ---
        self.model_dir: Optional[Path] = None
        self.model_files: List[Path] = []
        self.selected_model: Optional[Path] = None

        self.input_videos: List[Path] = []
        self.output_dir: Path = Path("outputs")
        self.lift_type: str = "none"

        self.encode_video: bool = True
        self.technique_analysis: bool = True

        self._is_running: bool = False
        self._pipeline_task: Optional[asyncio.Task[Any]] = None
        self._cancel_event = threading.Event()

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

        # Input videos section
        section_title = toga.Label(
            "Input Videos", style=Pack(font_weight="bold", margin=(10, 0, 6, 0))
        )
        content.add(section_title)

        button_row = toga.Box(style=Pack(direction="row", margin_bottom=6))
        self.btn_add_videos = toga.Button(
            "Add Videos",
            on_press=self.on_browse_video,
            style=Pack(margin_right=6, flex=1),
        )
        self.btn_clear_videos = toga.Button(
            "Clear Videos",
            on_press=self.on_clear_videos,
            enabled=False,
            style=Pack(flex=1),
        )
        button_row.add(self.btn_add_videos)
        button_row.add(self.btn_clear_videos)
        content.add(button_row)

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

        # Model selector (horizontal buttons)
        content.add(
            toga.Label(
                "Select Model", style=Pack(font_weight="bold", margin=(10, 0, 6, 0))
            )
        )

        self.model_button_row = toga.Box(
            style=Pack(direction="row", flex=0, margin_bottom=6)
        )
        content.add(self.model_button_row)

        self.model_hint_label = toga.Label(
            "Models are loaded from `barpath/models` if present. Add models there to see buttons here.",
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
        self.cancel_button = toga.Button(
            "Cancel",
            on_press=self.on_cancel_analysis,
            enabled=False,
            style=Pack(margin_right=6),
        )

        controls.add(self.run_button)
        controls.add(self.cancel_button)
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

        def _set(btn, tip, is_active: bool):
            btn.style.update(**(active_btn if is_active else inactive_btn))
            tip.style.update(**tip_style)
            # parent row is the tip's parent container; we can walk up via stored attribute if present
            # (we don't rely on it; background on button + text already indicates state)

        if files_ok and settings_ok and analyze_ok and analysis_ok:
            _set(self.tab_btn_files, self.tab_tip_files, active == "files")
            _set(self.tab_btn_settings, self.tab_tip_settings, active == "settings")
            _set(self.tab_btn_analyze, self.tab_tip_analyze, active == "analyze")
            _set(self.tab_btn_analysis, self.tab_tip_analysis, active == "analysis")

    # ----------------------------
    # Tab navigation (swap active page)
    # ----------------------------

    def _select_tab(self, tab: str) -> None:
        """Select a tab by removing and adding pages (eliminates redraw flashing)."""
        if tab not in ("files", "settings", "analyze", "analysis"):
            tab = "files"

        # Debounce redundant selections (avoids unnecessary churn)
        if getattr(self, "_current_tab", None) == tab:
            return
        self._current_tab = tab

        # Clear the page host and add only the selected page
        self.page_host.clear()

        if tab == "files":
            self.page_host.add(self.page_files)
        elif tab == "settings":
            self.page_host.add(self.page_settings)
        elif tab == "analyze":
            self.page_host.add(self.page_analyze)
        else:  # analysis
            self.page_host.add(self.page_analysis)
            self._render_analysis()

        # Update visual state of tabs
        self._apply_tab_styles(tab)

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
        model = self._resolve_selected_model()
        self._log("[bold]Configuration:[/bold]")
        self._log(f"  Input Videos: [cyan]{len(self.input_videos)}[/cyan]")
        if len(self.input_videos) <= 8:
            for i, vid in enumerate(self.input_videos, 1):
                self._log(f"    {i}. [dim]{vid.name}[/dim]")
        else:
            for i, vid in enumerate(self.input_videos[:5], 1):
                self._log(f"    {i}. [dim]{vid.name}[/dim]")
            self._log(f"    ... [dim]+{len(self.input_videos) - 5} more[/dim]")
        self._log(
            f"  Model:        [cyan]{model if model else '(not selected)'}[/cyan]"
        )
        self._log(f"  Lift Type:    [cyan]{self.lift_type}[/cyan]")
        self._log(f"  Output Dir:   [cyan]{self._effective_output_dir()}[/cyan]")
        self._log("")

    # ----------------------------
    # Model discovery + selection UI
    # ----------------------------

    def _populate_default_model_dir(self) -> None:
        models_dir = Path(__file__).parent / "models"
        if models_dir.exists() and models_dir.is_dir():
            self._populate_model_files(models_dir)

    def _populate_model_files(self, directory: Path) -> None:
        self.model_dir = directory

        pt_files = list(directory.glob("*.pt"))
        onnx_files = list(directory.glob("*.onnx"))
        openvino_dirs = [
            p
            for p in directory.iterdir()
            if p.is_dir() and "openvino" in p.name.lower()
        ]
        candidates = pt_files + onnx_files + openvino_dirs
        self.model_files = sorted(candidates, key=lambda p: p.name.lower())

        # If selection no longer valid, reset
        if self.model_files:
            if self.selected_model not in self.model_files:
                self.selected_model = self.model_files[0]
        else:
            self.selected_model = None

    def _refresh_settings_buttons(self) -> None:
        """Update selection visuals in-place to reduce flashing (avoid clear/rebuild)."""
        # Keep a stable set of buttons after first build
        if not hasattr(self, "_model_buttons"):
            self._model_buttons = {}  # type: ignore[attr-defined]
        if not hasattr(self, "_lift_buttons"):
            self._lift_buttons = {}  # type: ignore[attr-defined]

        # --- Model buttons ---
        # If empty, show a single placeholder (mounted once)
        if not self.model_files:
            if not hasattr(self, "_no_models_label"):
                self._no_models_label = toga.Label(  # type: ignore[attr-defined]
                    "(No models found)",
                    style=Pack(color="#5B6472", margin=6),
                )
                self.model_button_row.add(self._no_models_label)  # type: ignore[attr-defined]
            # Remove any existing model buttons (if any were previously rendered)
            for btn in list(self._model_buttons.values()):  # type: ignore[attr-defined]
                if btn in self.model_button_row.children:
                    self.model_button_row.remove(btn)
            self._model_buttons.clear()  # type: ignore[attr-defined]
        else:
            # Remove placeholder label if present
            if (
                hasattr(self, "_no_models_label")
                and self._no_models_label in self.model_button_row.children
            ):  # type: ignore[attr-defined]
                self.model_button_row.remove(self._no_models_label)  # type: ignore[attr-defined]

            desired = [p.name for p in self.model_files]

            # Remove buttons that are no longer needed
            for name in list(self._model_buttons.keys()):  # type: ignore[attr-defined]
                if name not in desired:
                    btn = self._model_buttons.pop(name)  # type: ignore[attr-defined]
                    if btn in self.model_button_row.children:
                        self.model_button_row.remove(btn)

            # Add missing buttons
            for model_path in self.model_files:
                name = model_path.name
                if name not in self._model_buttons:  # type: ignore[attr-defined]
                    btn = toga.Button(
                        name,
                        on_press=lambda w, mp=model_path: self._set_selected_model(mp),
                        style=self._pill_style(selected=False),
                    )
                    self._model_buttons[name] = btn  # type: ignore[attr-defined]
                    self.model_button_row.add(btn)

            # Update styles in place (no rebuild)
            for model_path in self.model_files:
                name = model_path.name
                btn = self._model_buttons.get(name)  # type: ignore[attr-defined]
                if btn is not None:
                    btn.style.update(
                        **self._pill_style_dict(
                            selected=(self.selected_model == model_path)
                        )
                    )

        # --- Lift buttons ---
        for lift in ("none", "clean", "snatch"):
            key = lift
            if key not in self._lift_buttons:  # type: ignore[attr-defined]
                btn = toga.Button(
                    lift.capitalize(),
                    on_press=lambda w, lt=lift: self._set_lift_type(lt),
                    style=self._pill_style(selected=False),
                )
                self._lift_buttons[key] = btn  # type: ignore[attr-defined]
                self.lift_button_row.add(btn)

        for lift in ("none", "clean", "snatch"):
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

    def _set_selected_model(self, model_path: Path) -> None:
        self.selected_model = model_path
        self._refresh_settings_buttons()
        self._log(f"[green]✓[/green] Selected model: [cyan]{model_path.name}[/cyan]")

    def _set_lift_type(self, lift_type: str) -> None:
        self.lift_type = lift_type
        self._refresh_settings_buttons()
        self._log(f"[green]✓[/green] Lift type: [cyan]{self.lift_type}[/cyan]")

    def _resolve_selected_model(self) -> Optional[Path]:
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

            self.btn_clear_videos.enabled = len(self.input_videos) > 0
            if added:
                self._log(f"[green]✓[/green] Added [cyan]{added}[/cyan] video(s)")
        except Exception as e:
            await self.main_window.dialog(  # type: ignore[attr-defined]
                toga.ErrorDialog("Error", f"Could not select file(s): {e}")
            )

    def on_clear_videos(self, widget: toga.Widget) -> None:
        self.input_videos.clear()
        self.video_list_box.clear()
        self.btn_clear_videos.enabled = False
        self._log("[yellow]![/yellow] Cleared all videos")

    def _add_video_row(self, video_path: Path) -> None:
        row = toga.Box(
            style=Pack(
                direction="row", margin_bottom=4, margin=6, background_color="#F2F3F7"
            )
        )

        remove_btn = toga.Button(
            "Remove",
            on_press=lambda w, vp=video_path: self.on_remove_video(w, vp),
            style=Pack(width=80, margin_right=8),
        )
        row.add(remove_btn)

        row.add(toga.Label(str(video_path), style=Pack(flex=1, color="#222")))

        self.video_list_box.add(row)

    def on_remove_video(self, widget: toga.Widget, video_path: Path) -> None:
        if video_path in self.input_videos:
            self.input_videos.remove(video_path)

        self.video_list_box.clear()
        for vp in self.input_videos:
            self._add_video_row(vp)

        self.btn_clear_videos.enabled = len(self.input_videos) > 0
        self._log(f"[yellow]–[/yellow] Removed: [dim]{video_path.name}[/dim]")

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

    # ----------------------------
    # Event handlers: Analyze
    # ----------------------------

    def on_run_analysis(self, widget: toga.Widget) -> None:
        if self._is_running:
            return

        if not self.input_videos:
            self._select_tab("files")
            self._log(
                "[bold red]ERROR[/bold red] Please add at least one video in Files."
            )
            return

        selected_model = self._resolve_selected_model()
        if not selected_model:
            self._select_tab("settings")
            self._log("[bold red]ERROR[/bold red] Please select a model in Settings.")
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
        self._cancel_event.clear()

        # Switch to Analyze tab
        self._select_tab("analyze")

        # Kick off background task
        self._pipeline_task = asyncio.create_task(self._run_pipeline_async())

    async def _run_pipeline_async(self) -> None:
        try:
            run_pipeline = _get_run_pipeline()
            selected_model = self._resolve_selected_model()
            if selected_model is None:
                raise RuntimeError("No model selected")

            is_batch = len(self.input_videos) > 1
            total_videos = len(self.input_videos)

            for video_idx, input_video in enumerate(self.input_videos, 1):
                if self._cancel_event.is_set():
                    break

                self._log(
                    f"[bold cyan]Processing video {video_idx}/{total_videos}[/bold cyan]: [dim]{input_video.name}[/dim]"
                )

                # video-specific output dir
                out_base = self._effective_output_dir()
                if is_batch:
                    video_output_dir = out_base / input_video.stem
                    video_output_dir.mkdir(parents=True, exist_ok=True)
                else:
                    video_output_dir = out_base
                    video_output_dir.mkdir(parents=True, exist_ok=True)

                # output video path (kept consistent with previous GUI behavior)
                output_video_path = (
                    video_output_dir / "output.mp4" if self.encode_video else None
                )

                # Consume generator updates
                for step_name, progress_value, message in run_pipeline(
                    input_video=str(input_video),
                    model_path=str(selected_model),
                    output_video=str(output_video_path) if output_video_path else None,
                    lift_type=self.lift_type,
                    output_dir=str(video_output_dir),
                    encode_video=self.encode_video,
                    technique_analysis=(self.lift_type != "none"),
                    cancel_event=self._cancel_event,
                ):
                    # Throttle frame spam
                    if "frame" not in str(message).lower() or progress_value is None:
                        # Rich-ish formatting similar to CLI
                        if progress_value is not None:
                            self._log(f"[dim]{step_name}[/dim] {message}")
                        else:
                            self._log(
                                f"[green]✓[/green] [dim]{step_name}[/dim] {message}"
                            )

                    if progress_value is not None:
                        # overall progress across all videos
                        video_progress = (video_idx - 1) / max(total_videos, 1)
                        step_progress = float(progress_value) / max(total_videos, 1)
                        overall_progress = video_progress + step_progress

                        self.progress_bar.value = int(overall_progress * 100)
                        self.progress_label.text = (
                            f"[{video_idx}/{total_videos}] {message}"
                        )
                    else:
                        self.progress_label.text = (
                            f"[{video_idx}/{total_videos}] ✓ {message}"
                        )

                    await asyncio.sleep(0.01)

                self._log(f"[green]✓[/green] Completed: [dim]{input_video.name}[/dim]")
                self._log("")

            if self._cancel_event.is_set():
                self._log("[yellow]![/yellow] Cancellation requested; stopping.")
                self.progress_label.text = "Cancelled"
                self.progress_bar.value = 0
                return

            self._log("[bold green]✓ All Videos Complete![/bold green]")
            self.progress_bar.value = 100
            self.progress_label.text = "Analysis complete!"

            # Enable view analysis if report exists (last video if batch)
            if is_batch and self.input_videos:
                last_video = self.input_videos[-1]
                analysis_path = (
                    self._effective_output_dir() / last_video.stem / "analysis.md"
                )
            else:
                analysis_path = self._effective_output_dir() / "analysis.md"

            if analysis_path.exists():
                self._log(
                    f"[green]✓[/green] Found report: [cyan]{analysis_path}[/cyan]"
                )
                # Refresh the analysis tab
                self._render_analysis()
            else:
                self._log(
                    f"[yellow]![/yellow] No analysis report found at: [dim]{analysis_path}[/dim]"
                )

        except InterruptedError:
            self._log("\n[yellow]![/yellow] Pipeline cancelled.")
            self.progress_label.text = "Cancelled"
            self.progress_bar.value = 0
        except Exception as e:
            self._log(f"\n[bold red]ERROR[/bold red] Pipeline failed: {e}")
            import traceback

            self._log(traceback.format_exc())
            self.progress_label.text = f"Error: {e}"
        finally:
            self._is_running = False
            self.run_button.enabled = True
            self.cancel_button.enabled = False
            self._pipeline_task = None

    def on_cancel_analysis(self, widget: toga.Widget) -> None:
        if self._is_running:
            self._log("[yellow]![/yellow] Cancellation requested...")
            self._cancel_event.set()
            self.cancel_button.enabled = False

    # ----------------------------
    # View analysis (unchanged logic, lightly styled)
    # ----------------------------
    # Utility
    # ----------------------------

    def _validate_environment(self) -> Tuple[bool, str]:
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
