#!/usr/bin/env python3
"""Helper module for rendering Rich markup as HTML for the log viewer.

This module handles:
- Converting Rich-style markup tags to HTML
- Loading and rendering HTML templates
- Managing log line formatting
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import List


class LogRenderer:
    """Handles Rich markup to HTML conversion and log rendering."""

    def __init__(self):
        """Initialize the log renderer with empty log lines."""
        self._log_html_lines: List[str] = []
        self._log_seq: int = 0
        self._template_cache: str | None = None

    def add_log_line(self, text: str) -> None:
        """Add a Rich-markup line to the log."""
        self._log_seq += 1
        self._log_html_lines.append(self._rich_markup_to_html(text))

    def clear_logs(self) -> None:
        """Clear all log lines."""
        self._log_html_lines.clear()
        self._log_seq = 0

    def render_html(self) -> str:
        """Render the full HTML document with all log lines."""
        lines_html = "\n".join(self._log_html_lines)

        # Load the HTML template from file
        template_path = Path(__file__).parent / "templates" / "log_viewer.html"
        try:
            if self._template_cache is None:
                with open(template_path, "r", encoding="utf-8") as f:
                    self._template_cache = f.read()

            # Insert the log content into the template
            doc = self._template_cache.replace(
                "<!-- Log content will be inserted here -->", lines_html
            )
            return doc
        except Exception:
            # Fallback: use inline template if file can't be loaded
            return self._fallback_template(lines_html)

    def _rich_markup_to_html(self, text: str) -> str:
        """Convert Rich markup tags to HTML for beautiful display.

        Supported tags:
        - [bold]...[/bold] or [bold]...[/]
        - [dim]...[/dim]
        - [cyan], [green], [yellow], [red], [blue], [magenta]
        - Combined tags like [bold green]...[/bold green]

        Args:
            text: Text with Rich-style markup tags

        Returns:
            HTML string with CSS classes for styling
        """
        # Escape HTML first
        s = html.escape(text)

        # Map Rich tags to CSS classes
        tag_map = {
            "bold": "bold",
            "dim": "dim",
            "cyan": "cyan",
            "green": "green",
            "yellow": "yellow",
            "red": "red",
            "blue": "blue",
            "magenta": "magenta",
        }

        # Handle combined opening tags like [bold green]
        def replace_combined_open(match):
            parts = match.group(1).split()
            classes = [tag_map[p] for p in parts if p in tag_map]
            if classes:
                return f'<span class="{" ".join(classes)}">'
            return match.group(0)

        # Replace combined opening tags first
        s = re.sub(r"\[([a-zA-Z ]+)\]", replace_combined_open, s)

        # Handle all closing tags (including combined ones like [/bold green])
        # Replace any [/anything] or [/anything anything] with closing span
        s = re.sub(r"\[/[^\]]+\]", "</span>", s)

        # Also handle the generic [/] closing tag
        s = s.replace("[/]", "</span>")

        return f'<div class="line">{s}</div>'

    def _fallback_template(self, lines_html: str) -> str:
        """Return a minimal inline HTML template as fallback.

        Args:
            lines_html: The rendered log lines HTML

        Returns:
            Complete HTML document string
        """
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{ height: 100%; width: 100%; overflow: hidden; }}
body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #FFFFFF; color: #1f2937; }}
.log-container {{ height: 100%; width: 100%; background: #f9fafb; padding: 12px; overflow-y: auto; border: 1px solid #e5e7eb; }}
.line {{ font-family: 'Consolas', 'Monaco', 'Courier New', monospace; font-size: 13px; margin: 3px 0; white-space: pre-wrap; word-wrap: break-word; line-height: 1.5; }}
.bold {{ font-weight: 700; }}
.dim {{ opacity: 0.5; color: #6b7280; }}
.cyan {{ color: #0891b2; }}
.green {{ color: #16a34a; }}
.yellow {{ color: #ca8a04; }}
.red {{ color: #dc2626; }}
.blue {{ color: #2563eb; }}
.magenta {{ color: #9333ea; }}
</style>
</head>
<body>
<div class="log-container" id="log">{lines_html}</div>
<script>
var log = document.getElementById('log');
log.scrollTop = log.scrollHeight;
</script>
</body>
</html>"""
