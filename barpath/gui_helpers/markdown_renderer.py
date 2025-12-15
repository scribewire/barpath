#!/usr/bin/env python3
"""Helper module for rendering Markdown to HTML.

This module handles:
- Converting Markdown to clean, semantic HTML
- Loading and rendering HTML templates for analysis display
- Managing analysis content
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


class MarkdownRenderer:
    """Handles Markdown to HTML conversion and analysis rendering."""

    def __init__(self):
        """Initialize the markdown renderer."""
        self._template_cache: Optional[str] = None

    def render_markdown_file(self, markdown_path: Path) -> str:
        """Render a markdown file to HTML using the analysis template.

        Args:
            markdown_path: Path to the markdown file to render

        Returns:
            Complete HTML document string
        """
        try:
            with open(markdown_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()

            html_content = self._markdown_to_html(markdown_content)
            return self._render_with_template(html_content)
        except FileNotFoundError:
            return self._render_no_analysis()
        except Exception:
            return self._render_error()

    def render_no_analysis(self) -> str:
        """Render the 'no analysis' placeholder page."""
        return self._render_no_analysis()

    def _markdown_to_html(self, markdown: str) -> str:
        """Convert markdown text to HTML.

        Supports:
        - Headers (# ## ### ####)
        - Bold (**text** or __text__)
        - Italic (*text* or _text_)
        - Lists (ordered and unordered)
        - Links [text](url)
        - Code blocks (```code```)
        - Inline code (`code`)
        - Blockquotes (> text)
        - Horizontal rules (---)
        - Paragraphs

        Args:
            markdown: Markdown text to convert

        Returns:
            HTML string
        """
        html = markdown

        # Escape HTML entities first
        html = html.replace("&", "&amp;")
        html = html.replace("<", "&lt;")
        html = html.replace(">", "&gt;")

        # Code blocks (must come before inline code)
        def replace_code_block(match):
            code = match.group(1)
            # Unescape for code blocks
            code = code.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
            return f"<pre><code>{code}</code></pre>"

        html = re.sub(
            r"```(?:\w+)?\n(.*?)```", replace_code_block, html, flags=re.DOTALL
        )

        # Inline code
        html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)

        # Headers
        html = re.sub(r"^#### (.*?)$", r"<h4>\1</h4>", html, flags=re.MULTILINE)
        html = re.sub(r"^### (.*?)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.*?)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"^# (.*?)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)

        # Bold (must come before italic)
        html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"__(.*?)__", r"<strong>\1</strong>", html)

        # Italic
        html = re.sub(r"\*(.*?)\*", r"<em>\1</em>", html)
        html = re.sub(r"_(.*?)_", r"<em>\1</em>", html)

        # Links
        html = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r'<a href="\2">\1</a>', html)

        # Horizontal rules
        html = re.sub(r"^---+$", r"<hr>", html, flags=re.MULTILINE)
        html = re.sub(r"^\*\*\*+$", r"<hr>", html, flags=re.MULTILINE)

        # Blockquotes
        def replace_blockquote(match):
            lines = match.group(0).split("\n")
            content = "\n".join(
                line.lstrip("> ").strip() for line in lines if line.strip()
            )
            return f"<blockquote>{content}</blockquote>"

        html = re.sub(r"^>.*(?:\n^>.*)*", replace_blockquote, html, flags=re.MULTILINE)

        # Lists - Unordered
        def replace_unordered_list(match):
            lines = match.group(0).split("\n")
            items = []
            for line in lines:
                if line.strip():
                    item = re.sub(r"^[\*\-\+]\s+", "", line.strip())
                    items.append(f"<li>{item}</li>")
            return f"<ul>{''.join(items)}</ul>"

        html = re.sub(
            r"^[\*\-\+]\s+.*(?:\n^[\*\-\+]\s+.*)*",
            replace_unordered_list,
            html,
            flags=re.MULTILINE,
        )

        # Lists - Ordered
        def replace_ordered_list(match):
            lines = match.group(0).split("\n")
            items = []
            for line in lines:
                if line.strip():
                    item = re.sub(r"^\d+\.\s+", "", line.strip())
                    items.append(f"<li>{item}</li>")
            return f"<ol>{''.join(items)}</ol>"

        html = re.sub(
            r"^\d+\.\s+.*(?:\n^\d+\.\s+.*)*",
            replace_ordered_list,
            html,
            flags=re.MULTILINE,
        )

        # Paragraphs - split by double newlines, but avoid wrapping existing block elements
        lines = html.split("\n\n")
        processed_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Don't wrap if it's already a block element
            if not re.match(r"^<(h[1-6]|ul|ol|pre|blockquote|hr|table)", line):
                processed_lines.append(f"<p>{line}</p>")
            else:
                processed_lines.append(line)

        html = "\n".join(processed_lines)

        return html

    def _render_with_template(self, html_content: str) -> str:
        """Render HTML content with the analysis template.

        Args:
            html_content: The HTML content to insert

        Returns:
            Complete HTML document string
        """
        template_path = Path(__file__).parent / "templates" / "analysis_viewer.html"
        try:
            if self._template_cache is None:
                with open(template_path, "r", encoding="utf-8") as f:
                    self._template_cache = f.read()

            # Replace the placeholder with actual content
            doc = self._template_cache.replace(
                '<!-- Analysis content will be inserted here -->\n    <div class="no-analysis">\n      <h2>No Analysis Available</h2>\n      <p>Run an analysis to see results here.</p>\n    </div>',
                html_content,
            )
            return doc
        except Exception:
            return self._fallback_template(html_content)

    def _render_no_analysis(self) -> str:
        """Render the template with no analysis message."""
        template_path = Path(__file__).parent / "templates" / "analysis_viewer.html"
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return self._fallback_template(
                '<div class="no-analysis"><h2>No Analysis Available</h2><p>Run an analysis to see results here.</p></div>'
            )

    def _render_error(self) -> str:
        """Render an error message."""
        error_html = '<div class="no-analysis"><h2>Error Loading Analysis</h2><p>There was an error loading the analysis file.</p></div>'
        return self._render_with_template(error_html)

    def _fallback_template(self, html_content: str) -> str:
        """Return a minimal inline HTML template as fallback.

        Args:
            html_content: The HTML content to insert

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
.analysis-container {{ height: 100%; width: 100%; background: #ffffff; padding: 24px; overflow-y: auto; line-height: 1.6; }}
.analysis-content {{ max-width: 800px; margin: 0 auto; }}
h1 {{ font-size: 28px; font-weight: 700; color: #111827; margin-bottom: 24px; padding-bottom: 12px; border-bottom: 2px solid #e5e7eb; }}
h2 {{ font-size: 22px; font-weight: 600; color: #1f2937; margin-top: 32px; margin-bottom: 16px; }}
h3 {{ font-size: 18px; font-weight: 600; color: #374151; margin-top: 24px; margin-bottom: 12px; }}
p {{ margin-bottom: 16px; color: #374151; }}
ul, ol {{ margin-left: 24px; margin-bottom: 16px; color: #374151; }}
li {{ margin-bottom: 8px; }}
strong {{ font-weight: 600; color: #111827; }}
code {{ font-family: 'Consolas', 'Monaco', 'Courier New', monospace; background: #f3f4f6; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; color: #dc2626; }}
.no-analysis {{ text-align: center; padding: 48px 24px; color: #6b7280; }}
</style>
</head>
<body>
<div class="analysis-container">
  <div class="analysis-content">{html_content}</div>
</div>
</body>
</html>"""
