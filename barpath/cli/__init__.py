"""
Command-line interface for barpath.

This module provides the CLI entry point for the barpath pipeline.

Usage:
    $ barpath --input_video lift.mp4 --model models/best.pt --output_video out.mp4
"""

from .barpath_cli import main

__all__ = ['main']