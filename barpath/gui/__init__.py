"""
Graphical user interface for barpath.

This module provides the GUI entry point for the barpath pipeline.
Built with PyQt6 for a modern, cross-platform interface.

Usage:
    $ barpath-gui
    
Or from Python:
    >>> from gui import main
    >>> main()
"""

from .barpath_gui import main, BarpathGUI

__all__ = ['main', 'BarpathGUI']