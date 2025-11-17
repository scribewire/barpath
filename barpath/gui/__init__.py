"""Graphical user interface for barpath.

This package exposes a Toga-based GUI as the primary entrypoint.

Usage (once `toga` is installed):

    $ python -m barpath.gui.barpath_gui
"""

from .barpath_gui import main, BarpathTogaApp

__all__ = ["main", "BarpathTogaApp"]