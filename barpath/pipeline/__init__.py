"""
barpath - Offline Weightlifting Technique Analysis
"""
__version__ = "1.0.0"

# This package exposes pipeline steps as standalone scripts (1_collect_data.py, etc.).
# We intentionally do not import them here because their filenames are not valid Python
# identifiers. Consumers (CLI/GUI) should load the numbered scripts dynamically.
__all__ = ['__version__']