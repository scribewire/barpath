"""
Standardized logging for the barpath pipeline.

Provides consistent logging across all pipeline steps with configurable
verbosity levels.
"""

import logging
import sys
from typing import Optional

from config import LOG_LEVEL


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger for a pipeline module.
    
    Args:
        name: Logger name (typically __name__)
        level: Log level override (defaults to config.LOG_LEVEL)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S"
            )
        )
        logger.addHandler(handler)
    
    log_level = level or LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    return logger


def log_step_start(logger: logging.Logger, step_name: str):
    """Log the start of a pipeline step."""
    logger.info(f"--- {step_name} ---")


def log_step_complete(logger: logging.Logger, step_name: str, details: str = ""):
    """Log the completion of a pipeline step."""
    msg = f"{step_name} Complete"
    if details:
        msg += f". {details}"
    logger.info(msg)


def log_warning(logger: logging.Logger, message: str):
    """Log a warning message."""
    logger.warning(message)


def log_error(logger: logging.Logger, message: str, exc: Optional[Exception] = None):
    """Log an error message with optional exception info."""
    if exc:
        logger.error(f"{message}: {exc}")
    else:
        logger.error(message)


def log_progress(logger: logging.Logger, current: int, total: int, message: str = ""):
    """Log progress for long-running operations."""
    percent = (current / total * 100) if total > 0 else 0
    msg = f"Progress: {current}/{total} ({percent:.1f}%)"
    if message:
        msg += f" - {message}"
    logger.debug(msg)


_pipeline_logger: Optional[logging.Logger] = None


def get_pipeline_logger() -> logging.Logger:
    """Get the global pipeline logger."""
    global _pipeline_logger
    if _pipeline_logger is None:
        _pipeline_logger = get_logger("barpath")
    return _pipeline_logger
