"""
Shared utility functions for the barpath pipeline.

This module contains helper functions used across multiple pipeline steps
to avoid code duplication and maintain consistency.
"""

import numpy as np
import os
import sys
from pathlib import Path


# ============================================================================
# GEOMETRIC CALCULATIONS
# ============================================================================

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle (in degrees) between three 2D points.
    
    Args:
        p1 (np.ndarray): First point [x, y]
        p2 (np.ndarray): Vertex point [x, y]
        p3 (np.ndarray): Third point [x, y]
    
    Returns:
        float: Angle in degrees, or np.nan if calculation fails
    
    Example:
        >>> p1 = np.array([0, 0])
        >>> p2 = np.array([1, 0])
        >>> p3 = np.array([1, 1])
        >>> calculate_angle(p1, p2, p3)
        90.0
    """
    # Check for NaN values
    if np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any():
        return np.nan
    
    # Calculate vectors from vertex (p2) to the other points
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calculate dot product and norms
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    # Handle numerical instability or zero-length vectors
    if norm == 0:
        return np.nan
    
    # Clamp cosine value to [-1, 1] to avoid domain errors
    cosine_angle = np.clip(dot / norm, -1.0, 1.0)
    
    # Return angle in degrees
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def calculate_lifter_angle(left_shoulder, right_shoulder):
    """
    Calculate the lifter's orientation angle using shoulder positions.
    
    Uses the (x, z) coordinates to determine if the lifter is facing
    straight (90°) or at an angle to the camera.
    
    Args:
        left_shoulder (tuple): (x, y, z, visibility) for left shoulder
        right_shoulder (tuple): (x, y, z, visibility) for right shoulder
    
    Returns:
        float: Orientation angle in degrees (90° = perpendicular to camera)
    """
    if left_shoulder is None or right_shoulder is None:
        return np.nan
    
    try:
        # Use (x, z) coordinates (indices 0 and 2)
        delta_x = left_shoulder[0] - right_shoulder[0]
        delta_z = left_shoulder[2] - right_shoulder[2]
        
        angle_rad = np.arctan2(delta_z, delta_x)
        angle_deg = 90 - abs(np.degrees(angle_rad))
        return angle_deg
    except (IndexError, TypeError):
        return np.nan


# ============================================================================
# DATA VALIDATION
# ============================================================================

def check_file_exists(filepath, file_description="File"):
    """
    Check if a file exists and exit with error message if not.
    
    Args:
        filepath (str): Path to check
        file_description (str): Human-readable description for error message
    
    Raises:
        SystemExit: If file doesn't exist
    """
    if not os.path.exists(filepath):
        print(f"❌ Error: {file_description} not found at '{filepath}'")
        sys.exit(1)


def validate_video_path(video_path):
    """
    Validate that video path exists and has valid extension.
    
    Args:
        video_path (str): Path to video file
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at '{video_path}'")
        return False
    
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    ext = os.path.splitext(video_path)[1].lower()
    
    if ext not in valid_extensions:
        print(f"❌ Error: Unsupported video format '{ext}'")
        print(f"   Supported formats: {', '.join(valid_extensions)}")
        return False
    
    return True


def validate_model_path(model_path):
    """
    Validate that model path exists and is a valid YOLO model.
    
    Args:
        model_path (str): Path to model file
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at '{model_path}'")
        print("\n💡 Tip: Download models with:")
        print("   python models/download_models.py")
        return False
    
    if not model_path.endswith('.pt'):
        print(f"❌ Error: Model must be a .pt file, got '{model_path}'")
        return False
    
    # Check file size (should be >1 MB for real model, not a pointer)
    file_size = os.path.getsize(model_path)
    if file_size < 1_000_000:  # Less than 1 MB
        print(f"⚠️  Warning: Model file is only {file_size/1024:.1f} KB")
        print("   This might be a Git LFS pointer file, not the actual model.")
        print("   Run: git lfs pull")
        return False
    
    return True


# ============================================================================
# COORDINATE TRANSFORMATIONS
# ============================================================================

def normalize_to_pixel_coords(normalized_x, normalized_y, frame_width, frame_height):
    """
    Convert normalized coordinates [0, 1] to pixel coordinates.
    
    Args:
        normalized_x (float): X coordinate normalized to [0, 1]
        normalized_y (float): Y coordinate normalized to [0, 1]
        frame_width (int): Frame width in pixels
        frame_height (int): Frame height in pixels
    
    Returns:
        tuple: (pixel_x, pixel_y) or (np.nan, np.nan) if inputs are invalid
    """
    if np.isnan(normalized_x) or np.isnan(normalized_y):
        return np.nan, np.nan
    
    pixel_x = normalized_x * frame_width
    pixel_y = normalized_y * frame_height
    
    return pixel_x, pixel_y


def pixel_to_normalized_coords(pixel_x, pixel_y, frame_width, frame_height):
    """
    Convert pixel coordinates to normalized coordinates [0, 1].
    
    Args:
        pixel_x (float): X coordinate in pixels
        pixel_y (float): Y coordinate in pixels
        frame_width (int): Frame width in pixels
        frame_height (int): Frame height in pixels
    
    Returns:
        tuple: (normalized_x, normalized_y)
    """
    normalized_x = pixel_x / frame_width
    normalized_y = pixel_y / frame_height
    
    return normalized_x, normalized_y


# ============================================================================
# PATH UTILITIES
# ============================================================================

def ensure_dir_exists(directory):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Directory path to create
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_project_root():
    """
    Get the project root directory.
    
    Returns:
        Path: Project root directory
    """
    return Path(__file__).parent.parent


def get_models_dir():
    """
    Get the models directory path.
    
    Returns:
        Path: Models directory
    """
    return get_project_root() / "models"


# ============================================================================
# PROGRESS & LOGGING
# ============================================================================

def print_header(text, width=70):
    """
    Print a formatted header for console output.
    
    Args:
        text (str): Header text
        width (int): Total width of header
    """
    print("\n" + "=" * width)
    padding = (width - len(text) - 2) // 2
    print(" " * padding + text)
    print("=" * width + "\n")


def print_section(text):
    """
    Print a section divider.
    
    Args:
        text (str): Section text
    """
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print('─' * 70)


def print_success(text):
    """Print a success message with checkmark."""
    print(f"✅ {text}")


def print_warning(text):
    """Print a warning message."""
    print(f"⚠️  {text}")


def print_error(text):
    """Print an error message."""
    print(f"❌ {text}")


def print_info(text):
    """Print an info message."""
    print(f"ℹ️  {text}")


# ============================================================================
# STATISTICS & METRICS
# ============================================================================

def calculate_detection_rate(detected_count, total_count):
    """
    Calculate detection rate percentage.
    
    Args:
        detected_count (int): Number of successful detections
        total_count (int): Total number of frames
    
    Returns:
        float: Detection rate as percentage
    """
    if total_count == 0:
        return 0.0
    return (detected_count / total_count) * 100


def format_file_size(size_bytes):
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes (int): Size in bytes
    
    Returns:
        str: Formatted size (e.g., "45.23 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


# ============================================================================
# CONSTANTS
# ============================================================================

# MediaPipe landmark names (used across pipeline)
LANDMARK_NAMES = {
    'left_shoulder', 'right_shoulder', 
    'left_hip', 'right_hip', 
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 
    'left_elbow', 'right_elbow', 
    'left_wrist', 'right_wrist',
}

# Color scheme for visualization
COLOR_SCHEME = {
    "Torso": (255, 255, 0),      # Cyan
    "Left Arm": (0, 165, 255),   # Orange
    "Right Arm": (0, 255, 255),  # Yellow
    "Left Leg": (255, 0, 128),   # Purple
    "Right Leg": (0, 255, 0),    # Green
    "Barbell Box": (255, 0, 255),# Magenta
    "Barbell Path": (0, 0, 255), # Red
}