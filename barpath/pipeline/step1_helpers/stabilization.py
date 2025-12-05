"""
Stabilization helpers for Step 1: Data Collection.

This module contains functions for estimating camera motion (shake/stabilization)
using optical flow and background feature tracking.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


# --- Stabilization Parameters ---
class StabilizationParams:
    """Parameters for the global motion stabilization model."""

    # Feature detection parameters
    feature_max_corners = 200
    feature_quality_level = 0.01
    feature_min_distance = 10
    feature_block_size = 7

    # Lucas-Kanade optical flow parameters (for feature tracking)
    lk_win_size = (21, 21)
    lk_max_level = 3
    lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    # Minimum number of matched features for motion estimation
    min_inliers = 10


def create_background_mask(segmentation_mask: np.ndarray) -> np.ndarray:
    """
    Create a background mask for feature detection.

    Inverts the segmentation mask (1 for background, 0 for foreground) and applies
    erosion to create a safety margin around the person.

    Args:
        segmentation_mask (np.ndarray): Binary segmentation mask from MediaPipe
                                       (1 for person, 0 for background)

    Returns:
        np.ndarray: Background mask (uint8) suitable for cv2.goodFeaturesToTrack
    """
    # Invert segmentation mask: 1 for background, 0 for foreground
    background_mask = (1 - segmentation_mask) * 255
    background_mask = background_mask.astype(np.uint8)

    # Dilate the foreground mask to create a safety margin
    kernel = np.ones((15, 15), np.uint8)
    background_mask = cv2.erode(background_mask, kernel, iterations=1)

    return background_mask


def detect_features(
    gray: np.ndarray,
    background_mask: Optional[np.ndarray] = None,
    params: Optional[StabilizationParams] = None,
) -> Optional[np.ndarray]:
    """
    Detect good features to track in the image.

    Uses cv2.goodFeaturesToTrack to find corner features suitable for
    optical flow tracking. Can optionally mask out foreground regions.

    Args:
        gray (np.ndarray): Grayscale image
        background_mask (np.ndarray, optional): Binary mask (255 for trackable regions)
        params (StabilizationParams, optional): Parameters for feature detection

    Returns:
        np.ndarray: Array of detected feature points, or None if no features found
    """
    if params is None:
        params = StabilizationParams()

    features = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=params.feature_max_corners,
        qualityLevel=params.feature_quality_level,
        minDistance=params.feature_min_distance,
        mask=background_mask,
        blockSize=params.feature_block_size,
    )

    return features


def track_features(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_features: np.ndarray,
    params: Optional[StabilizationParams] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Track features from previous frame to current frame using Lucas-Kanade optical flow.

    Args:
        prev_gray (np.ndarray): Grayscale image from previous frame
        curr_gray (np.ndarray): Grayscale image from current frame
        prev_features (np.ndarray): Features detected in previous frame
        params (StabilizationParams, optional): Parameters for optical flow

    Returns:
        tuple: (current_features, status, error) or (None, None, None) if tracking fails
    """
    if params is None:
        params = StabilizationParams()

    if prev_features is None or len(prev_features) == 0:
        return None, None, None

    try:
        curr_features, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_features,
            None,  # type: ignore[arg-type]
            winSize=params.lk_win_size,
            maxLevel=params.lk_max_level,
            criteria=params.lk_criteria,
        )
        return curr_features, status, err
    except cv2.error:
        return None, None, None


def estimate_motion(
    prev_features: np.ndarray,
    curr_features: np.ndarray,
    status: np.ndarray,
    params: Optional[StabilizationParams] = None,
) -> Tuple[float, float, Optional[np.ndarray]]:
    """
    Estimate global motion (translation) from tracked features.

    Uses median displacement of matched features to robustly estimate
    camera motion, ignoring outliers.

    Args:
        prev_features (np.ndarray): Features from previous frame
        curr_features (np.ndarray): Features tracked to current frame
        status (np.ndarray): Status array indicating which tracks succeeded
        params (StabilizationParams, optional): Parameters with min_inliers threshold

    Returns:
        tuple: (dx, dy, good_new_features) where:
            - dx, dy: Estimated horizontal and vertical motion in pixels
            - good_new_features: Array of successfully tracked features
    """
    if params is None:
        params = StabilizationParams()

    # Select good matches
    good_new = curr_features[status.flatten() == 1]
    good_old = prev_features[status.flatten() == 1]

    # Estimate translation-only motion using median displacement
    if len(good_new) >= params.min_inliers:
        try:
            displacements = good_new - good_old
            median_dx = float(np.median(displacements[:, 0, 0]))
            median_dy = float(np.median(displacements[:, 0, 1]))
            return median_dx, median_dy, good_new
        except (cv2.error, ValueError, IndexError):
            return 0.0, 0.0, None

    return 0.0, 0.0, None


def update_features(
    curr_features: Optional[np.ndarray],
    new_features: Optional[np.ndarray],
    min_features: int = 50,
) -> Optional[np.ndarray]:
    """
    Update feature array by merging current features with newly detected ones.

    When the number of tracked features drops below the minimum threshold,
    new features are detected and added to maintain tracking quality.

    Args:
        curr_features (np.ndarray): Currently tracked features, or None
        new_features (np.ndarray): Newly detected features
        min_features (int): Minimum number of features to maintain

    Returns:
        np.ndarray: Updated features array
    """
    if curr_features is None or len(curr_features) < min_features:
        if new_features is not None:
            if curr_features is not None:
                # Merge with existing features
                return np.vstack((curr_features, new_features))
            else:
                return new_features

    return curr_features
