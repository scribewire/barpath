"""
Fast Analysis: Bar Path Similarity using Dynamic Time Warping.

This module provides DTW-based trajectory comparison for holistic
bar path similarity scoring and temporal deviance curves.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from dtw import dtw

    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    dtw = None  # type: ignore
    print("Warning: dtw-python not installed. Fast Analysis will be disabled.")


def load_fast_analysis_model(
    model_dir: Path,
) -> Tuple[Optional[List[np.ndarray]], Optional[Dict]]:
    """
    Load pro reference trajectories and config for Fast Analysis.

    Args:
        model_dir: Path to model directory containing:
            - fast_analysis_trajectories.npy
            - fast_analysis_config.json

    Returns:
        Tuple of (trajectories, config) or (None, None) if not found.
    """
    trajectories_path = model_dir / "fast_analysis_trajectories.npy"
    config_path = model_dir / "fast_analysis_config.json"

    if not trajectories_path.exists() or not config_path.exists():
        return None, None

    try:
        trajectories_data = np.load(trajectories_path, allow_pickle=True)
        if isinstance(trajectories_data, np.ndarray):
            if trajectories_data.dtype == object:
                trajectories = list(trajectories_data)
            else:
                trajectories = [trajectories_data]
        else:
            trajectories = list(trajectories_data)

        with open(config_path, "r") as f:
            config = json.load(f)

        return trajectories, config
    except Exception as e:
        print(f"Error loading Fast Analysis model: {e}")
        return None, None


def distance_to_similarity(normalized_distance: float, scale: float = 1.0) -> float:
    """
    Convert DTW normalized distance to 0-1 similarity.

    scale is calibrated from the distribution of pro-vs-pro distances:
      scale = median(all pairwise pro distances)

    This means a user who matches as well as pros match each other scores ~0.5,
    and closer matches approach 1.0.
    """
    return float(np.exp(-normalized_distance / scale))


def extract_per_frame_cost(
    dtw_result: Dict,
    user_trajectory: np.ndarray,
    pro_trajectory: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Extract per-frame similarity from the DTW warping path.

    The warping path maps each user frame to one or more pro frames.
    For each user frame, the cost is the Euclidean distance to its
    aligned pro frame(s). This is converted to a similarity value.

    Returns: np.ndarray of shape (N_user_frames,), values in [0, 1].
    """
    n_user = len(user_trajectory)
    frame_costs = np.zeros(n_user)
    frame_counts = np.zeros(n_user)

    index1 = dtw_result.get("index1", [])
    index2 = dtw_result.get("index2", [])

    for i1, i2 in zip(index1, index2):
        if i1 < n_user and i2 < len(pro_trajectory):
            cost = np.linalg.norm(user_trajectory[i1] - pro_trajectory[i2])
            frame_costs[i1] += cost
            frame_counts[i1] += 1

    frame_counts[frame_counts == 0] = 1
    avg_costs = frame_costs / frame_counts

    temporal_similarity = np.exp(-avg_costs / scale)

    return temporal_similarity


def run_fast_analysis(
    user_trajectory: np.ndarray,
    pro_trajectories: List[np.ndarray],
    config: Optional[Dict] = None,
    top_k: int = 5,
) -> Dict:
    """
    Compare user trajectory against all pro references.

    Return similarity score and per-frame deviance from best match.

    Args:
        user_trajectory: shape (N, 2), normalized barbell trajectory
        pro_trajectories: list of shape (M_i, 2), normalized pro trajectories
        config: dict with 'scale' and 'top_k' parameters
        top_k: number of best matches to consider

    Returns:
        Dict with:
            - similarity: overall 0-1 similarity score
            - temporal_similarity: per-frame similarity curve
            - best_match_distance: normalized DTW distance of best match
            - top_k_distances: list of top-k normalized distances
            - available: bool indicating if analysis was possible
    """
    if not DTW_AVAILABLE:
        return {
            "similarity": None,
            "temporal_similarity": None,
            "best_match_distance": None,
            "top_k_distances": [],
            "available": False,
            "error": "dtw-python not installed",
        }

    if not pro_trajectories or len(pro_trajectories) == 0:
        return {
            "similarity": None,
            "temporal_similarity": None,
            "best_match_distance": None,
            "top_k_distances": [],
            "available": False,
            "error": "No pro trajectories available",
        }

    if config is None:
        config = {}

    scale = config.get("scale", 1.0)
    top_k = config.get("top_k", top_k)

    results = []

    for pro_traj in pro_trajectories:
        try:
            alignment = dtw(  # type: ignore
                user_trajectory,
                pro_traj,
                dist_method="euclidean",
                keep_internals=True,
            )
            results.append(
                {
                    "distance": float(alignment.distance),  # type: ignore
                    "normalized_distance": float(alignment.normalizedDistance),  # type: ignore
                    "index1": alignment.index1,  # type: ignore
                    "index2": alignment.index2,  # type: ignore
                    "alignment": alignment,
                }
            )
        except Exception as e:
            print(f"DTW comparison failed: {e}")
            continue

    if not results:
        return {
            "similarity": None,
            "temporal_similarity": None,
            "best_match_distance": None,
            "top_k_distances": [],
            "available": False,
            "error": "All DTW comparisons failed",
        }

    results.sort(key=lambda r: r["normalized_distance"])
    best_matches = results[:top_k]

    best = best_matches[0]
    similarity = distance_to_similarity(best["normalized_distance"], scale)

    best_traj_idx = pro_trajectories.index(
        min(pro_trajectories, key=lambda t: len(t) if len(t) > 0 else float("inf"))
    )
    for i, pro_traj in enumerate(pro_trajectories):
        if len(pro_traj) == len(best_matches[0].get("alignment", {}).get("index2", [])):
            best_traj_idx = i
            break

    temporal_curve = extract_per_frame_cost(
        best,
        user_trajectory,
        pro_trajectories[best_traj_idx]
        if best_traj_idx < len(pro_trajectories)
        else pro_trajectories[0],
        scale,
    )

    return {
        "similarity": similarity,
        "temporal_similarity": temporal_curve,
        "best_match_distance": best["normalized_distance"],
        "top_k_distances": [m["normalized_distance"] for m in best_matches],
        "available": True,
    }
