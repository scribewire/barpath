#!/usr/bin/env python3
"""
Test script to validate and visualize the improved stabilization implementation.

This script processes a video and generates diagnostic visualizations showing:
1. Feature tracking quality
2. Transform stability
3. Comparison of stabilization accuracy
4. Background/foreground separation quality

Usage:
    python test_stabilization.py --video path/to/video.mp4 --model path/to/model.pt

Output:
    - stabilization_diagnostics.pkl: Raw diagnostic data
    - stabilization_report.txt: Text report with statistics
    - stabilization_viz.mp4: Video with diagnostic overlays (optional)
"""

import argparse
import pickle
import sys
from pathlib import Path

try:
    import cv2
    import mediapipe as mp
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install opencv-python numpy mediapipe")
    sys.exit(1)


def test_stabilization(video_path, max_frames=300):
    """
    Test the stabilization implementation and collect diagnostics.

    Args:
        video_path: Path to input video
        max_frames: Maximum number of frames to process (for quick testing)

    Returns:
        dict: Diagnostic data including feature counts, inliers, transforms, etc.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)

    print(f"Video: {frame_width}x{frame_height} @ {fps:.1f} fps")
    print(f"Processing {total_frames} frames...")

    # Initialize MediaPipe Pose with segmentation
    mp_pose = mp.solutions.pose.Pose(  # type: ignore[attr-defined]
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        enable_segmentation=True,
    )

    # Stabilization parameters (same as improved implementation)
    prev_gray = None
    prev_background_features = None

    feature_max_corners = 300
    feature_quality_level = 0.01
    feature_min_distance = 10
    feature_block_size = 7

    lk_win_size = (21, 21)
    lk_max_level = 3
    lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    ransac_threshold = 3.0
    min_inliers = 10

    # Diagnostic data collection
    diagnostics = {
        "frame_data": [],
        "metadata": {
            "frame_width": frame_width,
            "frame_height": frame_height,
            "fps": fps,
            "total_frames": total_frames,
        },
    }

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get segmentation mask
        results = mp_pose.process(frame_rgb)
        segmentation_mask = None
        has_person = False

        if results and results.segmentation_mask is not None:
            segmentation_mask = (results.segmentation_mask > 0.5).astype(np.uint8)
            has_person = True

        # Create background mask
        background_mask = None
        if segmentation_mask is not None:
            background_mask = (1 - segmentation_mask) * 255
            background_mask = background_mask.astype(np.uint8)
            kernel = np.ones((15, 15), np.uint8)
            background_mask = cv2.erode(background_mask, kernel, iterations=1)

        # Track features and estimate transform
        shake_dx, shake_dy = 0.0, 0.0
        num_tracked = 0
        num_inliers = 0
        transform_matrix = None
        curr_background_features = None

        if prev_gray is not None and prev_background_features is not None:
            # Track features
            curr_features, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                prev_background_features,
                None,  # type: ignore[arg-type]
                winSize=lk_win_size,
                maxLevel=lk_max_level,
                criteria=lk_criteria,
            )

            if curr_features is not None and status is not None:
                good_new = curr_features[status.flatten() == 1]
                good_old = prev_background_features[status.flatten() == 1]
                num_tracked = len(good_new)

                # Estimate transform
                if len(good_new) >= min_inliers:
                    try:
                        transform_matrix, inliers = cv2.estimateAffinePartial2D(
                            good_old,
                            good_new,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=ransac_threshold,
                            confidence=0.99,
                            maxIters=2000,
                        )

                        if transform_matrix is not None and inliers is not None:
                            num_inliers = np.sum(inliers)

                            if num_inliers >= min_inliers:
                                shake_dx = float(transform_matrix[0, 2])
                                shake_dy = float(transform_matrix[1, 2])
                                curr_background_features = good_new[
                                    inliers.flatten() == 1
                                ].reshape(-1, 1, 2)
                    except cv2.error:
                        pass

        # Detect new features if needed
        num_detected = 0
        if curr_background_features is None or len(curr_background_features) < 50:
            if background_mask is not None:
                new_features = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=feature_max_corners,
                    qualityLevel=feature_quality_level,
                    minDistance=feature_min_distance,
                    mask=background_mask,
                    blockSize=feature_block_size,
                )
            else:
                new_features = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=feature_max_corners,
                    qualityLevel=feature_quality_level,
                    minDistance=feature_min_distance,
                    mask=None,
                    blockSize=feature_block_size,
                )

            if new_features is not None:
                num_detected = len(new_features)
                if curr_background_features is not None:
                    curr_background_features = np.vstack(
                        (curr_background_features, new_features)
                    )
                else:
                    curr_background_features = new_features

        # Store diagnostic data
        frame_diag = {
            "frame_idx": frame_idx,
            "has_person": has_person,
            "has_background_mask": background_mask is not None,
            "num_features": len(curr_background_features)
            if curr_background_features is not None
            else 0,
            "num_tracked": num_tracked,
            "num_detected": num_detected,
            "num_inliers": num_inliers,
            "shake_dx": shake_dx,
            "shake_dy": shake_dy,
            "shake_magnitude": np.sqrt(shake_dx**2 + shake_dy**2),
            "has_transform": transform_matrix is not None,
        }

        if transform_matrix is not None:
            # Extract rotation and scale from affine matrix
            a, b = transform_matrix[0, 0], transform_matrix[0, 1]
            scale = np.sqrt(a**2 + b**2)
            rotation_deg = np.degrees(np.arctan2(b, a))
            frame_diag["scale"] = scale
            frame_diag["rotation_deg"] = rotation_deg
        else:
            frame_diag["scale"] = None
            frame_diag["rotation_deg"] = None

        diagnostics["frame_data"].append(frame_diag)

        # Update state
        prev_gray = gray
        prev_background_features = curr_background_features

        # Progress
        if (frame_idx + 1) % 30 == 0:
            print(f"  Processed {frame_idx + 1}/{total_frames} frames...")

    cap.release()
    mp_pose.close()

    return diagnostics


def generate_report(diagnostics, output_path):
    """Generate a text report with statistics and analysis."""

    frame_data = diagnostics["frame_data"]
    metadata = diagnostics["metadata"]

    if not frame_data:
        print("No frame data to analyze!")
        return

    # Calculate statistics
    num_frames = len(frame_data)
    frames_with_person = sum(1 for f in frame_data if f["has_person"])
    frames_with_mask = sum(1 for f in frame_data if f["has_background_mask"])
    frames_with_transform = sum(1 for f in frame_data if f["has_transform"])

    feature_counts = [f["num_features"] for f in frame_data]
    inlier_counts = [f["num_inliers"] for f in frame_data if f["num_inliers"] > 0]
    shake_magnitudes = [
        f["shake_magnitude"] for f in frame_data if f["shake_magnitude"] > 0
    ]

    avg_features = np.mean(feature_counts) if feature_counts else 0
    avg_inliers = np.mean(inlier_counts) if inlier_counts else 0
    avg_shake = np.mean(shake_magnitudes) if shake_magnitudes else 0
    max_shake = max(shake_magnitudes) if shake_magnitudes else 0

    # Generate report
    report_lines = [
        "=" * 70,
        "STABILIZATION DIAGNOSTIC REPORT",
        "=" * 70,
        "",
        "VIDEO INFORMATION:",
        f"  Resolution: {metadata['frame_width']}x{metadata['frame_height']}",
        f"  Frame rate: {metadata['fps']:.1f} fps",
        f"  Frames analyzed: {num_frames}",
        "",
        "SEGMENTATION QUALITY:",
        f"  Frames with person detected: {frames_with_person}/{num_frames} ({100 * frames_with_person / num_frames:.1f}%)",
        f"  Frames with background mask: {frames_with_mask}/{num_frames} ({100 * frames_with_mask / num_frames:.1f}%)",
        "",
        "FEATURE TRACKING:",
        f"  Average features per frame: {avg_features:.1f}",
        f"  Min features: {min(feature_counts)}",
        f"  Max features: {max(feature_counts)}",
        "",
        "TRANSFORM ESTIMATION:",
        f"  Frames with valid transform: {frames_with_transform}/{num_frames} ({100 * frames_with_transform / num_frames:.1f}%)",
        f"  Average inliers: {avg_inliers:.1f}",
        "",
        "CAMERA SHAKE:",
        f"  Average shake magnitude: {avg_shake:.2f} pixels",
        f"  Maximum shake magnitude: {max_shake:.2f} pixels",
        "",
        "QUALITY ASSESSMENT:",
    ]

    # Quality assessment
    if frames_with_transform / num_frames > 0.9:
        report_lines.append(
            "  ✓ EXCELLENT: Transform estimation successful in >90% of frames"
        )
    elif frames_with_transform / num_frames > 0.7:
        report_lines.append(
            "  ✓ GOOD: Transform estimation successful in >70% of frames"
        )
    else:
        report_lines.append("  ⚠ POOR: Transform estimation failed in many frames")

    if avg_inliers > 50:
        report_lines.append("  ✓ EXCELLENT: High inlier count (>50 average)")
    elif avg_inliers > 20:
        report_lines.append("  ✓ GOOD: Adequate inlier count (>20 average)")
    else:
        report_lines.append(
            "  ⚠ POOR: Low inlier count - may need more features or lower threshold"
        )

    if max_shake < 10:
        report_lines.append("  ✓ EXCELLENT: Low camera shake (<10 pixels max)")
    elif max_shake < 30:
        report_lines.append("  ✓ GOOD: Moderate camera shake (<30 pixels max)")
    else:
        report_lines.append("  ⚠ WARNING: High camera shake detected (>30 pixels)")

    report_lines.extend(
        [
            "",
            "RECOMMENDATIONS:",
        ]
    )

    if avg_features < 100:
        report_lines.append(
            "  • Consider increasing maxCorners parameter for more features"
        )

    if avg_inliers < 20:
        report_lines.append(
            "  • Consider reducing ransac_threshold for stricter outlier rejection"
        )
        report_lines.append("  • Or increase qualityLevel to detect stronger features")

    if frames_with_mask / num_frames < 0.5:
        report_lines.append(
            "  • Low segmentation success - ensure good lighting and clear subject"
        )

    if max_shake > 30:
        report_lines.append(
            "  • High camera shake - consider using tripod or stabilizer"
        )

    report_lines.extend(
        [
            "",
            "=" * 70,
        ]
    )

    # Write report
    report_text = "\n".join(report_lines)
    with open(output_path, "w") as f:
        f.write(report_text)

    # Also print to console
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(
        description="Test and validate stabilization improvements"
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum number of frames to process (default: 300)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output files (default: current directory)",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("STABILIZATION DIAGNOSTIC TEST")
    print("=" * 70)
    print(f"\nInput video: {video_path}")
    print(f"Output directory: {output_dir}\n")

    # Run stabilization test
    print("Running stabilization analysis...")
    diagnostics = test_stabilization(str(video_path), args.max_frames)

    # Save diagnostics data
    diag_path = output_dir / "stabilization_diagnostics.pkl"
    with open(diag_path, "wb") as f:
        pickle.dump(diagnostics, f)
    print(f"\n✓ Diagnostic data saved to: {diag_path}")

    # Generate report
    report_path = output_dir / "stabilization_report.txt"
    generate_report(diagnostics, report_path)
    print(f"✓ Report saved to: {report_path}")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
