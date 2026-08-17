"""Step 5: Render final analysis video with overlays.

This module renders the visualization video with:
- Colored bar path (phase-based)
- Skeleton overlay
- Legend and HUD elements (sparkline, power band, error markers)
"""

import argparse
import gc
import os
import subprocess
from typing import Any, cast

import cv2
import numpy as np
import pandas as pd
from config import (
    BARBELL_BOX_THICKNESS,
    GC_INTERVAL_FRAMES,
)
from step5_helpers import HUDConfig
from step5_helpers.hud_renderer import (
    LEGEND_COLORS,
    PHASE_COLOR_SCHEMES,
    PHASE_NAMES,
    draw_hud_overlay,
)
from step5_helpers.overlay_metrics import OverlayMetrics
from utils import (
    COLOR_SCHEME,
    draw_legend,
    parse_barbell_box,
)


def step_5_render_video(
    df: pd.DataFrame,
    video_path: str,
    output_video_path: str,
    draw_pose: bool = True,
    lift_type: str = "snatch",
    analysis_result=None,
    hud_config=None,
):
    """
    Render the final visualization video.

    Args:
        df: DataFrame with kinematic data
        video_path: Path to source video
        output_video_path: Path to save output video
        draw_pose: Whether to draw skeleton overlay
        lift_type: Type of lift (snatch, clean, jerk, clean_jerk) for phase naming
    """
    print("--- Step 5: Rendering Final Video ---")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    overlay_metrics = OverlayMetrics.for_frame(frame_width, frame_height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    pose_enabled = draw_pose

    # Create default HUDConfig if not provided
    if hud_config is None:
        hud_config = HUDConfig()

    position_sources = [
        ("barbell_x_smooth", "barbell_y_smooth", "smoothed"),
        ("barbell_x_stable", "barbell_y_stable", "stabilized"),
    ]
    selected_source = None
    for x_col, y_col, label in position_sources:
        if x_col in df.columns and y_col in df.columns:
            selected_source = (x_col, y_col, label)
            break
    if selected_source is None:
        cap.release()
        raise ValueError("Missing barbell position columns in CSV. Please re-run Step 2.")

    position_x_col, position_y_col, source_label = selected_source
    if "bar_phase" not in df.columns:
        cap.release()
        raise ValueError("Missing bar_phase column in CSV. Please re-run Step 2.")

    print(f"Rendering bar path using {source_label} coordinates.")

    path_df = df[[position_x_col, position_y_col, "bar_phase"]].dropna()
    path_indices = np.asarray(cast(Any, path_df.index).to_numpy(dtype=float), dtype=float)
    path_points = np.asarray(
        cast(Any, path_df[[position_x_col, position_y_col]]).to_numpy(dtype=float),
        dtype=float,
    )
    path_phases = np.asarray(cast(Any, path_df["bar_phase"]).to_numpy(dtype=float), dtype=float)

    first_idx = np.asarray(cast(Any, df.index).to_numpy(), dtype=float)
    first_analyzed_frame = int(first_idx.min()) if first_idx.size > 0 else 0
    last_analyzed_frame = int(first_idx.max()) if first_idx.size > 0 else 0

    extra_frames = int(fps)

    start_frame = first_analyzed_frame
    end_frame = min(last_analyzed_frame + extra_frames, total_frames)
    frames_to_render = end_frame - start_frame

    print(f"Rendering frames {start_frame} to {end_frame} ({frames_to_render} total frames)...")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    last_shake_x = 0.0
    last_shake_y = 0.0

    for frame_idx in range(frames_to_render):
        frame_count = start_frame + frame_idx
        points_to_draw = None
        success, frame = cap.read()
        if not success:
            print(f"Warning: Could not read frame {frame_count}")
            break

        if frame_count in df.index:
            row = df.loc[frame_count]

            if not pd.isna(row.get("total_shake_x")):
                last_shake_x = float(row["total_shake_x"])
                last_shake_y = float(row["total_shake_y"])

            current_shake_x = last_shake_x
            current_shake_y = last_shake_y

            max_path_index = int(np.searchsorted(path_indices, frame_count, side="right"))

            draw_box = True

            landmarks_str = str(row.get("landmarks_str", "{}")) if pose_enabled else "{}"
            barbell_box_str = str(row.get("barbell_box_str", ""))

        else:
            current_shake_x = last_shake_x
            current_shake_y = last_shake_y

            max_path_index = len(path_points)

            draw_box = False

            landmarks_str = "{}"
            barbell_box_str = ""

        if max_path_index >= 2:
            points_to_draw = path_points[:max_path_index].copy()

            points_to_draw[:, 0] += current_shake_x  # type: ignore
            points_to_draw[:, 1] += current_shake_y  # type: ignore

            # Use HUD overlay orchestrator (bar path + skeleton + sparkline + power band + error markers)
            frame, _last_head_pos = draw_hud_overlay(
                frame,
                df,
                frame_width,
                frame_height,
                lift_type,
                hud_config,
                path_points,
                path_phases,
                max_path_index,
                current_shake_x,
                current_shake_y,
                landmarks_str,
                LEGEND_COLORS,
                analysis_result=analysis_result,
                current_frame=frame_count,
                overlay_metrics=overlay_metrics,
            )

        if draw_box:
            barbell_box = parse_barbell_box(barbell_box_str)
            if barbell_box:
                x1, y1, x2, y2 = barbell_box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    LEGEND_COLORS["Barbell Box"],
                    overlay_metrics.px(BARBELL_BOX_THICKNESS),
                )

        # Build dynamic legend based on lift type
        phase_names = PHASE_NAMES.get(lift_type, PHASE_NAMES["snatch"])
        phase_scheme = PHASE_COLOR_SCHEMES.get(lift_type, PHASE_COLOR_SCHEMES["snatch"])
        dynamic_legend = {
            "Barbell Box": COLOR_SCHEME["Barbell Box"],
        }
        for phase_id, phase_name in phase_names.items():
            dynamic_legend[phase_name] = phase_scheme.get(phase_id, (255, 255, 255))

        draw_legend(frame, dynamic_legend, overlay_metrics=overlay_metrics)

        out.write(frame)

        progress_fraction = (frame_idx + 1) / frames_to_render
        yield (
            "step5",
            progress_fraction,
            f"Rendering video: frame {frame_count} ({frame_idx + 1}/{frames_to_render})",
        )

        del frame
        if points_to_draw is not None:
            del points_to_draw
        if frame_idx % GC_INTERVAL_FRAMES == 0:
            gc.collect()

    cap.release()
    out.release()

    start_time = start_frame / fps
    duration = frames_to_render / fps

    temp_video_path = output_video_path + ".temp.mp4"
    os.replace(output_video_path, temp_video_path)

    try:
        print("Muxing audio from original video...")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_time),
            "-i",
            video_path,
            "-i",
            temp_video_path,
            "-t",
            str(duration),
            "-map",
            "1:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-shortest",
            output_video_path,
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            os.remove(temp_video_path)
            print(f"Step 5 Complete. Final video saved to '{output_video_path}'")
        else:
            os.replace(temp_video_path, output_video_path)
            print("Warning: ffmpeg audio muxing failed. Video saved without audio.")
            print(f"ffmpeg stderr: {result.stderr}")

    except FileNotFoundError:
        os.replace(temp_video_path, output_video_path)
        print("Warning: ffmpeg not found. Video saved without audio.")
    except Exception as e:
        if os.path.exists(temp_video_path):
            os.replace(temp_video_path, output_video_path)
        print(f"Warning: Error during audio muxing: {e}")


def main():
    parser = argparse.ArgumentParser(description="Step 5: Render final analysis video.")
    parser.add_argument(
        "--input_video", required=True, help="Path to the original source video file."
    )
    parser.add_argument(
        "--input_csv",
        default="final_analysis.csv",
        help="Path to the final analysis CSV from Step 2.",
    )
    parser.add_argument(
        "--output_video", required=True, help="Path to save the final visualized video."
    )
    parser.add_argument(
        "--lift_type",
        default="snatch",
        choices=["snatch", "clean", "jerk", "clean_jerk"],
        help="Type of lift for phase color mapping.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_video):
        print(f"Error: Input video not found at {args.input_video}")
        return
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV not found at {args.input_csv}")
        return

    try:
        df = pd.read_csv(args.input_csv)
        if "frame" in df.columns:
            df = df.set_index("frame")
        print(f"Loaded CSV with {len(df)} frames and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading CSV file {args.input_csv}: {e}")
        return

    for _ in step_5_render_video(df, args.input_video, args.output_video, lift_type=args.lift_type):
        pass


if __name__ == "__main__":
    main()
