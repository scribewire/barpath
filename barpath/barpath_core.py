"""
Core pipeline runner for barpath analysis.

This module orchestrates the 5-step barpath analysis pipeline:
1. Collect raw data from video
2. Analyze and enrich the data
3. Generate kinematic graphs
4. Render visualization video
5. Provide technique critique

The runner yields progress updates that can be consumed by CLI or GUI frontends.

``run_pipeline_from_folder`` is a lighter variant that skips step 1 (data
collection) and re-runs steps 2-5 from an existing output folder that
already contains a ``raw_data.pkl``.  This is useful for re-analysing
previously processed videos after changing analysis settings or code.
"""

# Import step functions - using importlib for dynamic loading
import importlib.util
import os
import pickle
import sys
from pathlib import Path

import pandas as pd


def _is_openvino_model_dir(path_str: str) -> bool:
    """Return True when the provided path looks like an OpenVINO export directory."""
    path = Path(path_str)
    if not path.is_dir():
        return False
    return any("openvino" in part.lower() for part in path.parts)


# Add pipeline directory to path for imports
pipeline_dir = Path(__file__).parent / "pipeline"
sys.path.insert(0, str(pipeline_dir))


def _import_step_function(step_file, function_name):
    """Dynamically import a function from a step file."""
    spec = importlib.util.spec_from_file_location("step_module", step_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {step_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


# Import the step functions
step_1_collect_data = _import_step_function(
    pipeline_dir / "1_collect_data.py", "step_1_collect_data"
)
step_2_analyze_data = _import_step_function(
    pipeline_dir / "2_analyze_data.py", "step_2_analyze_data"
)
step_3_generate_graphs = _import_step_function(
    pipeline_dir / "3_generate_graphs.py", "step_3_generate_graphs"
)
plot_superimposed_paths_compensated = _import_step_function(
    pipeline_dir / "3_generate_graphs.py", "plot_superimposed_paths_compensated"
)
plot_superimposed_paths_smoothed = _import_step_function(
    pipeline_dir / "3_generate_graphs.py", "plot_superimposed_paths_smoothed"
)
step_4_render_video = _import_step_function(
    pipeline_dir / "4_render_video.py", "step_4_render_video"
)
critique_lift = _import_step_function(
    pipeline_dir / "5_critique_lift.py", "critique_lift"
)


def run_pipeline_from_folder(
    output_folder,
    lift_type="none",
    encode_video=True,
    technique_analysis=True,
    raw_data_path="raw_data.pkl",
    analysis_csv_path="final_analysis.csv",
    cancel_event=None,
):
    """
    Re-run steps 2-5 of the barpath pipeline from an existing output folder.

    The folder must contain a ``raw_data.pkl`` produced by step 1.  The
    original video file is only required when ``encode_video=True``; its
    path is read from the pickle's metadata (``source_video`` key).  If
    the key is absent *or* the file no longer exists the video-render step
    is automatically skipped with a warning rather than raising an error.

    Yields progress updates as ``(step_name, progress_value, message)``
    tuples, identical to :func:`run_pipeline`.

    Args:
        output_folder (str | Path): Existing output directory that contains
            ``raw_data.pkl`` (and optionally a previous ``final_analysis.csv``
            and ``output.mp4``).
        lift_type (str): Lift type passed to the critique step.
        encode_video (bool): Whether to re-render the output video.
        technique_analysis (bool): Whether to re-run the technique critique.
        raw_data_path (str): Filename of the raw-data pickle inside
            ``output_folder``.  Defaults to ``"raw_data.pkl"``.
        analysis_csv_path (str): Filename for the re-written analysis CSV.
            Defaults to ``"final_analysis.csv"``.
        cancel_event (threading.Event, optional): Set this to abort.

    Yields:
        tuple: ``(step_name, progress_value, message)``
    """

    def check_cancel():
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Pipeline cancelled by user")

    output_folder = Path(output_folder)

    # Resolve pickle and CSV paths inside the folder
    pkl_path = (
        output_folder / raw_data_path
        if not Path(raw_data_path).is_absolute()
        else Path(raw_data_path)
    )
    csv_path = (
        output_folder / analysis_csv_path
        if not Path(analysis_csv_path).is_absolute()
        else Path(analysis_csv_path)
    )

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"raw_data.pkl not found in '{output_folder}'. "
            "This folder has not been processed by step 1 yet."
        )

    output_folder.mkdir(parents=True, exist_ok=True)

    # --- STEP 2: Analyze Data ---
    check_cancel()
    yield ("step2", None, "Loading raw data...")

    with open(pkl_path, "rb") as f:
        input_data = pickle.load(f)

    # Allow the caller to override the lift type stored in the pickle
    if lift_type != "none":
        input_data.setdefault("metadata", {})["lift_type"] = lift_type

    check_cancel()
    yield ("step2", None, "Starting data analysis...")
    step_2_analyze_data(input_data, str(csv_path))
    del input_data

    yield ("step2", None, f"Analysis complete. Saved to {csv_path}")

    # --- STEP 3: Generate Graphs ---
    check_cancel()
    yield ("step3", None, "Generating kinematic graphs...")

    df = pd.read_csv(str(csv_path))
    check_cancel()
    step_3_generate_graphs(df, str(output_folder))
    del df

    yield ("step3", None, f"Graphs generated in {output_folder}/")

    # --- STEP 4: Render Video ---
    check_cancel()
    if encode_video:
        # Try to find the original source video path from the pickle metadata
        with open(pkl_path, "rb") as f:
            _pkl_meta = pickle.load(f).get("metadata", {})
        source_video = _pkl_meta.get("source_video") or _pkl_meta.get("input_video")

        if source_video and os.path.exists(source_video):
            df = pd.read_csv(str(csv_path))
            if "frame" in df.columns:
                df = df.set_index("frame")

            output_video_path = output_folder / "output.mp4"
            pose_overlay_enabled = lift_type != "none"

            for update in step_4_render_video(
                df, source_video, str(output_video_path), draw_pose=pose_overlay_enabled
            ):
                check_cancel()
                yield update

            del df
        else:
            if source_video:
                yield (
                    "step4",
                    None,
                    f"Video rendering skipped — source video not found: {source_video}",
                )
            else:
                yield (
                    "step4",
                    None,
                    "Video rendering skipped — source video path not stored in raw_data.pkl",
                )
    else:
        yield ("step4", None, "Video rendering skipped")

    # --- STEP 5: Critique Lift ---
    check_cancel()
    if technique_analysis and lift_type != "none":
        yield ("step5", None, f"Analyzing {lift_type} technique...")

        df = pd.read_csv(str(csv_path))
        if "frame" in df.columns:
            df = df.set_index("frame")

        check_cancel()
        critiques = critique_lift(df, lift_type, str(output_folder))

        if not critiques:
            message = "Analysis complete (No phases detected?)"
        else:
            message = (
                f"Analysis complete. Report saved to "
                f"{os.path.join(str(output_folder), 'analysis.md')}"
            )

        yield ("step5", None, message)
    else:
        yield ("step5", None, "Technique analysis skipped")

    yield ("complete", 1.0, "Pipeline complete!")


def run_batch_postprocess(
    video_output_dirs,
    video_labels,
    batch_output_dir,
    use_filenames=False,
    analysis_csv_name="final_analysis.csv",
    cancel_event=None,
):
    """
    Run post-processing steps that operate across all videos in a batch.

    Currently this generates the superimposed bar-path graph.  Additional
    cross-video aggregations can be added here in future.

    Yields progress updates as (step_name, progress_value, message) tuples.

    Parameters
    ----------
    video_output_dirs : list of str or Path
        Per-video output directories (one per processed video), in order.
    video_labels : list of str
        Human-readable label for each video (filename stem or similar).
        Used when use_filenames=True.
    batch_output_dir : str or Path
        Top-level output directory where the combined graph is saved.
    use_filenames : bool
        Passed through to plot_superimposed_paths.
    analysis_csv_name : str
        Name of the analysis CSV file inside each per-video output dir.
    cancel_event : threading.Event, optional
        Checked before each step; raises InterruptedError if set.
    """

    def check_cancel():
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Pipeline cancelled by user")

    yield ("batch", None, "Generating superimposed bar-path graphs...")

    check_cancel()

    # Load each per-video analysis CSV
    video_data_list = []
    for label, video_dir in zip(video_labels, video_output_dirs):
        csv_path = Path(video_dir) / analysis_csv_name
        if not csv_path.exists():
            print(f"  Warning: no analysis CSV found at {csv_path} — skipping lift.")
            continue
        try:
            df = pd.read_csv(csv_path)
            video_data_list.append((label, df))
        except Exception as exc:
            print(f"  Warning: could not load {csv_path}: {exc} — skipping lift.")

    if len(video_data_list) < 2:
        yield (
            "batch",
            None,
            "Skipping superimposed graphs: fewer than 2 lifts with valid data.",
        )
        return

    check_cancel()

    os.makedirs(batch_output_dir, exist_ok=True)

    # --- Angle-compensated superimposed graph ---
    # Uses corrected cm traces for lifts with |yaw| >= 10°, smoothed px for
    # side-on lifts or those without correction data.
    try:
        plot_superimposed_paths_compensated(
            video_data_list,
            output_dir=str(batch_output_dir),
            use_filenames=use_filenames,
        )
        yield (
            "batch",
            None,
            f"Compensated superimposed graph saved to "
            f"{batch_output_dir}/superimposed_bar_paths_compensated.png",
        )
    except Exception as exc:
        yield (
            "batch",
            None,
            f"Warning: could not generate compensated superimposed graph: {exc}",
        )

    check_cancel()

    # --- Smoothed-only superimposed graph ---
    # Always uses the smoothed pixel-space traces, no angle compensation.
    try:
        plot_superimposed_paths_smoothed(
            video_data_list,
            output_dir=str(batch_output_dir),
            use_filenames=use_filenames,
        )
        yield (
            "batch",
            None,
            f"Smoothed superimposed graph saved to "
            f"{batch_output_dir}/superimposed_bar_paths_smoothed.png",
        )
    except Exception as exc:
        yield (
            "batch",
            None,
            f"Warning: could not generate smoothed superimposed graph: {exc}",
        )


def run_pipeline(
    input_video,
    model_path,
    output_video=None,
    lift_type="none",
    output_dir="outputs",
    encode_video=True,
    technique_analysis=True,
    raw_data_path="raw_data.pkl",
    analysis_csv_path="final_analysis.csv",
    cancel_event=None,
):
    """
    Run the complete barpath analysis pipeline.

    Yields progress updates as (step_name, progress_value, message) tuples.

    Args:
        input_video (str): Path to input video file
        model_path (str): Path to YOLO model file
        output_video (str, optional): Path for output video (if encode_video=True)
        lift_type (str): Type of lift for critique ('clean', 'none')
        output_dir (str): Directory to save outputs (graphs, analysis, etc.)
        encode_video (bool): Whether to render output video
        technique_analysis (bool): Whether to run technique critique
        raw_data_path (str): Path to save/load raw data pickle
        analysis_csv_path (str): Path to save/load analysis CSV
        cancel_event (threading.Event, optional): Event to signal cancellation

    Yields:
        tuple: (step_name, progress, message) where:
            - step_name: 'step1', 'step2', 'step3', 'step4', or 'step5'
            - progress: float 0.0-1.0 for steps with progress, or None for steps without
            - message: str describing current status
    """

    # Helper to check cancellation
    def check_cancel():
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Pipeline cancelled by user")

    # Validate inputs
    check_cancel()
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    # Handle OpenVINO directories - validate they contain both .xml and .bin files
    if _is_openvino_model_dir(model_path):
        xml_files = list(Path(model_path).glob("*.xml"))
        bin_files = list(Path(model_path).glob("*.bin"))
        if not xml_files:
            raise FileNotFoundError(
                f"OpenVINO directory '{model_path}' does not contain a .xml model file"
            )
        if not bin_files:
            raise FileNotFoundError(
                f"OpenVINO directory '{model_path}' does not contain a .bin weights file. "
                f"OpenVINO models require both .xml (model definition) and .bin (weights) files."
            )
    elif not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if encode_video and not output_video:
        raise ValueError("output_video required when encode_video=True")

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Store the source video path in the pickle metadata so
    # run_pipeline_from_folder can locate it for re-rendering.
    _source_video_abs = str(Path(input_video).resolve())

    # Update paths to be inside output_dir if they are defaults
    if raw_data_path == "raw_data.pkl":
        raw_data_path = os.path.join(output_dir, "raw_data.pkl")
    if analysis_csv_path == "final_analysis.csv":
        analysis_csv_path = os.path.join(output_dir, "final_analysis.csv")

    # Create output directory for video if needed (if absolute path provided)
    if encode_video and output_video:
        video_dir = os.path.dirname(output_video)
        if video_dir and not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)

    # --- STEP 1: Collect Data ---
    check_cancel()
    # step_1_collect_data yields progress internally
    for update in step_1_collect_data(
        input_video,
        model_path,
        raw_data_path,
        lift_type,
    ):
        check_cancel()
        yield update

    # Patch the pickle to include the absolute source video path so that
    # run_pipeline_from_folder can find the video for re-rendering later.
    try:
        with open(raw_data_path, "rb") as _f:
            _pkl = pickle.load(_f)
        _pkl.setdefault("metadata", {})["source_video"] = _source_video_abs
        with open(raw_data_path, "wb") as _f:
            pickle.dump(_pkl, _f)
        del _pkl
    except Exception as _e:
        print(f"  Warning: could not patch source_video into pickle: {_e}")

    # --- STEP 2: Analyze Data ---
    check_cancel()
    yield ("step2", None, "Starting data analysis...")

    # Load the raw data
    with open(raw_data_path, "rb") as f:
        input_data = pickle.load(f)

    check_cancel()
    # Run analysis (no progress reporting)
    step_2_analyze_data(input_data, analysis_csv_path)

    # Free memory
    del input_data

    yield ("step2", None, f"Analysis complete. Saved to {analysis_csv_path}")

    # --- STEP 3: Generate Graphs ---
    check_cancel()
    yield ("step3", None, "Generating kinematic graphs...")

    # Load analysis data
    df = pd.read_csv(analysis_csv_path)

    check_cancel()
    # Generate graphs (no progress reporting)
    step_3_generate_graphs(df, output_dir)

    # Free memory
    del df

    yield ("step3", None, f"Graphs generated in {output_dir}/")

    # --- STEP 4: Render Video ---
    check_cancel()
    if encode_video:
        # Load analysis data with frame index
        df = pd.read_csv(analysis_csv_path)
        if "frame" in df.columns:
            df = df.set_index("frame")

        pose_overlay_enabled = lift_type != "none"
        # step_4_render_video yields progress internally
        for update in step_4_render_video(
            df, input_video, output_video, draw_pose=pose_overlay_enabled
        ):
            check_cancel()
            yield update

        # Free memory
        del df
    else:
        yield ("step4", None, "Video rendering skipped")

    # --- STEP 5: Critique Lift ---
    check_cancel()
    if technique_analysis and lift_type != "none":
        yield ("step5", None, f"Analyzing {lift_type} technique...")

        # Load analysis data
        df = pd.read_csv(analysis_csv_path)
        if "frame" in df.columns:
            df = df.set_index("frame")

        check_cancel()
        # Run critique
        critiques = critique_lift(df, lift_type, output_dir)

        # Format results
        if not critiques:
            message = "✓ Analysis complete (No phases detected?)"
        else:
            # Short message for progress bar/log, since full report is in analysis.md
            message = f"Analysis complete. Report saved to {os.path.join(output_dir, 'analysis.md')}"

        yield ("step5", None, message)
    else:
        yield ("step5", None, "Technique analysis skipped")

    # Final completion
    yield ("complete", 1.0, "Pipeline complete!")


def run_pipeline_simple(
    input_video,
    model_path,
    output_video=None,
    lift_type="none",
    output_dir="outputs",
    encode_video=True,
    technique_analysis=True,
):
    """
    Simple wrapper that runs the pipeline and consumes all progress updates.

    Returns:
        dict: Summary of results
    """
    results = {
        "step1": None,
        "step2": None,
        "step3": None,
        "step4": None,
        "step5": None,
        "success": True,
        "error": None,
    }

    try:
        for step_name, progress, message in run_pipeline(
            input_video=input_video,
            model_path=model_path,
            output_video=output_video,
            lift_type=lift_type,
            output_dir=output_dir,
            encode_video=encode_video,
            technique_analysis=technique_analysis,
        ):
            results[step_name] = message
            print(f"[{step_name}] {message}")
    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        raise

    return results
