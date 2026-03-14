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
already contains a ``raw_data.pkl``.
"""

import importlib.util
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

print("barpath_core: Starting imports...", flush=True)


def _is_openvino_model_dir(path_str: str) -> bool:
    """Return True when the provided path looks like an OpenVINO export directory."""
    path = Path(path_str)
    if not path.is_dir():
        return False
    return any("openvino" in part.lower() for part in path.parts)


pipeline_dir = Path(__file__).parent / "pipeline"
sys.path.insert(0, str(pipeline_dir))
print(f"barpath_core: Added {pipeline_dir} to sys.path", flush=True)


def _import_step_function(step_file, function_name):
    """Dynamically import a function from a step file."""
    print(f"barpath_core: Loading {function_name} from {step_file}...", flush=True)
    try:
        spec = importlib.util.spec_from_file_location("step_module", step_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {step_file}")
        module = importlib.util.module_from_spec(spec)
        print(f"barpath_core: Executing module {step_file}...", flush=True)
        spec.loader.exec_module(module)
        result = getattr(module, function_name)
        print(f"barpath_core: Loaded {function_name}", flush=True)
        return result
    except Exception as e:
        print(f"barpath_core: ERROR loading {function_name}: {e}", flush=True)
        raise


print("barpath_core: Importing step functions...", flush=True)
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
print("barpath_core: All step functions loaded!", flush=True)


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
    """

    def check_cancel():
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Pipeline cancelled by user")

    output_folder = Path(output_folder)

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

    check_cancel()
    yield ("step2", None, "Loading raw data...")

    with open(pkl_path, "rb") as f:
        input_data = pickle.load(f)

    if lift_type != "none":
        input_data.setdefault("metadata", {})["lift_type"] = lift_type

    check_cancel()
    yield ("step2", None, "Starting data analysis...")
    step_2_analyze_data(input_data, str(csv_path))
    del input_data

    yield ("step2", None, f"Analysis complete. Saved to {csv_path}")

    check_cancel()
    yield ("step3", None, "Generating kinematic graphs...")

    df = pd.read_csv(str(csv_path))
    check_cancel()
    step_3_generate_graphs(df, str(output_folder))

    yield ("step3", None, f"Graphs generated in {output_folder}/")

    check_cancel()
    if encode_video:
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
    """

    def check_cancel():
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Pipeline cancelled by user")

    yield ("batch", None, "Generating superimposed bar-path graphs...")

    check_cancel()

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
    """

    def check_cancel():
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Pipeline cancelled by user")

    check_cancel()
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    _source_video_abs = str(Path(input_video).resolve())

    if raw_data_path == "raw_data.pkl":
        raw_data_path = os.path.join(output_dir, "raw_data.pkl")
    if analysis_csv_path == "final_analysis.csv":
        analysis_csv_path = os.path.join(output_dir, "final_analysis.csv")

    if encode_video and output_video:
        video_dir = os.path.dirname(output_video)
        if video_dir and not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)

    check_cancel()
    for update in step_1_collect_data(
        input_video,
        model_path,
        raw_data_path,
        lift_type,
    ):
        check_cancel()
        yield update

    try:
        with open(raw_data_path, "rb") as _f:
            _pkl = pickle.load(_f)
        _pkl.setdefault("metadata", {})["source_video"] = _source_video_abs
        with open(raw_data_path, "wb") as _f:
            pickle.dump(_pkl, _f)
        del _pkl
    except Exception as _e:
        print(f"  Warning: could not patch source_video into pickle: {_e}")

    check_cancel()
    yield ("step2", None, "Starting data analysis...")

    with open(raw_data_path, "rb") as f:
        input_data = pickle.load(f)

    check_cancel()
    step_2_analyze_data(input_data, analysis_csv_path)
    del input_data

    yield ("step2", None, f"Analysis complete. Saved to {analysis_csv_path}")

    check_cancel()
    yield ("step3", None, "Generating kinematic graphs...")

    df = pd.read_csv(analysis_csv_path)

    check_cancel()
    step_3_generate_graphs(df, output_dir)

    yield ("step3", None, f"Graphs generated in {output_dir}/")

    check_cancel()
    if encode_video:
        df = pd.read_csv(analysis_csv_path)
        if "frame" in df.columns:
            df = df.set_index("frame")

        pose_overlay_enabled = lift_type != "none"
        for update in step_4_render_video(
            df, input_video, output_video, draw_pose=pose_overlay_enabled
        ):
            check_cancel()
            yield update
    else:
        yield ("step4", None, "Video rendering skipped")

    check_cancel()
    if technique_analysis and lift_type != "none":
        yield ("step5", None, f"Analyzing {lift_type} technique...")

        df = pd.read_csv(analysis_csv_path)
        if "frame" in df.columns:
            df = df.set_index("frame")

        check_cancel()
        critiques = critique_lift(df, lift_type, output_dir)

        if not critiques:
            message = "Analysis complete (No phases detected?)"
        else:
            message = f"Analysis complete. Report saved to {os.path.join(output_dir, 'analysis.md')}"

        yield ("step5", None, message)
    else:
        yield ("step5", None, "Technique analysis skipped")

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
