"""
Core pipeline runner for barpath analysis.

This module orchestrates the 5-step barpath analysis pipeline:
1. Collect raw data from video
2. Analyze and enrich the data
3. Generate kinematic graphs
4. Render visualization video
5. Provide technique critique

The runner yields progress updates that can be consumed by CLI or GUI frontends.
"""

import os
import sys
import warnings
from pathlib import Path

# Suppress Google protobuf deprecation warnings (can't be fixed in our code)
warnings.filterwarnings("ignore", message=".*google._upb._message.*", category=DeprecationWarning)

# Add pipeline directory to path for imports
pipeline_dir = Path(__file__).parent / "pipeline"
sys.path.insert(0, str(pipeline_dir))

# Import step functions - using importlib for dynamic loading
import importlib.util

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
step_4_render_video = _import_step_function(
    pipeline_dir / "4_render_video.py", "step_4_render_video"
)
critique_clean = _import_step_function(
    pipeline_dir / "5_critique_lift.py", "critique_clean"
)

import pandas as pd
import pickle


def run_pipeline(
    input_video,
    model_path,
    output_video=None,
    lift_type="none",
    class_name="endcap",
    graphs_dir="graphs",
    encode_video=True,
    technique_analysis=True,
    raw_data_path="raw_data.pkl",
    analysis_csv_path="final_analysis.csv"
):
    """
    Run the complete barpath analysis pipeline.
    
    Yields progress updates as (step_name, progress_value, message) tuples.
    
    Args:
        input_video (str): Path to input video file
        model_path (str): Path to YOLO model file
        output_video (str, optional): Path for output video (if encode_video=True)
        lift_type (str): Type of lift for critique ('clean', 'none')
        class_name (str): YOLO class name for barbell endcap
        graphs_dir (str): Directory to save graphs
        encode_video (bool): Whether to render output video
        technique_analysis (bool): Whether to run technique critique
        raw_data_path (str): Path to save/load raw data pickle
        analysis_csv_path (str): Path to save/load analysis CSV
    
    Yields:
        tuple: (step_name, progress, message) where:
            - step_name: 'step1', 'step2', 'step3', 'step4', or 'step5'
            - progress: float 0.0-1.0 for steps with progress, or None for steps without
            - message: str describing current status
    """
    
    # Validate inputs
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if encode_video and not output_video:
        raise ValueError("output_video required when encode_video=True")
    
    # Create output directory if needed
    if encode_video and output_video:
        output_dir = os.path.dirname(output_video)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    # --- STEP 1: Collect Data ---
    # step_1_collect_data yields progress internally
    yield from step_1_collect_data(input_video, model_path, raw_data_path, class_name)
    
    # --- STEP 2: Analyze Data ---
    yield ('step2', None, 'Starting data analysis...')
    
    # Load the raw data
    with open(raw_data_path, 'rb') as f:
        input_data = pickle.load(f)
    
    # Run analysis (no progress reporting)
    step_2_analyze_data(input_data, analysis_csv_path)
    
    # Free memory
    del input_data
    
    yield ('step2', None, f'Analysis complete. Saved to {analysis_csv_path}')
    
    # --- STEP 3: Generate Graphs ---
    yield ('step3', None, 'Generating kinematic graphs...')
    
    # Load analysis data
    df = pd.read_csv(analysis_csv_path)
    
    # Generate graphs (no progress reporting)
    step_3_generate_graphs(df, graphs_dir)
    
    # Free memory
    del df
    
    yield ('step3', None, f'Graphs generated in {graphs_dir}/')
    
    # --- STEP 4: Render Video ---
    if encode_video:
        # Load analysis data with frame index
        df = pd.read_csv(analysis_csv_path)
        if 'frame' in df.columns:
            df = df.set_index('frame')
        
        # step_4_render_video yields progress internally
        yield from step_4_render_video(df, input_video, output_video)
        
        # Free memory
        del df
    else:
        yield ('step4', None, 'Video rendering skipped')
    
    # --- STEP 5: Critique Lift ---
    if technique_analysis and lift_type != 'none':
        yield ('step5', None, f'Analyzing {lift_type} technique...')
        
        # Load analysis data
        df = pd.read_csv(analysis_csv_path)
        if 'frame' in df.columns:
            df = df.set_index('frame')
        
        # Run critique
        critiques = []
        if lift_type == 'clean':
            critiques = critique_clean(df)
        
        # Format results
        if not critiques:
            message = "✓ No major technical issues detected"
        else:
            message = f"Found {len(critiques)} technical concerns:\n" + "\n".join(f"  {i+1}. {c}" for i, c in enumerate(critiques))
        
        yield ('step5', None, message)
    else:
        yield ('step5', None, 'Technique analysis skipped')
    
    # Final completion
    yield ('complete', 1.0, 'Pipeline complete!')


def run_pipeline_simple(
    input_video,
    model_path,
    output_video=None,
    lift_type="none",
    class_name="endcap",
    graphs_dir="graphs",
    encode_video=True,
    technique_analysis=True
):
    """
    Simple wrapper that runs the pipeline and consumes all progress updates.
    
    Returns:
        dict: Summary of results
    """
    results = {
        'step1': None,
        'step2': None,
        'step3': None,
        'step4': None,
        'step5': None,
        'success': True,
        'error': None
    }
    
    try:
        for step_name, progress, message in run_pipeline(
            input_video=input_video,
            model_path=model_path,
            output_video=output_video,
            lift_type=lift_type,
            class_name=class_name,
            graphs_dir=graphs_dir,
            encode_video=encode_video,
            technique_analysis=technique_analysis
        ):
            results[step_name] = message
            print(f"[{step_name}] {message}")
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
        raise
    
    return results
