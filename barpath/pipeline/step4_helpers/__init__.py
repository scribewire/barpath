"""Step 4 helpers for ML-based technique analysis.

This package contains:
- feature_extraction: Extract features from final_analysis.csv for analysis
- smart_analysis: Random Forest-based fault detection
- compiled_analyzer: Rule-based technique analyzer
"""

from .compiled_analyzer import CompiledAnalyzer, FAULT_DEFS, load_baselines_from_json
from .feature_extraction import (
    extract_technique_features,
    extract_trajectory,
    validate_phases,
)
from .smart_analysis import load_smart_analysis_model, run_smart_analysis

__all__ = [
    "CompiledAnalyzer",
    "FAULT_DEFS",
    "extract_trajectory",
    "extract_technique_features",
    "load_baselines_from_json",
    "load_smart_analysis_model",
    "run_smart_analysis",
    "validate_phases",
]
