"""
barpath - Offline Weightlifting Technique Analysis
"""
__version__ = "1.0.0"

from .collect_data import step_1_collect_data
from .analyze_data import step_2_analyze_data
from .generate_graphs import step_3_generate_graphs
from .render_video import step_4_render_video
from .critique_lift import critique_clean

__all__ = [
    'step_1_collect_data',
    'step_2_analyze_data',
    'step_3_generate_graphs',
    'step_4_render_video',
    'critique_clean'
]