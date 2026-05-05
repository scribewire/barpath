import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def mock_analysis_df():
    """Create a mock DataFrame mimicking final_analysis.csv structure."""
    n_frames = 100
    df = pd.DataFrame({
        'time_s': np.linspace(0, 3.0, n_frames),
        'vel_y_smooth': np.sin(np.linspace(0, 2*np.pi, n_frames)) * 2.0,
        'specific_power_y_smooth': np.abs(np.sin(np.linspace(0, 2*np.pi, n_frames))) * 50.0,
        'bar_phase': np.concatenate([np.zeros(33), np.ones(33), np.full(34, 2.0)]),
        'left_knee_angle': np.linspace(100, 140, n_frames),
        'right_knee_angle': np.linspace(105, 135, n_frames),
        'elbow_angle_left': np.linspace(140, 170, n_frames),
        'elbow_angle_right': np.linspace(145, 165, n_frames),
        'accel_y_smooth': np.cos(np.linspace(0, 2*np.pi, n_frames)) * 10.0,
        'barbell_x_smooth': np.full(n_frames, 320.0),
        'barbell_y_smooth': np.linspace(400, 200, n_frames),
    })
    return df


@pytest.fixture
def mock_frame():
    """Create a mock OpenCV frame (720p)."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def mock_fault_data():
    """Create mock analysis_result with fault data."""
    return {
        'compiled_faults': [
            {'id': 'early_arm_bend', 'name': 'Early Arm Bend', 'confidence': 75, 'severity': 'moderate', 'phase': 'pull'},
            {'id': 'hitching', 'name': 'Hitching', 'confidence': 60, 'severity': 'minor', 'phase': 'pull'},
            {'id': 'knee_cave', 'name': 'Knee Cave', 'confidence': 45, 'severity': 'minor', 'phase': 'pull'},
        ]
    }
