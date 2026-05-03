"""
Extract features from a window of live frames.
Adapts offline pipeline features to partial windows from live buffer.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from .live_buffer import FrameData
from barpath.pipeline.lift_detection_features import (
    extract_model_features_as_array,
    _safe_savgol,
)


def _to_float_array(values) -> np.ndarray:
    """Force input into a 1D float64 ndarray."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr


def _safe_float(value: float, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if isinstance(value, (int, float, np.number)):
        result = float(value)
    else:
        result = default
    return default if not np.isfinite(result) else result


class LiveFeatureExtractor:
    """Extract features compatible with the trained classifier from live frames.
    
    Converts FrameData windows to DataFrame format expected by
    extract_model_features_as_array().
    """
    
    def __init__(self, frame_width: int, frame_height: int, fps: float = 30.0):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        
    def extract(self, frames: List[FrameData]) -> pd.DataFrame:
        """
        Convert frame window to DataFrame matching final_analysis.csv format.
        
        Returns:
            DataFrame with columns needed by extract_model_features_as_array()
        """
        if len(frames) < 10:
            return pd.DataFrame()
            
        # Build DataFrame with required columns
        data = {
            'barbell_x_smooth': [],
            'barbell_y_smooth': [],
            'vel_y_smooth': [],
            'accel_y_smooth': [],
            'left_knee_angle': [],
            'right_knee_angle': [],
            'left_elbow_angle': [],
            'right_elbow_angle': [],
            'left_shoulder_x': [],
            'left_shoulder_y': [],
            'right_shoulder_x': [],
            'right_shoulder_y': [],
            'left_hip_x': [],
            'left_hip_y': [],
            'right_hip_x': [],
            'right_hip_y': [],
            'time_s': [],
            'frame_height': [],
            'bar_phase': [],
        }
        
        # Extract data from frames
        for frame in frames:
            # Barbell position
            if frame.barbell_center is not None:
                data['barbell_x_smooth'].append(frame.barbell_center[0])
                data['barbell_y_smooth'].append(frame.barbell_center[1])
            else:
                data['barbell_x_smooth'].append(np.nan)
                data['barbell_y_smooth'].append(np.nan)
                
            # Joint angles
            data['left_knee_angle'].append(frame.joint_angles.get('left_knee', 180.0))
            data['right_knee_angle'].append(frame.joint_angles.get('right_knee', 180.0))
            data['left_elbow_angle'].append(frame.joint_angles.get('left_elbow', 180.0))
            data['right_elbow_angle'].append(frame.joint_angles.get('right_elbow', 180.0))
            
            # Landmark positions
            landmarks = frame.landmarks
            data['left_shoulder_x'].append(self._get_landmark_x(landmarks, 11))
            data['left_shoulder_y'].append(self._get_landmark_y(landmarks, 11))
            data['right_shoulder_x'].append(self._get_landmark_x(landmarks, 12))
            data['right_shoulder_y'].append(self._get_landmark_y(landmarks, 12))
            data['left_hip_x'].append(self._get_landmark_x(landmarks, 23))
            data['left_hip_y'].append(self._get_landmark_y(landmarks, 23))
            data['right_hip_x'].append(self._get_landmark_x(landmarks, 24))
            data['right_hip_y'].append(self._get_landmark_y(landmarks, 24))
            
            # Timestamp
            data['time_s'].append(frame.timestamp_ms / 1000.0)
            
            # Frame height
            data['frame_height'].append(float(self.frame_height))
            
            # Placeholder for bar_phase (will be estimated)
            data['bar_phase'].append(0)
            
        df = pd.DataFrame(data)
        
        # Compute velocities and accelerations
        df = self._compute_kinematics(df)
        
        # Estimate bar_phase
        df = self._estimate_phases(df)
        
        return df
        
    def _get_landmark_x(self, landmarks: Dict, idx: int, default: float = 0.0) -> float:
        """Get x position from landmark."""
        lm = landmarks.get(idx)
        if lm and lm[3] > 0.3:  # visibility check
            return lm[0] * self.frame_width
        return default
        
    def _get_landmark_y(self, landmarks: Dict, idx: int, default: float = 0.0) -> float:
        """Get y position from landmark."""
        lm = landmarks.get(idx)
        if lm and lm[3] > 0.3:
            return lm[1] * self.frame_height
        return default
        
    def _compute_kinematics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute velocity and acceleration from positions."""
        if len(df) < 3:
            return df
            
        y = df['barbell_y_smooth'].values.astype(float)
        
        # Interpolate missing values
        y = pd.Series(y).interpolate(method='linear').bfill().ffill().values
        
        # Smooth positions
        y_smooth = _safe_savgol(_to_float_array(y), max_win=min(11, len(y)), polyorder=3)
        
        # Compute velocity
        timestamps = df['time_s'].values
        if len(timestamps) > 1:
            dt = np.diff(timestamps)
            dt = np.where(dt == 0, 1.0 / self.fps, dt)
            
            vel = np.diff(y_smooth) / dt
            vel = np.concatenate([[0], vel])
            df['vel_y_smooth'] = _safe_savgol(_to_float_array(vel), max_win=min(9, len(vel)), polyorder=2)
        else:
            df['vel_y_smooth'] = 0.0
            
        # Compute acceleration
        if len(df) > 2:
            vel = df['vel_y_smooth'].values.astype(float)
            vel = pd.Series(vel).interpolate().bfill().ffill().values
            
            if len(timestamps) > 2:
                dt = np.diff(timestamps[:-1])
                dt = np.where(dt == 0, 1.0 / self.fps, dt)
                
                accel = np.diff(vel[:-1]) / dt
                accel = np.concatenate([[0, 0], accel])
                df['accel_y_smooth'] = _safe_savgol(_to_float_array(accel), max_win=7, polyorder=2)
            else:
                df['accel_y_smooth'] = 0.0
        else:
            df['accel_y_smooth'] = 0.0
            
        return df
        
    def _estimate_phases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate bar_phase from velocity profile."""
        if len(df) < 10:
            return df
            
        y = df['barbell_y_smooth'].values.astype(float)
        vel = df['vel_y_smooth'].values.astype(float)
        
        # Find key indices
        vel_max_idx = int(np.argmin(vel))  # Most negative (fastest up)
        y_min_idx = int(np.argmin(y))  # Highest point
        
        # Ensure proper ordering
        if y_min_idx <= vel_max_idx:
            search_from = vel_max_idx + 1
            if search_from < len(y):
                y_min_idx = search_from + int(np.argmin(y[search_from:]))
            else:
                y_min_idx = vel_max_idx + 1
                
        # Assign phases
        phases = np.zeros(len(df), dtype=int)
        phases[:vel_max_idx] = 0  # Pull
        phases[vel_max_idx:y_min_idx] = 1  # Pull-under  
        phases[y_min_idx:] = 2  # Recovery
        
        df['bar_phase'] = phases
        return df
        
    def window_to_features(self, frames: List[FrameData]) -> np.ndarray:
        """
        Full pipeline: frames -> DataFrame -> 37 features.
        
        Returns:
            1D numpy array of 37 features ready for classifier
        """
        df = self.extract(frames)
        if len(df) < 10:
            return np.zeros(37, dtype=np.float64)
            
        try:
            return extract_model_features_as_array(df)
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(37, dtype=np.float64)
            
    def window_to_df(self, frames: List[FrameData]) -> pd.DataFrame:
        """Convenience method: frames -> DataFrame."""
        return self.extract(frames)