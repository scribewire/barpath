"""
Central configuration for the barpath analysis pipeline.

All magic numbers, thresholds, and tunable parameters are defined here
so they can be adjusted in one place and easily tested.
"""

# ---------------------------------------------------------------------------
# Step 1: Data Collection
# ---------------------------------------------------------------------------

# Frame decoding queue size (buffer ahead of inference)
DECODE_QUEUE_SIZE = 8

# YOLO detection confidence threshold
YOLO_CONFIDENCE_THRESHOLD = 0.25

# MediaPipe pose estimation settings
MEDIAPIPE_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_TRACKING_CONFIDENCE = 0.5
MEDIAPIPE_MODEL_COMPLEXITY = 1

# Stabilization parameters
STAB_MIN_FEATURES = 50
STAB_FEATURE_QUALITY = 0.01
STAB_FEATURE_MIN_DISTANCE = 30
STAB_LK_WINDOW_SIZE = 15
STAB_LK_MAX_LEVEL = 2
STAB_LK_CRITERIA_COUNT = 10
STAB_LK_CRITERIA_EPS = 0.03
STAB_MOTION_OUTLIER_THRESHOLD = 3.0

# ---------------------------------------------------------------------------
# Step 2: Data Analysis
# ---------------------------------------------------------------------------

# Landmarks to track (used for unpacking and angle calculations)
LANDMARKS_TO_TRACK = {
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_eye",
    "right_eye",
}

# Smoothing window sizes (Savitzky-Golay)
SAVGOL_POSITION_WINDOW = 11
SAVGOL_VELOCITY_WINDOW = 15
SAVGOL_POLY_ORDER = 3

# Phase detection thresholds
PHASE_VEL_THRESHOLD_FACTOR = 0.05  # fraction of max velocity
PHASE_HIP_SMOOTH_WINDOW = 9
PHASE_HIP_DROP_STD_FACTOR = 0.1

# Jerk-specific phase detection thresholds
JERK_DIP_VELOCITY_THRESHOLD = 20.0  # px/s threshold for downward movement
JERK_DRIVE_VELOCITY_THRESHOLD = 20.0  # px/s threshold for upward movement
JERK_MIN_KNEE_BEND_ANGLE = 5.0  # minimum knee angle change (degrees) to confirm dip

# Truncation settings
TRUNCATION_BUFFER_SECONDS = 1.0  # seconds before knee pass to keep

# Barbell endcap real-world width (metres)
BARBELL_ENDCAP_WIDTH_M = 0.05  # 50mm

# ---------------------------------------------------------------------------
# Step 3: Graph Generation
# ---------------------------------------------------------------------------

# Graph DPI and sizing
GRAPH_DPI = 150
GRAPH_WIDTH_PATH = 8
GRAPH_HEIGHT_PATH = 10
GRAPH_WIDTH_TIMESERIES = 12
GRAPH_HEIGHT_TIMESERIES = 6

# Phase colors (hex)
PHASE_COLORS = {
    0: "#e02020",  # Pull - vivid red
    1: "#f07800",  # Pull-under - vivid orange
    2: "#18a020",  # Recovery - vivid green
}

PHASE_LABELS = {
    0: "Pull",
    1: "Pull-under",
    2: "Recovery",
}

# 6-phase colors for clean+jerk (phases 3-5 are jerk phases)
PHASE_COLORS_6 = {
    0: "#e02020",  # Clean Pull - vivid red
    1: "#f07800",  # Clean Pull-under - vivid orange
    2: "#18a020",  # Clean Recovery - vivid green
    3: "#2060c0",  # Jerk Dip - blue
    4: "#9467bd",  # Jerk Drive - purple
    5: "#17becf",  # Jerk Recovery - cyan
}

PHASE_LABELS_6 = {
    0: "Clean Pull",
    1: "Clean Pull-under",
    2: "Clean Recovery",
    3: "Jerk Dip",
    4: "Jerk Drive",
    5: "Jerk Recovery",
}

PHASE_FILL_ALPHA = 0.12

# Start/end marker colors
START_MARKER_COLOR = "white"
START_MARKER_EDGE = "black"
END_MARKER_COLOR = "black"
END_MARKER_EDGE = "white"

# Multi-lift palette for superimposed graphs
LIFT_PALETTE = [
    "#1f77b4",  # muted blue
    "#9467bd",  # muted purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # medium grey
    "#bcbd22",  # yellow-green
    "#17becf",  # teal
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    "#c5b0d5",  # light purple
]

# DTW similarity scaling factor
DTW_SIMILARITY_K = 5.0

# ---------------------------------------------------------------------------
# Step 4: Video Rendering
# ---------------------------------------------------------------------------

# Memory management: run garbage collection every N frames
GC_INTERVAL_FRAMES = 50

# Phase colors in BGR (OpenCV format)
PHASE_COLORS_BGR = {
    0: (32, 32, 224),  # Pull - vivid red
    1: (0, 120, 240),  # Pull-under - vivid orange
    2: (32, 160, 24),  # Recovery - vivid green
}

# 6-phase colors in BGR for clean+jerk
PHASE_COLORS_6_BGR = {
    0: (32, 32, 224),  # Clean Pull - red
    1: (0, 120, 240),  # Clean Pull-under - orange
    2: (32, 160, 24),  # Clean Recovery - green
    3: (224, 96, 32),  # Jerk Dip - blue
    4: (240, 32, 150),  # Jerk Drive - purple
    5: (240, 200, 32),  # Jerk Recovery - cyan
}

# Skeleton line thickness
SKELETON_LINE_THICKNESS = 3
LANDMARK_RADIUS = 5

# Barbell box line thickness
BARBELL_BOX_THICKNESS = 2

# ---------------------------------------------------------------------------
# Step 5: Critique
# ---------------------------------------------------------------------------

# Scale factor clamping for uniform scaling
SCALE_MIN = 0.4
SCALE_MAX = 2.5

# ---------------------------------------------------------------------------
# Step 5: HUD Overlay
# ---------------------------------------------------------------------------

# HUD Sparkline
SPARKLINE_WIDTH_RATIO = 0.20       # 20% of frame width
SPARKLINE_HEIGHT_RATIO = 0.15      # 15% of frame height
SPARKLINE_MARGIN_X = 20            # pixels from right edge
SPARKLINE_MARGIN_Y = 20            # pixels from top edge
SPARKLINE_LINE_THICKNESS = 2
SPARKLINE_AXIS_COLOR_BGR = (80, 80, 80)  # gray axis lines

# HUD Power Zone Band
POWER_BAND_HEIGHT = 15             # pixels tall
POWER_BAND_GAP = 8                 # pixels below sparkline

# HUD Joint Angles
ANGLE_TEXT_POSITION_Y_RATIO = 0.92  # 92% down from top (near bottom)
ANGLE_FONT_SCALE = 0.7
ANGLE_FONT_THICKNESS = 2
ANGLE_GREEN_BGR = (0, 255, 0)
ANGLE_YELLOW_BGR = (0, 255, 255)
ANGLE_RED_BGR = (0, 0, 255)
ANGLE_FALLBACK_MIN = 90.0          # degrees (green lower bound when no baseline)
ANGLE_FALLBACK_MAX = 135.0         # degrees (green upper bound when no baseline)
ANGLE_BORDERLINE_MARGIN = 0.10     # 10% boundary margin for yellow

# HUD Error Markers
ERROR_TRIANGLE_SIZE = 12           # pixels (side length)
ERROR_TEXT_Y_OFFSET = 25           # pixels below triangle apex for label

# HUD Fault Type Colors (BGR) - Distinct per fault category
FAULT_COLORS_BGR = {
    "arm": (0, 165, 255),          # Orange
    "extension": (255, 0, 255),    # Magenta
    "path": (255, 255, 0),         # Cyan
    "knee_leg": (0, 255, 255),     # Yellow
    "catch": (0, 0, 255),          # Red
}

# Live Coaching Tip
COACHING_TIP_DURATION_S = 5.0
COACHING_TIP_CONFIDENCE_THRESHOLD = 0.6
COACHING_TIP_FONT_SCALE = 0.8
COACHING_TIP_FONT_THICKNESS = 2
COACHING_TIP_COLOR_BGR = (0, 255, 255)  # Yellow

# Coaching Tip Text
COACHING_TIP_FALLBACK = "Lift looks good"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# Log level: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL = "INFO"
