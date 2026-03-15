import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import (
    DTW_SIMILARITY_K,
    END_MARKER_COLOR,
    END_MARKER_EDGE,
    GRAPH_DPI,
    GRAPH_HEIGHT_PATH,
    GRAPH_HEIGHT_TIMESERIES,
    GRAPH_WIDTH_PATH,
    GRAPH_WIDTH_TIMESERIES,
    LIFT_PALETTE,
    PHASE_COLORS,
    PHASE_FILL_ALPHA,
    PHASE_LABELS,
    SCALE_MAX,
    SCALE_MIN,
    START_MARKER_COLOR,
    START_MARKER_EDGE,
)


def _phase_legend_handles():
    """Return a list of legend patch handles for the three phases."""
    return [
        mpatches.Patch(color=PHASE_COLORS[i], label=PHASE_LABELS[i])
        for i in sorted(PHASE_COLORS)
    ]


def _add_phase_shading(ax, phase_series, time_series):
    """
    Add a soft background shading band for each phase region on a time-series
    axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    phase_series : pd.Series  (index aligned with time_series)
    time_series  : pd.Series
    """
    if phase_series is None or len(phase_series) == 0:
        return

    combined = pd.DataFrame({"t": time_series, "phase": phase_series}).dropna()
    if combined.empty:
        return

    current_phase = int(combined["phase"].iloc[0])
    seg_start_t = combined["t"].iloc[0]

    for _, row in combined.iloc[1:].iterrows():
        p = int(row["phase"])
        if p != current_phase:
            ax.axvspan(
                seg_start_t,
                row["t"],
                color=PHASE_COLORS[current_phase % len(PHASE_COLORS)],
                alpha=PHASE_FILL_ALPHA,
                linewidth=0,
            )
            seg_start_t = row["t"]
            current_phase = p

    # Close off the last segment
    ax.axvspan(
        seg_start_t,
        combined["t"].iloc[-1],
        color=PHASE_COLORS[current_phase % len(PHASE_COLORS)],
        alpha=PHASE_FILL_ALPHA,
        linewidth=0,
    )


def _add_phase_vlines(ax, phase_series, time_series):
    """
    Draw a vertical dashed line at every phase transition.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    phase_series : pd.Series
    time_series  : pd.Series
    """
    combined = pd.DataFrame({"t": time_series, "phase": phase_series}).dropna()
    if combined.empty:
        return

    prev_phase = int(combined["phase"].iloc[0])
    for _, row in combined.iloc[1:].iterrows():
        p = int(row["phase"])
        if p != prev_phase:
            # Color the divider line with the *incoming* phase color
            ax.axvline(
                x=row["t"],
                color=PHASE_COLORS[p % len(PHASE_COLORS)],
                linestyle="--",
                alpha=0.70,
                linewidth=1.4,
            )
            prev_phase = p


def _plot_phase_path(ax, x_vals, y_vals, phase_vals, linewidth=2.0):
    """
    Draw a bar-path polyline with segments colored per phase.
    Returns a list of Line2D objects (one per phase segment drawn).

    Parameters
    ----------
    ax          : matplotlib.axes.Axes
    x_vals      : np.ndarray  shape (N,)
    y_vals      : np.ndarray  shape (N,)
    phase_vals  : np.ndarray  shape (N,)  integer phase labels
    linewidth   : float
    """
    # Keep track of which phase labels have already been added to the legend
    seen_phases = set()
    n = len(x_vals)
    if n < 2:
        return

    current_phase = int(phase_vals[0])
    seg_start = 0

    for i in range(1, n):
        new_phase = int(phase_vals[i])
        is_last = i == n - 1

        if new_phase != current_phase or is_last:
            end_idx = i + 1 if is_last else i + 1
            seg_x = x_vals[seg_start:end_idx]
            seg_y = y_vals[seg_start:end_idx]

            color = PHASE_COLORS[current_phase % len(PHASE_COLORS)]
            label = (
                PHASE_LABELS[current_phase]
                if current_phase not in seen_phases
                else "_nolegend_"
            )
            seen_phases.add(current_phase)

            ax.plot(
                seg_x,
                seg_y,
                color=color,
                linewidth=linewidth,
                label=label,
                solid_capstyle="round",
            )

            seg_start = i
            current_phase = new_phase


def _set_path_axis_limits(ax, x_vals, y_vals, inverted_y=True):
    """
    Set axis limits for a bar-path plot with generous horizontal padding.

    Horizontal padding is based on the vertical range (the dominant axis for
    any bar path) so the breathing room is consistent even when the bar moves
    only a tiny amount sideways.  A minimum absolute pad is also enforced so
    short lifts never look cramped.

    Parameters
    ----------
    ax        : matplotlib.axes.Axes
    x_vals    : np.ndarray
    y_vals    : np.ndarray
    inverted_y: bool  – when True the Y axis is already inverted; limits are
                        set accordingly.
    """
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Horizontal pad: 30 % of the vertical range, at least 15 % of the
    # horizontal range, and never less than a small absolute floor so the
    # path is never pinned to the edges.
    abs_floor = max(y_range * 0.05, x_range * 0.05, 5.0)
    pad_x = max(y_range * 0.30, x_range * 0.15, abs_floor)

    # Vertical pad: modest – just enough so markers aren't clipped.
    pad_y = max(y_range * 0.08, abs_floor * 0.5)

    if inverted_y:
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_max + pad_y, y_min - pad_y)  # inverted
    else:
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)


def _draw_start_end_markers(ax, x_vals, y_vals):
    """
    Plot a clearly distinguishable start marker (white circle, black edge)
    and end marker (black diamond, white edge) that do not conflict with any
    phase color.
    """
    ax.plot(
        x_vals[0],
        y_vals[0],
        marker="o",
        color=START_MARKER_COLOR,
        markeredgecolor=START_MARKER_EDGE,
        markeredgewidth=1.5,
        markersize=11,
        zorder=5,
        label="Start",
    )
    ax.plot(
        x_vals[-1],
        y_vals[-1],
        marker="D",
        color=END_MARKER_COLOR,
        markeredgecolor=END_MARKER_EDGE,
        markeredgewidth=1.5,
        markersize=9,
        zorder=5,
        label="End",
    )


# ---------------------------------------------------------------------------
# Per-graph generators
# ---------------------------------------------------------------------------


def plot_barbell_lateral_corrected(df, output_dir):
    """
    Plot perspective-corrected bar path in real-world centimetres.

    Uses barbell_x_corrected_cm / barbell_y_corrected_cm produced by the
    shoulder-geometry px→m conversion in step 2.  Both axes are in the same
    physical unit (cm) so the aspect ratio is always believable.
    """
    path_cols = ["barbell_x_corrected_cm", "barbell_y_corrected_cm", "bar_phase"]
    if not all(col in df.columns for col in path_cols):
        print("Skipping corrected path plot (no correction data available)")
        return

    path_data_df = df[path_cols].dropna()
    if len(path_data_df) < 2:
        print("Skipping corrected path plot (insufficient data points)")
        return

    x_vals = path_data_df["barbell_x_corrected_cm"].values
    y_vals = path_data_df["barbell_y_corrected_cm"].values
    phase_vals = path_data_df["bar_phase"].values.astype(int)

    # ------------------------------------------------------------------
    # Figure sizing: derive a sensible figure height from the data range
    # so the plot is neither too tall nor too wide.
    # ------------------------------------------------------------------
    x_range = x_vals.max() - x_vals.min()
    y_range = y_vals.max() - y_vals.min()
    # Give at least 5 cm of padding on each axis for readability
    x_span = max(x_range, 5.0)
    y_span = max(y_range, 5.0)
    # Base width ~6 inches; height scaled to match the data aspect ratio
    fig_width = 6.0
    fig_height = max(4.0, min(12.0, fig_width * (y_span / x_span)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    _plot_phase_path(ax, x_vals, y_vals, phase_vals)
    _draw_start_end_markers(ax, x_vals, y_vals)

    ax.set_title(
        "Angle-Compensated Bar Path\n(Pull / Pull-under / Recovery)",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Horizontal Displacement (cm)", fontsize=12)
    ax.set_ylabel("Vertical Displacement (cm)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Y axis: upward = positive (bar rises), so invert because y_corrected_cm
    # increases downward in image-space.
    ax.invert_yaxis()
    ax.set_aspect("equal")

    _set_path_axis_limits(ax, x_vals, y_vals, inverted_y=True)

    # Annotation box: camera yaw + scale info
    annotation_lines = []
    if "camera_yaw_deg" in df.columns:
        camera_yaw_series = df["camera_yaw_deg"].dropna()
        if len(camera_yaw_series) > 0:
            yaw_val = camera_yaw_series.iloc[0]
            if not pd.isna(yaw_val):
                annotation_lines.append(f"Camera yaw: {float(yaw_val):.1f}\u00b0")
    if "px_to_m_scale" in df.columns:
        scale_series = df["px_to_m_scale"].dropna()
        if len(scale_series) > 0:
            median_scale_mm = float(scale_series.median()) * 1000.0
            annotation_lines.append(f"Scale: {median_scale_mm:.2f} mm/px")

    if annotation_lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
            fontsize=9,
        )

    # Legend: phase patches + start/end markers
    phase_handles = _phase_legend_handles()
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    marker_handles = [
        h
        for h, lbl in zip(existing_handles, existing_labels)
        if lbl in ("Start", "End")
    ]
    ax.legend(handles=phase_handles + marker_handles, loc="best", fontsize=9)

    output_path = Path(output_dir) / "barbell_lateral_corrected_path.png"
    plt.savefig(output_path, dpi=GRAPH_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Generated: {output_path}")


def _plot_xy_path(df, x_col, y_col, title, filename, output_dir):
    """
    Shared helper: plot a 2-D bar path (X vs Y) colored by phase.
    """
    path_cols = [x_col, y_col, "bar_phase"]
    if not all(col in df.columns for col in path_cols):
        print(f"Warning: Missing columns for '{title}'. Skipping.")
        return None

    path_data_df = df[path_cols].dropna()
    if len(path_data_df) < 2:
        print(
            f"Warning: Insufficient data for '{title}' ({len(path_data_df)} points). Skipping."
        )
        return None

    x_vals = path_data_df[x_col].values
    y_vals = path_data_df[y_col].values
    phase_vals = path_data_df["bar_phase"].values.astype(int)

    fig, ax = plt.subplots(figsize=(GRAPH_WIDTH_PATH, GRAPH_HEIGHT_PATH))

    _plot_phase_path(ax, x_vals, y_vals, phase_vals)
    _draw_start_end_markers(ax, x_vals, y_vals)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Horizontal Position (px)", fontsize=12)
    ax.set_ylabel("Vertical Position (px)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    _set_path_axis_limits(ax, x_vals, y_vals, inverted_y=True)

    # Legend: phase patches + start/end markers
    phase_handles = _phase_legend_handles()
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    marker_handles = [
        h
        for h, lbl in zip(existing_handles, existing_labels)
        if lbl in ("Start", "End")
    ]
    ax.legend(handles=phase_handles + marker_handles, loc="best", fontsize=9)

    graph_path = os.path.join(output_dir, filename)
    plt.savefig(graph_path, dpi=GRAPH_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Generated: {graph_path}")
    return graph_path


def _plot_timeseries(df, y_col, title, y_label, output_dir):
    """
    Shared helper: plot a kinematic time-series with phase shading, vertical
    transition lines, and a proper phase legend.
    """
    if y_col not in df.columns:
        print(f"Warning: Column '{y_col}' not found. Skipping graph.")
        return None

    valid_data = df[["time_s", y_col]].dropna()
    if len(valid_data) < 2:
        print(
            f"Warning: Insufficient data for '{title}' ({len(valid_data)} points). Skipping."
        )
        return None

    fig, ax = plt.subplots(figsize=(GRAPH_WIDTH_TIMESERIES, GRAPH_HEIGHT_TIMESERIES))

    # --- Phase shading (drawn first so it sits behind the data line) ---
    if "bar_phase" in df.columns:
        phase_aligned = df["bar_phase"].reindex(valid_data.index)
        _add_phase_shading(ax, phase_aligned, valid_data["time_s"])
        _add_phase_vlines(ax, phase_aligned, valid_data["time_s"])

    # --- Data line ---
    ax.plot(
        valid_data["time_s"],
        valid_data[y_col],
        color="#2060c0",
        linewidth=1.6,
        label=y_label,
        zorder=3,
    )

    # Zero reference line (only if the data crosses zero)
    y_min, y_max = ax.get_ylim()
    if y_min < 0 < y_max:
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.35, linewidth=0.9)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, alpha=0.3, zorder=0)

    # Legend: phase shading patches + data line
    phase_handles = _phase_legend_handles()
    data_handle, _ = ax.get_legend_handles_labels()
    # The data line is the last handle added
    ax.legend(
        handles=phase_handles + [data_handle[-1]] if data_handle else phase_handles,
        fontsize=9,
        loc="best",
    )

    graph_path = os.path.join(output_dir, f"{y_col}_graph.png")
    plt.savefig(graph_path, dpi=GRAPH_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Generated: {graph_path}")
    return graph_path


# ---------------------------------------------------------------------------
# DTW similarity and reference-scale helpers
# ---------------------------------------------------------------------------


def _dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    Compute the Dynamic Time Warping distance between two 2-D point sequences.

    Each sequence is shaped ``(N, 2)`` where columns are (x, y).  The DTW
    cost matrix is built with Euclidean point-to-point distances and the
    standard DP recurrence.  The raw accumulated cost at the corner is
    returned (not normalised — use :func:`_dtw_similarity_pct` for a
    percentage score).

    Parameters
    ----------
    seq_a, seq_b : ndarray, shape (N, 2) and (M, 2)

    Returns
    -------
    float
        DTW distance (lower = more similar).
    """
    raw, _ = _dtw_distance_with_steps(seq_a, seq_b)
    return raw


def _dtw_distance_with_steps(seq_a: np.ndarray, seq_b: np.ndarray) -> tuple:
    """
    Compute the DTW distance and the length of the optimal warping path.

    Uses the standard DP recurrence with Euclidean point-to-point cost.
    After filling the cost matrix the optimal path is traced back from
    ``(N, M)`` to ``(1, 1)`` to count the number of warping steps ``W``.

    Returns
    -------
    (distance, n_steps) : (float, int)
        ``distance``  – accumulated DTW cost along the optimal path.
        ``n_steps``   – number of cells in the optimal path (>= max(N, M)).
    """
    n, m = len(seq_a), len(seq_b)
    # Accumulated cost matrix (1-indexed; row/col 0 are sentinels)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(seq_a[i - 1] - seq_b[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    # Trace back the optimal path to count steps
    i, j = n, m
    n_steps = 0
    while i > 0 or j > 0:
        n_steps += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            best = min(dtw[i - 1, j - 1], dtw[i - 1, j], dtw[i, j - 1])
            if dtw[i - 1, j - 1] == best:
                i -= 1
                j -= 1
            elif dtw[i - 1, j] == best:
                i -= 1
            else:
                j -= 1

    return float(dtw[n, m]), max(n_steps, 1)


def _path_length(seq: np.ndarray) -> float:
    """
    Return the total arc-length of a 2-D point sequence ``(N, 2)``.
    Used to normalise DTW distance so it is independent of sequence length.
    """
    if len(seq) < 2:
        return 1.0
    diffs = np.diff(seq, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1))) or 1.0


def _dtw_similarity_pct(
    ref_xy: np.ndarray,
    other_xy: np.ndarray,
) -> float:
    """
    Return a percentage similarity score in [0, 100] between two 2-D bar
    paths using Dynamic Time Warping.

    Normalisation
    -------------
    The raw DTW cost is the sum of Euclidean distances along the optimal
    warping path.  We normalise it by two factors to make it independent of
    both sequence length and absolute path scale:

    1. **Step count** – divide by the number of warping steps ``W``
       (the length of the optimal warping path, ``1 <= W <= N+M-1``).
       This gives the *mean per-step deviation* in the same units as the
       path coordinates.

    2. **Path scale** – divide by the bounding-box diagonal of the
       reference path.  This converts the mean deviation into a
       dimensionless fraction of the overall path extent, so a 3 cm
       deviation on a 160 cm path scores differently from a 3 cm
       deviation on a 10 cm path.

    The resulting ``d_norm`` is the mean warping deviation as a fraction
    of the path's spatial extent.  A value of 0 means the paths are
    identical; a value of 1 means the average warp step deviates by the
    full diagonal of the bounding box.

    Mapping to percentage
    ---------------------
    An exponential decay converts ``d_norm`` to a human-readable score:

        similarity = 100 * exp(-k * d_norm)

    ``k = 5`` is calibrated so the scale is intuitive for bar-path work:

    * d_norm = 0.00  → 100 %  (identical)
    * d_norm = 0.05  →  78 %  (very similar — minor style differences)
    * d_norm = 0.10  →  61 %  (noticeable shape difference)
    * d_norm = 0.14  →  50 %  (substantially different)
    * d_norm = 0.20  →  37 %  (very different)

    Parameters
    ----------
    ref_xy   : ndarray (N, 2)  – reference lift path
    other_xy : ndarray (M, 2)  – comparison lift path

    Returns
    -------
    float in [0, 100]
    """
    if len(ref_xy) < 2 or len(other_xy) < 2:
        return 0.0

    raw, n_steps = _dtw_distance_with_steps(ref_xy, other_xy)

    # Mean per-step deviation in path coordinate units
    mean_dev = raw / max(n_steps, 1)

    # Characteristic scale: bounding-box diagonal of the reference path
    ref_x_range = float(ref_xy[:, 0].max() - ref_xy[:, 0].min())
    ref_y_range = float(ref_xy[:, 1].max() - ref_xy[:, 1].min())
    bbox_diag = float(np.sqrt(ref_x_range**2 + ref_y_range**2))
    if bbox_diag < 1e-6:
        bbox_diag = 1.0

    d_norm = mean_dev / bbox_diag

    return float(100.0 * np.exp(-DTW_SIMILARITY_K * d_norm))


def _find_phase_transition_points(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    phase_vals: np.ndarray,
) -> dict:
    """
    Return a dict mapping ``'A->B'`` strings to ``(x, y)`` tuples for every
    phase transition found in *phase_vals*.
    """
    pts: dict = {}
    for i in range(1, len(phase_vals)):
        if phase_vals[i] != phase_vals[i - 1]:
            key = f"{phase_vals[i - 1]}->{phase_vals[i]}"
            pts[key] = (float(x_vals[i]), float(y_vals[i]))
    return pts


def _uniform_scale_to_reference(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    ref_phases: np.ndarray,
    other_x: np.ndarray,
    other_y: np.ndarray,
    other_phases: np.ndarray,
) -> float:
    """
    Find the single uniform scale factor ``s`` that best maps the
    *other* lift onto the *reference* lift by minimising the sum of
    squared distances between matching phase-transition markers.

    Both paths must already be translated so the pull-under start sits
    at the origin ``(0, 0)``.  The pull-under origin itself is the same
    for both (always (0,0)) and so is excluded from the optimisation —
    only the ``1->2`` (pull-under→recovery) and ``0->1`` markers that are
    not at the origin are used.

    If no usable non-origin markers are found the function falls back to
    matching the total arc-length of the two paths (a reasonable proxy for
    overall scale).

    Parameters
    ----------
    ref_x, ref_y, ref_phases     : reference path arrays
    other_x, other_y, other_phases : path to be scaled

    Returns
    -------
    float  – scale factor (multiply *other* x/y by this value)
    """
    ref_pts = _find_phase_transition_points(ref_x, ref_y, ref_phases)
    other_pts = _find_phase_transition_points(other_x, other_y, other_phases)

    # Collect matching non-origin marker pairs
    numerator = 0.0
    denominator = 0.0

    common_keys = set(ref_pts.keys()) & set(other_pts.keys())
    for key in common_keys:
        rx, ry = ref_pts[key]
        ox, oy = other_pts[key]
        # Skip the pull-under anchor itself (it is (0,0) for both)
        if abs(ox) < 1e-9 and abs(oy) < 1e-9:
            continue
        # Least-squares 1-D scale: s = dot(other_pt, ref_pt) / dot(other_pt, other_pt)
        numerator += ox * rx + oy * ry
        denominator += ox * ox + oy * oy

    if denominator > 1e-9:
        scale = numerator / denominator
        return float(np.clip(scale, SCALE_MIN, SCALE_MAX))

    ref_len = _path_length(np.column_stack([ref_x, ref_y]))
    other_len = _path_length(np.column_stack([other_x, other_y]))
    if other_len > 1e-9:
        return float(np.clip(ref_len / other_len, SCALE_MIN, SCALE_MAX))

    return 1.0


def _get_lift_yaw(df: pd.DataFrame) -> float:
    """Return the camera yaw (degrees) stored in df, or NaN if unavailable."""
    if "camera_yaw_deg" not in df.columns:
        return float("nan")
    valid = df["camera_yaw_deg"].dropna()
    if valid.empty:
        return float("nan")
    return float(valid.iloc[0])


def _find_pull_under_anchor(phase_vals: np.ndarray) -> int:
    """
    Return the index (into phase_vals) of the first frame where the phase
    transitions from Pull (0) to Pull-under (1).

    If no such transition exists — e.g. phase detection failed or the lift
    only has a single phase — returns 0 so callers always get a valid index.
    """
    for i in range(1, len(phase_vals)):
        if phase_vals[i - 1] == 0 and phase_vals[i] == 1:
            return i
    return 0


def _extract_lift_path(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    phase_col: str,
    y_trunc_col: str,
):
    """
    Extract (x_vals, y_vals, phase_vals) from *df* using the given columns,
    truncating at peak bar height and translating so the pull-under start
    point sits at (0, 0).

    The anchor point is the first frame where the phase transitions from
    Pull (0) to Pull-under (1).  If that transition is not present the first
    data point is used as the origin instead (graceful fallback).

    Returns None if there are fewer than 2 usable data points.
    """
    working_df = df
    if y_trunc_col in df.columns and bool(df[y_trunc_col].notna().any()):
        peak_idx = df[y_trunc_col].idxmin()
        working_df = df.loc[:peak_idx]

    path_df = working_df[[x_col, y_col, phase_col]].dropna()
    if len(path_df) < 2:
        return None

    x_vals = np.asarray(path_df[x_col], dtype=float)
    y_vals = np.asarray(path_df[y_col], dtype=float)
    phase_vals = np.asarray(path_df[phase_col], dtype=int)

    anchor = _find_pull_under_anchor(phase_vals)
    x_vals = x_vals - x_vals[anchor]
    y_vals = y_vals - y_vals[anchor]
    return x_vals, y_vals, phase_vals


def _draw_superimposed_figure(
    lift_paths,
    output_dir,
    filename,
    title,
    unit_label,
    use_filenames,
    similarity_scores: Optional[list] = None,
):
    """
    Shared rendering helper for superimposed bar-path graphs.

    Parameters
    ----------
    lift_paths : list of (label, x_vals, y_vals, phase_vals)
        Pre-extracted, origin-normalised path arrays (one per lift).
        For non-reference lifts these are the *scaled* arrays so the visual
        comparison reflects the scale normalisation.
    output_dir : str or Path
    filename : str
        Output PNG filename (no directory).
    title : str
        Graph title (may contain newlines).
    unit_label : str
        Axis unit string, e.g. ``"cm"`` or ``"px"``.
    use_filenames : bool
        When True use the raw label; when False use "Lift N".
    similarity_scores : list of float or None, optional
        Per-lift DTW similarity percentages (same length as ``lift_paths``).
        Index 0 (the reference lift) should be ``None``; subsequent entries
        are the score vs the reference.  When provided the percentage is
        appended to the legend label.
    """
    if not lift_paths:
        print(f"Skipping {filename}: no lift paths to draw.")
        return

    if similarity_scores is None:
        similarity_scores = [None] * len(lift_paths)

    fig, ax = plt.subplots(figsize=(GRAPH_WIDTH_PATH, GRAPH_HEIGHT_PATH))

    all_x = np.concatenate([p[1] for p in lift_paths])
    all_y = np.concatenate([p[2] for p in lift_paths])

    lift_line_handles = []
    seen_phases_global: set = set()

    for lift_idx, (file_label, x_vals, y_vals, phase_vals) in enumerate(lift_paths):
        lift_color = LIFT_PALETTE[lift_idx % len(LIFT_PALETTE)]
        base_label = file_label if use_filenames else f"Lift {lift_idx + 1}"

        # Append DTW similarity to every lift except the reference (lift 0)
        score = (
            similarity_scores[lift_idx] if lift_idx < len(similarity_scores) else None
        )
        if score is not None:
            legend_label = f"{base_label}  [{score:.1f}% match]"
        else:
            legend_label = base_label

        (line_handle,) = ax.plot(
            x_vals,
            y_vals,
            color=lift_color,
            linewidth=2.0,
            alpha=0.85,
            solid_capstyle="round",
            label=legend_label,
            zorder=3,
        )
        lift_line_handles.append(line_handle)

        # Phase-transition dots: fill = phase color, edge = lift color
        for i in range(1, len(phase_vals)):
            if phase_vals[i] != phase_vals[i - 1]:
                transition_phase = int(phase_vals[i])
                phase_color = PHASE_COLORS[transition_phase % len(PHASE_COLORS)]
                ax.plot(
                    x_vals[i],
                    y_vals[i],
                    marker="o",
                    color=phase_color,
                    markersize=7,
                    markeredgecolor=lift_color,
                    markeredgewidth=1.2,
                    zorder=5,
                    label="_nolegend_",
                )
                seen_phases_global.add(transition_phase)

    # Phase-dot legend entries (one per observed phase)
    phase_dot_handles = []
    for phase_id in sorted(PHASE_COLORS):
        if phase_id in seen_phases_global:
            phase_dot_handles.append(
                mlines.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=PHASE_COLORS[phase_id],
                    markeredgecolor="#555555",
                    markeredgewidth=0.8,
                    markersize=8,
                    label=f"\u2192 {PHASE_LABELS[phase_id]}",
                )
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"Horizontal Displacement ({unit_label})", fontsize=12)
    ax.set_ylabel(f"Vertical Displacement ({unit_label})", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    _set_path_axis_limits(ax, all_x, all_y, inverted_y=True)

    ax.legend(
        handles=lift_line_handles + phase_dot_handles,
        loc="best",
        fontsize=9,
        title="Lifts / Phase transitions",
        title_fontsize=9,
    )

    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=GRAPH_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Generated: {output_path}")


def _compute_scaled_paths_and_similarity(lift_paths):
    """
    Given a list of ``(label, x_vals, y_vals, phase_vals)`` tuples:

    1. Keep the first lift (index 0) as the reference — its coordinates are
       unchanged.
    2. For every subsequent lift, compute the best uniform scale factor that
       maps its phase-transition markers onto the reference markers
       (least-squares, see :func:`_uniform_scale_to_reference`), then scale
       its x and y arrays by that factor.
    3. Compute the DTW similarity percentage between each scaled non-reference
       path and the reference path (on the *scaled* coordinates so the
       comparison reflects the shape rather than the absolute size).

    Returns
    -------
    scaled_paths : list of (label, x_vals, y_vals, phase_vals)
        Reference lift unchanged; others have scaled x/y.
    similarity_scores : list of float or None
        Index 0 is ``None`` (reference has no score vs itself); subsequent
        entries are the DTW percentage for that lift vs the reference.
    scale_factors : list of float
        Scale factor applied to each lift (1.0 for the reference).
    """
    if not lift_paths:
        return lift_paths, [], []

    scaled_paths = []
    similarity_scores: list = []
    scale_factors: list = []

    ref_label, ref_x, ref_y, ref_ph = lift_paths[0]
    scaled_paths.append((ref_label, ref_x, ref_y, ref_ph))
    similarity_scores.append(None)  # reference has no score vs itself
    scale_factors.append(1.0)

    ref_xy = np.column_stack([ref_x, ref_y])

    for label, x, y, ph in lift_paths[1:]:
        # Find the best uniform scale
        s = _uniform_scale_to_reference(ref_x, ref_y, ref_ph, x, y, ph)
        sx = x * s
        sy = y * s
        scaled_paths.append((label, sx, sy, ph))
        scale_factors.append(s)

        # DTW similarity on the scaled path vs the reference
        other_xy = np.column_stack([sx, sy])
        pct = _dtw_similarity_pct(ref_xy, other_xy)
        similarity_scores.append(pct)

        print(
            f"  Superimposed scaling: '{label}' scale={s:.4f}, "
            f"DTW similarity={pct:.1f}%"
        )

    return scaled_paths, similarity_scores, scale_factors


def plot_superimposed_paths_compensated(
    video_data_list, output_dir, use_filenames=False
):
    """
    Superimposed bar-path graph using real-world centimetre traces.

    Every lift uses its ``barbell_x/y_corrected_cm`` columns when they exist
    and contain at least one non-NaN value.  These columns are now produced
    for all lifts — angled-view lifts use the shoulder-width scale (Path A)
    and side-on lifts use the hip-to-shoulder vertical scale (Path B) — so
    all traces are in the same physical unit (cm) and can be directly compared.

    A lift falls back to its smoothed pixel columns only if the corrected_cm
    columns are missing or entirely NaN (e.g. world_landmarks were absent).
    Lifts where neither column set exists are silently skipped.

    The Y axis is labelled "cm" whenever at least one lift has corrected data,
    and "px" only if every lift fell back to pixels.

    Non-reference lifts are uniformly scaled to best match the reference
    lift's phase-transition marker positions (no distortion — both x and y
    are multiplied by the same scalar).  DTW similarity percentages vs the
    reference are shown in the legend.

    Parameters
    ----------
    video_data_list : list of (label, df) tuples
    output_dir : str or Path
    use_filenames : bool
    """
    if not video_data_list:
        print("Skipping compensated superimposed path: no data provided.")
        return

    SMOOTH_COLS = ["barbell_x_smooth", "barbell_y_smooth", "bar_phase"]
    CORR_COLS = [
        "barbell_x_corrected_cm",
        "barbell_y_corrected_cm",
        "bar_phase",
    ]

    raw_lift_paths = []
    any_cm = False

    for lift_idx, (file_label, df) in enumerate(video_data_list):
        # Use corrected_cm for any lift that has valid (non-NaN) cm data,
        # regardless of camera yaw — side-on lifts now produce cm columns too.
        use_corrected = (
            all(c in df.columns for c in CORR_COLS)
            and df["barbell_x_corrected_cm"].notna().any()
        )

        if use_corrected:
            x_col, y_col, phase_col = CORR_COLS
            y_trunc_col = "barbell_y_corrected_cm"
            any_cm = True
            method = (
                df["scale_method"].iloc[0]
                if "scale_method" in df.columns
                else "unknown"
            )
            print(
                f"  Lift {lift_idx + 1} ({file_label}): using corrected_cm [{method}]"
            )
        elif (
            all(c in df.columns for c in SMOOTH_COLS)
            and df["barbell_x_smooth"].notna().any()
        ):
            x_col, y_col, phase_col = SMOOTH_COLS
            y_trunc_col = "barbell_y_smooth"
            print(
                f"  Lift {lift_idx + 1} ({file_label}): falling back to smoothed px (no cm data)"
            )
        else:
            print(f"  Skipping lift {lift_idx + 1}: no usable path columns.")
            continue

        result = _extract_lift_path(df, x_col, y_col, phase_col, y_trunc_col)
        if result is None:
            print(f"  Skipping lift {lift_idx + 1}: insufficient data points.")
            continue

        raw_lift_paths.append((file_label, result[0], result[1], result[2]))

    if not raw_lift_paths:
        print(
            "Skipping compensated superimposed path: all lifts had insufficient data."
        )
        return

    # Scale non-reference lifts and compute DTW similarity
    lift_paths, similarity_scores, scale_factors = _compute_scaled_paths_and_similarity(
        raw_lift_paths
    )

    unit_label = "cm" if any_cm else "px"
    title = (
        "Superimposed Bar Paths — Real-World Scale\n"
        "(origin-normalised at pull-under start; non-reference lifts uniformly scaled to reference)"
    )

    _draw_superimposed_figure(
        lift_paths,
        output_dir,
        filename="superimposed_bar_paths_compensated.png",
        title=title,
        unit_label=unit_label,
        use_filenames=use_filenames,
        similarity_scores=similarity_scores,
    )


def plot_superimposed_paths_smoothed(video_data_list, output_dir, use_filenames=False):
    """
    Superimposed bar-path graph using the smoothed pixel-space traces for
    every lift, regardless of whether angle-compensated data is available.

    Non-reference lifts are uniformly scaled to best match the reference
    lift's phase-transition marker positions.  DTW similarity percentages
    vs the reference are shown in the legend.

    Parameters
    ----------
    video_data_list : list of (label, df) tuples
    output_dir : str or Path
    use_filenames : bool
    """
    if not video_data_list:
        print("Skipping smoothed superimposed path: no data provided.")
        return

    SMOOTH_COLS = ["barbell_x_smooth", "barbell_y_smooth", "bar_phase"]

    raw_lift_paths = []
    for lift_idx, (file_label, df) in enumerate(video_data_list):
        if not all(c in df.columns for c in SMOOTH_COLS):
            print(f"  Skipping lift {lift_idx + 1}: smoothed columns not found.")
            continue
        if not df["barbell_x_smooth"].notna().any():
            print(f"  Skipping lift {lift_idx + 1}: smoothed columns are all NaN.")
            continue

        result = _extract_lift_path(
            df,
            x_col="barbell_x_smooth",
            y_col="barbell_y_smooth",
            phase_col="bar_phase",
            y_trunc_col="barbell_y_smooth",
        )
        if result is None:
            print(f"  Skipping lift {lift_idx + 1}: insufficient data points.")
            continue

        raw_lift_paths.append((file_label, result[0], result[1], result[2]))

    if not raw_lift_paths:
        print("Skipping smoothed superimposed path: all lifts had insufficient data.")
        return

    # Scale non-reference lifts and compute DTW similarity
    lift_paths, similarity_scores, scale_factors = _compute_scaled_paths_and_similarity(
        raw_lift_paths
    )

    _draw_superimposed_figure(
        lift_paths,
        output_dir,
        filename="superimposed_bar_paths_smoothed.png",
        title="Superimposed Bar Paths — Smoothed\n(origin-normalised; non-reference lifts uniformly scaled to reference)",
        unit_label="px",
        use_filenames=use_filenames,
        similarity_scores=similarity_scores,
    )


# ---------------------------------------------------------------------------
# Main step function
# ---------------------------------------------------------------------------


def step_3_generate_graphs(df, output_dir):
    """
    Takes the final analysis DataFrame and generates all kinematic graphs.
    """
    print("--- Step 3: Generating Kinematic Graphs ---")
    if df.empty:
        print("Error: No data in DataFrame.")
        return

    if "time_s" not in df.columns:
        print("Error: 'time_s' column not found in data.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # --- Truncate data at maximum bar height ---
    if "barbell_y_smooth" in df.columns and df["barbell_y_smooth"].notna().any():
        peak_height_idx = df["barbell_y_smooth"].idxmin()
        print(f"Truncating graphs data at peak height (index {peak_height_idx}).")
        df = df.loc[:peak_height_idx]

    graph_files = []
    skipped = []

    # ------------------------------------------------------------------
    # 1. Kinematic time-series graphs
    # ------------------------------------------------------------------
    kinematics = [
        (
            "Smoothed Vertical Velocity (px/s)",
            "vel_y_smooth",
            "Velocity (px/s)",
        ),
        (
            "Smoothed Vertical Acceleration (px/s²)",
            "accel_y_smooth",
            "Acceleration (px/s²)",
        ),
        (
            "Smoothed Vertical Specific Power",
            "specific_power_y_smooth",
            "Specific Power (px²/s³)",
        ),
    ]

    for title, col, y_label in kinematics:
        result = _plot_timeseries(df, col, title, y_label, output_dir)
        if result:
            graph_files.append(result)
        else:
            skipped.append(title)

    # ------------------------------------------------------------------
    # 2. Smoothed bar path (X-Y)
    # ------------------------------------------------------------------
    result = _plot_xy_path(
        df,
        x_col="barbell_x_smooth",
        y_col="barbell_y_smooth",
        title="Smoothed Bar Path (Pull / Pull-under / Recovery)",
        filename="barbell_xy_stable_path.png",
        output_dir=output_dir,
    )
    if result:
        graph_files.append(result)
    else:
        skipped.append("Smoothed Bar Path")

    # ------------------------------------------------------------------
    # 3. Unsmoothed (stabilized) bar path (X-Y)
    # ------------------------------------------------------------------
    result = _plot_xy_path(
        df,
        x_col="barbell_x_stable",
        y_col="barbell_y_stable",
        title="Unsmoothed Bar Path (Pull / Pull-under / Recovery)",
        filename="barbell_xy_stable_path_unsmoothed.png",
        output_dir=output_dir,
    )
    if result:
        graph_files.append(result)
    else:
        skipped.append("Unsmoothed Bar Path")

    # ------------------------------------------------------------------
    # 4. Perspective-corrected lateral path (optional)
    # ------------------------------------------------------------------
    if (
        "barbell_x_corrected_cm" in df.columns
        and df["barbell_x_corrected_cm"].notna().any()
    ):
        try:
            plot_barbell_lateral_corrected(df, output_dir)
        except Exception as e:
            print(f"Warning: Could not generate corrected path graph: {e}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\nStep 3 Complete.")
    print(f"  Generated: {len(graph_files)} graphs in '{output_dir}'")
    if skipped:
        print(f"  Skipped:   {len(skipped)} graphs due to missing/insufficient data")
        for title in skipped:
            print(f"    - {title}")

    plt.close("all")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Generate kinematic graphs from analysis CSV."
    )
    parser.add_argument(
        "--input",
        default="final_analysis.csv",
        help="Path to the final analysis CSV file from Step 2.",
    )
    parser.add_argument(
        "--output_dir",
        default="graphs",
        help="Directory to save the generated graph PNGs.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        return

    try:
        df = pd.read_csv(args.input)
        print(f"Loaded data: {len(df)} frames, {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading CSV file {args.input}: {e}")
        return

    step_3_generate_graphs(df, args.output_dir)


if __name__ == "__main__":
    main()
