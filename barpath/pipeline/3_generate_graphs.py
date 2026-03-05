import argparse
import os
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import pandas as pd

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Phase color definitions (matplotlib named colors / hex)
# Keep these in one place so every graph is consistent.
# ---------------------------------------------------------------------------
PHASE_COLORS = {
    0: "#e02020",  # Pull        — vivid red
    1: "#f07800",  # Pull-under  — vivid orange
    2: "#18a020",  # Recovery    — vivid green
}
PHASE_LABELS = {
    0: "Pull",
    1: "Pull-under",
    2: "Recovery",
}

# Lighter, transparent fills used for background shading on time-series plots
PHASE_FILL_ALPHA = 0.12

# Start / end marker styles that do NOT conflict with phase colors
START_MARKER_COLOR = "white"
START_MARKER_EDGE = "black"
END_MARKER_COLOR = "black"
END_MARKER_EDGE = "white"


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
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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

    fig, ax = plt.subplots(figsize=(8, 10))

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
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
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

    fig, ax = plt.subplots(figsize=(12, 6))

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
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Generated: {graph_path}")
    return graph_path


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
