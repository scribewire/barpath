# Phase 02: Live Preview HUD Upgrade - Context

**Gathered:** 2026-05-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Upgrade the offline output video HUD (Step 5, `barpath/pipeline/5_render_video.py`) with full telemetry overlay: bar path with phase coloring, velocity sparkline, power zone band, knee joint angles (color-coded vs pro baselines), and fault error markers (top 3 faults as colored triangles on bar path). Minimal live preview change: add a single coaching tip overlay (top-right) post-lift, based on buffered lift data with heuristic fault checks (confidence threshold 0.6). Skeleton overlay shown on both offline and live views.

</domain>

<decisions>
## Implementation Decisions

### Scope & Architecture
- **D-01:** Primary target is Step 5 offline video render (`5_render_video.py`). Live preview gets minimal change only — a single coaching tip text overlay in top-right corner, shown for 5 seconds after lift completes (instant appear/disappear, no animation).
- **D-02:** New `barpath/pipeline/step5_helpers/` package with `hud_renderer.py` for HUD drawing functions. Follows existing `step{1,2,4}_helpers/` pattern. `5_render_video.py` stays as the render orchestrator.
- **D-03:** Orchestrator (`barpath_core.py`) loads analysis data (CSV + Step 4 fault output) and passes it to `step_5_render_video()` as parameters. Step 5 stays a renderer, not a data loader.
- **D-04:** Render by mutating frames directly (OpenCV drawing functions) — same pattern as existing Step 5.

### HUD Layout (All 6 Elements)
- **D-05:** Overlay on video (consistent with existing Step 5 pattern). Full layout:
  - **Bar path trail:** Center (phase-colored: Pull=red, Pull-under=orange, Recovery=green)
  - **Phase markers:** On bar path at transition points
  - **Velocity text:** Near barbell position per frame (existing Step 5 behavior)
  - **Phase legend:** Bottom-left (existing Step 5 behavior)
  - **Velocity sparkline:** Top-right, polyline with phase-colored segments, full curve from frame 1 (pre-computed, not building up). Proportional sizing: 20% width, 15% height.
  - **Power zone band:** Below sparkline, same time axis, single-hue intensity (light=low→dark=high), normalized to lift max power (relative scale)
  - **Joint angles:** Bottom-center row, knees only (left + right), color-coded vs phase-specific thresholds from pro baseline data (green=good, yellow=borderline, red=outside range)
  - **Error markers:** Small filled triangles on bar path at fault frames, color-coded by fault type, top 3 faults only (remaining listed as text in legend)
  - **Skeleton overlay:** Shown on both offline and live views
- **D-06:** Proportional sizing (relative to frame size) — avoids breakage across different webcam/video resolutions.
- **D-07:** Distinct color palette per HUD element (not reusing phase colors for everything). Phase colors reserved for bar path, sparkline, and legend.

### Implementation Phasing
- **D-08:** Implement in order: (1) bar path trail + phase labels (existing, may need refinement), (2) velocity sparkline, (3) power zone band, (4) joint angles display, (5) error markers. Each element independently testable.
- **D-09:** CLI toggles for individual HUD elements: `--no-skeleton`, `--no-sparkline`, `--no-power-zones`, `--no-angles`, `--no-error-markers`. Existing `--no-video` still skips all rendering.

### Live Preview Coaching Tip
- **D-10:** Show tip only after lift completes (DISPLAYING state). Buffer lift data during detection, compute joint angles and kinematics from buffer on completion, run heuristic fault checks. Show tip instantly, persist 5 seconds, disappear.
- **D-11:** Tip text: most probable fault name (e.g., "Early arm bend") if confidence > 0.6, otherwise "Lift looks good". Single line, top-right overlay.
- **D-12:** Tip shows instantly (no fade animation). Uses simple heuristic checks — no ML required for live tip. Same fault categories as Step 4's compiled analyzer, but computed from buffered lift data in real-time.

### Fault Markers (Offline Video)
- **D-13:** Fault data from Step 4's `compiled_analyzer` and `smart_analysis` output. Top 3 faults shown as colored triangles on bar path; remaining faults listed as text in legend area.
- **D-14:** Triangle markers color-coded by fault type (distinct colors per fault category — arm, extension, path deviation, etc.).

### the agent's Discretion
- Sparkline rendering implementation (pre-allocate NumPy buffer vs incremental cv2.polylines)
- Exact pixel offsets and margins within proportional size constraints
- Heuristic fault check implementation for live tip (which simplified checks to run on buffered data)
- Power zone band rendering approach (heatmap strip vs horizontal bar)
- Color palette selection within the "distinct per element" principle

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project & Phase
- `.planning/PROJECT.md` — Project roadmap with phase 02 goals and implementation steps
- `.planning/codebase/ARCHITECTURE.md` — Pipeline architecture, live detection subsystem, Step 5 current behavior
- `.planning/codebase/STRUCTURE.md` — File naming conventions, where to add new code (step5_helpers pattern)
- `.planning/codebase/STACK.md` — OpenCV version, available libraries for rendering

### Current Step 5 Rendering
- `barpath/pipeline/5_render_video.py` — Existing offline video renderer (421 lines). Reference for drawing patterns (skeleton, bar path, phase coloring, legend, velocity text). This file will be refactored into orchestrator + step5_helpers.
- `barpath/pipeline/utils.py` — Shared video drawing helpers (video_writer setup, font constants)
- `barpath/pipeline/config.py` — Central constants (colors, thresholds, font sizes)

### Live Detection System
- `barpath/pipeline/live_detection_system.py` — Live detection state machine (DetectionState enum: IDLE→DETECTING→COMPLETE→JERK_WATCH→DISPLAYING). Hooks for live coaching tip.
- `barpath/pipeline/live_buffer.py` — CircularFrameBuffer for frame storage during lift detection. Used for buffering lift data for post-lift analysis.
- `barpath/pipeline/lift_classifier.py` — Lift type classification (Random Forest). May inform which heuristic checks to run for live tip.

### Analysis Output (Data Sources for HUD)
- `barpath/pipeline/2_analyze_data.py` — Produces final_analysis.csv (barbell position, velocity, acceleration, power, joint angles, phases)
- `barpath/pipeline/4_critique_lift.py` — Produces analysis.md + fault probability data from compiled_analyzer and smart_analysis
- `barpath/pipeline/step2_helpers/kinematics.py` — Barbell kinematics calculations (velocity, acceleration, power)
- `barpath/pipeline/step2_helpers/landmark_processing.py` — Joint angle calculations (knee angles, elbow angles)
- `barpath/pipeline/step4_helpers/compiled_analyzer.py` — Rule-based fault detection with pro baselines. Fault definitions and threshold logic.
- `barpath/pipeline/step4_helpers/feature_extraction.py` — Feature extraction for technique analysis

### Orchestrator
- `barpath/barpath_core.py` — Pipeline orchestrator. run_pipeline() calls step_5_render_video(). Needs updated to pass analysis data to Step 5.
- `barpath/barpath_cli.py` — CLI entry point. New --no-* flags for HUD element toggles.
- `barpath/barpath_gui.py` — GUI entry point. Settings tab may need HUD toggle checkboxes.

### Pro Baseline Data (Angle Thresholds)
- `barpath/models/analysis/{lifter}/{lift_type}/pro_baseline_report.json` — Pro athlete baseline percentiles used for joint angle reference ranges

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `barpath/pipeline/5_render_video.py:step_5_render_video()` — Existing frame rendering loop. Reads CSV, iterates frames, applies OpenCV drawing. The skeleton drawing, bar path trail, phase coloring, and legend rendering logic can be extracted into step5_helpers.
- `barpath/pipeline/3_generate_graphs.py` — Matplotlib graph generation (881 lines). Sparkline and power band could reference the same data computation patterns (not the matplotlib rendering, but how data is structured).
- `barpath/pipeline/step2_helpers/kinematics.py` — Velocity, acceleration, and power calculations already exist. HUD reads these values from CSV, doesn't recompute.
- `barpath/pipeline/step2_helpers/landmark_processing.py` — Joint angle calculation functions. Can be reused for live tip heuristic checks.
- `barpath/pipeline/live_buffer.py:CircularFrameBuffer` — Bounded deque pattern. Can be extended or reused for buffering lift data (bar position + joint landmarks) during live detection for post-lift analysis.

### Established Patterns
- Pipeline step helper packages: `step{N}_helpers/` with `__init__.py` re-exports. New `step5_helpers/` follows this.
- `cv2.line`, `cv2.circle`, `cv2.putText`, `cv2.polylines` — Standard OpenCV drawing used throughout Step 5.
- Generator-based progress reporting: Step 5 yields `(step_name, progress, message)` tuples. Keep this pattern.
- `cv2.FONT_HERSHEY_SIMPLEX` for all text. Configurable font scale and thickness via `config.py`.
- Config module constants: All magic numbers in `barpath/pipeline/config.py`. Add new HUD constants (sparkline size ratio, power band height, angle text position) there.

### Integration Points
- `barpath/barpath_core.py:run_pipeline()` line ~430 — Calls `step_5_render_video()`. Currently passes model_path, output_video_path, analysis_csv_path, render_video flag, exercise, lift_type. Must add analysis data from Step 4 output.
- `barpath/barpath_core.py:_import_step_function()` — Dynamic import of step functions. No change needed — Step 5 import still works via this mechanism.
- `barpath/pipeline/live_detection_system.py:DetectionState.DISPLAYING` — State where live tip should appear. Hook here for tip rendering.
- `barpath/barpath_gui.py` — Live preview rendering loop. Needs to call tip overlay drawing function when in DISPLAYING state.
- `barpath/barpath_cli.py` — argparse setup. Add `--no-skeleton`, `--no-sparkline`, `--no-power-zones`, `--no-angles`, `--no-error-markers` flags.

</code_context>

<specifics>
## Specific Ideas

No specific external references or examples — user prefers pragmatic overlay approach consistent with existing Step 5 patterns.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-Live Preview HUD Upgrade*
*Context gathered: 2026-05-03*
