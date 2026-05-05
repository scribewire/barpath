# Phase 02: Live Preview HUD Upgrade - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-05 (update session)
**Phase:** 02-live-preview-hud-upgrade
**Areas discussed:** Step 4→Step 5 Data Flow, Live Coaching Tip Buffering, CLI Flag Integration, step5_helpers Extraction Timing, Sparkline Rendering, Power Band Rendering, Live Fault Checks, Color Palette Location, GUI Toggles

---

## Step 4→Step 5 Data Flow

| Option | Description | Selected |
|--------|-------------|----------|
| Function parameter | Add analysis_result parameter to step_5_render_video(). Clean interface, Step 5 stays pure renderer. | ✓ |
| Step 5 reads analysis.md | Step 5 reads and parses analysis.md directly. No orchestrator change needed. | |
| Separate JSON file | Step 4 writes faults.json, Step 5 reads it. Clean separation but adds file. | |

**User's choice:** Function parameter — orchestrator passes analysis_result dict to Step 5.

---

## Live Coaching Tip Data Buffering

| Option | Description | Selected |
|--------|-------------|----------|
| Extend CircularFrameBuffer | Add barbell_position and landmarks fields to existing buffer. Minimal new code. | ✓ |
| New LiftDataBuffer class | New class storing full LiftData. Cleaner separation but more code. | |
| Detection state object | Store in live_detection_system.py state. No new classes but couples state to storage. | |

**User's choice:** Extend CircularFrameBuffer with barbell position + joint landmarks fields.

---

## CLI Flag Integration Pattern

| Option | Description | Selected |
|--------|-------------|----------|
| Through orchestrator | CLI → barpath_core.run_pipeline() → step_5_render_video(). Follows existing parameter flow. | ✓ |
| HUDConfig object | Create dataclass with toggle flags. More scalable but adds abstraction. | |
| Config module constants | Step 5 reads from config.py. Simple but uses global state. | |

**User's choice:** Pass flags through orchestrator to Step 5.

---

## step5_helpers Extraction Timing

| Option | Description | Selected |
|--------|-------------|----------|
| Extract first | Create step5_helpers/ package before building HUD elements. Clean from start. | ✓ |
| Build then extract | Add HUD elements to 5_render_video.py, verify, then extract. Faster but messy. | |
| Incremental extraction | Extract existing code now, add new elements as helpers incrementally. | |

**User's choice:** Extract step5_helpers/ package first, then implement HUD elements inside it.

---

## Sparkline Rendering

| Option | Description | Selected |
|--------|-------------|----------|
| Pre-allocate numpy buffer | Pre-allocate array, draw polyline once. Faster for offline rendering. | ✓ |
| Incremental cv2.line | Draw segments frame-by-frame. Simpler but slower. | |
| Separate image paste | Render to small image, paste onto frame. Cleaner but adds compositing. | |

**User's choice:** Pre-allocate numpy buffer for full curve, draw with cv2.polylines.

---

## Power Zone Band Rendering

| Option | Description | Selected |
|--------|-------------|----------|
| Per-column rectangles | Draw 1px-wide rectangles per time column with intensity color. Precise control. | ✓ |
| Numpy gradient strip | Create numpy array with gradient, paste onto frame. Faster but less precise. | |
| Single bar with legend | Horizontal bar with color gradient. Simpler but less informative. | |

**User's choice:** Per-column rectangles with intensity-based color (matches UI-SPEC.md formula).

---

## Live Fault Check Implementation

| Option | Description | Selected |
|--------|-------------|----------|
| Simplified compiled rules | Reuse simplified versions of compiled_analyzer rules. Consistent with offline analysis. | ✓ |
| Standalone heuristic checks | Hardcode 3-4 specific checks. Faster but diverges from Step 4 logic. | |
| Full CompiledAnalyzer | Run full analyzer on buffered data. Most accurate but may be slow. | |

**User's choice:** Simplified compiled rules — consistent with Step 4 but optimized for real-time.

---

## Color Palette Location

| Option | Description | Selected |
|--------|-------------|----------|
| config.py constants | Define all HUD colors as named constants in config.py. Consistent with existing pattern. | ✓ |
| Color utility module | Utility module with palette generation. More flexible but adds complexity. | |
| Inline in helpers | Colors inline in drawing functions. Simplest but harder to maintain. | |

**User's choice:** All HUD colors defined as constants in config.py.

---

## GUI Toggles

| Option | Description | Selected |
|--------|-------------|----------|
| Settings tab checkboxes | Add HUD element checkboxes to existing Settings tab. Consistent with backend dropdown. | ✓ |
| CLI flags mirrored in GUI | GUI builds argparse-like config from settings. | |
| CLI-only for now | HUD toggles only via CLI, GUI shows full HUD always. | |

**User's choice:** Settings tab checkboxes alongside existing backend dropdown.

---

## the agent's Discretion

- Exact pixel offsets and margins within proportional size constraints
- Power zone band color formula implementation details (UI-SPEC.md provides exact BGR formula)
- Frame mapping heuristics for placing fault triangles on bar path (RESEARCH.md Section 10)

## Deferred Ideas

None — discussion stayed within phase scope.
