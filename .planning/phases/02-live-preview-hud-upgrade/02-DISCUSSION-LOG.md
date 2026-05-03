# Phase 02: Live Preview HUD Upgrade - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-03
**Phase:** 02-live-preview-hud-upgrade
**Areas discussed:** HUD Layout & Design, Rendering Performance, HUD/Data Integration, Activation & Lifecycle, Power Zones Visualization, Error Markers & Coaching Cues, Joint Angles Display

---

## HUD Layout & Design

| Option | Description | Selected |
|--------|-------------|----------|
| Overlay on video (like Step 5) | Everything drawn directly on the video frame — same pattern as offline renderer | ✓ |
| Side panel (split view) | Video on left, telemetry panel on right | |
| Floating mini-dashboard | Semi-transparent compact panel overlaid on corner | |

**User's choice:** Overlay on video consistent with Step 5 pattern.

---

| Option | Description | Selected |
|--------|-------------|----------|
| All six from the start | Full telemetry all elements visible | |
| Phase it in: core first | Start with bar path trail + velocity sparkline + phase labels, then add power zones, joint angles, error markers | ✓ |

**User's choice:** Core elements first, iterate to full HUD.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Keep Step 5 positions, add sparkline top-right | Bar path center, sparkline top-right, phase legend bottom-left | ✓ |
| Consolidated instrument panel | Cluster velocity + phase + timing in bottom-center panel | |
| Two-zone layout | Bar path center-left, sparkline + phase bar top-right | |

**User's choice:** Keep existing Step 5 element positions, add velocity sparkline top-right.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Simple polyline + phase colors | Velocity history as connected polyline, color-coded by phase | ✓ |
| Minimal sparkline (single color) | Single-color thin line showing velocity trend | |

**User's choice:** Phase-colored polyline sparkline.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Just note where they'll go | Define layout slots without implementing | |
| Design full layout now | Sketch all 6 elements' positions for clean future additions | ✓ |

**User's choice:** Design full 6-element layout upfront.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Power/zones + angles panel | Power zone below sparkline (top-right stack), joint angles bottom-center, error markers as icons on bar path | ✓ |
| Distributed across corners | Each element in its own corner | |

**User's choice:** Top-right stack (sparkline + power), bottom-center angles, icons on bar path.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Proportional to frame size | Sparkline = 20% width, 15% height of frame | ✓ |
| Fixed pixel size, clamped | 300x100px, clamped to 30% of frame | |

**User's choice:** Proportional sizing to scale across webcam/video resolutions.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse phase colors, neutral elsewhere | Phase colors for bar path/sparkline, white/black for text | |
| Distinct palette per element | Bar path = phase colors, sparkline = different, power = different, etc. | ✓ |

**User's choice:** Distinct color palette per HUD element.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Show skeleton too | Full body skeleton overlay + HUD elements | ✓ |
| No skeleton on live HUD | Skip skeleton, cleaner view | |

**User's choice:** Show skeleton overlay on both offline and live.

---

## Rendering Performance

**User clarified scope:** Primary target is Step 5 offline video render, not live preview. Performance concerns are minimal for offline rendering.

| Option | Description | Selected |
|--------|-------------|----------|
| Mutate frame directly (like Step 5) | Draw HUD onto frame with cv2.line/circle/putText | ✓ |
| Separate overlay, then composite | Render to separate array, cv2.addWeighted to composite | |

**User's choice:** Mutate frames directly — consistent with existing Step 5 pattern.

---

## HUD/Data Integration

| Option | Description | Selected |
|--------|-------------|----------|
| Read CSV + pass via orchestrator | barpath_core.py loads analysis data, passes to step_5_render_video as parameters | ✓ |
| Step 5 reads everything itself | Step 5 reads CSV + Step 4 output directly | |

**User's choice:** Orchestrator loads data, passes to Step 5. Step 5 stays a pure renderer.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Heuristic quick check | Run simple real-time checks frame-by-frame during lift | |
| Show tip only after lift completes | Buffer lift data, run mini fault analysis on completion, then show tip | ✓ |

**User's choice:** Show tip after lift completes with mini-analysis on buffered data.

---

| Option | Description | Selected |
|--------|-------------|----------|
| 0.6 — cautious | Only show tip when fault probability > 60% | ✓ |
| 0.4 — inclusive | Show tip at > 40% probability | |

**User's choice:** 0.6 threshold — fewer tips, higher confidence.

---

| Option | Description | Selected |
|--------|-------------|----------|
| New step5_helpers/ package | Extract HUD drawing functions, keep 5_render_video.py as orchestrator | ✓ |
| Extend 5_render_video.py inline | Keep everything in one file | |

**User's choice:** New step5_helpers/ package following existing helper pattern.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Save buffered lift data | Buffer bar + joint data during lift, compute on completion | ✓ |
| Incremental computation during lift | Compute angles/velocity frame-by-frame during lift | |

**User's choice:** Buffer data during detection, compute on completion.

---

## Activation & Lifecycle

| Option | Description | Selected |
|--------|-------------|----------|
| Appear after analysis, persist until next lift | Tip stays visible until new lift detected | |
| Appear briefly, fade out | Tip appears for 5 seconds after analysis, then fades out | ✓ |

**User's choice:** Brief appear, 5 second duration, then disappear.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Smooth opacity fade | 0.5s fade in/out with cv2.addWeighted | |
| Instant appear/disappear | Pop in, pop out | ✓ |

**User's choice:** Instant — no fade animation.

---

| Option | Description | Selected |
|--------|-------------|----------|
| No toggles — full HUD always | All elements rendered, no user control | |
| CLI flags for elements | --no-skeleton, --no-sparkline, --no-angles switches | ✓ |

**User's choice:** CLI toggles for individual HUD elements.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Build up frame-by-frame | Sparkline grows as video plays | |
| Full curve from start | Complete velocity curve visible from frame 1 | ✓ |

**User's choice:** Full curve from start — viewer sees complete trajectory immediately.

---

## Power Zones Visualization

| Option | Description | Selected |
|--------|-------------|----------|
| Colored bar path (heatmap) | Bar path changes color by power output | |
| Power zone band beside sparkline | Horizontal color band below sparkline on same time axis | ✓ |
| Max power callout only | Just peak power text label | |

**User's choice:** Power zone band below sparkline, stacked charts on same time axis.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Diverging gradient (blue→white→red) | Low=blue, med=white, high=red | |
| Single-hue intensity (light→dark) | Single color that darkens with power | ✓ |

**User's choice:** Single-hue intensity gradient.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Relative — normalized to lift max | Band intensity scaled 0-100% of peak power | ✓ |
| Absolute — show actual values | Raw specific_power values | |

**User's choice:** Relative scale, normalized to lift max power.

---

## Error Markers & Coaching Cues

| Option | Description | Selected |
|--------|-------------|----------|
| Icons at fault frame on bar path | Warning icon on bar path at fault frame | ✓ |
| Color-coded path segments | Bar path changes color at fault regions | |
| Text callout list in legend area | Faults listed as text in legend | |

**User's choice:** Icons (triangles) at fault frames on the bar path.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Colored circles with letters | Circle with letter (A=arm, E=extension) | |
| Simple triangle markers | Colored filled triangles pointing to fault location | ✓ |

**User's choice:** Simple colored triangles.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Top 3 faults only | Markers for 3 highest-probability faults, rest in legend | ✓ |
| All faults above 50% probability | Show every fault above 50% | |

**User's choice:** Top 3 faults as markers on path, remaining as legend text.

---

## Joint Angles Display

| Option | Description | Selected |
|--------|-------------|----------|
| All 4 — knees + elbows | Left/right knee and elbow angles | |
| Knees only | Knee angles only — most critical for Olympic lifting | ✓ |

**User's choice:** Knees only (left + right).

---

| Option | Description | Selected |
|--------|-------------|----------|
| Raw numbers only | Display angle value, no comparison | |
| Color-coded vs threshold | Green/yellow/red based on phase-specific ideal range | ✓ |

**User's choice:** Color-coded vs pro baseline thresholds.

---

## the agent's Discretion

- Sparkline rendering implementation (NumPy buffer vs incremental cv2.polylines)
- Exact pixel offsets and margins within proportional size constraints
- Heuristic fault check implementation for live coaching tip
- Power zone band rendering approach
- Color palette selection within "distinct per element" principle

## Deferred Ideas

None — discussion stayed within phase scope.
