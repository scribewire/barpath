# Barpath Project Roadmap

## Project Overview
Barpath is a weightlifting analysis tool that processes video of lifts (clean, snatch, jerk), detects barbell with YOLO, tracks body pose with MediaPipe, analyzes kinematics, generates graphs, produces technique critique reports, and renders output video with overlays. It has both a CLI and a Toga-based GUI, plus a live webcam preview system.

## User Goals
1. **Code Reorganization**: Consolidate live preview code into `barpath/pipeline/realtime_processing/` and move training/model generation scripts into `barpath/scripts/`, preserving all functionality.
2. **Live Preview HUD Upgrade**: Full telemetry overlay showing velocity curves, bar path trajectory, power zones, joint angles, and error markers on the video feed post-lift-recognition.
3. **Analysis Graphs Enhancement**: Interactive Plotly/Bokeh graphs with WebView embedding in Toga GUI, showing annotated smoothed bar path, velocity, and power graphs with error markers. Generate `analysis.html` for browser/CLI use.

## Key Decisions
- **Live HUD**: Full telemetry overlay on video feed
- **Interactive HTML**: Plotly/Bokeh with WebView for Toga GUI
- **Testing**: Add tests throughout each phase
- **Documentation**: Full documentation updates

## Phases

### Phase 1: Code Reorganization
**Goal**: Restructure codebase without breaking any functionality.

**Plans:** 4 plans

Plans:
- [ ] 01-01-PLAN.md — Create realtime_processing package, move live_*.py files, update internal imports
- [ ] 01-02-PLAN.md — Create barpath/scripts package, move scripts, remove sys.path manipulation
- [ ] 01-03-PLAN.md — Update external consumer imports (barpath_gui.py, kinematic_completion.py, tests)
- [ ] 01-04-PLAN.md — Verify file structure, import resolution, and run pytest

1.1 Create `barpath/pipeline/realtime_processing/` package
1.2 Move live preview files from `barpath/pipeline/`:
  - `live_lift_recognition.py`
  - `live_window_features.py`
  - `live_training_data.py`
  - `live_feature_extractor.py`
  - `live_detection_system.py`
  - `live_buffer.py`
1.3 Update all imports across the codebase
1.4 Create `barpath/scripts/` package
1.5 Move training/retraining scripts from root `scripts/`:
  - `retrain_lift_classifier.py`
  - `retrain_live_classifier.py`
1.6 Update script paths and references
1.7 Run tests to verify no breakage

### Phase 2: Live Preview HUD Upgrade
**Goal**: Full telemetry overlay on live webcam preview post-lift-recognition.

2.1 Analyze existing analysis output data structures
2.2 Design HUD overlay system for real-time rendering
2.3 Implement velocity curve overlay
2.4 Implement bar path trajectory overlay
2.5 Implement power zones visualization
2.6 Implement joint angles display
2.7 Implement error markers on video feed
2.8 Integrate with existing live preview pipeline
2.9 Optimize for real-time performance

### Phase 3: Interactive Analysis Graphs
**Goal**: Replace static matplotlib graphs with interactive Plotly/Bokeh visualizations.

3.1 Analyze existing `3_generate_graphs.py` output
3.2 Create Plotly-based graph generation module
3.3 Implement smoothed bar path graph with annotations
3.4 Implement velocity graph with phase markers
3.5 Implement power graph with error indicators
3.6 Create `analysis.html` generator for browser/CLI
3.7 Integrate Plotly WebView into Toga GUI
3.8 Add fallback static image generation for environments without WebView

### Phase 4: Testing & Documentation
**Goal**: Comprehensive test coverage and full documentation.

4.1 Add unit tests for reorganized modules
4.2 Add integration tests
4.3 Add tests for HUD overlay system
4.4 Add tests for interactive graph generation
4.5 Update README with setup guides
4.6 Document new HUD features
4.7 Create developer documentation for reorganized structure

## Success Criteria
- All live preview code lives in `barpath/pipeline/realtime_processing/` with no broken imports
- All training scripts live in `barpath/scripts/` and are executable
- Live preview displays full telemetry overlay (velocity, bar path, power, joint angles, errors)
- Analysis view shows interactive graphs with error annotations
- `analysis.html` is generated for browser/CLI use
- All tests pass
- Documentation is complete and accurate

## Risks & Mitigations
- **Import breakage during reorganization**: Mitigated by running tests after each move and using GitNexus impact analysis before changes
- **Real-time performance with HUD overlay**: May need to optimize rendering or reduce overlay complexity; mitigation is to profile and adjust
- **Toga WebView compatibility**: WebView support varies by platform; mitigation is to add static image fallback
