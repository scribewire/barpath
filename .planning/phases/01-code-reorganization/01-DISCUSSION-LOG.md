# Phase 1: Code Reorganization — Discussion Log

**Date:** 2026-05-02
**Phase:** 1 — Code Reorganization

## Discussion Areas

### Import Strategy
**Question:** How should imports be structured after moving live preview files to realtime_processing/?

**Options Presented:**
1. Absolute imports (barpath.pipeline.realtime_processing.*)
2. Relative imports within package
3. __init__.py re-exports (Recommended)

**Decision:** __init__.py re-exports — all public classes and functions re-exported from `barpath/pipeline/realtime_processing/__init__.py`. Consumers import via `from barpath.pipeline.realtime_processing import LiveDetectionSystem`. Within package, files use relative imports.

### Dependency Boundaries
**Question:** Which files should move to realtime_processing/? The live detection system depends on lift_classifier.py, heuristic_classifier.py, and kinematic_completion.py which also serve the offline pipeline.

**Options Presented:**
1. Move only live_*.py files (Recommended)
2. Move all related files including shared classifiers
3. Move with compatibility shims

**Decision:** Move only live_*.py files. Shared classifiers (lift_classifier.py, heuristic_classifier.py, kinematic_completion.py, lift_detection_features.py) stay in pipeline/ root. Moved files import shared classifiers via absolute imports.

### Scripts Structure
**Question:** How should training scripts in barpath/scripts/ be structured?

**Options Presented:**
1. barpath/scripts/ as package (Recommended)
2. Keep scripts/ at repo root
3. Installable CLI commands

**Decision:** barpath/scripts/ as package with __init__.py. Scripts moved from repo root scripts/ to barpath/scripts/. Run via `python -m barpath.scripts.retrain_lift_classifier` or direct execution. Remove sys.path manipulation — rely on package installation.

### Test File Placement
**Question:** What should happen to the existing test files (tests/test_live_*.py)?

**Options Presented:**
1. Keep tests/ in place, update imports (Recommended)
2. Move tests alongside modules
3. Create tests/ subdirectory

**Decision:** Keep tests/ in place, update imports to point to new realtime_processing location.

## Summary
All four gray areas resolved with recommended options. Clear decisions captured for downstream researcher and planner agents.
