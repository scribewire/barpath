# External Integrations

**Analysis Date:** 2026-05-01

## APIs & External Services

**No external web APIs detected.** This application is fully offline/desktop — it does not call any REST APIs, GraphQL endpoints, or cloud services.

**Model Zoo (pre-trained models bundled in-repo):**
- YOLO barbell detection: `barpath/models/std_nano.pt` (YOLO26 NMS-free model)
- OpenVINO export: `barpath/models/std_nano_openvino_model/` (`.xml` + `.bin`)
- MediaPipe Pose Landmarker: `barpath/models/pose_landmarker_heavy.task`
- Lift detection model: `barpath/models/lift_detection/lift_detection_model.pkl` (scikit-learn)
- Analysis baselines: `barpath/models/analysis/pro_baseline_report.json` (statistical feature data from 298 pro lifts)
- Per-lifter analysis models in `barpath/models/analysis/` for lifters: botev, generic (`generic/` and `generic - Copy/`), ilyin, juniansyah, liao, lovchev, lu, nasar, sagir, sincraian, talakhadze, tian
- Smart Analysis models: Random Forest `.pkl` + features JSON + faults JSON per lifter/lift-type directory

**Training Data Reference (external, not in repo):**
- Barbell detection trained on Roboflow datasets:
  - "Bar path (2025) bar path detection unified (v6)" — `roboflow.com` (cited in README)
  - "barbelldetection (2024) barbell detection (v2)" — `roboflow.com` (cited in README)

## Data Storage

**Databases:**
- None (no SQLite, PostgreSQL, MySQL, etc.)
- Data stored as flat files:
  - **Pickle** (`.pkl`) — Raw per-frame detection data (`raw_data.pkl`)
  - **CSV** (`.csv`) — Enriched kinematic analysis (`final_analysis.csv`)
  - **Markdown** (`.md`) — Technique analysis reports (`analysis.md`)
  - **PNG** (`.png`) — Kinematic graphs in `graphs/` subfolder
  - **MP4** (`.mp4`) — Annotated video output (`output.mp4`)
  - **JSON** (`.json`) — Pro baseline statistics, model configurations

**File Storage:**
- Local filesystem only
- Output directory: `outputs/` (configurable via `--output_dir`)
- All data is ephemeral (produced by pipeline, consumed by user locally)

**Caching:**
- No external caching service (no Redis, Memcached, etc.)
- Pipeline uses a bounded frame queue (size 8, in `1_collect_data.py`) as internal producer-consumer buffer

## Authentication & Identity

**Auth Provider:**
- None (no user authentication, no accounts, no sessions)
- Application runs fully offline with no user identity system

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry, Rollbar, Datadog, or similar)
- Errors print to stderr via Python's traceback and Rich console

**Logs:**
- Print-based logging to stdout/stderr
- No structured logging framework
- Log level configurable via `barpath/pipeline/config.py` `LOG_LEVEL` setting (default: "INFO")

## CI/CD & Deployment

**Hosting:**
- Not hosted (desktop application only)
- Distribution via GitHub releases with Briefcase-built installers

**CI Pipeline:**
- **Ruff Lint** (`.github/workflows/ruff.yml`):
  - Trigger: push/PR to `main`
  - Runner: `ubuntu-latest`
  - Steps: checkout → setup Python 3.11 → install ruff → `ruff check --fix . && ruff format .`
  - No test running, no coverage, no build/deploy steps in CI

**Build System:**
- Briefcase (BeeWare) for standalone installer packaging:
  - Windows: `.msi` via `briefcase package windows`
  - macOS: `.dmg` via `briefcase package macos`
  - Linux: `.deb` via `briefcase package linux`
  - Config: `pyproject.toml` (Briefcase section) and `docs/BUILD_INSTRUCTIONS.md`

## Environment Configuration

**Required env vars:**
- None (no `DATABASE_URL`, `API_KEY`, or other environment variables required)
- The `.gitignore` references `.env` files as ignored, but no `.env` files exist in the repo

**Secrets location:**
- Not applicable (no secrets, no API keys, no credentials)

## Hardware Acceleration Integrations

**Hardware Detection:**
- `barpath/hardware_detection.py` — Detects OS, CPU brand (Intel/AMD), NVIDIA GPU, Intel GPU at runtime
- Falls back to CPU if no acceleration available

**Supported Accelerators:**
| Accelerator | Requirements | Packages |
|-------------|--------------|----------|
| PyTorch CUDA | NVIDIA GPU + CUDA toolkit | `torch` + CUDA index |
| ONNX Runtime GPU | NVIDIA GPU | `onnxruntime-gpu` |
| ONNX Runtime (CPU) | Any | `onnxruntime` |
| OpenVINO | Intel CPU | `openvino` |
| TensorRT | NVIDIA GPU | `tensorrt` + `pycuda` |

**Device Selection Priority** (in `1_collect_data.py`):
1. TensorRT engine (`.engine` file) — no device selection
2. Intel GPU for OpenVINO
3. NVIDIA GPU via CUDA (`torch.cuda.is_available()`)
4. CPU fallback

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

---

*Integration audit: 2026-05-01*
