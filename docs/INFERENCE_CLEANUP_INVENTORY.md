# Inference implementation inventory — 2026-10-31

Before implementation: `git status --short` and `git rev-parse HEAD` both report
"not a git repository". `_review/git_status.txt` is a historical snapshot, not
current Git state. No Git branch or commit can be recorded in this copy.
Baseline test output: `artifacts/inference_runs/implementation_audit/baseline.txt`.

## Planned, non-destructive changes

| File | Purpose |
| --- | --- |
| `inference/pipeline.py` | Keep public interfaces, add ROI filtering, finite-output checks and measured timing; reuse original normalizer/selector/ByteTrack. |
| `inference/run.py` | Preserve CLI; explicit full-dataset default, new output root, opt-in recording, richer timing and provenance. |
| `inference/models.py` | Central validated model catalog and provenance, reading existing metadata. |
| `inference/telemetry.py` | Bounded timing windows and honest aggregate statistics. |
| `inference/runtime.py` | Isolated session workers, latest-frame capture/display and reliable shutdown. |
| `inference/gui.py` | Single PySide6 inference GUI, reusing the pipeline. |
| `tests/test_inference_desktop.py` | Offline lifecycle, ROI, reset, dispatch and multi-source tests. |
| `scripts/verify_inference.py` | Read-only bundle parity, preservation and offline GUI verification. |
| `docs/INFERENCE_WINDOWS.md` | Verified Windows launch/use/troubleshooting. |
| `docs/INFERENCE_VERIFICATION.md` | Final evidence and limitations. |
| `README.md`, `requirements.txt`, `.gitignore` | Current inference entrypoint, avoid conflicting ONNX packages, protect local outputs. |

## Audit findings and retained historical components

* Actual entrypoints: `collector/main.py`, `collector/process_dataset.py`,
  `collector/audit_recordings.py`, `inference/run.py`, research/training scripts
  and six notebooks. No existing `inference/gui.py` exists. The OpenCV dashboard
  is the existing inference UI; collector UI is recording-specific.
* Existing causal contract is `pose13_xy_conf_mask_causal_v2`, 52 floats, six
  Vietnamese labels. `shared/pose_features.py` remains authoritative and unchanged.
* `PoseRunner` uses official ByteTrack and preserves original detector values.
  `SinglePersonSelector` abstains on ambiguous people. Both remain unchanged.
* Existing `CameraReader` retains only the latest frame, but has a daemon thread
  and a two-second join; backend reads may outlast it. Desktop sessions will use
  process containment so a blocked native backend can be stopped safely.
* Existing CLI always records camera events and offline MP4; exports will become
  explicit user choices. It chooses the newest checkpoint by modification time;
  default will become the requested full-dataset bundle.
* Existing `inference/pipeline.py` includes a geometry experiment path and
  `draw_dashboard`. Keep compatibility, do not delete public helpers.
* Shell launch/install scripts target Ubuntu and historical `lab2_imp`; Colab
  `/content/drive/...` paths in notebooks/README are research lineage, not Windows
  runtime defaults. Keep and identify them rather than rewriting history.
* Existing `docs/TRAINING_WINDOWS.md` and research code remain historical
  reproducibility instructions, not permission to train.
* Root requirements install CPU and GPU ONNX Runtime together; choose one in
  dependency declarations. Installed environment will not be changed.
* `_review`, `reports`, previous inference outputs, model duplicate exports,
  checkpoints, raw/processed datasets, manifests, backups and notebooks remain.
* No source tree restructuring, model moves, deletion or archival is proposed.

Pre-edit copies of affected existing source files and a SHA-256 manifest of
existing protected files are stored under `implementation_audit`. Generated
bytecode/test caches are excluded. Verification checks content, additions and
removals in protected trees after testing. No physical devices are probed.
