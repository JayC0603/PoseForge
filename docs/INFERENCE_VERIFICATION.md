# PoseForge final inference verification

## A. Repository cleanup

Git is installed, but this working copy has no `.git` directory. Current status/commit and a dedicated branch are unavailable. `_review/git_status.txt` is historical. No repository was initialized or history fabricated.

Existing files modified:

* `README.md`: current inference entrypoint and historical Ubuntu/Colab context.
* `requirements.txt`: remove simultaneous CPU/GPU ONNX Runtime declarations; identify Windows inference. The installed Conda environment was not changed.
* `inference/pipeline.py`: finite logits/features, authoritative label/architecture validation, explicit FP32 YOLO, ROI filtering, selected-person box/state handling and measured stage latency. Shared normalization and collector code are unchanged.
* `inference/run.py`: explicit Full Dataset default, opt-in video recording, timing/provenance, safe source redaction, capture cleanup and the new output root.

Files created:

* `.gitignore`: protect local artifacts, private videos, weights, camera configuration and caches.
* `inference/models.py`: validated deployment catalog and SHA-256 provenance.
* `inference/telemetry.py`: bounded latency windows and actual Qt paint rates.
* `inference/runtime.py`: isolated native sessions, shared-memory preview mailbox, independent capture/state and bounded shutdown.
* `inference/gui.py`: single PySide6 inference desktop application.
* `tests/test_inference_desktop.py`: 17 desktop/runtime/contract tests.
* `scripts/verify_inference.py`: read-only deployed bundle, development-video, output and preservation checks.
* `docs/INFERENCE_CLEANUP_INVENTORY.md`, `docs/INFERENCE_WINDOWS.md`, `docs/INFERENCE_VERIFICATION.md`: audit, usage and evidence.

No files were deleted, moved, renamed or archived. No `src/` tree was introduced. Public Python entrypoints/interfaces remain. Intentional CLI behavior changes: explicit full-dataset default; video export requires `--export-video`; new outputs go to `artifacts/inference_runs/`. Existing model/dataset paths and historical outputs are preserved. Pre-edit copies of the four changed existing files are in `implementation_audit/source_before/`.

## B. Application

Verified PowerShell launch:

```powershell
Set-Location C:\Users\tienq\PoseForge_Reposory
& C:\Users\tienq\miniforge3\envs\ferlab-infer\python.exe -m inference.gui
```

Implemented: large aspect-preserving preview, original skeleton/selected box, Unicode labels, real confidence and six probabilities, START/END, explicit model/backend selection, manual webcam/DroidCam/RTSP/HTTP sources, MP4 chooser, ROI drag/toggle/reset, responsive one-to-four source layouts, source removal, optional confidence display threshold, per-source telemetry, opt-in local exports and operator reference labels. Model/source settings are locked during a session; END then change them and START with fresh state. Four-source probabilities/details start collapsed to fit the preview grid.

No core requested feature is deferred. Physical hardware verification remains pending. Limitations: separate models per process consume more VRAM; four sources run slower; stream codecs depend on OpenCV; live MP4 export uses nominal 25-FPS playback; MP4 uses numeric class IDs while the GUI/logs retain Vietnamese names. No independent accuracy estimate is claimed.

## C. Inference and bundle evidence

Default: Full Dataset MLP v001; ONNX Runtime CPU; YOLO11n-Pose CUDA:0; official ByteTrack; original ambiguity-aware selected-person logic; unchanged `shared.pose_features.CausalPoseNormalizer`; `pose13_xy_conf_mask_causal_v2`; 52 inputs, six outputs; FP32. Actual device/dtype/provider are recorded in each new desktop run’s `runtime_environment.json`. The recorded provider is `CPUExecutionProvider`, YOLO is `torch.float32` on `cuda:0`, and FP16 is false.

All four installed bundles load PT and ONNX. The catalog validates each backend sequentially and releases it. Each selected live model owns fresh per-stream state. Raw outputs remain unchanged by optional UNCERTAIN display behavior.

| Model | PT/ONNX maximum probability difference (32 seeded finite probes) |
| --- | ---: |
| MLP Wide – Full Dataset v001 (default) | 8.34e-07 |
| MLP Wide – Original v001 | 3.87e-07 |
| GRU-16 – Research | 2.98e-07 |
| GRU-32 – Research | 1.19e-07 |

The same 71-frame CHANH development video was also processed end to end with PT and ONNX. Both produced finite 52-feature/six-class outputs on all frames, identical class IDs, and maximum probability difference **2.38e-07**. The ONNX annotated output was decoded again: 71 frames, original 1280×720 dimensions. PT run without video export produced no MP4. This is functional verification using existing development material, not independent accuracy.

## D. Measured performance

Windows desktop Qt windows displaying the existing DGD label-0 development MP4, encoded at 20 FPS. RTX 2050, FP32 YOLO CUDA:0, ONNX CPU, annotated MP4 plus CSV/JSONL explicitly enabled. Each test ran for eight seconds after every panel had painted its first frames; model initialization happened beforehand. No other regression/performance job was running during these final measurements.

Model FPS is reciprocal mean measured call latency. Pipeline FPS is observed completion throughput; Live FPS below is average unique Qt paints over the active interval. Latency samples use the last at most 120 processed frames. These values are not webcam FPS or monitor scanout rates.

| Visible session | YOLO Model FPS | Classifier FPS | Pipeline FPS | Average Live FPS | Painted frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| One source | 49.0 | 4784.6 | 20.0 | 19.2 | 155 |
| Four sources, panel 1 | 17.8 | 1276.2 | 10.6 | 10.6 | 87 |
| Four sources, panel 2 | 17.9 | 1058.1 | 10.7 | 10.8 | 89 |
| Four sources, panel 3 | 17.7 | 1255.5 | 10.6 | 10.7 | 88 |
| Four sources, panel 4 | 17.7 | 887.7 | 10.5 | 10.7 | 88 |

Single-source median/p95 latency: YOLO 20.41/24.44 ms; classifier 0.182/0.348 ms; pipeline 29.35/33.15 ms. Display freshness median/p95 78.0/79.7 ms from host receipt, not exposure time. Single-source capture skipped 12 frames; GUI painted 155 of 161 processed frames.

Four-source throughput is limited on this GPU and is reported visibly. No sources are silently removed. Ended panels can show no current Live FPS after the two-second window expires; average active display FPS is retained in `display.json`. The headless CLI reports Live FPS as null.

## E. Reliability

17 offscreen tests pass: initialization without camera access; validated selector/defaults; explicit backend failure; source selection; duplicate START protection; START/END/restart; model state reset; no-person/ambiguous/uncertain display; disconnect/reconnect state reset; letterbox ROI drag and interrupted drag; ROI state isolation; one-to-four-source state and responsive layout; GRU warmup/track reset with separate stream buffers; measured FPS and percentiles; real spawned-process shutdown; native read-hang containment; rapid START/END without child leaks; local-file EOF and release; finite logits; authoritative labels; actual delegated FP32 backend inspection; bounded non-blocking shared preview transport.

Visible one- and four-source final runs exited without forced termination, and worker exit codes/shutdown traces were recorded. Intermediate tests exposed a Windows native-runtime teardown stall after all I/O had closed. Workers now explicitly flush closed exports and terminal metadata, then exit their owned process; the parent joins them. Large frames use one shared-memory slot, not queued image messages. Native capture hangs remain bounded by process termination and are reported honestly. No physical disconnection or camera-driver tests are claimed.

## F. Regression

The final two-source development-video run also passed offscreen: both sources
stopped cleanly, with 97 and 103 Qt paints and no forced termination. These
offscreen paints verify rendering/lifecycle and are excluded from the visible
Live FPS performance table.

* Untouched baseline in the default console: **138 passed, 4 skipped, 2 failed** (Windows console Unicode encoding failures).
* Complete suite with `PYTHONUTF8=1`, Windows Qt platform: **156 passed, 4 skipped, 1 failed**. All tests were collected; no deselection or weakened existing assertions.
* Final offscreen desktop suite: **17 passed**. A targeted pipeline/GRU/desktop run also passed (34 passed, 2 skipped before the final two added checks).
* Remaining failure: `collector/tests/test_asset_layout.py::test_shared_assets_copy_link_and_verify`, WinError 1314 (symlink privilege). Safe workaround: enable Windows Developer Mode or use an appropriately elevated test shell yourself. Inference requires neither; no Windows privileges were changed.
* Four pre-existing skips: two external legacy training-bundle fixtures absent; two YOLO test-image fixtures absent. Separate checks loaded the real installed bundles and processed real local development video.
* Offscreen full-suite experiments exposed collector layout differences; collector was left unchanged and the complete final suite used its original Windows platform. Intermediate failures and diagnostic runs remain in the local audit directory.

## G. Runtime artifacts

All new artifacts are local under `artifacts/inference_runs/`. Authoritative reports:

* `implementation_audit/baseline.txt`, `baseline.xml`.
* `implementation_audit/regression_final.txt`, `regression_final.xml`, `gui_tests.txt`, `gui_tests.xml`.
* `implementation_audit/model_verification.json`: deployed PT/ONNX checks, metadata, hashes and environment.
* `implementation_audit/offline_verification.json`: original video parity and output decoding.
* `implementation_audit/gui_smoke_1.json`, `gui_smoke_4.json`: visible final runs and measurements.
* `implementation_audit/gui_smoke_2.json`: final two-source functional run (check its visible/platform flag; offscreen paints do not establish physical Live FPS).
* `implementation_audit/gui_1_sources.png`, `gui_4_sources.png`: real application preview screenshots.
* `implementation_audit/preservation_before.json`, `preservation_after.json`: original-file integrity.
* `desktop_offline_smoke_v001/`: annotated MP4, CSV, JSONL, timing and provenance.
* `desktop_offline_pt_v001/`: PT predictions/timing/provenance, no video.

Final visible run directories:

* `desktop_20261031_235019_v001`: one source.
* `desktop_20261031_235347_v001`: four sources.

Desktop source folders contain `run_config.json`, `provenance.json`, `runtime_environment.json`, `performance.json`, `display.json`, `shutdown.json`, `shutdown_trace.json` and only explicitly enabled prediction/video exports. Synthetic automated-test artifacts live in pytest temporary directories.

## H. Pending interactive hardware verification

No physical camera was opened or probed. Launch the documented GUI, enter your known webcam/DroidCam index or supplied stream URL, then explicitly click START. Test the six postures and inspect selected identity, keypoints, ROI and orientation. END releases the session. Real webcam acquisition, device-driver behavior, phone network reconnection and six-posture observations remain for your interactive test. Such observations are not formal external accuracy estimates.

## I. Training freeze and preservation

No production/research training job, dataset mutation, checkpoint replacement, export replacement, model-directory move or environment/package installation occurred. The existing regression suite exercises tiny synthetic models and temporary fixtures as part of its tests; it does not retrain the deployed models. SHA-256 verification covers **4,702 existing protected files (571,215,460 bytes)** across artifacts, collector, shared code, training/research scripts, notebooks, manifests, reports and configuration: **zero changed and zero missing**. The four intentionally changed existing application/root files have pre-edit backups. No external-test artifact was altered.

If real-camera predictions look wrong, follow the stage-by-stage diagnosis order in the Windows guide. Retraining requires a separate evidence-based diagnosis and explicit approval.
