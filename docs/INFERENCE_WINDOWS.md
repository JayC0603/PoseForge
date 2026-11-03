# PoseForge inference on Windows

Run from `C:\Users\tienq\PoseForge_Reposory`. Training is frozen. This application
does not train, replace checkpoints, or edit datasets.

## Activate and launch

Verified in PowerShell with the existing `ferlab-infer` environment:

```powershell
Set-Location C:\Users\tienq\PoseForge_Reposory
(& 'C:\Users\tienq\miniforge3\Scripts\conda.exe' 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate ferlab-infer
python -m inference.gui
```

The direct interpreter form also works when Conda is absent from PATH:

```powershell
& C:\Users\tienq\miniforge3\envs\ferlab-infer\python.exe -m inference.gui
```

The window validates local bundles before showing. No physical camera is opened
or scanned during launch. START initializes models and then opens the selected
sources. Allow several seconds for CUDA startup. END releases sources and
clears predictions. Wait for sources to be released before START again. Closing
the window performs the same shutdown. A blocked native backend is contained in
a child process and terminated after a three-second cooperative shutdown deadline;
the UI remains responsive and reports incomplete exports in `shutdown.json`.
Owned workers flush exports and terminal telemetry before explicitly exiting
their process, avoiding a verified Windows CUDA/ORT interpreter-teardown stall.
`shutdown_trace.json` records cleanup stages; the parent joins each process.

## Model and source controls

Default bundle: `artifacts/models/mlp_wide_full_real_dataset/v001`.
Default execution: YOLO11n-Pose (`artifacts/pretrained/yolo11n-pose.pt`) on
CUDA:0, official ByteTrack, existing selected-person logic, original shared
`pose13_xy_conf_mask_causal_v2` normalizer, 52 inputs and six outputs. Everything
runs in FP32. There is no added keypoint smoothing or interpolation.

The model selector also offers Original MLP v001, GRU-16 Research and GRU-32
Research when their metadata/checkpoints pass validation. Research models are
not ranked above the default. Bundles are loaded sequentially for startup
validation and released. Each active source loads only the selected model.
END before switching models or source settings; the next START creates fresh
tracking, normalization and temporal state. ONNX and PT are explicit choices in
Advanced Settings. Classifiers run on CPU. Unavailable ONNX produces an error;
choose PT explicitly if its validation passed. There is no silent fallback.

For a Windows webcam, leave source `0`, or enter its known camera index, then
START. For DroidCam installed as a Windows virtual camera, enter its camera
index. Alternatively paste a supported HTTP/MJPEG or RTSP stream URL supplied by
your DroidCam/stream application. This app uses OpenCV/FFmpeg and does not invent
a phone URL or scan devices. Confirm phone and PC connectivity yourself.
Credentials and URL query strings are omitted from saved configuration, titles
and application logs. Native stream diagnostics are suppressed inside workers.

Use **Video…** to choose an existing local MP4, or type its absolute path.
Files play at their encoded rate using a latest-frame mailbox and stop at EOF;
frames can be skipped under load and this is counted. Use the offline CLI to
process every file frame. GRU file playback requires the original adjacent
`camNN_timestamps.csv` matching `camNN_raw.mp4`; encoded FPS is not substituted
for capture timestamps. Live GRU uses host frame receipt times and abstains
until a complete valid 16/32-frame causal context exists.

**Add Source** accepts additional indices, file paths or stream URLs, up to four
total. **Remove Last Source** removes the most recently added source while
stopped. START opens all configured sources. One source gets the large preview;
two get two panels; three/four use a two-column grid on wide windows and stack on
smaller windows. Probabilities and timing details can be expanded per panel;
they start collapsed for multiple sources to preserve preview space.
Each source has independent capture, ByteTrack, selector,
normalizer, ROI, temporal buffer and telemetry. The current PoseRunner owns
mutable Ultralytics predictor/tracker state, so workers also own separate model
instances. This costs more GPU memory than a shared stateless forward service.
Four-source development playback was verified but is slower on this RTX 2050;
the status reports limited throughput. Prefer one primary source.
Preview transfer uses one shared-memory frame per source, bounded to 1920×1080
with aspect ratio preserved. Higher-resolution source frames still enter the
pipeline and video export at original dimensions. No large image messages can
accumulate in the Qt event queue or process mailbox.

## ROI and predictions

While frames are visible, click **Select ROI** and drag a rectangle on that
source's preview. The letterbox margins are excluded when mapping coordinates.
ROI is stored in normalized original-frame coordinates and survives widget
resizing. **Enable ROI** toggles it; **Reset ROI** clears it. Detector boxes whose
centers fall inside ROI are passed to the original ambiguity-aware selector.
YOLO still sees the full, unscaled source frame. Changing ROI resets that
source's state; queued results from the previous ROI are discarded.

The actual Vietnamese label names and ordering come from the model bundle:

| ID | Label |
| --- | --- |
| 0 | Ngồi thẳng |
| 1 | Cúi gập / gù lưng |
| 2 | Gục xuống bàn |
| 3 | Ngả người ra sau |
| 4 | Nghiêng / quay sang bên |
| 5 | Giơ tay phát biểu |

Class 4 includes left/right sideways leaning or turning. No-person, ambiguous
selection and GRU warmup clear the previous prediction. Confidence and the six
bars are raw classifier probabilities. Optional Advanced Settings confidence
threshold displays UNCERTAIN below the chosen value; zero disables this display
rule. It does not change raw probabilities, keypoints or evaluation logs. A high
softmax score is not independent evidence of accuracy.

## Four measured FPS values

* **YOLO Model FPS**: reciprocal mean synchronized Ultralytics forward latency;
  excludes decoding, preprocessing, tracking and display.
* **Classifier FPS**: reciprocal mean active ONNX/PT classifier call latency,
  including its input/output adapter and softmax; excludes pose extraction and
  temporal buffer construction. This is model-call capacity, not camera FPS.
* **Pipeline FPS**: observed intervals between processed frame completions;
  includes relevant pipeline work, source pacing and export overhead. The
  separate Pipeline latency measures pose/tracking, selection, normalization,
  classifier and overlay for each processed frame.
* **Live FPS**: unique frames actually painted by Qt, measured over two seconds.
  It excludes duplicate repaints. It is not camera exposure or monitor scanout.

Model and pipeline latency show median/p95 over at most 120 processed frames.
Display freshness measures host receipt to Qt paint, not sensor-to-display
latency. Capture skipped counts decoded frames bypassed by the latest-frame
policy; processed-but-not-painted counts GUI/mailbox omissions. Missing samples
and initial zero-frame windows display a dash. Offline headless runs have no
Live FPS. GUI reports include active-period average Live FPS for ended sources.

## Local outputs and optional labeled sessions

Every START creates `artifacts/inference_runs/desktop_<timestamp>_vNNN/source_NN/`
with sanitized run configuration, model/pose SHA-256 provenance, performance,
actual backend/device/precision, display and shutdown reports. Private camera recordings stay local and the
entire artifacts tree is Git ignored.

Video recording is OFF by default. Advanced Settings can enable annotated MP4
and/or raw prediction CSV + JSONL. For an optional labeled session, select an
operator reference label before START, enable prediction export and optionally
video, then END before changing the reference posture. Each row retains the raw
six probabilities, reference label, track/selection state and measured timing.
Operator labels are unverified annotations, not a formal accuracy estimate.
Local files export CFR video at encoded FPS; live annotated export uses nominal
25 FPS playback. Dropped frames can shorten exported playback; JSONL receipt
timestamps preserve timing when enabled. GUI labels support Vietnamese; MP4
overlay uses numeric class ID and confidence because OpenCV's default font does
not render Vietnamese reliably. Labels remain in CSV/JSONL and model provenance.

The preserved CLI was verified with the original development clip:

```powershell
python -m inference.run --source artifacts/dataset_working/Real_Posture_Dataset/recordings/CHANH/session_20261027_1124/label_0/take_001/cam01_raw.mp4 --mode video --model artifacts/models/mlp_wide_full_real_dataset/v001 --device cuda:0 --backend onnx --cls-device cpu --headless --export-video --max-frames 120 --name desktop_offline_smoke
```

CLI video recording also requires `--export-video`. CLI prediction logs remain
enabled. Previously generated outputs under `artifacts/inference/` are retained;
new CLI and desktop runs use `artifacts/inference_runs/`.

## Troubleshooting and real camera verification

* CUDA errors: verify the activated interpreter is `ferlab-infer`, close other
  GPU jobs, reduce sources, or explicitly choose YOLO CPU in Advanced Settings.
  No device or precision change happens silently. Keep the existing CUDA torch
  installation; no package upgrade is required by this implementation.
* ONNX errors: select the explicitly validated PT backend, or repair ONNX Runtime
  in a separate environment maintenance action. Avoid installing CPU and GPU
  ONNX Runtime distributions together. This application defaults to CPU ORT.
* Missing camera: check Windows camera privacy access, release the camera in
  other apps, confirm its known index, and try START again. Do not run a scanner.
* RTSP/HTTP disconnection: status becomes RECONNECTING, predictions clear, and
  capture retries with bounded backoff. A new connection resets all causal state.
  After 30 seconds without acquisition, the session reports an actionable ERROR;
  check the connection and START again. END remains available during retries.
* Unsupported video/codec: use a locally playable MP4 supported by this OpenCV
  build. No file extension alone guarantees that its codec is available.
* Qt errors: launch from the same environment. The installed PySide6 is 6.11.3.
  Remove `QT_QPA_PLATFORM=offscreen` from an interactive shell if a previous test
  set it. Offscreen is only for automated tests.
* Environment reproduction: the tested environment reports torch 2.6.0+cu124,
  OpenCV 5.0.0 and ONNX Runtime 1.31.1. The preserved collector requirements still
  declare OpenCV below 5; that pre-existing declaration differs from this working
  environment. No dependency reinstall or fresh-environment recreation was
  performed. Use the existing `ferlab-infer` interpreter for this delivery.
* Windows console Unicode errors in legacy scripts: set `PYTHONUTF8=1` for the
  test process. The known asset-migration test additionally requires Windows
  symlink privileges (Developer Mode or an appropriately elevated test shell).
  Inference does not require creating symlinks; no privileges were changed.

Real hardware remains an interactive verification step: launch the GUI, choose
your known webcam/DroidCam source, click START, and perform the six postures.
Check orientation, frame size, selected box/track, keypoints and ROI before
interpreting scores. END before closing or changing settings. Report source
type, posture, screenshot and the local run report if something looks wrong.
Live predictions alone do not establish independent accuracy. Diagnosis follows
acquisition → scaling → identity → ROI → keypoints → normalization → feature
contract → labels → backend → confidence → camera domain → generalization.
Any proposed retraining requires a separate diagnosis report and explicit approval.

## Reproduce the local checks

These commands were executed with the existing environment, from the repository:

```powershell
$env:PYTHONUTF8 = '1'
$env:QT_QPA_PLATFORM = 'windows'
python -m pytest -q --junitxml=artifacts/inference_runs/implementation_audit/regression_final.xml
```

The complete suite includes the known Windows symlink-privilege failure. No
tests are deselected or assertions weakened. Four pre-existing skips require
legacy external bundle/test-image fixtures; the installed deployment bundles and
real development video are verified separately.

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
python -m pytest tests/test_inference_desktop.py -q --junitxml=artifacts/inference_runs/implementation_audit/gui_tests.xml
```

```powershell
$env:QT_QPA_PLATFORM = 'windows'
python scripts/verify_inference.py --gui-smoke --sources 1 --visible
python scripts/verify_inference.py --gui-smoke --sources 2
python scripts/verify_inference.py --gui-smoke --sources 4 --visible
```

These smoke commands explicitly process only the hardcoded existing DGD
development MP4. They never select hardware. Visible tests activate their own
window; leave it visible until completion so Qt paint throughput can be counted.
The two-source command runs offscreen and verifies functionality without making
a physical display-FPS claim.
See `INFERENCE_VERIFICATION.md` for final counts, measurements and evidence.

The read-only bundle/output/integrity check was also executed:

```powershell
python scripts/verify_inference.py --models --offline-check --preservation
```
