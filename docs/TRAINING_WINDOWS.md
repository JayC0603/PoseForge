# PoseForge recovered real-data training on Windows

The executed run is `artifacts/training/real_v001`. Its frozen dataset ID is
`d8c7da82dd54e1d70532778c733c94f0042c9bdd2ee2ebc96fbee4040cab5f7c`.
Measured results and the final checkpoint location are in
`artifacts/reports/TRAINING_FINAL_REPORT.md` (generated after execution).

## Environment and commands

Run from PowerShell in `C:\Users\tienq\PoseForge_Reposory`.
This session did not have `conda` on PATH. The existing environment executable was
used directly, with no package installation or downgrade:

```powershell
Set-Location C:\Users\tienq\PoseForge_Reposory
$Python = 'C:\Users\tienq\miniforge3\envs\ferlab-infer\python.exe'
$env:PYTHONUTF8 = '1'
& $Python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
& $Python -m pytest -q

& $Python scripts/preflight_dataset.py --data artifacts/dataset_working/Real_Posture_Dataset --decode-videos --json artifacts/reports/recovered_integrity_preflight.json
& $Python scripts/validate_dataset.py --data artifacts/dataset_working/Real_Posture_Dataset --deep
& $Python scripts/train_poseforge.py --data artifacts/dataset_working/Real_Posture_Dataset --stage audit
& $Python scripts/train_poseforge.py --stage prepare
& $Python scripts/train_poseforge.py --stage baseline --run artifacts/training/real_v001 --device cuda:0
& $Python scripts/train_poseforge.py --stage improve --run artifacts/training/real_v001 --device cuda:0
& $Python scripts/train_poseforge.py --stage finalize --run artifacts/training/real_v001 --device cuda:0
& $Python scripts/train_poseforge.py --stage export --run artifacts/training/real_v001
```

The initial test run was also executed without `PYTHONUTF8`; its Windows encoding
failures were diagnosed before implementing training. Captured test and validation
outputs are under `artifacts/reports/`. Each workflow command saves a timestamped
log under the reports directory or the prepared run's `logs/` directory. An
environment with Conda initialized may use `conda activate ferlab-infer` and
`python` for the same commands.

## New datasets, configuration, and interruption

Existing audit reports cannot be overwritten. For a changed dataset, use a new
report directory for **both** audit and prepare:

```powershell
& $Python scripts/train_poseforge.py --stage audit --reports artifacts/reports/preflight_v002
& $Python scripts/train_poseforge.py --stage prepare --reports artifacts/reports/preflight_v002
```

Prepare always allocates a new `real_vNNN` directory. It hashes source videos,
poses, metadata, timestamps, labels, and extraction summaries; copies and verifies
the non-video inputs; then seals the snapshot and processed features using SHA-256
inventories and Windows read-only file attributes. Sources recently changed
(within 60 seconds), sources changed during audit/copy, and structural mismatches
stop the workflow. Training reads the snapshot's derived features, never live
recording files. Original files and prior models are not modified.

Optional arguments:

* `--data PATH`, `--reports PATH`, `--output PATH`: input and derived destinations.
* `--validation-subjects PERSON ...`, `--test-subjects PERSON ...`: prepare only;
  require all six classes and at least 16 eligible frames per evaluation class.
* `--seed 302`: deterministic initial split and baseline training seed.
* `--config configs/training_real_baseline.json`: JSON baseline defaults.
* `--device cuda:0`: requires working CUDA when requested; no silent CPU fallback.
* `--resume`: restore a candidate's last completed epoch, AdamW state, early
  stopping state, and CPU/CUDA RNG states. Completed experiments with matching
  configuration are skipped. A different configuration cannot overwrite a run.

For example, after interruption:

```powershell
& $Python scripts/train_poseforge.py --stage baseline --run artifacts/training/real_v001 --device cuda:0 --resume
& $Python scripts/train_poseforge.py --stage improve --run artifacts/training/real_v001 --device cuda:0 --resume
```

Changing hyperparameters after training requires a new prepared run. Optimizer
resume files are trusted local PyTorch files; do not resume downloaded optimizer
checkpoints. `best.pt` and deployment `model_final.pt` use weights-only loading.
Interruption during an epoch restarts that epoch from its saved boundary. A CPU
integration test verifies resumed weights and history exactly match uninterrupted
training. No mixed precision is used for these small MLPs.

Finalization freezes the selected configuration, retrains from fresh seed 10302
on train + validation, and chooses its epoch count from the validation-only
multi-seed confirmation. It writes a new `artifacts/models/mlp_<name>/vNNN`
checkpoint before accessing test predictions. A persistent
`test_evaluation_started.json` guard prevents automatically repeating an interrupted
test evaluation. Inspect that marker and saved evidence if interrupted at that
point. Completed finalization returns its existing final model, without rerunning
test or replacing models.

## Data and feature policy

All six official IDs remain unchanged. Class 4 combines both side directions.
There is no VLM relabeling, pose retraining, keypoint re-extraction, or additional
keypoint smoothing. The shared contract is
`pose13_xy_conf_mask_causal_v2`, shape `[13,4]`, flattened dimension 52. Confidence
threshold is 0.35 and the existing causal scale EMA alpha is 0.25.

Every original frame remains represented in the snapshot and feature arrays,
including missing observations. Training eligibility excludes no-selection and
uncertain-selection frames, fewer than four valid joints, camera errors, explicit
curation exclusions, and exact cross-clip video-frame duplicates. Partial-joint
frames otherwise remain eligible. Multi-box observations require the existing
selector's geometric duplicate-box proof; spatially ambiguous selections are
review-required. An uncertain selection enters normalization as absent, avoiding
contamination of later causal state. Raw observations are retained unchanged.

Causal state resets at each camera clip, at a changed selected track ID, and after
more than eight absent frames, matching deployment. Prefix-invariance checks and
deployment replay tests cover these transitions. There is no feature-statistic
fitting on test. Source frame indices and original capture ticks are preserved;
irregular timestamps are neither reordered nor interpolated.

Without an original clean curation manifest, the snapshot writes an explicitly
automated `training_curation_manifest.json`; it does not claim human approval or
invent source edits. Existing clean lineage and curation edits are read if
present. No overlapping segments are created. Both views and every take for a
subject stay in one split, checked before training.

## Experiments and metrics

Baseline Small, Base, and Wide use ordinary CE, dropout 0.15, AdamW learning rate
0.001 and weight decay 0.001, batch 128, at most 100 epochs, patience 12. Baselines
use all eligible real training frames with no weighting, smoothing, augmentation,
oversampling, or source balancing. The baseline stage first runs an eight-epoch
Small smoke test and verifies loss decreases.

Controlled improvements use the best baseline architecture and seed 302:
square-root inverse **source-take/camera** class counts (weights capped at 0.5–2),
smoothing 0.05/0.10 (0.00 is baseline), supported keypoint augmentation, a quota of
48 unique frames per source take/camera/epoch, and that quota with mild square-root
balancing. No oversampling is used. Only changes improving subject Macro-F1 by
at least 0.005 can enter the combined candidate; incompatible alternatives are
not stacked. The top two configurations are confirmed at seeds 303 and 304 in
addition to 302. Ranking uses mean validation subject Macro-F1 across these seeds.

Each completed experiment includes `best.pt`, `last.pt`, `config.json`,
`history.csv`, `training_counts.json`, `metrics.json`, `classification_report.csv`,
`per_class_metrics.csv`, `per_subject_metrics.csv`, `camera_comparison.csv`,
`confusion_matrix.png`, `learning_curves.png`, and frame predictions. Metrics use
the fixed eligible validation frames, six-class Macro-F1 with missing-class
zero-division handling, mean clip probabilities, and source-take probabilities.
Validation loss is plain unsmoothed, unweighted CE for fair comparison; train
loss is the configured objective. ECE uses 10 bins; Brier score is also saved.

`comparison.csv`, `seed_confirmation.csv`, `confirmed_ranking.csv`, and
`selection.json` summarize selection. `training_summary.json` and the final model
directory record independent test results and provenance. Test scores describe
one held-out subject and are not a population confidence interval.

## Deployment

Read `artifacts/training/real_v001/final_model.json` for the exact versioned model
directory. The final report contains the exact executed video-inference command.
The general PowerShell command is:

```powershell
$FinalModel = (Get-Content artifacts/training/real_v001/final_model.json -Raw | ConvertFrom-Json).path
& $Python inference/run.py --source 'YOUR_VIDEO.mp4' --mode video --model $FinalModel --pose artifacts/pretrained/yolo11n-pose.pt --device cuda:0 --backend onnx --cls-device cpu --headless
```

ONNX is considered verified only when `onnx_real_parity.json` reports a maximum
absolute real-feature probability difference at most 1e-5. Export verifies both
deployment loaders and six-class probabilities on train/validation features;
it does not run test predictions again. End-to-end video validation uses the
existing YOLO11n-Pose, ByteTrack, selector, causal normalizer, and classifier.
Its annotated output stays local, outside shareable reports/model bundles.

## Notebooks

The three V2 notebooks now call the same reusable CLI and inspect dynamic
reports/features. Set `POSEFORGE_ROOT`, `POSEFORGE_DATA`, and `POSEFORGE_RUN` as
needed. They do not require Colab, Drive, a particular ZIP, or historical counts.
Notebook 03 defaults `RUN_FINAL=False` to avoid unintentionally consuming test.
Original notebook copies were preserved under
`artifacts/reports/notebook_backups/`.

## Model improvement before external testing (2026-10-30)

This phase uses only the twelve development subjects. QBAO is prohibited by the
new development loader and was neither trained on nor evaluated again. The original
TRUONG/VMTRI validation split and eligibility masks are preserved. The independent
external test remains pending user authorization.

The following commands were executed in the existing `ferlab-infer` environment:

```powershell
Set-Location C:\Users\tienq\PoseForge_Reposory
$Python = 'C:\Users\tienq\miniforge3\envs\ferlab-infer\python.exe'
$env:PYTHONUTF8 = '1'
& $Python scripts/improve_poseforge.py --help
& $Python scripts/improve_poseforge.py --stage prepare
& $Python scripts/improve_poseforge.py --stage screen --run artifacts/improvement/dev_v001 --device cuda:0
& $Python scripts/improve_poseforge.py --stage confirm --run artifacts/improvement/dev_v001 --device cuda:0
& $Python scripts/improve_poseforge.py --stage cv --run artifacts/improvement/dev_v001 --device cuda:0
& $Python scripts/improve_poseforge.py --stage cv-robustness --run artifacts/improvement/dev_v001 --device cuda:0
& $Python scripts/improve_poseforge.py --stage finalize --run artifacts/improvement/dev_v001 --device cuda:0
& $Python scripts/improve_poseforge.py --stage deploy --run artifacts/improvement/dev_v001 --device cuda:0
& $Python scripts/improve_poseforge.py --stage report --run artifacts/improvement/dev_v001 --device cuda:0
& $Python -m pytest -q
& $Python -m pytest -q tests/test_model_improvement.py tests/test_real_training.py tests/test_phase2_v2.py tests/test_phase3.py
```

Preparation writes a development-only `dev_vNNN` directory containing copies of
the original sealed feature arrays for those twelve subjects, source hashes,
quality/exclusion reports, a frozen comparison protocol, and subject folds. It
does not copy QBAO arrays or private videos. Existing raw data and v001 artifacts
are never overwritten. Subsequent stages require an explicit `--run` and accept
`--resume`. Resume uses the same trusted local optimizer/RNG checkpoints as the
original workflow.

Screening tests the Wide control, optional geometry features, residual MLP,
LayerNorm MLP, and quality-gated arm augmentation. A new geometry contract is
explicitly named `pose13_xy_conf_mask_causal_v2_geometry_v1`: 52 original values,
22 calculated geometry values, and 22 masks (96 dimensions). The original feature
normalizer, 52-dimensional contract, and Small/Base/Wide definitions remain intact.
Both contracts share the same geometry implementation across NumPy deployment and
Torch training, with explicit masks and safe denominators.

The top two challengers and control are confirmed at seeds 302/303/304. Three
subject folds use seeds 302/303 and fixed epoch budgets selected from matched-split
development confirmation. Fold validation never chooses a checkpoint. CV is
development-selection evidence, not a new independent test. The bounded
`cv-robustness` follow-up evaluates an already-confirmed candidate with a strong
synthetic robustness gain; it saves a separate plan and preserves the first CV plan.
All sessions, cameras and source takes for a subject remain together.

No candidate met the recorded promotion/non-degradation criteria. Geometry's
matched-split gain did not persist across folds. Targeted augmentation improved
synthetic arm occlusion but degraded clean fold performance. The recommended
production model remains `artifacts/models/mlp_wide/v001`. A fresh baseline
development reproduction is saved separately at
`artifacts/improvement/dev_v001/unpromoted_development_model/v001/model_final.pt`.
It is an unpromoted research artifact, not a replacement release. It was fit on
all twelve development subjects at seed 10302 for four epochs and has no new
test metrics. The deploy stage checks PT/ONNX parity on development features,
original v001 compatibility, model latency, and a local CHANH video smoke.

The full results and decisions are in
`artifacts/reports/MODEL_IMPROVEMENT_REPORT.md`. Per-frame exclusions, error records,
ablations, seed/fold results, per-class/camera metrics, learning curves, confusion
matrices, latency, logs and checkpoints are in `artifacts/improvement/dev_v001`.
Do not call the original `train_poseforge.py --stage finalize` to continue this
phase: it belongs to the earlier held-out-test workflow. The new improvement CLI
has no final/external-test evaluation stage.

## Temporal split and all-13-subject release candidate

This subsequent phase explicitly authorizes including QBAO in all-data training.
QBAO is no longer held out for this candidate. No historical QBAO score is an
evaluation of the new model; independent external evaluation remains pending.

Actual commands executed from the repository:

```powershell
$Python = 'C:\Users\tienq\miniforge3\envs\ferlab-infer\python.exe'
$env:PYTHONUTF8 = '1'
& $Python scripts/train_poseforge.py --help
& $Python scripts/train_poseforge.py --split-strategy temporal_within_take --stage audit
& $Python scripts/train_poseforge.py --split-strategy temporal_within_take --stage prepare --run artifacts/experiments/temporal_within_take_80_10_10/v001
& $Python scripts/train_poseforge.py --split-strategy temporal_within_take --stage baseline --run artifacts/experiments/temporal_within_take_80_10_10/v001
& $Python scripts/train_poseforge.py --split-strategy temporal_within_take --stage internal-test --run artifacts/experiments/temporal_within_take_80_10_10/v001
& $Python scripts/train_poseforge.py --split-strategy temporal_within_take --stage full-data --run artifacts/experiments/temporal_within_take_80_10_10/v001 --device cuda:0
& $Python scripts/train_poseforge.py --split-strategy temporal_within_take --stage export --run artifacts/experiments/temporal_within_take_80_10_10/v001
& $Python scripts/train_poseforge.py --split-strategy temporal_within_take --stage report --run artifacts/experiments/temporal_within_take_80_10_10/v001
& $Python -m pytest -q tests/test_temporal_training.py tests/test_real_training.py tests/test_phase2_v2.py tests/test_model_improvement.py tests/test_phase3.py
& $Python -m pytest -q --deselect=collector/tests/test_asset_layout.py::test_shared_assets_copy_link_and_verify
```

`--split-strategy` defaults to `subject_wise`, preserving the original CLI and
notebooks. Temporal audit creates a new `vNNN` run; use its returned path for all
later stages. Run audit again into a new version when the source dataset changes.
Temporal options include `--ratios 0.8 0.1 0.1`, `--purge-seconds 0.25`,
`--min-eval-frames 8`, `--min-eval-seconds 0.10`, and minimum evaluation takes and
subjects per class (both two). Options are frozen at audit/prepare, rather than
silently changed by subsequent stage arguments. Training accepts `--resume` for
the existing trusted local optimizer/RNG checkpoints.

Boundaries use one shared host-reception monotonic timeline per original take,
verified against UTC/metadata clocks, with both cameras mapped to the same 80% and
90% boundaries. Repeated ticks remain together; encoded FPS is not used. This is
receipt-time alignment, not proof of exposure synchronization. `purge_seconds` is
explicitly a symmetric radius: ±0.25 seconds around each boundary. It was not
reduced. Short/sparse evaluation segments are excluded rather than moved to Train.

The actual split was infeasible: Train 10,201 frames; Validation 55 frames from one
subject and labels 1/2 only; Internal Test 145 frames with no label 4. The baseline
and internal-test commands therefore recorded **skipped/infeasible** status and
did not train a temporal classifier or compute test scores. The split, leakage
checks, frame provenance, counts and limitations are retained for reproducibility.
Effective temporal training usage was 71.56% of eligible frames after exclusions.

The full-data stage used the authorized fallback: original 52-feature MLP Wide,
four fixed epochs from historical subject-wise development confirmation (best
epochs 8/4/1), fresh seed 10302, AdamW lr 0.001, weight decay 0.001, batch 128,
dropout 0.15. All **14,256 verified eligible observations from 13 subjects** were
used in every epoch, including frames excluded only by the temporal embargo or
short evaluation segments. Existing pose-quality exclusions remained intact.
The immutable `full_snapshot` contains copied pose/timestamp/metadata files and
features with hashes; no source videos are copied into reports/model bundles.

Candidate: `artifacts/models/mlp_wide_full_real_dataset/v001/`.
It includes `model_final.pt`, `model_final.onnx`, identical loader-compatible
`model.onnx`, full-data manifest/history, configuration, labels, feature contract,
and deployment verification. PT/ONNX real-feature parity and a 71-frame existing
video smoke passed. No held-out score is reported for the all-data model.
Original `artifacts/models/mlp_wide/v001` and prior research artifacts are retained.

Inference on already available footage was executed using:

```powershell
& $Python inference/run.py --source artifacts/dataset_working/Real_Posture_Dataset/recordings/CHANH/session_20261027_1124/label_0/take_001/cam01_raw.mp4 --mode video --model artifacts/models/mlp_wide_full_real_dataset/v001 --pose artifacts/pretrained/yolo11n-pose.pt --device cuda:0 --backend onnx --cls-device cpu --headless --name full_real_dataset_smoke
```

See `artifacts/reports/TEMPORAL_SPLIT_AND_FULL_DATA_TRAINING.md` for the distinct
historical subject-wise evidence, infeasible temporal protocol, full-data fit,
actual counts and deployment results. `temporal_confusion_matrix.png` is explicitly
a no-test status figure because no temporal confusion matrix could be measured.
The broad regression command deselects only the existing Windows privilege-dependent
symlink migration test; these new workflows use no symlinks or administrator actions.
No independent external test was performed.

## Take-level MLP and real GRU training (2026-10-31)

This phase replaces the infeasible within-video split with source-take grouping.
It does not remove the historical implementation. Both camera views and all
derived data of an original take stay together. The frozen allocation is 61 Train,
13 Validation, 13 Internal Test takes, with one wholly pose-empty take excluded.
All six labels occur in every usable partition. Train includes all 13 subjects.

Actual executed commands, using the existing environment and no reinstalls:

```powershell
Set-Location C:\Users\tienq\PoseForge_Reposory
$Python = 'C:\Users\tienq\miniforge3\envs\ferlab-infer\python.exe'
$env:PYTHONUTF8 = '1'
& $Python scripts/train_poseforge.py --help
& $Python scripts/train_poseforge.py --split-strategy take_level --stage audit
& $Python scripts/train_poseforge.py --split-strategy take_level --stage sequences --run artifacts/experiments/take_level_gru/v001
& $Python scripts/train_poseforge.py --split-strategy take_level --stage baseline --run artifacts/experiments/take_level_gru/v001 --device cuda:0
& $Python scripts/train_poseforge.py --split-strategy take_level --stage gru16 --run artifacts/experiments/take_level_gru/v001 --device cuda:0
& $Python scripts/train_poseforge.py --split-strategy take_level --stage gru32 --run artifacts/experiments/take_level_gru/v001 --device cuda:0
& $Python scripts/train_poseforge.py --split-strategy take_level --stage compare --run artifacts/experiments/take_level_gru/v001 --device cuda:0
& $Python scripts/train_poseforge.py --split-strategy take_level --stage internal-test --run artifacts/experiments/take_level_gru/v001 --device cuda:0
& $Python scripts/train_poseforge.py --split-strategy take_level --stage finalize --run artifacts/experiments/take_level_gru/v001 --device cuda:0
& $Python scripts/train_poseforge.py --split-strategy take_level --stage export --run artifacts/experiments/take_level_gru/v001 --device cuda:0
& $Python scripts/train_poseforge.py --split-strategy take_level --stage report --run artifacts/experiments/take_level_gru/v001 --device cuda:0
& $Python scripts/train_poseforge.py --split-strategy take_level --stage verify --run artifacts/experiments/take_level_gru/v001
& $Python -m pytest -q
& $Python -m pytest -q tests/test_take_gru.py tests/test_phase3.py tests/test_real_training.py tests/test_model_improvement.py
& $Python scripts/evaluate_real_world.py --help
& $Python inference/run.py --help
```

The audit ranks 512 deterministic candidate allocations using data coverage/balance
only and freezes the winner before training. No temporal embargo or frame split
is used. `prepare` is an alias for `sequences`; `verify` checks the frozen source
groups and current source/release hashes. New audits always allocate a new version.
Train commands accept `--resume`, restoring the last completed epoch, optimizer,
early-stopping state and RNG state. Existing subject-wise modes remain supported.

Sequence options are frozen by audit: `--train-stride 4` (or 8),
`--min-sequence-coverage 0.8`, `--max-sequence-gap 0.25`, `--max-missing-run 3`,
`--per-source-window-cap 48`, `--split-candidates 512`, and `--seed 302`.
Lengths 16/32 are selected by their explicit training stages. Evaluation stride is
one. This baseline deliberately rejects partial/padded contexts instead of inventing
extra frames. Repeated ticks are retained; backward ticks, large gaps, track/session
changes and long missing runs reset context. Every rejected window and its lineage
is logged. Each sequence uses one source take and camera; no future-frame access,
cross-camera state sharing, keypoint interpolation or new smoothing is introduced.

MLP and both GRUs trained from scratch on CUDA at seeds 302/303/304. GRUs use one
unidirectional 64-hidden-unit layer, head dropout 0.15, batch 64, AdamW lr 0.001,
weight decay 0.001, gradient clipping 1, and validation early stopping. The MLP
retains its established architecture/configuration. A decreasing-loss real CUDA
smoke precedes each model's three full runs.

The matched comparison uses exactly the same 1,065 frozen validation target frames:
MLP mean frame Macro-F1 0.9597, GRU16 0.9032, GRU32 0.8774. Own coverage and MLP
all-eligible-frame results are also saved. GRUs did not improve flicker or class
recall, so neither was promoted. Only the frozen fresh MLP winner was evaluated
once on the internal held-out takes; the old full-data checkpoint was never scored
on them. The internal result is labeled **INTERNAL TAKE-LEVEL TEST – SEEN SUBJECTS,
UNSEEN TAKES**, not unseen-subject or external accuracy.

The existing `artifacts/models/mlp_wide_full_real_dataset/v001` remains the full-data
deployment candidate, including all 13 participants / 14,256 eligible observations.
It was not unnecessarily retrained. Its four-epoch historical subject-development
schedule is preserved; the new take-control median best epoch (14) does not silently
replace that broader-development choice. Actual trained GRU research checkpoints,
ONNX models, labels, feature/sequence contracts and provenance are at:

* `artifacts/experiments/take_level_gru/v001/gru16/deployment/`
* `artifacts/experiments/take_level_gru/v001/gru32/deployment/`

These GRUs trained on the take-level Train partition only and are **not promoted**
full-data releases. PT/ONNX parity passed on real Train/Validation sequences.
ONNX sequence lengths are fixed at 16/32; batch is dynamic and tested. The old MLP
PT/ONNX paths remain compatible. Each PosePipeline owns its rolling context buffer;
the shared classifier is stateless. First 15/31 frames abstain, as do invalid current
poses. Person/stream changes and gaps reset context, avoiding stale predictions.

The following real-video smoke commands were executed on an existing **Train**
clip, not the internal test. Camera capture sidecars are auto-detected. These are
functional runs, not independent accuracy tests:

```powershell
& $Python inference/run.py --source artifacts/dataset_working/Real_Posture_Dataset/recordings/CHANH/session_20261027_1124/label_1/take_001/cam01_raw.mp4 --mode video --model artifacts/models/mlp_wide_full_real_dataset/v001 --pose artifacts/pretrained/yolo11n-pose.pt --device cuda:0 --backend onnx --cls-device cpu --headless --name take_gru_selected_full_model
& $Python inference/run.py --source artifacts/dataset_working/Real_Posture_Dataset/recordings/CHANH/session_20261027_1124/label_1/take_001/cam01_raw.mp4 --mode video --model artifacts/experiments/take_level_gru/v001/gru16/deployment --pose artifacts/pretrained/yolo11n-pose.pt --device cuda:0 --backend onnx --cls-device cpu --headless --name take_gru_research_gru16
& $Python inference/run.py --source artifacts/dataset_working/Real_Posture_Dataset/recordings/CHANH/session_20261027_1124/label_1/take_001/cam01_raw.mp4 --mode video --model artifacts/experiments/take_level_gru/v001/gru32/deployment --pose artifacts/pretrained/yolo11n-pose.pt --device cuda:0 --backend onnx --cls-device cpu --headless --name take_gru_research_gru32
```

Offline GRU video requires original capture timestamps: either the camera sidecar
or `--timestamps PATH`. Encoded MP4 FPS is not silently used as acquisition time.
CSV and JSONL contain every processed frame, including warmup/abstention; aggregate
clip prediction, prediction coverage, switches, latency and runtime FPS are saved
in `performance.json`. `--display` uses the existing preview, with explicit GRU
warmup status. No second GUI was created.

### External and live testing: PENDING, not executed

The user explicitly deferred all external accuracy and real webcam testing.
No hardware was opened or probed. Only software guards and command parsers were
tested. Do not execute the following future examples until the user supplies new
consented recordings or explicitly authorizes a camera:

```powershell
# FUTURE ONLY — not run; requires explicit camera authorization.
& $Python inference/run.py --source 0 --mode camera --model artifacts/experiments/take_level_gru/v001/gru16/deployment --device cuda:0 --backend onnx --display --max-seconds 15

# FUTURE ONLY — not run; define these paths after collecting a genuinely new,
# consented, physically isolated dataset and freezing its manifest/model.
& $Python scripts/evaluate_real_world.py --data $AuthorizedExternalRoot --manifest $AuthorizedExternalManifest --model artifacts/models/mlp_wide_full_real_dataset/v001 --backend onnx --device cuda:0 --display
```

The external JSON manifest requires `dataset_role: "independent_external_test"`,
`consent_confirmed: true`, and a `recordings` array. Each record requires
`person_id`, `session_id`, `take_id`, `camera_id`, integer `label` 0–5,
`video_path` and `timestamps_path` relative to the isolated external root. Old
participant IDs, training-video hashes, duplicate videos and missing lineage are
rejected. An evaluation registry records the frozen model/manifest before running.
External performance reports include abstention coverage and the six-class metrics;
further tuning after reviewing failures makes those data development evidence.

Reports: `artifacts/reports/TAKE_LEVEL_GRU_TRAINING_REPORT.md` and
`artifacts/reports/REAL_WORLD_TEST_REPORT.md`. All prior recordings, recovery files,
checkpoints and reports remain intact. The full test suite was run without hiding
the known Windows symlink-permission failure; the new workflow requires no admin
rights or symlink creation.
