# ByteTrack integration

## Run and configure

```powershell
Set-Location 'C:\Users\tienq\PoseForge_Real_Posture_Collector_V1'
& .\.venv\Scripts\python.exe .\main.py
```

The local environment includes Ultralytics 8.4.190, LAP and CPU PyTorch. Both official `models/yolo11n-pose.pt` and `models/yolo11m-pose.pt` were downloaded for verification. The app still requires a local file and does not download models during inference. Select a model in Camera Sources, choose YOLO-Pose + ByteTrack and an inference device, then enable tracking. CPU is verified here; CUDA requires a compatible CUDA-enabled PyTorch installation and GPU.

For another environment, install `requirements-pose.txt` (also includes the optional MediaPipe backend). Ultralytics is pinned because the raw-data adapter and global allocator isolation are verified against this version.

## Architecture and isolation

- `pose.py`: calls `model.track(source=frame, persist=True, tracker="bytetrack.yaml", conf=0.25, verbose=False, classes=[0])`. No custom association, motion estimation, temporal filtering or coordinate interpolation is implemented.
- Every live camera owns a separate `PoseRunner`, YOLO model, predictor, ByteTrack instance, track lists, ID allocation state and tracking-session UUID. A single inference thread calls those instances serially. They are created lazily, once per camera, not every frame. MediaPipe also gets separate runners.
- Upstream `BaseTrack._count` is process-global. A shared lock scopes that allocator to its owning runner for the duration of an official tracking/reset call, then restores the surrounding allocator. IDs are allocated by ByteTrack itself; they are neither fabricated nor remapped. The same lock covers offline and live calls, preventing allocator races. Tests specifically reset CAM02 while CAM01 gains another person and verify CAM01 cannot reuse an active ID.
- A callback snapshots original detector boxes, confidence and keypoint arrays **before** Ultralytics filters unconfirmed detections or updates tracking boxes. Association reads ByteTrack's own `STrack.idx` detection index. This preserves unmatched people and original detector coordinates without implementing another association algorithm.
- `LatestFrames` retains at most one pending frame per configured camera and services cameras in FIFO order. In-flight results carry a generation token and are discarded if their source was replaced. Inference runs off the Qt thread. Capture and raw writers never wait for the tracking lock or inference.
- Tracking FPS measures completed inference samples separately from camera receive FPS; the configured raw writer FPS remains unchanged. Skipped live inference frames are expected. Offline extraction reads **every** available video frame sequentially.

Models consume memory independently per camera. YOLO11m costs more memory than YOLO11n; model count scales with enabled cameras. Auto device uses Ultralytics device selection. If GPU memory is insufficient, choose CPU or YOLO11n, which retains identical camera-state isolation. CUDA/multi-GPU memory behavior could not be measured on this CPU-only environment. Slow/low-rate tracking can increase ID switches; raw recording is prioritized.

## Reset and target selection

Each capture object has a unique stream ID, and its connection generation advances whenever the source opens/reopens, including local-file loops. A changed token resets only that camera's tracker while retaining loaded weights. Restarting preview workers creates fresh model instances. Changing dataset Person ID or Session starts a new tracking context as well. Labels/takes for the same participant/session may retain continuity within the same uninterrupted live stream; this is intentional.

Exactly one assigned person can become the initial candidate. Multiple people require choosing an ID in that camera's header. Once selected, an absent ID stays `missing`, even if another person becomes the only detection. Only the return of that same ID through ByteTrack's own association, or an explicit operator selection, resumes selection. Reconnect/reset after a selection requires explicit reselection; a new stream's numeric ID 1 never automatically inherits the old target. Target choices can be changed while recording; model/source/identity settings are locked.

IDs are scoped by camera and tracking session. CAM01/ID 1 and CAM02/ID 1 do not assert the same identity. A selected ID is an operator choice, not proof of participant identity. There is no cross-camera re-identification and no guarantee of ID stability through arbitrary occlusion/disconnection.

## Files and schema

Folder layout and existing Person ID/session/label/take metadata remain compatible:

```text
recordings/TQB/session_.../label_0/take_001/
  cam01_raw.mp4
  cam01_timestamps.csv
  metadata.json
  tracking_selection.json
  keypoints/
    cam01_yolo17_live_tracks.txt
    cam01_yolo17_raw.txt
```

`*_yolo17_live_tracks.txt` is **sampled preview output**, not full-frame extraction. Each JSONL row contains `schema_version`, `event`, `camera_id`, source `frame_index` (explicit `frame_index_kind`), receive monotonic timestamp/UTC, `track_session`, `stream_token`, `persons`, selected ID/state and any error. Selection changes, camera loss/reset and recording boundaries are explicit events. `tracking_selection.json` contains the dataset identity, model/device, final per-camera selection/session, dropped-preview-sample count and errors. A bounded background journal writes these files independently of video writers; sample overflow is reported, never hidden.

`*_yolo17_raw.txt` remains JSON Lines with the legacy fields used by normalization. Version 2 adds:

| Field | Meaning |
| --- | --- |
| `frame_index`, `relative_ms`, `host_utc` | Existing video writer-frame index and timing fields. |
| `camera_id`, `timestamp`, `host_recv_monotonic_ns` | Camera ID, original host UTC (fallback: relative milliseconds), original sidecar monotonic timestamp when available. |
| `schema_version`, `tracker`, `track_session` | `2`, `bytetrack`, UUID scoped to this offline camera/video run. |
| `persons` | Every original detection, including unassigned detections. Each has `detection_index`, nullable `track_id`, `tracking_state`, detector `bbox` xyxy, `bbox_confidence`, `keypoints`, `keypoints_17` and `keypoint_confidences`. |
| `selected_track_id`, `selection_state` | Latched target and its current selection/missing state. |
| `track_id`, `bbox`, `bbox_confidence` | Current selected person's data; null if absent. |
| `keypoints` | Legacy selected-person 17×3 `[x_px,y_px,confidence]`; null for missing/ambiguous/unassigned/invalid targets. |
| `keypoints_17`, `keypoint_confidences` | Original 17×2 XY values and 17 confidence values, also present per person; no temporal changes. |

Empty frames keep `persons: []`. Unassigned detections use `track_id: null`; no synthetic IDs or confidence values are generated. A missing selected ID remains in `selected_track_id`, while current `track_id`/`keypoints` are null. Original raw MP4s and timestamp CSVs are never annotated or rewritten. Normalization still reads only legacy `keypoints`; missing targets remain masked. Subject-wise splitting still uses dataset Person IDs.

Offline IDs start fresh for each camera/video and cannot be assumed equal to sampled live IDs. Default extraction writes all detections; when multiple people appear before a candidate is selected, top-level training keypoints stay null until an explicit offline choice is supplied. Inspect the `persons` records before selecting offline IDs. An explicit camera choice applies to each corresponding video in that extraction request; review the target in every clip rather than assuming that ID implies identity.

```powershell
# First export all detections; ambiguous target keypoints remain masked.
& .\.venv\Scripts\python.exe .\process_dataset.py extract --backend yolo --model .\models\yolo11n-pose.pt --device cpu

# After reviewing offline IDs, select independently per camera and re-extract.
& .\.venv\Scripts\python.exe .\process_dataset.py extract --backend yolo --model .\models\yolo11m-pose.pt --track-id cam01=1 --track-id cam02=2 --device cpu
& .\.venv\Scripts\python.exe .\process_dataset.py normalize --backend yolo
& .\.venv\Scripts\python.exe .\process_dataset.py split
& .\.venv\Scripts\python.exe .\process_dataset.py sequences --backend yolo
```

Dataset Tools also accepts optional offline `cam01=1, cam02=2` choices. MediaPipe export retains its prior format and does not use ByteTrack. Old raw TXT files remain readable by normalization; omit `--only-missing` when deliberately upgrading an existing extraction.

## Verification

The original 17 regression tests passed before edits. The final suite has **30 passing tests**. Added tests use **official BYTETracker** with deterministic detector rows (not a fake association implementation), plus actual inference with both pretrained YOLO11n-Pose and YOLO11m-Pose on the official Ultralytics bus image. Controlled movement has zero ID switches over ten frames; a short disappearance recovers the retained ID in the tested case. Tests cover distinct people, independent camera states/counters, another camera's reset, no detections, unassigned rows, exact detector values, target loss/reselection, generation changes, stale-worker results, bounded queues, export/normalization, raw MP4 preservation and existing S/E/dataset behavior.

`tools/validate_tracking.py` runs the native desktop window with two prerecorded sources, real CPU models, target selections, S/E recording, full offline extraction, normalization, reconnect/reselection and safe close. Screenshots and real results are in `ui_validation/bytetrack/`. Test media/recordings are isolated and must not be included in a training dataset. See that folder's results for actual counts and timing.

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe .\tools\validate_tracking.py
& .\.venv\Scripts\python.exe .\tools\validate_tracking.py --entrypoint
```

Unverified: physical DroidCam/network behavior, CUDA and large camera-count/GPU memory limits; crowded long sequences and long occlusions. Actual MediaPipe inference was not retested; its format/normalization regression still passes.

Official references: [Ultralytics tracking interface](https://docs.ultralytics.com/modes/track/), [tracking callbacks and detection indices](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/track.py), [BaseTrack allocator](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/basetrack.py). Test assets: [YOLO11n weights](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt), [YOLO11m weights](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt), [Ultralytics bus image](https://github.com/ultralytics/assets/blob/main/im/bus.jpg).
