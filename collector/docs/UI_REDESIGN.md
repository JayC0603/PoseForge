# PoseForge UI implementation and validation

Historical report for the UI redesign. Subsequent ByteTrack changes extend pose, capture-generation and export handling; see [TRACKING.md](TRACKING.md) for the current implementation and verification.

Implemented and tested on Windows, 27 October 2026.

## A. Root causes

- The original collector's two tall forms, preview grid and log shared an unscrollable vertical layout. Its measured minimum client height was **1008 logical pixels**, exceeding common Windows work areas. Asking for 1366×720 still produced a 1366×1008 window.
- Camera previews had a 320×220 minimum per camera. Two columns and proportional form stretch competed for width and height.
- Large global QSS padding combined with compressed widget allocations clipped input and button text. Applying background colors to every QWidget also painted distracting strips behind labels.
- Wrapped labels, unconstrained nested layouts, expanding header badges and inconsistent size policies compounded the minimum-size pressure.
- Qt already handles Windows DPI. The missing piece was adapting to the smaller available logical client area at 125% and 150%, rather than assuming display pixels were usable layout space.
- Camera reconfiguration and shutdown waited for workers on the GUI thread. Keyboard handling was duplicated, and Start was enabled without checking all required fields/camera readiness.

## B. Changes made

| File | Responsibility |
| --- | --- |
| `posture_collector/ui_components.py` | Shared dark theme, labels/buttons/cards, scroll containers, aspect-preserving camera painting and offline/stalled states. |
| `posture_collector/ui_layout.py` | Sidebar, responsive header, adaptive camera grid, three scrollable control cards and bounded collapsible log. |
| `posture_collector/ui_assets/` | Bundled SVG dropdown and checkbox indicators. Navigation uses Qt standard icons. |
| `posture_collector/ui.py` | Existing action wiring retained; validates Start, locks recording identity/source settings, measures received FPS, handles shortcuts once, manages background camera operations and safe asynchronous shutdown. |
| `tests/test_ui.py` | Layout, subject/camera forms, progress, keyboard, raw recording, timestamps, workers and shutdown regressions. |
| `tests/test_pipeline_cli.py` | Both keypoint formats, extraction wiring, existing processing CLI and raw-video preservation. |
| `tools/validate_ui.py` | Repeatable native Qt screenshot/state validation, isolated synthetic feeds and saved metrics. |

The right column is 348–380 logical pixels; the sidebar is 184–216 depending on available width. Camera cards switch between two columns and one. The Event Log is 150 logical pixels and initially collapses on short windows. The minimum window is 960×560; larger control content scrolls. Session selection supports existing sessions and new typed IDs. Capture counts come from saved metadata for the selected person across sessions; amber counts identify review flags through their tooltips. Notes use the existing consent-reference field.

**No changes were made to `capture.py`, `storage.py`, `process.py`, `pose.py`, the dataset schema, or backend APIs.** Preview skeletons are painted on a copied QImage. Writers still receive raw camera frames. Test recordings are isolated under `ui_validation/` and pytest temporary directories; existing user recordings/configuration were not edited.

## C. UI validation

See [screenshot index and measurements](ui_validation/README.md). Six profiles each have 11 final state captures: disconnected, connected, recording, scrolled progress, Camera Sources, Dataset Tools, log toggle, stalled, reconnected, simulated overlay and four sources.

- 1920×1080 display profiles at effective Qt DPR 1.0, 1.25 and 1.5.
- 1600×900 and 1366×768 client sizes at DPR 1.0.
- Native maximized window on this machine's 1920×1080 display at Windows 125% scaling.
- Normal and maximized `main.py` entry-point smoke runs.

100%/150% use a **process-local Qt scale multiplier**, not a change to Windows settings. Native Windows scaling is 125%. PNGs capture the actual client area: Windows limited the full-screen-sized normal clients to 1920×1055 physical pixels; native maximized client capture is 1920×991 because the native frame and taskbar consume space. Exact logical and physical dimensions are saved per capture in `metrics.json`. No OS-level 1600×900 or 1366×768 display-mode switch was performed; those are actual window/client sizes. Horizontal overflow was zero across all final captures.

The supplied screenshot was not present in this chat; the original application was run and captured in `ui_validation/before_1366.png` before edits. Final captures were inspected for button visibility, header containment, readable labels, card padding and scroll access. Scrolled cards can naturally extend beyond their viewport and remain accessible by scrolling.

## D. Functional tests

Baseline: **5 passed**. Final project environment: **17 passed in 22.82s**, using `.venv/Scripts/python.exe -m pytest -q`.

- Two local OpenCV sources connect, disconnect, stall and reconnect through the real GUI/backend.
- S/E starts/stops; auto-repeat and duplicate commands are ignored; typing in inputs does not record.
- Missing/invalid identity or session, offline sources and processing disable Start. Stop and identity/source locking track real recording/finalization states.
- Both independent raw MP4s, correct person/session/label/take paths and timestamp CSVs are verified. Every decoded frame in the overlay-isolation test remains the original flat camera color despite bright preview skeletons.
- Metadata-derived counts, next take, session suggestions, subject consent/note and camera add/remove work.
- Extraction through deterministic inference stubs covers YOLO 17 and MediaPipe 33 formats. Actual `normalize`, `split`, `sequences` and `manifest` CLI commands complete for both backends; hashes confirm raw MP4s remain unchanged.
- Slow pose work and camera reconfiguration leave Qt responsive; closing during recording waits for writers, and background processing/pose workers are handled outside the GUI thread.

**Not verified:** real DroidCam/network devices; actual YOLO/MediaPipe pretrained inference (no local weights supplied); changing the operating system's global DPI/display modes. Synthetic footage/landmarks are marked test-only and must not be used as training data.

## E. How to run

A local `.venv` with PySide6, OpenCV, NumPy and pytest is installed. From PowerShell:

```powershell
Set-Location 'C:\Users\tienq\PoseForge_Real_Posture_Collector_V1'
& .\.venv\Scripts\python.exe .\main.py
```

The existing `run_windows.bat` also uses this environment. Optional pose models/dependencies are still configured as described in the main README.

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe .\tools\validate_ui.py --width 1920 --height 1080 --scale 1.25
```
