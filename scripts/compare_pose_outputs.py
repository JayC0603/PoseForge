#!/usr/bin/env python3
"""Read-only comparison of saved YOLO17 keypoints against YOLO11n/m Pose.

Run from the project root after applying the accompanying Git patch. Uses the
collector's exact PoseRunner (YOLO-Pose + ByteTrack) rather than raw YOLO.predict.
Does not extract/write keypoint files or modify labels/recordings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "collector"))
from posture_collector.pose import PoseRunner  # noqa: E402

WEIGHTS = ("yolo11n-pose.pt", "yolo11m-pose.pt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def box_iou(a, b) -> float:
    if a is None or b is None or len(a) != 4 or len(b) != 4:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def choose_raw(data: Path, person: str, camera: str, minimum_frames: int) -> Path:
    candidates = sorted((data / "recordings" / person).glob(
        f"*/label_*/take_*/keypoints/{camera}_yolo17_raw.txt"
    ))
    for raw in candidates:
        video = raw.parent.parent / f"{camera}_raw.mp4"
        if not video.is_file():
            continue
        with raw.open(encoding="utf-8") as stream:
            head = [json.loads(line) for _, line in zip(range(minimum_frames), stream) if line.strip()]
        if len(head) >= minimum_frames and sum(row.get("keypoints") is not None for row in head) >= minimum_frames // 2:
            return raw
    raise FileNotFoundError(
        f"No suitable {camera} raw keypoints/video found for {person}. "
        "Pass --raw to select a specific existing file or reduce --frames."
    )


def read_reference(raw: Path, frames: int):
    rows = {}
    with raw.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["frame_index"])
            if index >= frames:
                break
            if index in rows:
                raise ValueError(f"Duplicated frame index: {index}")
            rows[index] = row
    return rows


def compare(video: Path, reference: dict, weights: Path, frames: int, device: str, confidence: float):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode video: {video}")
    errors_px = []
    errors_norm = []
    conf_errors = []
    matched_iou = []
    reference_valid = 0
    matched_frames = 0
    decoded = 0
    try:
        # Exactly the collector's tracking path and its conf=.25/default imgsz.
        with PoseRunner("yolo", weights, video=True, device=device) as runner:
            for frame_index in range(frames):
                ok, frame = cap.read()
                if not ok:
                    break
                decoded += 1
                detections = runner.track_frame(frame)
                old = reference.get(frame_index)
                if not old or old.get("keypoints") is None:
                    continue
                original_box = old.get("bbox")
                if not original_box:
                    continue
                original = np.asarray(old["keypoints"], dtype=np.float32)
                if original.shape != (17, 3):
                    continue
                reference_valid += 1
                if not detections:
                    continue
                chosen = max(detections, key=lambda p: box_iou(original_box, p.get("bbox")))
                similarity = box_iou(original_box, chosen.get("bbox"))
                # Avoid comparing two different people in multi-person frames.
                if similarity < 0.25 or chosen.get("keypoints") is None:
                    continue
                new = np.asarray(chosen["keypoints"], dtype=np.float32)
                if new.shape != (17, 3):
                    continue
                matched_frames += 1
                matched_iou.append(similarity)
                good = (
                    (original[:, 2] >= confidence)
                    & (new[:, 2] >= confidence)
                    & np.all(np.isfinite(original), axis=1)
                    & np.all(np.isfinite(new), axis=1)
                )
                if not np.any(good):
                    continue
                delta = np.linalg.norm(original[good, :2] - new[good, :2], axis=1)
                errors_px.extend(delta.tolist())
                diagonal = float(np.hypot(frame.shape[1], frame.shape[0]))
                errors_norm.extend((delta / diagonal * 100).tolist())
                conf_errors.extend(np.abs(original[good, 2] - new[good, 2]).tolist())
    finally:
        cap.release()
    return {
        "decoded": decoded,
        "reference_valid": reference_valid,
        "matched_frames": matched_frames,
        "points": len(errors_px),
        "median_px": float(np.median(errors_px)) if errors_px else float("nan"),
        "p90_px": float(np.percentile(errors_px, 90)) if errors_px else float("nan"),
        "median_pct_diagonal": float(np.median(errors_norm)) if errors_norm else float("nan"),
        "median_conf_delta": float(np.median(conf_errors)) if conf_errors else float("nan"),
        "median_box_iou": float(np.median(matched_iou)) if matched_iou else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Existing Real_Posture_Dataset directory")
    parser.add_argument("--person", default="CHANH", help="Subject ID used for automatic test clip selection")
    parser.add_argument("--camera", default="cam01", help="Camera used for automatic test clip selection")
    parser.add_argument("--raw", type=Path, help="Explicit existing camXX_yolo17_raw.txt to compare")
    parser.add_argument("--frames", type=int, default=24, help="Sequential video frames to replay (default 24)")
    parser.add_argument("--device", choices=["auto", "cpu"], default="cpu")
    parser.add_argument("--confidence", type=float, default=0.50, help="Min keypoint confidence in BOTH outputs")
    args = parser.parse_args()
    if args.frames < 2 or not 0 < args.confidence < 1:
        parser.error("--frames must be >=2 and --confidence strictly between 0 and 1")
    data = args.data.expanduser().resolve()
    raw = args.raw.expanduser().resolve() if args.raw else choose_raw(data, args.person, args.camera, args.frames)
    if not raw.is_file() or not raw.name.endswith("_yolo17_raw.txt"):
        parser.error(f"Invalid raw file: {raw}")
    camera = raw.name.removesuffix("_yolo17_raw.txt")
    video = raw.parent.parent / f"{camera}_raw.mp4"
    if not video.is_file():
        parser.error(f"Missing source video: {video}")
    reference = read_reference(raw, args.frames)
    summary = raw.with_suffix(".summary.json")
    info = json.loads(summary.read_text(encoding="utf-8")) if summary.is_file() else {}
    print(f"Reference : {raw}")
    print(f"Video     : {video}")
    print(f"Reported source model: {info.get('model_path', 'unknown')}")
    print(f"Video frames to compare: {args.frames} | device={args.device} | confidence>={args.confidence}")
    print("Keypoints matched by frame index and person bounding-box IoU (not Track ID).")
    results = {}
    for name in WEIGHTS:
        weight = ROOT / "artifacts" / "pretrained" / name
        if not weight.is_file():
            parser.error(f"Weight missing: {weight}")
        print(f"\nRunning {name} | SHA256 {sha256(weight)}", flush=True)
        result = compare(video, reference, weight, args.frames, args.device, args.confidence)
        results[name] = result
        print(
            f"  matched: {result['matched_frames']}/{result['reference_valid']} valid reference frames "
            f"(decoded {result['decoded']}) | compared keypoints: {result['points']}\n"
            f"  median error: {result['median_px']:.2f} px "
            f"({result['median_pct_diagonal']:.3f}% image diagonal), p90: {result['p90_px']:.2f} px\n"
            f"  box IoU median: {result['median_box_iou']:.3f} "
            f"| confidence absolute delta median: {result['median_conf_delta']:.4f}"
        )
    ranked = sorted(
        ((name, info["median_px"]) for name, info in results.items()
         if info["points"] >= 20 and info["matched_frames"] >= 5 and np.isfinite(info["median_px"])),
        key=lambda item: item[1],
    )
    if len(ranked) == 2 and ranked[0][1] * 1.5 < ranked[1][1]:
        print(f"\n[LIKELY] Historical keypoints are closer to {ranked[0][0]} on this clip.")
    else:
        print("\n[INCONCLUSIVE] Compare more frames/clips, and verify environment/checkpoint hashes.")
    print("[READ ONLY] No original dataset file was created or changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
