"""PoseForge: existing YOLO+ByteTrack/selector/causal features, interchangeable PT/ONNX classifier."""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from collector.posture_collector.pose import PoseRunner
from collector.posture_collector.single_person import SinglePersonSelector
from shared.pose_features import CausalPoseNormalizer, FEATURE_VERSION
from shared.pose_geometry import GEOMETRY_VERSION, geometry_features, validate_contract
from shared.sequence_data import TemporalSequenceBuffer

SKELETON = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 11), (6, 12))
REQUIRED_JOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                   'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                   'left_wrist', 'right_wrist', 'left_hip', 'right_hip']
# Distinct BGR colors, ordered by class ID. OpenCV does the chart drawing; no matplotlib.
CLASS_COLORS = ((225, 159, 70), (72, 183, 239), (175, 118, 241),
                (96, 191, 108), (222, 129, 200), (83, 204, 224))
SHORT_LABELS = ('UP', 'HUNCH', 'DESK', 'BACK', 'SIDE', 'HAND')


class PostureClassifier:
    """Classify 52 normalized floats using original PT or ONNX (auto prefers exported ONNX)."""

    def __init__(self, model_dir: Path, backend='auto', device='cpu'):
        model_dir = Path(model_dir)
        contract = json.loads((model_dir / 'feature_contract.json').read_text(encoding='utf-8'))
        self.input_dim = validate_contract(contract)
        self.feature_version = contract['version']
        self.sequence_contract = json.loads((model_dir/'sequence_contract.json').read_text(encoding='utf-8')) if (model_dir/'sequence_contract.json').exists() else None
        self.is_temporal = self.sequence_contract is not None
        self.sequence_length = int(self.sequence_contract['sequence_length']) if self.is_temporal else 1
        if self.is_temporal and (self.sequence_contract.get('input_dim')!=52 or self.sequence_length not in (16,32) or self.feature_version!=FEATURE_VERSION):
            raise ValueError('Unsupported sequence feature contract')
        if self.is_temporal and (self.sequence_contract.get('hidden_dim',64)!=64
                or self.sequence_contract.get('layers',1)!=1 or self.sequence_contract.get('bidirectional',False)):
            raise ValueError('Unsupported GRU architecture metadata')
        self.conf_thresh = float(contract['confidence_threshold'])
        self.ema_alpha = float(contract['ema_alpha'])
        if backend not in ('auto', 'pt', 'onnx'):
            raise ValueError('Backend must be auto, pt, or onnx')
        self.backend = ('onnx' if (model_dir / 'model.onnx').is_file() else 'pt') if backend == 'auto' else backend
        self.device = device
        if self.backend == 'onnx':
            import onnxruntime as ort
            labels_path = model_dir / 'inference_labels.json'
            if not labels_path.is_file():
                raise FileNotFoundError('Missing inference_labels.json; run export_onnx.py first')
            self.labels = json.loads(labels_path.read_text(encoding='utf-8'))
            available = ort.get_available_providers()
            if device.startswith('cuda') and 'CUDAExecutionProvider' not in available:
                raise RuntimeError('ONNX CUDA provider unavailable; choose --cls-device cpu or install onnxruntime-gpu')
            providers = (['CUDAExecutionProvider', 'CPUExecutionProvider'] if device.startswith('cuda')
                         else ['CPUExecutionProvider'])
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(model_dir / 'model.onnx'),
                                                 sess_options=opts, providers=providers)
            if (self.session.get_inputs()[0].name != 'features'
                    or self.session.get_outputs()[0].name != 'logits'):
                raise ValueError('Unexpected ONNX model signature')
            if self.session.get_inputs()[0].shape[-1] != self.input_dim:
                raise ValueError('ONNX input width differs from feature contract')
            if (self.session.get_inputs()[0].type != 'tensor(float)'
                    or self.session.get_outputs()[0].type != 'tensor(float)'
                    or self.session.get_outputs()[0].shape[-1] != 6):
                raise ValueError('Expected FP32 ONNX input and six FP32 logits')
            expected_rank=3 if self.is_temporal else 2
            if len(self.session.get_inputs()[0].shape)!=expected_rank or (self.is_temporal and self.session.get_inputs()[0].shape[1]!=self.sequence_length):
                raise ValueError('ONNX sequence dimensions differ from runtime contract')
            metadata = self.session.get_modelmeta().custom_metadata_map
            if metadata.get('feature_version', self.feature_version) != self.feature_version:
                raise ValueError('ONNX feature metadata differs from feature contract')
        else:
            import torch
            from shared.mlp_model import build_classifier_model
            ckpt = torch.load(model_dir / 'model_final.pt', weights_only=True, map_location='cpu')
            if ckpt.get('stage') != 'final' or ckpt['feature_contract'] != contract:
                raise ValueError('PT checkpoint feature contract mismatch')
            self.labels = ckpt['label_names']
            config_file = model_dir / 'model_config.json'
            if config_file.is_file():
                config = json.loads(config_file.read_text(encoding='utf-8'))
                if config.get('model', ckpt['model_name']) != ckpt['model_name']:
                    raise ValueError('PT architecture differs from model_config.json')
            if ckpt['model_name'].startswith('gru') != self.is_temporal:
                raise ValueError('Architecture and sequence contract disagree')
            if self.is_temporal and (int(ckpt['model_config'].get('sequence_length',ckpt['model_name'][3:]))!=self.sequence_length
                                    or int(ckpt['model_config'].get('hidden_dim',64))!=64):
                raise ValueError('Checkpoint architecture and runtime sequence contract disagree')
            self.model = build_classifier_model(ckpt['model_name'], ckpt['model_config']['dropout'], self.input_dim)
            self.model.load_state_dict(ckpt['model_state_dict'], strict=True)
            if device.startswith('cuda') and not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable for PT classifier; choose --cls-device cpu')
            self.model = self.model.to(device).eval()
        if len(self.labels) != 6:
            raise ValueError('Expected six label names')
        label_map = model_dir / 'label_map.json'
        if label_map.is_file():
            entries = json.loads(label_map.read_text(encoding='utf-8'))['labels']
            if [r['id'] for r in entries] != list(range(6)) or [r['name'] for r in entries] != self.labels:
                raise ValueError('Classifier labels disagree with label_map.json')
        self.model_dir = model_dir

    def predict(self, features):
        if self.is_temporal:
            raw=np.asarray(features,np.float32)
            if raw.shape!=(self.sequence_length,52) or not np.isfinite(raw).all():raise ValueError('GRU requires a full finite causal sequence')
            a=raw.reshape(1,self.sequence_length,52)
        else:a = np.asarray(features, np.float32).reshape(1, -1)
        if self.feature_version == GEOMETRY_VERSION and a.shape[1] == 52:
            a = geometry_features(a)
        if a.shape[-1] != self.input_dim or not np.isfinite(a).all():
            raise ValueError('Classifier input does not match feature contract')
        if self.backend == 'onnx':
            logits = self.session.run(['logits'], {'features': a})[0][0]
        else:
            import torch
            with torch.inference_mode():
                logits = self.model(torch.from_numpy(a).to(self.device)).cpu().numpy()[0]
        if logits.shape != (6,) or not np.isfinite(logits).all():
            raise ValueError('Classifier returned non-finite or invalid six-class logits')
        logits = logits - np.max(logits)
        exps = np.exp(logits)
        return exps / exps.sum()


class StableClass:
    def __init__(self, window=5, min_conf=.55):
        self.window = deque(maxlen=window)
        self.min_conf = min_conf

    def reset(self):
        self.window.clear()

    def update(self, probs):
        index = int(np.argmax(probs))
        self.window.append(index if float(probs[index]) >= self.min_conf else -1)
        if len(self.window) == self.window.maxlen and len(set(self.window)) == 1 and index != -1:
            return index
        return None


class PosePipeline:
    def __init__(self, pose_file, classifier, device='auto', stable_frames=5, min_conf=.55):
        self.pose = PoseRunner('yolo', pose_file, video=True, device=device)
        overrides = getattr(getattr(self.pose, 'engine', None), 'overrides', None)
        if isinstance(overrides, dict):
            overrides['half'] = False
        self.classifier = classifier
        self.normalizer = CausalPoseNormalizer(classifier.conf_thresh, classifier.ema_alpha)
        self.selector = SinglePersonSelector()
        self.stable = StableClass(stable_frames, min_conf)
        self.last_track_id = None
        self.last_box = None
        self.roi = None
        self.missing = 0
        self.temporal = TemporalSequenceBuffer(classifier.sequence_contract) if getattr(classifier,'is_temporal',False) else None

    def reset(self):
        self.pose.reset_tracking()
        self.selector = SinglePersonSelector()
        self.normalizer.reset()
        self.stable.reset()
        self.last_track_id = None
        self.last_box = None
        self.missing = 0
        if self.temporal is not None:self.temporal.reset()

    def close(self):
        self.pose.close()

    def set_roi(self, roi):
        """Normalized full-frame rectangle; detection coordinates never get rescaled."""
        value = None if roi is None else tuple(float(x) for x in roi)
        if value is not None and (len(value) != 4 or not all(np.isfinite(value))
                or not (0 <= value[0] < value[2] <= 1 and 0 <= value[1] < value[3] <= 1)):
            raise ValueError('ROI must be normalized x1,y1,x2,y2 within the frame')
        if value != self.roi:
            self.roi = value
            self.reset()

    def process(self, frame, timestamp_ns=None):
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        timestamp_ns=time.monotonic_ns() if timestamp_ns is None else int(timestamp_ns)
        people = self.pose.track_frame(frame)
        pose_ms = (time.perf_counter() - t0) * 1000
        # Ultralytics Profile synchronizes CUDA and measures the model forward
        # separately from preprocessing, postprocessing and ByteTrack.
        results = getattr(getattr(getattr(self.pose, 'engine', None), 'predictor', None), 'results', None)
        yolo_ms = results[0].speed.get('inference') if results else None
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            def inside(person):
                box = person.get('bbox')
                return (box is not None and x1*w <= (box[0]+box[2])/2 <= x2*w
                        and y1*h <= (box[1]+box[3])/2 <= y2*h)
            people = [person for person in people if inside(person)]
        selected, reason = self.selector.select(people)
        kp = None if selected is None else selected.get('keypoints')
        track_id = None if selected is None else selected.get('track_id')
        info = {'class_id': None, 'label': None, 'score': 0.0, 'stable_id': None,
                'probs': None, 'keypoints_valid': 0, 'selection_reason': reason,
                'track_id': track_id, 'bbox': None if selected is None else selected.get('bbox'),
                'yolo_ms': yolo_ms, 'pose_tracking_ms': pose_ms, 'classifier_ms': None,
                'feature_dim': None, 'roi': self.roi}
        def finish(keypoints):
            out = draw_overlay(frame, info, keypoints)
            info['inference_ms'] = (time.perf_counter() - t0) * 1000
            info['pipeline_ms'] = info['inference_ms']
            return info, out
        if kp is None:
            absent=self.normalizer.transform(None, w, h)[0]
            if self.temporal is not None:self.temporal.push(absent,timestamp_ns,False)
            self.missing += 1
            self.stable.reset()
            if self.missing > 8:
                self.normalizer.reset()
                self.last_track_id = None
                self.last_box = None
                if self.temporal is not None:self.temporal.reset()
            return finish(None)

        from collector.posture_collector.single_person import overlap
        box = selected.get('bbox')
        identity_changed = (track_id is not None and self.last_track_id is not None
                            and track_id != self.last_track_id)
        # Unassigned detections are valid in the original selector. A spatially
        # disjoint replacement must not inherit the previous person's state.
        if self.last_box is not None and box is not None and (track_id is None or self.last_track_id is None):
            identity_changed = identity_changed or overlap(self.last_box, box)[0] == 0
        if identity_changed:
            self.normalizer.reset()
            self.stable.reset()
            if self.temporal is not None:self.temporal.reset()
        if track_id is not None:
            self.last_track_id = track_id
        else:
            self.last_track_id = None
        self.last_box = box
        self.missing = 0
        features, _, _ = self.normalizer.transform(kp, w, h)
        info['feature_dim'] = int(features.size)
        if features.size != 52 or not np.isfinite(features).all():
            raise ValueError('Causal normalizer must produce 52 finite values')
        info['keypoints_valid'] = int(features[:, 3].sum())
        # Face-only poses must not reach the posture classifier.
        visible = features[:, 3].astype(bool)
        pose_ready = (visible[5] and visible[6]
                      and ((visible[7] and visible[9]) or (visible[8] and visible[10])))
        info['pose_quality'] = 'ready' if pose_ready else 'insufficient_upper_body'
        if not pose_ready:
            self.stable.reset()
            if self.temporal is not None:
                self.temporal.reset()
            return finish(kp)
        classifier_input=features
        if self.temporal is not None:
            classifier_input=self.temporal.push(features,timestamp_ns,info['keypoints_valid']>=4,track_id)
            info['temporal_state']='ready' if classifier_input is not None else 'warmup_or_invalid_context'
            if classifier_input is None:
                self.stable.reset()
                return finish(kp)
        if info['keypoints_valid']:
            if getattr(self.classifier, 'device', 'cpu').startswith('cuda'):
                import torch
                torch.cuda.synchronize(self.classifier.device)
            cls_start = time.perf_counter()
            probs = self.classifier.predict(classifier_input)
            if getattr(self.classifier, 'device', 'cpu').startswith('cuda'):
                torch.cuda.synchronize(self.classifier.device)
            info['classifier_ms'] = (time.perf_counter() - cls_start) * 1000
            index = int(np.argmax(probs))
            info.update(class_id=index, label=self.classifier.labels[index],
                        score=float(probs[index]), stable_id=self.stable.update(probs),
                        probs=[float(p) for p in probs])
        else:
            self.stable.reset()
        return finish(kp)


def draw_overlay(frame, info, kp):
    out = frame.copy()
    if info.get('bbox') is not None:
        x1, y1, x2, y2 = map(int, info['bbox'])
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 180, 60), 2)
    if info.get('roi') is not None:
        h, w = out.shape[:2]
        x1, y1, x2, y2 = info['roi']
        cv2.rectangle(out, (round(x1*w), round(y1*h)), (round(x2*w), round(y2*h)), (180, 100, 255), 2)
    if kp is not None:
        pts = np.asarray(kp, np.float32)
        if pts.shape == (17, 3):
            ok = np.isfinite(pts).all(axis=1) & (pts[:, 2] >= .35)
            for i, j in SKELETON:
                if ok[i] and ok[j]:
                    cv2.line(out, tuple(pts[i, :2].astype(int)), tuple(pts[j, :2].astype(int)),
                             (0, 220, 0), 2)
            for i in range(13):
                if ok[i]:
                    cv2.circle(out, tuple(pts[i, :2].astype(int)), 3, (0, 240, 255), -1)
    return out


def draw_dashboard(frame, info, labels, fps=0., status='LIVE'):
    """Compact live window: FPS, chosen class/confidence, six distinct probability columns."""
    if info.get('temporal_state')=='warmup_or_invalid_context':status='GRU WARMUP'
    h, w = frame.shape[:2]
    # Bound GUI width without resizing source data passed to YOLO or saved events.
    factor = min(1., 960. / w, 700. / h)
    if factor < 1.:
        frame = cv2.resize(frame, (max(1, round(w * factor)), max(1, round(h * factor))),
                           interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
    panel_w, canvas_h = 360, max(h, 450)
    canvas = np.full((canvas_h, w + panel_w, 3), (28, 29, 33), np.uint8)
    canvas[:h, :w] = frame
    x = w + 18
    def line(text, y, size=.65, color=(235, 235, 235), thick=2):
        cv2.putText(canvas, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    size, color, thick, cv2.LINE_AA)

    line('POSEFORGE  /  LIVE', 38, .69)
    state_color = (88, 210, 118) if status == 'LIVE' else (70, 160, 240)
    line(f'{status}   FPS {fps:5.1f}', 75, .62, state_color)

    probs = info.get('probs')
    cls = info.get('class_id')
    stable = info.get('stable_id')
    if probs is None:
        message = ('INSUFFICIENT UPPER-BODY POSE' if info.get('pose_quality') == 'insufficient_upper_body' else 'NO POSE / NO PREDICTION')
        line(message, 125, .48, (90, 175, 245))
        line('Confidence: --', 160, .56)
    else:
        decision = int(stable) if stable is not None else int(cls)
        label = str(labels[decision])
        state = 'STABLE' if stable is not None else 'WAITING'
        label = label[:25]
        line(f'{state}: {label}', 125, .55, CLASS_COLORS[decision])
        line(f'Confidence: {float(info["score"])*100:5.1f}%', 160, .58)
    line('CLASS PROBABILITIES', 208, .54, (180, 180, 180), 1)

    chart_left, top, bottom = x + 5, 235, 370
    cv2.line(canvas, (chart_left, bottom), (w + panel_w - 18, bottom), (92, 92, 92), 1)
    values = np.asarray(probs if probs is not None else np.zeros(6), np.float32)
    bar_width, step = 32, 50
    for i in range(6):
        bx = chart_left + i * step + 3
        height = int(np.clip(values[i], 0, 1) * (bottom - top))
        if height:
            cv2.rectangle(canvas, (bx, bottom - height), (bx + bar_width, bottom - 1),
                          CLASS_COLORS[i], -1)
        cv2.putText(canvas, f'{values[i]*100:.0f}', (bx + 1, max(top - 6, bottom - height - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, SHORT_LABELS[i], (bx - 4, bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, CLASS_COLORS[i], 1, cv2.LINE_AA)
    line('Press Q to quit', 428, .52, (145, 145, 145), 1)
    return canvas
