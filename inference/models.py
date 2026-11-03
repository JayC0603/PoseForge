"""Explicit deployment catalog. Existing bundle metadata defines labels/features."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / 'artifacts/models/mlp_wide_full_real_dataset/v001'
POSE_MODEL = ROOT / 'artifacts/pretrained/yolo11n-pose.pt'
OUTPUT_ROOT = ROOT / 'artifacts/inference_runs'


@dataclass(frozen=True)
class ModelEntry:
    name: str
    path: Path
    backends: tuple
    labels: tuple
    research: bool = False


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def provenance(model_dir, pose_path=POSE_MODEL):
    folder = Path(model_dir)
    files = {p.name: sha256(p) for p in folder.iterdir()
             if p.name in {'model_final.pt', 'model.onnx', 'feature_contract.json',
                           'sequence_contract.json', 'model_config.json', 'label_map.json', 'inference_labels.json'}}
    return dict(model=str(folder), files=files, pose_model=str(pose_path), pose_sha256=sha256(pose_path))


def validate_bundle(path):
    """Load each backend sequentially, check contracts, finite outputs and parity."""
    from inference.pipeline import PostureClassifier
    from shared.pose_features import FEATURE_VERSION
    path = Path(path)
    contract = json.loads((path/'feature_contract.json').read_text(encoding='utf-8'))
    config = json.loads((path/'model_config.json').read_text(encoding='utf-8'))
    if contract['version'] != FEATURE_VERSION or contract['flatten_dim'] != 52:
        raise ValueError('Desktop catalog requires original causal 52-feature models')
    expected = 'gru'+str(json.loads((path/'sequence_contract.json').read_text())['sequence_length']) if (path/'sequence_contract.json').exists() else 'mlp_wide'
    if config['model'] != expected:
        raise ValueError('Architecture metadata mismatch')
    outputs = {}; labels = None; errors = {}
    rng = np.random.default_rng(20261031)
    sample = rng.normal(size=(int(expected[3:]), 52) if expected.startswith('gru') else (13, 4)).astype(np.float32)
    for backend, filename in [('pt', 'model_final.pt'), ('onnx', 'model.onnx')]:
        if not (path/filename).is_file():
            errors[backend] = 'Missing '+filename
            continue
        try:
            classifier = PostureClassifier(path, backend, 'cpu')
            output = classifier.predict(sample)
            if labels is not None and labels != classifier.labels:
                raise ValueError('Backend label ordering mismatch')
            labels = classifier.labels
            outputs[backend] = output
            del classifier
        except (ImportError, RuntimeError, OSError, ValueError, KeyError) as error:
            errors[backend] = str(error)
    if not outputs:
        raise ValueError('No validated backend: '+str(errors))
    if len(outputs) == 2:
        np.testing.assert_allclose(outputs['pt'], outputs['onnx'], atol=1e-5, rtol=1e-5)
    return tuple(outputs), tuple(labels), errors


def discover_models():
    specs = [('MLP Wide – Full Dataset v001 (default)', DEFAULT_MODEL, False),
             ('MLP Wide – Original v001', ROOT/'artifacts/models/mlp_wide/v001', False),
             ('GRU-16 – Research', ROOT/'artifacts/experiments/take_level_gru/v001/gru16/deployment', True),
             ('GRU-32 – Research', ROOT/'artifacts/experiments/take_level_gru/v001/gru32/deployment', True)]
    entries, errors = [], []
    for name, path, research in specs:
        try:
            backends, labels, warnings = validate_bundle(path)
            entries.append(ModelEntry(name, path, backends, labels, research))
            errors.extend(f'{name}: {k}: {v}' for k, v in warnings.items())
        except Exception as error:
            errors.append(f'{name}: {error}')
    return entries, errors
