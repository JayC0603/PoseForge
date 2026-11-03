"""Optional current-frame 2D geometry; base causal v2 remains unchanged.

One implementation handles NumPy deployment and Torch training. Every calculated
scalar has its own validity flag. No camera/depth inference or imputation.
"""
from __future__ import annotations

import numpy as np

from shared.pose_features import FEATURE_VERSION

GEOMETRY_VERSION = FEATURE_VERSION + '_geometry_v1'
GEOMETRY_NAMES = [
    'torso_direction_x', 'torso_direction_y',
    'head_shoulder_x', 'head_shoulder_y', 'head_hip_x', 'head_hip_y',
    'shoulder_direction_x', 'shoulder_direction_y',
    'left_elbow_cosine', 'right_elbow_cosine',
    'left_wrist_shoulder_x', 'left_wrist_shoulder_y',
    'right_wrist_shoulder_x', 'right_wrist_shoulder_y',
    'hip_direction_x', 'hip_direction_y',
    'torso_side_difference_x', 'torso_side_difference_y',
    'left_elbow_shoulder_y', 'right_elbow_shoulder_y',
    'left_wrist_head_y', 'right_wrist_head_y',
]
GEOMETRY_DIM = 52 + 2 * len(GEOMETRY_NAMES)


def geometry_features(features):
    """[... ,52] -> [... ,96], clipping calculated values only to [-4,4]."""
    is_numpy = isinstance(features, np.ndarray)
    if is_numpy:
        xp = np
        stack = lambda xs: np.stack(xs, axis=-1)
        cat = lambda xs: np.concatenate(xs, axis=-1)
        norm = lambda x: np.sqrt(np.sum(x*x, axis=-1))
        clamp = lambda x: np.clip(x, -4., 4.)
        safe = lambda x: np.maximum(x, 1e-6)
        cast = lambda x: x.astype(np.float32)
    else:
        import torch
        xp = torch
        stack = lambda xs: torch.stack(xs, dim=-1)
        cat = lambda xs: torch.cat(xs, dim=-1)
        norm = lambda x: (x*x).sum(dim=-1).sqrt()
        clamp = lambda x: x.clamp(-4., 4.)
        safe = lambda x: x.clamp(min=1e-6)
        cast = lambda x: x.to(dtype=features.dtype)
    if features.shape[-1] != 52:
        raise ValueError('Geometry expects flattened original 52 features')
    a = features.reshape(*features.shape[:-1], 13, 4)
    xy, valid = a[..., :2], a[..., 3] > .5
    point = lambda j: xy[..., j, :]
    present = lambda j: valid[..., j]
    shoulders = (point(5) + point(6)) * .5
    hips = (point(11) + point(12)) * .5
    sm, hm = present(5) & present(6), present(11) & present(12)
    values, masks = [], []
    def scalar(value, mask):
        values.append(clamp(value) * cast(mask)); masks.append(cast(mask))
    def vector(value, mask, unit=False):
        if unit:
            length = norm(value)
            mask = mask & (length > 1e-6)
            value = value / safe(length)[..., None]
        scalar(value[..., 0], mask); scalar(value[..., 1], mask)
    vector(shoulders - hips, sm & hm, unit=True)
    vector(point(0) - shoulders, present(0) & sm)
    vector(point(0) - hips, present(0) & hm)
    vector(point(6) - point(5), sm, unit=True)
    for shoulder, elbow, wrist in [(5,7,9), (6,8,10)]:
        u, v = point(shoulder)-point(elbow), point(wrist)-point(elbow)
        denominator = norm(u)*norm(v)
        mask = present(shoulder) & present(elbow) & present(wrist) & (denominator > 1e-6)
        cosine = (u*v).sum(axis=-1) if is_numpy else (u*v).sum(dim=-1)
        scalar(cosine/safe(denominator), mask)
    vector(point(9)-point(5), present(9)&present(5))
    vector(point(10)-point(6), present(10)&present(6))
    vector(point(12)-point(11), hm, unit=True)
    vector((point(5)-point(11))-(point(6)-point(12)), sm & hm)
    for elbow, shoulder in [(7,5),(8,6)]:
        scalar(point(elbow)[...,1]-point(shoulder)[...,1], present(elbow)&present(shoulder))
    for wrist in [9,10]:
        scalar(point(wrist)[...,1]-point(0)[...,1], present(wrist)&present(0))
    return cat([features, stack(values), stack(masks)])


def validate_contract(contract):
    from shared.dataset_pipeline import COCO13
    version = contract.get('version')
    expected = 52 if version == FEATURE_VERSION else GEOMETRY_DIM if version == GEOMETRY_VERSION else None
    if (expected is None or contract.get('flatten_dim') != expected
            or contract.get('shape_per_frame') != [13,4]
            or contract.get('channels') != ['x_norm','y_norm','confidence','valid_mask']
            or contract.get('joints') != COCO13):
        raise ValueError('Model feature contract incompatible with supported causal features')
    if version == GEOMETRY_VERSION and (contract.get('geometry_names') != GEOMETRY_NAMES
                                      or contract.get('geometry_layout') != 'base52,values22,masks22'):
        raise ValueError('Geometry feature ordering mismatch')
    return expected
