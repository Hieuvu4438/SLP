"""Per-clip physical objectives; metres normalized by neutral shoulder width."""

import torch
import torch.nn.functional as F

from cprflow.geometry.joint_maps import BODY_NAMES
from cprflow.geometry.rotations import geodesic_distance, rotation_6d_to_matrix

GROUPS = ("body", "left", "right", "left_local", "right_local")
WEIGHTS = (1.0, 1.0, 1.0, 2.0, 2.0)


def shoulder_width(geo):
    names = geo["names"]
    joints = geo["joints"]
    return (
        joints[..., names.index("right_shoulder"), :] - joints[..., names.index("left_shoulder"), :]
    ).norm(dim=-1)


def fk_loss(pred, target):
    scale = shoulder_width(target).clamp_min(0.05)
    return sum(
        w
        * F.huber_loss(
            pred[g] / scale[:, None, None], target[g] / scale[:, None, None], reduction="mean"
        )
        for w, g in zip(WEIGHTS, GROUPS)
    ) / sum(WEIGHTS)


def velocity_loss(pred, target, expr_pred, expr_target, times, scales):
    if len(times) < 2:
        return expr_pred.sum() * 0
    dt = times[1:] - times[:-1]
    if (dt <= 0).any():
        raise ValueError("Velocity requires verified increasing timestamps")
    terms = []
    for g in GROUPS:
        pv = (pred[g][1:] - pred[g][:-1]) / dt[:, None, None]
        tv = (target[g][1:] - target[g][:-1]) / dt[:, None, None]
        terms.append(((pv - tv) / max(float(scales[g]), 1e-3)).square().mean())
    pv = (expr_pred[1:] - expr_pred[:-1]) / dt[:, None]
    tv = (expr_target[1:] - expr_target[:-1]) / dt[:, None]
    terms.append(((pv - tv) / max(float(scales["expression"]), 1e-3)).square().mean())
    return torch.stack(terms).mean()


def rotation_loss(pred, target):
    pr = rotation_6d_to_matrix(pred[:, :246].reshape(-1, 41, 6))
    tr = rotation_6d_to_matrix(target[:, :246].reshape(-1, 41, 6))
    angles = geodesic_distance(pr, tr) / torch.pi
    return torch.stack([v.mean() for v in angles.split((10, 15, 15, 1), -1)]).mean()


def activities(x, geo, times, motion_normalizer, scales):
    """Activity samples live at adjacent-frame midpoint timestamps."""
    dt = (times[1:] - times[:-1]).clamp_min(1e-8)
    outputs = []
    for side in ("left", "right"):
        local = geo[side + "_local"]
        fingers = (local[1:] - local[:-1]) / dt[:, None, None]
        wrist = (geo[side][1:, 0] - geo[side][:-1, 0]) / dt[:, None]
        outputs.append(torch.cat((fingers.flatten(1), wrist), -1).norm(dim=-1))
    rotations = rotation_6d_to_matrix(x[:, :246].reshape(-1, 41, 6))
    head = BODY_NAMES.index("head")
    outputs.append(geodesic_distance(rotations[1:, head], rotations[:-1, head]) / dt)
    expression = motion_normalizer(x)[:, 246:]
    expr_vel = (expression[1:] - expression[:-1]) / dt[:, None]
    jaw_vel = geodesic_distance(rotations[1:, 40], rotations[:-1, 40]) / dt
    outputs.append(
        torch.linalg.vector_norm(torch.cat((expr_vel, jaw_vel[:, None]), -1), dim=-1) / (51**0.5)
    )
    return torch.stack(
        [
            a / max(float(scales[g]), 1e-3)
            for a, g in zip(
                outputs, ("activity_left", "activity_right", "activity_head", "activity_face")
            )
        ],
        -1,
    )
