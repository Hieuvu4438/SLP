"""Training objectives only. Final benchmark metrics remain in slp.evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .codec import MotionPosterior
from .geometry import PART_SLICES, learning256_to_soke133, rotation6d_to_matrix
from .geometry import rotation_distance
from .tensors import check_sequence, masked_mean, valid_mask, zero_padding


@dataclass(frozen=True)
class LossWeights:
    fk: float = 1.0
    velocity: float = 0.1
    kl: float = 1e-4
    expression: float = 1.0

    def __post_init__(self):
        if any(not torch.isfinite(torch.tensor(v)) or v < 0
               for v in (self.fk, self.velocity, self.kl, self.expression)):
            raise ValueError("loss weights must be finite and nonnegative")


def motion_reconstruction(predicted: Tensor, target: Tensor, lengths: Tensor, *,
                          weights: LossWeights = LossWeights(),
                          fk: nn.Module | None = None,
                          rotation_offsets: Tensor | None = None) -> dict[str, Tensor]:
    """Inputs are INVERSE-normalized learning features [B,T,256].

    Rotations are scored geodesically per part; velocity uses projected joint
    positions when FK is active. FK cannot silently disappear with a positive
    coefficient. When disabled explicitly (ablation/tests), feature velocity is
    used and reported separately.
    """
    check_sequence(predicted, lengths, 256)
    check_sequence(target, lengths, 256)
    if predicted.shape != target.shape:
        raise ValueError("paired reconstruction tensors must have identical shapes")
    if weights.fk > 0 and fk is None:
        raise ValueError("positive FK weight requires an explicit kinematics backend")
    mask = valid_mask(lengths, target.shape[1])
    predicted, target = zero_padding(predicted, lengths), zero_padding(target, lengths)
    # Select valid frames BEFORE nonlinear geometry to prevent padded NaN leakage.
    pred_valid, target_valid = predicted[mask], target[mask]
    rotations = []
    for part in PART_SLICES:
        stop = min(part.stop, 246)
        a = pred_valid[:, part.start:stop].reshape(len(pred_valid), -1, 6)
        b = target_valid[:, part.start:stop].reshape(len(target_valid), -1, 6)
        rotations.append(rotation_distance(rotation6d_to_matrix(a),
                                            rotation6d_to_matrix(b)).mean())
    rotation_loss = torch.stack(rotations).sum()
    expression_loss = F.smooth_l1_loss(pred_valid[:, 246:], target_valid[:, 246:])
    fk_loss = rotation_loss * 0
    if fk is not None:
        if rotation_offsets is None:
            rotation_offsets = getattr(fk, "rotation_offsets", None)
        pred_joints = fk(learning256_to_soke133(pred_valid, rotation_offsets))
        with torch.no_grad():
            target_joints = fk(learning256_to_soke133(target_valid, rotation_offsets))
        fk_loss = torch.linalg.vector_norm(pred_joints - target_joints, dim=-1).mean()
        pred_motion = predicted.new_zeros(*predicted.shape[:2], *pred_joints.shape[1:])
        target_motion = target.new_zeros(*target.shape[:2], *target_joints.shape[1:])
        pred_motion[mask] = pred_joints
        target_motion[mask] = target_joints
        velocity_space = "joint_velocity"
    else:
        pred_motion, target_motion = predicted, target
        velocity_space = "feature_velocity"
    pair_mask = mask[:, 1:] & mask[:, :-1]
    delta_pred = pred_motion[:, 1:] - pred_motion[:, :-1]
    delta_target = target_motion[:, 1:] - target_motion[:, :-1]
    velocity_loss = masked_mean((delta_pred - delta_target).square(), pair_mask)
    total = (rotation_loss + weights.expression * expression_loss +
             weights.fk * fk_loss + weights.velocity * velocity_loss)
    return dict(total=total, rotation=rotation_loss, expression=expression_loss,
                fk=fk_loss, velocity=velocity_loss, **{velocity_space: velocity_loss})


def posterior_kl(posterior: MotionPosterior) -> Tensor:
    value = 0.5 * (posterior.mean.square() + posterior.logvar.exp() - 1 - posterior.logvar)
    frame_mean = value.flatten(2).mean(-1)
    valid = valid_mask(posterior.lengths, value.shape[1])
    return (torch.where(valid, frame_mean, torch.zeros_like(frame_mean)).sum(-1) /
            posterior.lengths).mean()
