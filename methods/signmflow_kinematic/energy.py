"""Matched kinematic metrics and bounded density weighting for correction learning."""

import torch
from torch import nn

from . import geometry as G
from .kinematics import GROUPS, axis_jacobian, tangent_pushforward


def group_weights(weights, *, dtype=torch.float32):
    unknown = set(weights) - set(GROUPS)
    if unknown or any(not 0 <= value < float("inf") for value in weights.values()):
        raise ValueError(f"Invalid observable group weights, unexpected keys {unknown}")
    return torch.cat([torch.full((s.stop - s.start,), weights.get(name, 0.) / (s.stop - s.start),
                                 dtype=dtype) for name, s in GROUPS.items()])


class KinematicEnergy(nn.Module):
    def __init__(self, fk, variant, weights, base=0.1, strength=1., jacobian_chunk=8):
        super().__init__()
        if variant not in {"fk_full", "fk_diagonal", "fk_joint_block"}:
            raise ValueError(variant)
        if not (0 < base < float("inf") and 0 <= strength < float("inf")) or jacobian_chunk < 1:
            raise ValueError("Expected positive finite base, nonnegative finite strength and a positive chunk")
        self.fk = fk
        self.variant = variant
        self.base = base
        self.strength = strength
        self.jacobian_chunk = jacobian_chunk
        self.register_buffer("weights", group_weights(weights))

    def forward(self, q, dq):
        rotation = G.quat_to_matrix(q)
        velocity = G.body_velocity(q, dq)
        if self.variant == "fk_full":
            result = (tangent_pushforward(self.fk, rotation, velocity).square() * self.weights).sum(-1)
        else:
            jacobian = axis_jacobian(self.fk, rotation, self.jacobian_chunk)
            if self.variant == "fk_diagonal":
                slope_sq = (jacobian.square() * self.weights[:, None, None]).sum(-3)
                result = (slope_sq * velocity.square()).sum((-1, -2))
            else:
                per_joint = (jacobian * velocity[:, None]).sum(-1)
                result = (per_joint.square() * self.weights[:, None]).sum((-1, -2))
        return self.base * velocity.square().sum((-1, -2)) + self.strength * result


class DensityEnergy(nn.Module):
    """Bounded isotropic mass cost, calibrated once against train source/target pairs."""

    def __init__(self, measure, median_log_mass, bounds=(0.5, 2.), power=1.):
        super().__init__()
        lo, hi = bounds
        if not 0 < lo <= 1 <= hi < float("inf") or not 0 < power < float("inf"):
            raise ValueError("Invalid density bounds or power")
        self.measure = measure
        self.register_buffer("median_log_mass", median_log_mass)
        self.lo, self.hi, self.power = lo, hi, power

    def forward(self, q, dq):
        import math

        log_weight = self.power * (self.median_log_mass - self.measure.log_mass(q))
        weight = log_weight.clamp(math.log(self.lo), math.log(self.hi)).exp()
        return (weight * G.body_velocity(q, dq).square().sum(-1)).sum(-1)
