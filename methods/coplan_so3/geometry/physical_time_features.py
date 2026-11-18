"""Sequence features and physical-second derivatives, with train-only RMS scales."""

from __future__ import annotations

import torch
from torch import nn

from ..data.dataset import TrainMoments


def difference(x, times, mask):
    valid = mask[:, 1:] & mask[:, :-1]
    dt = times[:, 1:] - times[:, :-1]
    if ((dt <= 0) & valid).any():
        raise ValueError("Valid physical time intervals must be positive")
    dt = torch.where(valid, dt, torch.ones_like(dt))
    out = (x[:, 1:] - x[:, :-1]) / dt[..., None]
    return out * valid[..., None], valid, (times[:, 1:] + times[:, :-1]) / 2


def orders(x, times, mask):
    result = [(x * mask[..., None], mask)]
    for _ in range(2):
        x, mask, times = difference(x, times, mask)
        result.append((x, mask))
    return result


class SequenceFeatures(nn.Module):
    """All feature weights depend on the current state and frozen train scales.

    Groups preserve both global signing location and local handshape.
    RMS normalization is independent for positions, velocity and acceleration.
    """

    def __init__(self, fk, dimensions, weights=(1.0, 0.1, 0.01), group_weights=None):
        super().__init__()
        if len(weights) != 3 or any(w < 0 for w in weights):
            raise ValueError("Three nonnegative physical-order weights required")
        self.fk, self.weights = fk, tuple(weights)
        self.group_weights = group_weights or {k: 1.0 for k in dimensions}
        if set(self.group_weights) != set(dimensions) or min(self.group_weights.values()) < 0:
            raise ValueError("Nonnegative weights for each feature group required")
        self.stats = nn.ModuleDict(
            {f"{k}_{i}": TrainMoments(d) for k, d in dimensions.items() for i in range(3)}
        )

    def raw(self, r, e):
        p = self.fk(r, e)
        left, right = p["left"], p["right"]
        lw, rw = left[..., 0, :], right[..., 0, :]
        lr, rr = p["left_rotation"], p["right_rotation"]
        # Hand-local coordinates rotate with the wrist; global wrist features remain.
        ll = (
            lr.transpose(-1, -2)[..., None, :, :] @ (left[..., 1:, :] - lw[..., None, :])[..., None]
        )[..., 0]
        rh = (
            rr.transpose(-1, -2)[..., None, :, :]
            @ (right[..., 1:, :] - rw[..., None, :])[..., None]
        )[..., 0]
        return {
            "joints": p["body"].flatten(-2),
            "wrists": torch.cat((lw - p["torso"], rw - p["torso"]), -1),
            "handshape": torch.cat((ll.flatten(-2), rh.flatten(-2)), -1),
            "palms": torch.cat((lr[..., :, 1:].flatten(-2), rr[..., :, 1:].flatten(-2)), -1),
            "relative": torch.cat(
                (rw - lw, (right[..., 1:, :] - left[..., 1:, :]).flatten(-2)), -1
            ),
        }

    @torch.no_grad()
    def fit_batch(self, r, e, mask, times, *, split):
        for name, x in self.raw(r, e).items():
            for i, (value, valid) in enumerate(orders(x, times, mask)):
                self.stats[f"{name}_{i}"].update(value, valid, split=split)

    def finalize(self, *, split):
        for stat in self.stats.values():
            # Very short clips may provide no acceleration observations.
            if stat.count == 0:
                raise ValueError("Train statistics require at least one valid frame triple")
            stat.finalize(split=split)

    def forward(self, r, e, mask, times):
        result = {}
        for name, x in self.raw(r, e).items():
            for i, (value, valid) in enumerate(orders(x, times, mask)):
                stat = self.stats[f"{name}_{i}"]
                if not stat.ready:
                    raise RuntimeError("Fit and freeze feature RMS on train before KFM")
                rms = (stat.mean.square() + stat.m2 / stat.count).clamp_min(1e-8).sqrt().to(value)
                scale = (self.weights[i] * self.group_weights[name]) ** 0.5
                # Counts depend on lengths only and cannot see the hidden endpoint.
                denom = (valid.sum(1) * value.shape[-1]).clamp_min(1).sqrt()
                result[f"{name}_{i}"] = value / rms * scale / denom[:, None, None]
        return result
