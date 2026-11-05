"""Frame-aligned plan conditioning, with TRAINING-ONLY optional duration rescaling."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .plans import PlanBatch
from .tensors import valid_mask


@dataclass
class TemporalCondition:
    features: Tensor             # [B,L,C+2]: content, phase, log effective duration
    lengths: Tensor
    subcell_units: Tensor        # diagnostics for exposure rescaling


def _step_integral(x: Tensor, half_width: Tensor) -> Tensor:
    """Integral from -infinity of a linearly smoothed Heaviside step."""
    w = half_width.clamp_min(1e-12)
    smooth = torch.where(x <= -w, torch.zeros_like(x),
                         torch.where(x >= w, x, (x + w).square() / (4 * w)))
    return torch.where(half_width > 0, smooth, x.clamp_min(0))


def interval_weights(durations: Tensor, canvas_length: int, *, rescale: bool = False,
                     overlap: float = 2.0) -> tuple[Tensor, Tensor, Tensor]:
    """Cell-integrated memberships for one plan, including fractional intervals.

    Boundaries share ramp widths <= half either neighboring duration; ramps never
    cross. Memberships integrate to one at every cell, even if units < one cell.
    This blends ADJACENT units of one plan, never independent sampled plans.
    """
    if durations.ndim != 1 or durations.dtype != torch.long or len(durations) == 0:
        raise ValueError("expected positive integer duration vector")
    if (bool((durations <= 0).any()) or canvas_length < 1 or
            not math.isfinite(overlap) or overlap < 0):
        raise ValueError("invalid duration, canvas, or overlap")
    total = int(durations.sum())
    if not rescale and total != canvas_length:
        raise ValueError("reference/canvas length is forbidden without explicit rescale=True")
    d = durations.to(torch.float64)
    if rescale:
        d = d * (canvas_length / total)
    boundaries = torch.cat((d.new_zeros(1), d.cumsum(0)))
    half_widths = d.new_zeros(len(d) + 1)
    if len(d) > 1:
        half_widths[1:-1] = torch.minimum(d[:-1], d[1:]).div(2).clamp_max(overlap)
    left = torch.arange(canvas_length, device=d.device, dtype=d.dtype)[:, None]
    right = left + 1
    integrated_steps = (_step_integral(right - boundaries, half_widths) -
                        _step_integral(left - boundaries, half_widths))
    weights = integrated_steps[:, :-1] - integrated_steps[:, 1:]
    if bool((weights < -1e-8).any()) or bool((weights.sum(-1) <= 0).any()):
        raise RuntimeError("invalid temporal membership construction")
    weights = weights.clamp_min(0)
    weights = weights / weights.sum(-1, keepdim=True)
    lo = torch.maximum(left, boundaries[:-1])
    hi = torch.minimum(right, boundaries[1:])
    midpoint = torch.where(hi > lo, (hi + lo) / 2, (left + right) / 2)
    phases = ((midpoint - boundaries[:-1]) / d).clamp(0, 1)
    return weights, phases, d


def temporal_condition(plan: PlanBatch, *, canvas_lengths: Tensor | None = None,
                       rescale: bool = False, overlap: float = 2.0) -> TemporalCondition:
    plan.validate()
    predicted_lengths = plan.latent_lengths
    if canvas_lengths is None:
        if rescale:
            raise ValueError("exposure rescaling requires an explicit training canvas")
        canvas_lengths = predicted_lengths
    if canvas_lengths.device != plan.contents.device or canvas_lengths.shape != plan.lengths.shape:
        raise ValueError("canvas lengths must match plan batch and device")
    valid_mask(canvas_lengths)  # validate nonempty integer lengths
    extent = int(canvas_lengths.max())
    features = plan.contents.new_zeros(len(plan.lengths), extent, plan.contents.shape[-1] + 2)
    subcell = torch.zeros_like(plan.lengths)
    for b, k in enumerate(plan.lengths.tolist()):
        length = int(canvas_lengths[b])
        w, phases, effective = interval_weights(plan.durations[b, :k], length,
                                                rescale=rescale, overlap=overlap)
        w = w.to(plan.contents)
        phases = phases.to(plan.contents)
        features[b, :length, :-2] = w @ plan.contents[b, :k]
        features[b, :length, -2] = (w * phases).sum(-1)
        features[b, :length, -1] = (w * effective.log().to(w)).sum(-1)
        subcell[b] = (effective < 1).sum()
    return TemporalCondition(features, canvas_lengths, subcell)
