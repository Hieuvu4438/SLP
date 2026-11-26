"""Flow targets and sampling in complementary subspaces."""

import hashlib
from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import Tensor

from .data import POSE_DIM
from .subspace import Projector, Representation


class Stage(NamedTuple):
    start: Tensor
    target: Tensor
    rank: Tensor
    constrain: Callable[[Tensor], Tensor]


def identity(value: Tensor) -> Tensor:
    return value


def training_stages(state: Tensor, noise: Tensor, projector: Projector,
                    scales: Tensor, lengths: Tensor, unconstrained: bool) -> tuple[Stage, ...]:
    coarse = projector.coarse(state) / scales[0]
    first = Stage(projector.coarse(noise), coarse, projector.coarse_rank, projector.coarse)
    if projector.mode == "single":
        return (first,)
    if unconstrained:
        detail = Stage(noise + coarse * (scales[0] / scales[1]), state / scales[1],
                       lengths * POSE_DIM, identity)
    else:
        detail = Stage(projector.detail(noise), projector.detail(state) / scales[1],
                       projector.detail_rank, projector.detail)
    return first, detail


def rank_loss(error: Tensor, rank: Tensor) -> Tensor:
    valid = rank > 0
    per_clip = error.square().sum((1, 2)) / rank.clamp_min(1)
    return (per_clip * valid).sum() / valid.sum().clamp_min(1)


def sample_noise(keys: list[str], lengths: list[int], seed: int, device) -> Tensor:
    rows = []
    generator = torch.Generator(device=device)
    for key, length in zip(keys, lengths, strict=True):
        digest = hashlib.blake2b(f"{seed}/{key}".encode(), digest_size=8).digest()
        generator.manual_seed(int.from_bytes(digest, "little"))
        rows.append(torch.randn(length, POSE_DIM, generator=generator, device=device))
    return torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)


def integrate(start, constrain, velocity, steps):
    value = start
    for step in range(steps):
        clock = value.new_full((len(value),), step / steps)
        value = value + constrain(velocity(value, clock)) / steps
    return constrain(value)


def sample(velocity, noise: Tensor, projector: Projector, representation: Representation,
           steps: tuple[int, int], unconstrained: bool, *, coarse_override: Tensor | None = None,
           residual_override: Tensor | None = None, coarse_only: bool = False) -> Tensor:
    """An override supplies a projected whitened component for an oracle diagnostic."""
    coarse_scale, detail_scale = representation.scales
    if coarse_override is None:
        coarse_steps = sum(steps) if projector.mode == "single" else steps[0]
        coarse = integrate(projector.coarse(noise), projector.coarse,
                           lambda x, u: velocity(x, u, 0, None), coarse_steps)
    else:
        coarse = projector.coarse(coarse_override) / coarse_scale
    coarse_state = coarse * coarse_scale

    if coarse_only:
        return coarse_state
    if residual_override is not None:
        return coarse_state + projector.detail(residual_override)
    if projector.mode == "single" or (not unconstrained and not bool(projector.detail_rank.any())):
        return coarse_state

    def detail_velocity(state, time):
        return velocity(state, time, 1, coarse)

    if unconstrained:
        clean = integrate(noise + coarse_state / detail_scale, identity, detail_velocity, steps[1])
        return clean * detail_scale
    detail = integrate(projector.detail(noise), projector.detail, detail_velocity, steps[1])
    return coarse_state + detail * detail_scale
