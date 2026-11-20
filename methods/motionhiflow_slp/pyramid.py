"""MotionHiFlow stage endpoints, linear resampling, and deterministic Euler inference."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from .data import valid_frames


def resample(
    values: Tensor, lengths: Tensor, target_lengths: Tensor, size: int | None = None
) -> Tensor:
    """Linearly resample each clip on its own temporal grid, without sampling padding."""
    if size is None:
        size = int(target_lengths.max().item())
    index = torch.arange(size, device=values.device, dtype=torch.float32)
    rate = lengths.float() / target_lengths
    position = ((index[None] + 0.5) * rate[:, None] - 0.5).clamp_min(0)
    position = torch.minimum(position, (lengths - 1)[:, None])
    left = position.floor().long()
    right = torch.minimum(left + 1, (lengths - 1)[:, None])
    mix = (position - left).to(values.dtype)
    flat = values.flatten(start_dim=2)
    left = flat.gather(1, left[..., None].expand(-1, -1, flat.shape[-1]))
    right = flat.gather(1, right[..., None].expand_as(left))
    out = torch.lerp(left, right, mix[..., None])
    out = out.masked_fill(~valid_frames(target_lengths, size)[..., None], 0)
    return out.reshape(values.shape[0], size, *values.shape[2:])


@dataclass(frozen=True)
class Stage:
    scale: float
    sigma_start: float
    sigma_end: float
    time_start: float
    time_width: float

    def time(self, progress: Tensor) -> Tensor:
        return self.time_start + self.time_width * progress


class PyramidFlow:
    def __init__(self, scales: list[float], gamma: float | None):
        if (not scales or scales[-1] != 1 or not all(math.isfinite(s) for s in scales)
                or any(not 0 < a < b for a, b in zip(scales, scales[1:]))
                or not 0 < scales[0] <= 1):
            raise ValueError("Flow scales must be strictly increasing in (0, 1] and end at 1")
        if gamma is not None and (not math.isfinite(gamma) or gamma <= 0):
            raise ValueError("flow.gamma must be positive or null")
        ends = [1 - scale for scale in scales]
        starts = [1.0, *ends[:-1]]
        if gamma is not None:
            # Optional start-sigma correction from the released MotionHiFlow scheduler.
            for i in range(1, len(starts)):
                clean = 1 - starts[i]
                starts[i] = 1 - clean / (math.sqrt(1 + 1 / gamma) * starts[i] + clean)
        widths = [start - end for start, end in zip(starts, ends)]
        total = sum(widths)
        time = 0.0
        self.stages = []
        for scale, start, end, width in zip(scales, starts, ends, widths):
            self.stages.append(Stage(scale, start, end, time, width / total))
            time += width / total
        self.probabilities = torch.tensor([stage.time_width for stage in self.stages])

    @staticmethod
    def lengths(lengths: Tensor, scale: float) -> Tensor:
        return (lengths * scale).floor().long().clamp_min(1)

    def start_state(
        self, i: int, previous_end: Tensor, previous_noise: Tensor, previous_lengths: Tensor,
        noise: Tensor, lengths: Tensor
    ) -> Tensor:
        """Recover coarse clean data, upsample it, and reapply pyramid noise."""
        coarse_sigma = self.stages[i - 1].sigma_end
        sigma = self.stages[i].sigma_start
        clean = (previous_end - coarse_sigma * previous_noise) / (1 - coarse_sigma)
        clean = resample(clean, previous_lengths, lengths, size=noise.shape[1])
        return (1 - sigma) * clean + sigma * noise

    def training_pair(
        self, clean: Tensor, noise: Tensor, lengths: Tensor, i: int, progress: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        stage = self.stages[i]
        stage_lengths = self.lengths(lengths, stage.scale)
        stage_noise = resample(noise, lengths, stage_lengths)
        stage_clean = resample(clean, lengths, stage_lengths, size=stage_noise.shape[1])
        end = (1 - stage.sigma_end) * stage_clean + stage.sigma_end * stage_noise
        start = stage_noise
        if i > 0:
            coarse = self.stages[i - 1]
            coarse_lengths = self.lengths(lengths, coarse.scale)
            coarse_noise = resample(noise, lengths, coarse_lengths)
            coarse_clean = resample(clean, lengths, coarse_lengths, size=coarse_noise.shape[1])
            coarse_end = (1 - coarse.sigma_end) * coarse_clean + coarse.sigma_end * coarse_noise
            start = self.start_state(i, coarse_end, coarse_noise, coarse_lengths,
                                     stage_noise, stage_lengths)
        return (
            torch.lerp(start, end, progress[:, None, None, None]),
            end - start,
            stage.time(progress),
            stage_lengths,
        )

    def step_counts(self, steps: int) -> list[int]:
        """Use exactly the requested NFE with at least one evaluation per stage."""
        if steps < len(self.stages):
            raise ValueError("sampler.steps must be at least the number of scales")
        remaining = steps - len(self.stages)
        quotas = [remaining * stage.time_width for stage in self.stages]
        counts = [1 + int(quota) for quota in quotas]
        order = sorted(range(len(quotas)), key=lambda i: quotas[i] % 1, reverse=True)
        for i in order[:steps - sum(counts)]:
            counts[i] += 1
        return counts

    def sample(
        self, velocity: Callable[[Tensor, Tensor, float, Tensor], Tensor],
        noise: Tensor, lengths: Tensor, steps: int,
    ) -> Tensor:
        counts = self.step_counts(steps)
        current = previous_noise = previous_lengths = None
        for i, (stage, count) in enumerate(zip(self.stages, counts)):
            stage_lengths = self.lengths(lengths, stage.scale)
            stage_noise = resample(noise, lengths, stage_lengths)
            if i == 0:
                current = stage_noise
            else:
                current = self.start_state(i, current, previous_noise, previous_lengths,
                                           stage_noise, stage_lengths)
            mask = valid_frames(stage_lengths, current.shape[1])[..., None, None]
            for step in range(count):
                time = stage.time(noise.new_full((len(noise),), step / count))
                update = velocity(current, time, stage.scale, stage_lengths).float()
                current = (current + update / count).masked_fill(~mask, 0)
            previous_noise, previous_lengths = stage_noise, stage_lengths
        return current
