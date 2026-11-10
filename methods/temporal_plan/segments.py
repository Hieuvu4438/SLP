"""Frozen motion-derived segment scores and exact distributions on partitions."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def phase_samples(sequence: Tensor, steps: int = 8) -> Tensor:
    """Sample a SINGLE unpadded [time,features] interval with aligned endpoints."""
    if sequence.ndim != 2 or len(sequence) < 1 or steps < 1:
        raise ValueError("phase sampling requires a non-empty interval")
    return F.interpolate(sequence.T[None], size=steps, mode="linear",
                         align_corners=True)[0].T


@dataclass(frozen=True)
class TeacherConfig:
    max_duration: int = 32
    phase_steps: int = 8
    radii: tuple[int, ...] = (1, 2, 4)
    alpha: float = 1.0
    beta: float = 0.5
    penalty: float = 0.3
    temperature: float = 0.5

    def __post_init__(self):
        if self.max_duration < 1 or self.phase_steps < 1 or not self.radii:
            raise ValueError("durations, phase steps and radii must be positive")
        if any(r < 1 for r in self.radii):
            raise ValueError("radii must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if any(not math.isfinite(v) or v < 0 for v in (self.alpha, self.beta, self.penalty)):
            raise ValueError("score coefficients must be finite and nonnegative")


@torch.no_grad()
def boundary_evidence(features: Tensor, radii: tuple[int, ...] = (1, 2, 4)) -> Tensor:
    """RMS mean-change per part/scale, then robust within-sequence calibration.

    Input contains TRAIN-standardized codec means [L,4,D]. Cut j separates
    features[:j] and features[j:]. Both endpoints have zero boundary reward.
    A constant signal has evidence .5 at interior cuts, not false certainty.
    This score is a heuristic, not calibrated linguistic boundary probability.
    """
    if features.ndim != 3 or features.shape[1] != 4 or len(features) < 1:
        raise ValueError("expected a non-empty [L,4,D] sequence")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("non-finite teacher input")
    length = len(features)
    evidence = features.new_zeros(length + 1)
    if length == 1:
        return evidence
    changes = []
    for radius in radii:
        values = []
        for j in range(1, length):
            left = features[max(0, j - radius):j].mean(0)
            right = features[j:min(length, j + radius)].mean(0)
            values.append((left - right).square().mean(-1).sqrt())  # four parts
        changes.append(torch.stack(values))
    raw = torch.stack(changes, -1).mean((-1, -2))
    median = raw.median()
    mad = (raw - median).abs().median().clamp_min(1e-4)
    evidence[1:length] = torch.sigmoid(((raw - median) / (1.4826 * mad)).clamp(-6, 6))
    return evidence


@torch.no_grad()
def segment_scores(features: Tensor, config: TeacherConfig) -> Tensor:
    """scores[j-1,d-1] for interval [j-d,j); invalid intervals are -inf.

    Distortion uses eight phase samples then aligned-endpoint interpolation back
    to the original interval. No temporal padding enters scores or pooling.
    """
    b = boundary_evidence(features, config.radii)
    flat = features.flatten(1)
    length = len(flat)
    table = flat.new_full((length, config.max_duration), -torch.inf)
    # Batch all starts at a given duration: only O(Dmax) interpolation launches,
    # rather than one per interval. The mathematical score is unchanged.
    for duration in range(1, min(config.max_duration, length) + 1):
        intervals = flat.unfold(0, duration, 1)  # [number_of_starts,features,duration]
        sampled = F.interpolate(intervals, size=config.phase_steps, mode="linear",
                                 align_corners=True)
        reconstructed = F.interpolate(sampled, size=duration, mode="linear",
                                       align_corners=True)
        distortion = (intervals-reconstructed).square().mean((1, 2))
        table[duration-1:, duration-1] = (
            -config.alpha*distortion + config.beta*b[duration:] - config.penalty
        )
    return table


class PartitionDistribution:
    """Exact distribution on all positive-duration covers, using log-space DP.

    No categorical independence assumption is made between consecutive cuts.
    The score table can require gradients: d logZ / d score gives segment
    marginals divided by temperature. Actual teacher targets are frozen.
    """
    def __init__(self, scores: Tensor, temperature: float = 0.5):
        if scores.ndim != 2 or min(scores.shape) < 1 or not scores.is_floating_point():
            raise ValueError("expected floating [L,Dmax] scores")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        length, width = scores.shape
        ends = torch.arange(1, length + 1, device=scores.device)[:, None]
        durations = torch.arange(1, width + 1, device=scores.device)[None]
        self.scores = scores.masked_fill(durations > ends, -torch.inf)
        if bool(torch.isnan(self.scores).any()) or bool(torch.isposinf(self.scores).any()):
            raise ValueError("scores may contain finite values or -inf, never NaN/+inf")
        self.temperature = temperature
        self.weights = self.scores / temperature
        self.length, self.max_duration = scores.shape
        forward = [scores.new_zeros(())]
        for end in range(1, self.length + 1):
            width = min(self.max_duration, end)
            terms = torch.stack(forward[end-width:end]).flip(0) + self.weights[end-1, :width]
            forward.append(torch.logsumexp(terms, 0))
        self.forward = torch.stack(forward)
        if not bool(torch.isfinite(self.forward[-1])):
            raise ValueError("score support has no complete partition")

    @property
    def log_normalizer(self) -> Tensor:
        return self.forward[-1]

    def log_prob(self, durations: list[int] | tuple[int, ...]) -> Tensor:
        if not durations or any(isinstance(d, bool) or not isinstance(d, int)
                                or not 1 <= d <= self.max_duration for d in durations):
            raise ValueError("durations must be positive integers in teacher support")
        if sum(durations) != self.length:
            raise ValueError("a partition must cover the entire sequence")
        end = 0
        terms = []
        for d in durations:
            end += d
            terms.append(self.weights[end-1, d-1])
        return torch.stack(terms).sum() - self.log_normalizer

    @torch.no_grad()
    def sample(self, generator: torch.Generator | None = None) -> list[int]:
        end, reverse = self.length, []
        while end:
            width = min(self.max_duration, end)
            logits = self.forward[end-width:end].flip(0) + self.weights[end-1, :width]
            d = int(torch.multinomial(logits.softmax(0), 1, generator=generator)) + 1
            reverse.append(d)
            end -= d
        return list(reversed(reverse))

    @torch.no_grad()
    def mode(self) -> list[int]:
        values = [self.scores.new_zeros(())]
        predecessors = []
        for end in range(1, self.length + 1):
            width = min(self.max_duration, end)
            terms = torch.stack(values[end-width:end]).flip(0) + self.weights[end-1, :width]
            values.append(terms.max())
            predecessors.append(terms.argmax()+1)
        predecessors = [0] + torch.stack(predecessors).cpu().tolist()
        end, reverse = self.length, []
        while end:
            d = predecessors[end]
            reverse.append(d)
            end -= d
        return list(reversed(reverse))

    def segment_marginals(self) -> Tensor:
        backward = [self.scores.new_full((), -torch.inf) for _ in range(self.length + 1)]
        backward[self.length] = self.scores.new_zeros(())
        for start in range(self.length - 1, -1, -1):
            terms = [self.weights[start+d-1, d-1] + backward[start+d]
                     for d in range(1, min(self.max_duration, self.length-start) + 1)]
            backward[start] = torch.logsumexp(torch.stack(terms), 0)
        rows = []
        for end in range(1, self.length + 1):
            row = []
            for d in range(1, self.max_duration + 1):
                if d > end:
                    row.append(self.scores.new_zeros(()))
                else:
                    row.append((self.forward[end-d] + self.weights[end-1, d-1] +
                                backward[end] - self.log_normalizer).exp())
            rows.append(torch.stack(row))
        return torch.stack(rows)


class SegmentCodec(nn.Module):
    def __init__(self, feature_width: int = 256, content_width: int = 128,
                 hidden: int = 512, phase_steps: int = 8):
        super().__init__()
        self.feature_width, self.content_width, self.phase_steps = (
            feature_width, content_width, phase_steps
        )
        width = feature_width * phase_steps
        self.encoder = nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
                                     nn.Linear(hidden, content_width))
        self.decoder = nn.Sequential(nn.Linear(content_width, hidden), nn.GELU(),
                                     nn.Linear(hidden, width))

    def encode(self, phases: Tensor) -> Tensor:
        if phases.shape[-2:] != (self.phase_steps, self.feature_width):
            raise ValueError("phase content shape mismatch")
        return self.encoder(phases.flatten(-2))

    def decode(self, codes: Tensor) -> Tensor:
        return self.decoder(codes).reshape(*codes.shape[:-1],
                                           self.phase_steps, self.feature_width)

    def forward(self, phases: Tensor) -> tuple[Tensor, Tensor]:
        codes = self.encode(phases)
        return self.decode(codes), codes
