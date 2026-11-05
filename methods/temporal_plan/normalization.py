"""Streaming train-only statistics with checkpointed fitted state."""
from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from .tensors import check_sequence, valid_mask


class FeatureNormalizer(nn.Module):
    def __init__(self, width: int, min_std: float = 1e-4):
        super().__init__()
        if width <= 0 or min_std <= 0:
            raise ValueError("width and min_std must be positive")
        self.min_std = min_std
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("std", torch.ones(width))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def fit(self, batches: Iterable[tuple[Tensor, Tensor]]) -> None:
        """Population standard deviation; padding and validation data are excluded."""
        count = 0
        mean = torch.zeros_like(self.mean, dtype=torch.float64)
        m2 = torch.zeros_like(mean)
        for x, lengths in batches:
            check_sequence(x, lengths, self.mean.numel())
            values = x[valid_mask(lengths, x.shape[1])].reshape(-1, self.mean.numel()).to(mean)
            n = len(values)
            other_mean = values.mean(0)
            other_m2 = (values - other_mean).square().sum(0)
            delta = other_mean - mean
            combined = count + n
            m2 += other_m2 + delta.square() * (count * n / combined)
            mean += delta * (n / combined)
            count = combined
        if count == 0:
            raise ValueError("cannot fit statistics without training frames")
        self.mean.copy_(mean)
        self.std.copy_((m2 / count).clamp_min(0).sqrt().clamp_min(self.min_std))
        self.count.fill_(count)

    def _check_fitted(self) -> None:
        if int(self.count) == 0:
            raise RuntimeError("normalization must be fitted on training data first")

    def forward(self, x: Tensor) -> Tensor:
        self._check_fitted()
        if x.shape[-1] != self.mean.numel():
            raise ValueError("feature width does not match fitted statistics")
        return (x - self.mean) / self.std

    def inverse(self, x: Tensor) -> Tensor:
        self._check_fitted()
        if x.shape[-1] != self.mean.numel():
            raise ValueError("feature width does not match fitted statistics")
        return x * self.std + self.mean
