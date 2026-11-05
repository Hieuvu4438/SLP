"""Joint clean-latent predictor: temporal, cross-part and full-text attention."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .tensors import check_sequence, valid_mask, zero_padding


def sinusoidal(values: Tensor, width: int, max_period: float = 10000) -> Tensor:
    half = width // 2
    frequencies = torch.exp(-math.log(max_period) *
                             torch.arange(half, device=values.device, dtype=values.dtype) /
                             max(1, half - 1))
    phase = values[..., None] * frequencies
    result = torch.cat((phase.sin(), phase.cos()), -1)
    if result.shape[-1] < width:
        result = torch.cat((result, torch.zeros_like(values[..., None])), -1)
    return result


class JointBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(hidden)
        self.part_norm = nn.LayerNorm(hidden)
        self.text_norm = nn.LayerNorm(hidden)
        self.ff_norm = nn.LayerNorm(hidden)
        self.time_modulation = nn.Linear(hidden, 2 * hidden)
        self.temporal = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.parts = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.text = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * hidden, hidden))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, time: Tensor, memory: Tensor, text_valid: Tensor,
                lengths: Tensor) -> Tensor:
        b, length, parts, hidden = x.shape
        mask = valid_mask(lengths, length)
        scale, shift = self.time_modulation(time).chunk(2, -1)
        h = self.temporal_norm(x) * (1 + scale[:, None, None]) + shift[:, None, None]
        h = h.transpose(1, 2).reshape(b * parts, length, hidden)
        temporal_mask = (~mask)[:, None].expand(-1, parts, -1).reshape(b * parts, length)
        h = self.temporal(h, h, h, key_padding_mask=temporal_mask, need_weights=False)[0]
        h = h.reshape(b, parts, length, hidden).transpose(1, 2)
        x = zero_padding(x + self.dropout(h), lengths)
        h = self.part_norm(x).reshape(b * length, parts, hidden)
        h = self.parts(h, h, h, need_weights=False)[0].reshape_as(x)
        x = zero_padding(x + self.dropout(h), lengths)
        h = self.text_norm(x).reshape(b, length * parts, hidden)
        h = self.text(h, memory, memory, key_padding_mask=~text_valid, need_weights=False)[0]
        x = zero_padding(x + self.dropout(h.reshape_as(x)), lengths)
        return zero_padding(x + self.dropout(self.ff(self.ff_norm(x))), lengths)


class JointDenoiser(nn.Module):
    def __init__(self, latent_width: int = 64, content_width: int = 128,
                 text_width: int = 512, hidden: int = 512, heads: int = 8,
                 layers: int = 8, dropout: float = 0.1):
        super().__init__()
        if min(latent_width, content_width, text_width, hidden, heads, layers) < 1:
            raise ValueError("denoiser dimensions must be positive")
        if hidden % heads:
            raise ValueError("hidden width must be divisible by attention heads")
        self.latent_width, self.content_width, self.hidden = latent_width, content_width, hidden
        self.input = nn.Linear(latent_width, hidden)
        self.part_embedding = nn.Parameter(torch.randn(4, hidden) * .02)
        self.plan_projection = nn.Linear(content_width + 2, hidden)
        self.text_projection = nn.Linear(text_width, hidden)
        self.time = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList(JointBlock(hidden, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, latent_width)

    def forward(self, noisy: Tensor, noise_times: Tensor, memory: Tensor,
                text_valid: Tensor, condition: Tensor, lengths: Tensor) -> Tensor:
        check_sequence(noisy, lengths, self.latent_width)
        check_sequence(condition, lengths, self.content_width + 2)
        b, length, parts, _ = noisy.shape
        if parts != 4 or condition.shape[:2] != (b, length):
            raise ValueError("expected joint four-part latent and aligned plan canvas")
        if noise_times.shape != (b,) or bool(((noise_times < 0) | (noise_times > 1)).any()):
            raise ValueError("noise times must be one normalized time per example")
        if not bool(torch.isfinite(noise_times).all()):
            raise ValueError("non-finite noise time")
        if memory.ndim != 3 or memory.shape[:2] != text_valid.shape or len(memory) != b:
            raise ValueError("invalid text memory")
        if text_valid.dtype != torch.bool or bool((~text_valid.any(-1)).any()):
            raise ValueError("each text must contain a valid token")
        if not bool(torch.isfinite(memory[text_valid]).all()):
            raise ValueError("non-finite valid text features")
        noisy, condition = zero_padding(noisy, lengths), zero_padding(condition, lengths)
        memory = torch.where(text_valid[..., None], memory, torch.zeros_like(memory))
        memory = self.text_projection(memory)
        position = sinusoidal(torch.arange(length, device=noisy.device, dtype=noisy.dtype),
                               self.hidden)
        x = (self.input(noisy) + self.part_embedding[None, None] +
             position[None, :, None] + self.plan_projection(condition)[:, :, None])
        x = zero_padding(x, lengths)
        time = self.time(sinusoidal(noise_times.to(noisy) * 1000, self.hidden))
        for block in self.blocks:
            x = block(x, time, memory, text_valid, lengths)
        return zero_padding(self.output(self.norm(x)), lengths)
