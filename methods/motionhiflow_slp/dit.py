"""Text-motion DiT with separate modality blocks followed by shared blocks."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .codec import PARTS
from .data import LANGUAGES, valid_frames


class TextCondition(NamedTuple):
    tokens: Tensor
    mask: Tensor
    pooled: Tensor


def frequencies(size: int) -> Tensor:
    return torch.exp(-math.log(10_000) * torch.arange(size, dtype=torch.float32) / size)


def modulate(norm: nn.LayerNorm, x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return norm(x) * (1 + scale[:, None]) + shift[:, None]


class RotaryAttention(nn.Module):
    def __init__(self, heads: int, width: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.dropout = dropout
        self.register_buffer("omega", frequencies(self.head_width // 4), persistent=False)

    def angles(self, frames: int, text: int, stride: float) -> tuple[Tensor, Tensor]:
        time = torch.arange(frames, device=self.omega.device).repeat_interleave(PARTS) * stride
        part = torch.arange(PARTS, device=self.omega.device).repeat(frames)
        angles = torch.cat((time[:, None] * self.omega, part[:, None] * self.omega), dim=-1)
        angles = F.pad(angles, (0, 0, 0, text))
        return angles.cos(), angles.sin()

    def rotate(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        even, odd = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        return torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2)

    def forward(
        self, qkv: Tensor, mask: Tensor, angles: tuple[Tensor, Tensor]
    ) -> Tensor:
        batch, tokens = qkv.shape[:2]
        q, k, v = qkv.reshape(batch, tokens, 3, self.heads, self.head_width).unbind(dim=2)
        q = F.rms_norm(q.transpose(1, 2), (self.head_width,))
        k = F.rms_norm(k.transpose(1, 2), (self.head_width,))
        output = F.scaled_dot_product_attention(
            self.rotate(q, *angles),
            self.rotate(k, *angles),
            v.transpose(1, 2),
            attn_mask=mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        return output.transpose(1, 2).reshape(batch, tokens, -1)


class Stream(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.modulation = nn.Linear(width, 6 * width)
        self.attn_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.ff_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(approximate="tanh"),
            nn.Linear(4 * width, width), nn.Dropout(dropout),
        )

    def prepare(self, x: Tensor, conditioning: Tensor):
        shift, scale, *residual = self.modulation(F.silu(conditioning)).chunk(6, dim=-1)
        return self.qkv(modulate(self.attn_norm, x, shift, scale)), residual

    def finish(self, x: Tensor, attended: Tensor, residual: list[Tensor]) -> Tensor:
        attn_gate, shift, scale, ff_gate = residual
        x = x + attn_gate[:, None] * self.out(attended)
        return x + ff_gate[:, None] * self.ff(modulate(self.ff_norm, x, shift, scale))


class DualBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.motion = Stream(width, dropout)
        self.text = Stream(width, dropout)
        self.attention = RotaryAttention(heads, width, dropout)

    def forward(self, motion, text, condition, mask, angles):
        m_qkv, m_residual = self.motion.prepare(motion, condition)
        t_qkv, t_residual = self.text.prepare(text, condition)
        attended = self.attention(torch.cat((m_qkv, t_qkv), dim=1), mask, angles)
        m_out, t_out = attended.split((motion.shape[1], text.shape[1]), dim=1)
        return (self.motion.finish(motion, m_out, m_residual),
                self.text.finish(text, t_out, t_residual))


class SharedBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.modulation = nn.Linear(width, 3 * width)
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention = RotaryAttention(heads, width, dropout)
        self.ff = nn.Linear(width, 4 * width)
        self.out = nn.Linear(5 * width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, condition, mask, angles):
        shift, scale, gate = self.modulation(F.silu(condition)).chunk(3, dim=-1)
        features = modulate(self.norm, x, shift, scale)
        attention = self.attention(self.qkv(features), mask, angles)
        ff = F.gelu(self.ff(features), approximate="tanh")
        return x + gate[:, None] * self.dropout(self.out(torch.cat((attention, ff), dim=-1)))


class MotionDiT(nn.Module):
    def __init__(
        self, latent_dim: int, text_dim: int, width: int, heads: int, double_blocks: int,
        single_blocks: int, dropout: float, time_patch: int,
    ):
        super().__init__()
        if heads < 1 or width < 1 or width % (4 * heads):
            raise ValueError("DiT width must be divisible by 4 * heads for time/part RoPE")
        if min(time_patch, latent_dim) < 1 or min(double_blocks, single_blocks) < 0:
            raise ValueError("Invalid DiT patch size or number of layers")
        if not double_blocks + single_blocks:
            raise ValueError("DiT requires at least one attention block")
        self.time_patch = time_patch
        self.latent_dim = latent_dim
        self.motion_proj = nn.Linear(latent_dim * time_patch, width)
        self.text_proj = nn.Linear(text_dim, width)
        self.part_embed = nn.Parameter(torch.zeros(PARTS, width))
        self.null_text = nn.Parameter(torch.zeros(width))
        self.language = nn.Embedding(len(LANGUAGES), width)
        self.register_buffer("omega", frequencies(width // 2), persistent=False)
        self.time_proj = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.scale_proj = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.text_pool = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.duration = nn.Sequential(
            nn.LayerNorm(text_dim + width), nn.Linear(text_dim + width, width),
            nn.SiLU(), nn.Linear(width, 1),
        )
        self.dual = nn.ModuleList(DualBlock(width, heads, dropout) for _ in range(double_blocks))
        self.shared = nn.ModuleList(SharedBlock(width, heads, dropout) for _ in range(single_blocks))
        self.position = RotaryAttention(heads, width, 0.0)
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.final_mod = nn.Linear(width, 2 * width)
        self.output = nn.Linear(width, time_patch * latent_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def pool(tokens: Tensor, mask: Tensor) -> Tensor:
        return tokens.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True)

    def prepare_text(self, tokens: Tensor, mask: Tensor) -> TextCondition:
        projected = self.text_proj(tokens)
        return TextCondition(projected, mask, self.pool(projected, mask))

    def log_duration(self, tokens: Tensor, mask: Tensor, languages: Tensor) -> Tensor:
        pooled = self.pool(tokens.detach(), mask)
        return self.duration(torch.cat((pooled, self.language(languages)), dim=-1)).squeeze(-1)

    def time_embedding(self, value: Tensor) -> Tensor:
        angle = 1000 * value[:, None] * self.omega
        return torch.cat((angle.cos(), angle.sin()), dim=-1)

    def forward(
        self, x: Tensor, time: Tensor, scale: float, lengths: Tensor, text: TextCondition,
        languages: Tensor, drop_text: Tensor,
    ) -> Tensor:
        batch, frames, parts, dim = x.shape
        patch = self.time_patch
        frame_mask = valid_frames(lengths, frames)
        x = x.masked_fill(~frame_mask[..., None, None], 0)
        x = F.pad(x, (0, 0, 0, 0, 0, (-frames) % patch))
        x = x.reshape(batch, -1, patch, parts, dim).transpose(2, 3).flatten(3)
        patches = x.shape[1]
        motion = (self.motion_proj(x) + self.part_embed).flatten(1, 2)
        motion_mask = valid_frames((lengths + patch - 1) // patch, patches)
        motion_mask = motion_mask.repeat_interleave(parts, dim=1)

        tokens = torch.where(drop_text[:, None, None], self.null_text, text.tokens)
        pooled = torch.where(drop_text[:, None], self.null_text, text.pooled)
        null_mask = torch.arange(tokens.shape[1], device=x.device)[None] == 0
        text_mask = torch.where(drop_text[:, None], null_mask, text.mask)
        mask = torch.cat((motion_mask, text_mask), dim=1)
        condition = (
            self.time_proj(self.time_embedding(time))
            + self.scale_proj(self.time_embedding(torch.full_like(time, scale)))
            + self.text_pool(pooled)
            + self.language(languages)
        )
        angles = self.position.angles(patches, tokens.shape[1], patch / scale)
        for block in self.dual:
            motion, tokens = block(motion, tokens, condition, mask, angles)
        features = torch.cat((motion, tokens), dim=1)
        for block in self.shared:
            features = block(features, condition, mask, angles)
        shift, gain = self.final_mod(F.silu(condition)).chunk(2, dim=-1)
        predicted = self.output(modulate(self.final_norm, features[:, :patches * parts], shift, gain))
        predicted = predicted.reshape(batch, patches, parts, patch, dim).transpose(2, 3)
        predicted = predicted.reshape(batch, patches * patch, parts, dim)[:, :frames]
        return predicted.masked_fill(~frame_mask[..., None, None], 0)
