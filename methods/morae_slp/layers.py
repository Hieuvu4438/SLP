"""Transformer building blocks with rotary positions and key-padding masks."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def rope_cache(length: int, head_dim: int, device, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    if head_dim % 2:
        raise ValueError("Rotary attention needs an even head dimension")
    inv = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    phase = torch.outer(torch.arange(length, device=device).float(), inv)
    return phase.cos().to(dtype), phase.sin().to(dtype)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x [B, H, T, Dh]; rotates interleaved channel pairs (as in the MoRAE decoder)."""
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)


def key_mask(valid: Tensor | None) -> Tensor | None:
    """[B, T] True = real frame -> SDPA boolean mask [B, 1, 1, T] (True = may attend)."""
    return None if valid is None else valid[:, None, None, :]


class Attention(nn.Module):
    """Multi-head self-attention. ``inner`` may differ from ``dim`` (MoRAE's compressor uses a
    3072-wide attention on 768-D features)."""

    def __init__(self, dim: int, heads: int, inner: int | None = None, rope: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        inner = inner or dim
        if inner % heads:
            raise ValueError(f"attention width {inner} is not divisible by {heads} heads")
        self.heads, self.head_dim, self.rope, self.dropout = heads, inner // heads, rope, dropout
        if rope and self.head_dim % 2:
            raise ValueError("Rotary attention needs an even head dimension")
        self.qkv = nn.Linear(dim, 3 * inner)
        self.proj = nn.Linear(inner, dim)

    def forward(self, x: Tensor, valid: Tensor | None) -> Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.rope:
            cos, sin = rope_cache(t, self.head_dim, x.device, q.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask(valid),
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(b, t, -1))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"width {dim} is not divisible by {heads} heads")
        self.heads, self.head_dim = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(context_dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, context: Tensor, context_valid: Tensor) -> Tensor:
        b, t, _ = x.shape
        q = self.q(x).reshape(b, t, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(context).reshape(b, context.shape[1], 2, self.heads,
                                        self.head_dim).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask(context_valid))
        return self.proj(out.transpose(1, 2).reshape(b, t, -1))


class Block(nn.Module):
    """Pre-LN Transformer block (attention + GELU MLP)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, rope: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, rope=rope)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid: Tensor | None) -> Tensor:
        x = x + self.drop(self.attn(self.norm1(x), valid))
        return x + self.drop(self.mlp(self.norm2(x)))


class Transformer(nn.Module):
    """Stack of blocks; ``forward`` returns every block output (for MoRAE's last-four-layer sum)."""

    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float = 4.0,
                 rope: bool = True, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio, rope, dropout)
                                    for _ in range(depth))

    def forward(self, x: Tensor, valid: Tensor | None) -> list[Tensor]:
        hidden = []
        for block in self.blocks:
            x = block(x, valid)
            hidden.append(x)
        return hidden


def init_weights(module: nn.Module) -> None:
    """Truncated-normal linear weights and zero biases (ViT / I-JEPA initialisation)."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def sinusoidal_table(length: int, dim: int) -> Tensor:
    pe = torch.zeros(length, dim)
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div[: dim // 2])
    return pe


def masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    """Mean of values [B, T, ...] over valid frames [B, T] (all trailing dims averaged too)."""
    while valid.dim() < values.dim():
        valid = valid.unsqueeze(-1)
    weight = valid.to(values.dtype).expand_as(values)
    return (values * weight).sum() / weight.sum().clamp_min(1)
