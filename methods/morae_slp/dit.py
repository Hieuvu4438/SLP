"""DiT-M latent flow transformer (MoRAE Sec. 3.3) with token-level text cross-attention.

Blocks follow the MARDM adaLN-zero transformer of the released MoRAE code. The condition
c = TimestepEmb(t) + W_p pool(text) + LangEmb(language) modulates every block, exactly as MoRAE
conditions on the CLIP sentence embedding. SLP needs word order, so ``text_mode: xattn`` adds a
cross-attention sub-layer to the mBART tokens in every block (zero-initialised output, so the
initial network equals the paper's). ``text_mode: adaln`` is the paper-faithful variant.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers import Attention, CrossAttention, sinusoidal_table
from .text import masked_pool

TEXT_MODES = ("adaln", "xattn")


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, width: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(nn.Linear(frequency_dim, width), nn.SiLU(),
                                 nn.Linear(width, width))

    def forward(self, t: Tensor) -> Tensor:
        half = self.frequency_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        # Scale t in [0, 1] by 1000 as in DiT/MARDM so the embedding resolves small steps.
        args = 1000 * t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class DiTBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff: int, dropout: float, cross: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(width, heads, rope=False, dropout=dropout)
        self.cross = cross
        if cross:
            self.norm_x = nn.LayerNorm(width, eps=1e-6)
            self.xattn = CrossAttention(width, width, heads)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, ff), nn.GELU(approximate="tanh"),
                                 nn.Dropout(dropout), nn.Linear(ff, width))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, c: Tensor, valid: Tensor, text: Tensor | None,
                text_valid: Tensor | None) -> Tensor:
        sa, ga, sha, sm, gm, shm = self.adaLN(c).chunk(6, dim=-1)
        x = x + ga.unsqueeze(1) * self.drop(self.attn(modulate(self.norm1(x), sha, sa), valid))
        if self.cross:
            x = x + self.drop(self.xattn(self.norm_x(x), text, text_valid))
        return x + gm.unsqueeze(1) * self.drop(self.mlp(modulate(self.norm2(x), shm, sm)))


class FinalLayer(nn.Module):
    def __init__(self, width: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(width, out_dim)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class TextContext(nn.Module):
    """Pooled condition vector and cross-attention tokens; dropped rows see only a null token."""

    def __init__(self, text_dim: int, width: int, cross: bool):
        super().__init__()
        self.pool = nn.Linear(text_dim, width)
        self.cross = cross
        if cross:
            self.proj = nn.Sequential(nn.Linear(text_dim, width), nn.LayerNorm(width))
            self.null = nn.Parameter(torch.randn(1, 1, width) * 0.02)

    def forward(self, tokens: Tensor, mask: Tensor, drop: Tensor
                ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        keep = (~drop).to(tokens.dtype)[:, None]
        pooled = self.pool(masked_pool(tokens, mask) * keep)
        if not self.cross:
            return pooled, None, None
        b = len(tokens)
        context = torch.cat([self.null.expand(b, 1, -1).to(tokens.dtype), self.proj(tokens)], 1)
        context_valid = torch.cat([drop[:, None], mask & ~drop[:, None]], dim=1)
        return pooled, context, context_valid


class LatentDiT(nn.Module):
    def __init__(self, latent_dim: int, text_dim: int, width: int = 512, depth: int = 12,
                 heads: int = 8, ff: int = 2048, dropout: float = 0.1, text_mode: str = "xattn",
                 languages: int = 3, max_frames: int = 1024):
        super().__init__()
        if text_mode not in TEXT_MODES:
            raise ValueError(f"flow.text_mode must be one of {TEXT_MODES}")
        cross = text_mode == "xattn"
        self.inp = nn.Linear(latent_dim, width)
        self.register_buffer("positions", sinusoidal_table(max_frames, width), persistent=False)
        self.pos_drop = nn.Dropout(dropout)
        self.time = TimestepEmbedder(width)
        self.language = nn.Embedding(languages, width)
        self.text = TextContext(text_dim, width, cross)
        self.blocks = nn.ModuleList(DiTBlock(width, heads, ff, dropout, cross)
                                    for _ in range(depth))
        self.final = FinalLayer(width, latent_dim)
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.time.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time.mlp[2].weight, std=0.02)
        nn.init.normal_(self.language.weight, std=0.02)
        for block in self.blocks:  # adaLN-zero
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
            if block.cross:
                nn.init.zeros_(block.xattn.proj.weight)
                nn.init.zeros_(block.xattn.proj.bias)
        nn.init.zeros_(self.final.adaLN[-1].weight)
        nn.init.zeros_(self.final.adaLN[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def condition(self, tokens: Tensor, mask: Tensor, languages: Tensor, drop: Tensor) -> dict:
        """Text-dependent tensors, computed once per batch and reused at every ODE step."""
        pooled, context, context_valid = self.text(tokens, mask, drop)
        return {"cond": pooled + self.language(languages), "context": context,
                "context_valid": context_valid}

    def forward(self, x: Tensor, t: Tensor, valid: Tensor, condition: dict) -> Tensor:
        if x.shape[1] > len(self.positions):
            raise ValueError(f"sequence of {x.shape[1]} frames exceeds max_frames")
        c = condition["cond"] + self.time(t)
        h = self.pos_drop(self.inp(x) + self.positions[: x.shape[1]].to(x.dtype))
        for block in self.blocks:
            h = block(h, c, valid, condition["context"], condition["context_valid"])
        return self.final(h, c)


class DurationHead(nn.Module):
    """log T from the pooled (never dropped) text features and the language."""

    def __init__(self, text_dim: int, hidden: int = 512, languages: int = 3):
        super().__init__()
        self.language = nn.Embedding(languages, text_dim)
        nn.init.zeros_(self.language.weight)
        self.mlp = nn.Sequential(nn.Linear(text_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def init_prior(self, mean_log_length: float) -> None:
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, mean_log_length)

    def forward(self, tokens: Tensor, mask: Tensor, languages: Tensor) -> Tensor:
        # FP32 on purpose: in bf16, log T ~ 5 has a resolution of ~3 % (coarse frame counts).
        with torch.autocast(tokens.device.type, enabled=False):
            pooled = masked_pool(tokens.float(), mask) + self.language(languages).float()
            return self.mlp(F.layer_norm(pooled, (pooled.shape[-1],))).squeeze(-1)
