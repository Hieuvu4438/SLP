from __future__ import annotations

import math
import torch
from torch import nn


def embedding(t, dim):
    f = torch.exp(
        torch.arange(dim // 2, device=t.device, dtype=t.dtype)
        * (-math.log(10000) / max(1, dim // 2 - 1))
    )
    x = t[..., None] * f
    return torch.cat((x.sin(), x.cos()), -1)


def mlp(ni, no, hidden):
    return nn.Sequential(nn.Linear(ni, hidden), nn.SiLU(), nn.Linear(hidden, no))


class LanguageAdapters(nn.Module):
    def __init__(self, dim, languages=3):
        super().__init__()
        self.adapters = nn.ModuleList([mlp(dim, dim, max(8, dim // 8)) for _ in range(languages)])
        for a in self.adapters:
            nn.init.zeros_(a[-1].weight)
            nn.init.zeros_(a[-1].bias)

    def forward(self, x, language):
        if ((language < 0) | (language >= len(self.adapters))).any():
            raise ValueError("Unknown sign language ID; expected DGS=0, CSL=1, ASL=2")
        weights = torch.nn.functional.one_hot(language, len(self.adapters)).to(x)
        branches = torch.stack([a(x) for a in self.adapters], 1)
        return x + (branches * weights.reshape(len(x), -1, *([1] * (x.ndim - 1)))).sum(1)


class AxialBlock(nn.Module):
    def __init__(self, dim, heads, ff, dropout):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(4)])
        self.spatial = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.temporal = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, dim)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask, context=None, context_mask=None):
        b, t, p, d = x.shape
        h = self.norms[0](x).reshape(b * t, p, d)
        x = x + self.drop(self.spatial(h, h, h, need_weights=False)[0].reshape(b, t, p, d))
        h = self.norms[1](x).transpose(1, 2).reshape(b * p, t, d)
        pad = ~mask[:, None, :].expand(b, p, t).reshape(b * p, t)
        h = self.temporal(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        x = x + self.drop(h.reshape(b, p, t, d).transpose(1, 2))
        if context is not None:
            h = self.norms[2](x).reshape(b, t * p, d)
            h = self.cross(h, context, context, key_padding_mask=~context_mask, need_weights=False)[
                0
            ]
            x = x + self.drop(h.reshape(b, t, p, d))
        x = x + self.drop(self.ff(self.norms[3](x)))
        return x * mask[..., None, None]


class Condition(nn.Module):
    def __init__(self, dim, text_dim, plan_dim):
        super().__init__()
        self.text = nn.Linear(text_dim, dim)
        self.plan = nn.Linear(plan_dim, dim)
        self.language = nn.Embedding(3, dim)
        self.length = mlp(1, dim, dim)
        self.dim = dim

    def forward(self, text, text_mask, language, lengths, z=None, zmask=None):
        token = self.language(language) + self.length(lengths.to(text).log()[:, None])
        pieces = [self.text(text), token[:, None]]
        masks = [text_mask, torch.ones(len(text), 1, dtype=torch.bool, device=text.device)]
        if z is not None:
            pieces.append(self.plan(z))
            masks.append(zmask)
        return torch.cat(pieces, 1), torch.cat(masks, 1)
