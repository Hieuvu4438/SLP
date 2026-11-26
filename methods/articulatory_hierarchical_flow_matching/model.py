"""Text-conditioned transformer shared by the coarse and detail flows."""

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import LANGUAGES, POSE_DIM
from .subspace import frame_intervals


def sinusoidal(values: Tensor, frequencies: Tensor) -> Tensor:
    angles = values[..., None] * frequencies
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class Condition(NamedTuple):
    tokens: Tensor
    mask: Tensor
    language: Tensor
    log_length: Tensor
    energy: Tensor | None


class Block(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.self_norm = nn.LayerNorm(width)
        self.cross_norm = nn.LayerNorm(width)
        self.ff_norm = nn.LayerNorm(width)
        self.temporal = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.text = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(width * 4, width), nn.Dropout(dropout))
        self.modulation = nn.Linear(width, width * 2)

    def forward(self, x, condition, motion_mask, token_mask, context):
        scale, shift = self.modulation(context).chunk(2, dim=-1)
        norm = self.self_norm(x) * (1 + scale[:, None]) + shift[:, None]
        x = x + self.temporal(norm, norm, norm, key_padding_mask=~motion_mask,
                              need_weights=False)[0]
        x = x + self.text(self.cross_norm(x), condition, condition, key_padding_mask=~token_mask,
                          need_weights=False)[0]
        return x + self.ff(self.ff_norm(x))


class MotionTransformer(nn.Module):
    def __init__(self, text_dim: int, width: int, heads: int, depth: int, dropout: float,
                 energy_rank: int | None, plan_dim: int | None = None, retrieval=False):
        super().__init__()
        self.text_projection = nn.Linear(text_dim, width)
        self.language = nn.Embedding(len(LANGUAGES), width)
        self.duration = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                                      nn.Linear(width, 1))
        self.null = nn.Parameter(torch.zeros(1, 1, width))
        self.in_pose = nn.ModuleList(nn.Linear(POSE_DIM, width) for _ in range(2))
        self.out_pose = nn.ModuleList(nn.Linear(width, POSE_DIM) for _ in range(2))
        self.coarse = nn.Linear(POSE_DIM, width)
        self.stage = nn.Embedding(2, width)
        self.progress = nn.Linear(1, width)
        self.clock = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(Block(width, heads, dropout) for _ in range(depth))
        self.out_norm = nn.LayerNorm(width)
        self.plan_projection = nn.Linear(plan_dim, width) if plan_dim is not None else None
        self.retrieve_projection = nn.Linear(POSE_DIM, width) if retrieval else None
        self.register_buffer("frequencies", 10000 ** (-torch.arange(0, width, 2).float() / width))
        if energy_rank is not None:
            self.energy_queries = nn.Parameter(torch.randn(4, width) * 0.02)
            self.energy_attention = nn.MultiheadAttention(width, heads, batch_first=True)
            self.energy_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                             nn.SiLU(), nn.Linear(width, energy_rank))
            self.energy_embed = nn.Linear(energy_rank, width)
        else:
            self.energy_head = None

    def prepare(self, text: Tensor, mask: Tensor, languages: Tensor) -> Condition:
        tokens = self.text_projection(text)
        language = self.language(languages)
        pooled = (tokens * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
        length = self.duration(pooled + language).squeeze(-1).float()
        energy = None
        if self.energy_head is not None:
            query = self.energy_queries[None] + language[:, None]
            read = self.energy_attention(query, tokens, tokens, key_padding_mask=~mask,
                                         need_weights=False)[0]
            energy = F.softplus(self.energy_head(query + read).float())
        return Condition(tokens, mask, language, length, energy)

    def forward(self, state: Tensor, time: Tensor, stage: int, condition: Condition,
                lengths: Tensor, motion_mask: Tensor, dropped: Tensor,
                coarse: Tensor | None = None, plan: Tensor | None = None,
                retrieval: Tensor | None = None) -> Tensor:
        tokens = torch.cat((condition.tokens, self.null.to(condition.tokens).expand(
            len(state), -1, -1,
        )), dim=1)
        token_mask = torch.cat((condition.mask & ~dropped[:, None], dropped[:, None]), dim=1)
        frames = torch.arange(state.shape[1], device=state.device)
        progress = frames[None, :, None] / (lengths - 1).clamp_min(1)[:, None, None]
        x = self.in_pose[stage](state) + sinusoidal(frames, self.frequencies)[None]
        x = x + self.progress(progress)
        context = (self.clock(sinusoidal(time * 1000, self.frequencies)) + condition.language
                   + self.stage.weight[stage])
        if coarse is not None:
            x = x + self.coarse(coarse)
        if stage == 0 and plan is not None:
            local = self.plan_projection(plan)
            x = x + local.masked_fill(dropped[:, None, None] | ~motion_mask[..., None], 0)
        if stage == 0 and retrieval is not None:
            hint = self.retrieve_projection(retrieval)
            x = x + hint.masked_fill(dropped[:, None, None] | ~motion_mask[..., None], 0)
        if stage == 0 and condition.energy is not None:
            intervals = frame_intervals(lengths, state.shape[1])
            energy = condition.energy.gather(1, intervals[..., None].expand(
                -1, -1, condition.energy.shape[-1],
            ))
            x = x + self.energy_embed(energy).masked_fill(dropped[:, None, None], 0)
        for block in self.blocks:
            x = block(x, tokens, motion_mask, token_mask, context)
        return self.out_pose[stage](self.out_norm(x)).float().masked_fill(
            ~motion_mask[..., None], 0,
        )
