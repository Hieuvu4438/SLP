"""Causal target-order planner with conditional content/duration/stop distributions."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .plans import PlanBatch
from .tensors import valid_mask


@dataclass
class PlannerOutput:
    mixture_logits: Tensor
    means: Tensor
    logvars: Tensor
    duration_logits: Tensor
    stop_logits: Tensor


@dataclass
class GeneratedPlans:
    plan: PlanBatch
    stop_reasons: tuple[str, ...]  # eos or capacity; cap-stop is NOT successful EOS


def mixture_nll(logits: Tensor, means: Tensor, logvars: Tensor, target: Tensor) -> Tensor:
    """Joint mixture density first, then per-dimension scaling (not independent mixes)."""
    log_density = -0.5 * (
        (target.unsqueeze(-2) - means).square() * (-logvars).exp() +
        logvars + math.log(2 * math.pi)
    ).sum(-1)
    return -torch.logsumexp(logits.log_softmax(-1) + log_density, -1) / target.shape[-1]


class TextToPlan(nn.Module):
    def __init__(self, content_width: int = 128, text_width: int = 512, hidden: int = 512,
                 heads: int = 8, layers: int = 4, mixtures: int = 4,
                 max_duration: int = 32, max_latents: int = 512,
                 dropout: float = 0.1, stop_threshold: float = 0.5):
        super().__init__()
        if min(content_width, hidden, heads, layers, mixtures, max_duration, max_latents) < 1:
            raise ValueError("planner dimensions and supports must be positive")
        if hidden % heads or not 0 < stop_threshold < 1:
            raise ValueError("invalid attention heads or stop threshold")
        self.content_width, self.mixtures = content_width, mixtures
        self.max_duration, self.max_latents, self.stop_threshold = (
            max_duration, max_latents, stop_threshold
        )
        self.bos = nn.Parameter(torch.zeros(hidden))
        self.duration_embedding = nn.Embedding(max_duration + 1, hidden, padding_idx=0)
        self.content_embedding = nn.Linear(content_width, hidden)
        self.position = nn.Embedding(max_latents, hidden)
        self.text_projection = nn.Linear(text_width, hidden)
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(hidden, heads, 4 * hidden, dropout=dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(layers)
        )
        self.norm = nn.LayerNorm(hidden)
        self.mixture = nn.Linear(hidden, mixtures * (1 + 2 * content_width))
        self.duration = nn.Sequential(nn.Linear(hidden + content_width, hidden), nn.GELU(),
                                       nn.Linear(hidden, max_duration))
        self.stop = nn.Sequential(nn.Linear(2 * hidden + content_width, hidden), nn.GELU(),
                                   nn.Linear(hidden, 1))

    def _states(self, memory: Tensor, text_valid: Tensor, contents: Tensor,
                durations: Tensor, lengths: Tensor) -> Tensor:
        b, k, _ = contents.shape
        if k > self.max_latents:
            raise ValueError("number of plan units exceeds the training-derived cap")
        if memory.ndim != 3 or memory.shape[:2] != text_valid.shape or memory.shape[0] != b:
            raise ValueError("invalid text memory/mask")
        if text_valid.dtype != torch.bool or bool((~text_valid.any(-1)).any()):
            raise ValueError("each text must have at least one valid token")
        valid = valid_mask(lengths, k)
        contents = torch.where(valid[..., None], contents, torch.zeros_like(contents))
        durations = torch.where(valid, durations, torch.zeros_like(durations))
        encoded = self.content_embedding(contents) + self.duration_embedding(durations)
        prefix = torch.cat((self.bos.expand(b, 1, -1), encoded[:, :-1]), 1)
        h = prefix + self.position(torch.arange(k, device=contents.device))[None]
        memory = torch.where(text_valid[..., None], memory, torch.zeros_like(memory))
        memory = self.text_projection(memory)
        causal = torch.ones(k, k, device=contents.device, dtype=torch.bool).triu(1)
        for layer in self.layers:
            h = layer(h, memory, tgt_mask=causal, tgt_key_padding_mask=~valid,
                      memory_key_padding_mask=~text_valid)
        return self.norm(h)

    def _mixture(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        params = self.mixture(h).reshape(*h.shape[:-1], self.mixtures,
                                        1 + 2 * self.content_width)
        return (params[..., 0], params[..., 1:1+self.content_width],
                params[..., 1+self.content_width:].clamp(-6, 2))

    def forward(self, memory: Tensor, text_valid: Tensor, plan: PlanBatch) -> PlannerOutput:
        plan.validate(self.max_duration)
        contents = plan.clean_contents()
        h = self._states(memory, text_valid, contents, plan.durations, plan.lengths)
        logits, means, logvars = self._mixture(h)
        duration = self.duration(torch.cat((h, contents), -1))
        stop = self.stop(torch.cat((h, contents, self.duration_embedding(plan.durations)), -1))
        return PlannerOutput(logits, means, logvars, duration, stop.squeeze(-1))

    def loss(self, memory: Tensor, text_valid: Tensor, plan: PlanBatch) -> dict[str, Tensor]:
        out = self(memory, text_valid, plan)
        valid = valid_mask(plan.lengths, plan.contents.shape[1])
        content_loss = mixture_nll(out.mixture_logits, out.means, out.logvars,
                                   plan.clean_contents())
        target_duration = (plan.durations - 1).clamp_min(0)
        duration_loss = F.cross_entropy(out.duration_logits.transpose(1, 2),
                                        target_duration, reduction="none")
        stops = torch.arange(plan.contents.shape[1], device=plan.lengths.device)[None]
        stop_target = (stops == plan.lengths[:, None] - 1).to(out.stop_logits)
        stop_loss = F.binary_cross_entropy_with_logits(out.stop_logits, stop_target,
                                                       reduction="none")
        def average(values):
            return (torch.where(valid, values, torch.zeros_like(values)).sum(-1) /
                    plan.lengths).mean()
        c, d, s = average(content_loss), average(duration_loss), average(stop_loss)
        return dict(total=c+d+s, content=c, duration=d, stop=s)

    @torch.no_grad()
    def generate(self, memory: Tensor, text_valid: Tensor, *,
                 generator: torch.Generator | None = None,
                 sample_content: bool = True) -> GeneratedPlans:
        if self.training:
            raise RuntimeError("planner generation requires eval() to disable dropout")
        codes, durations, reasons = [], [], []
        for b in range(len(memory)):
            prefix_codes = memory.new_zeros(0, self.content_width)
            prefix_durations: list[int] = []
            remaining = self.max_latents
            while remaining:
                # Dummy current unit is never visible to its own mixture prediction:
                # _states shifts all contents/durations right behind BOS.
                current = torch.cat((prefix_codes, prefix_codes.new_zeros(1, self.content_width)))
                ds = torch.tensor([prefix_durations + [0]], device=memory.device)
                lengths = torch.tensor([len(current)], device=memory.device)
                h = self._states(memory[b:b+1], text_valid[b:b+1], current[None], ds, lengths)
                state = h[:, -1]
                logits, means, logvars = self._mixture(state)
                if sample_content:
                    component = int(torch.multinomial(logits[0].softmax(-1), 1,
                                                       generator=generator))
                    noise = torch.randn(self.content_width, device=memory.device,
                                        dtype=memory.dtype, generator=generator)
                    e = means[0, component] + (.5 * logvars[0, component]).exp() * noise
                else:
                    e = means[0, int(logits[0].argmax())]
                duration_logits = self.duration(torch.cat((state, e[None]), -1))[0]
                support = min(remaining, self.max_duration)
                d = int(duration_logits[:support].argmax()) + 1
                de = self.duration_embedding(torch.tensor([d], device=memory.device))
                probability = self.stop(torch.cat((state, e[None], de), -1)).sigmoid().item()
                prefix_codes = torch.cat((prefix_codes, e[None]))
                prefix_durations.append(d)
                remaining -= d
                if probability >= self.stop_threshold:
                    reasons.append("eos")
                    break
            else:
                reasons.append("capacity")
            codes.append(prefix_codes)
            durations.append(prefix_durations)
        return GeneratedPlans(PlanBatch.from_sequences(codes, durations), tuple(reasons))
