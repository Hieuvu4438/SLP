"""Frozen text encoders, duration and monotone anchor heads, endpoint Transformers."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace

import torch
from torch import nn

from .anchors import Anchors


def sinusoidal(x, width):
    half = width // 2
    freq = torch.exp(-math.log(10000) * torch.arange(
        half, dtype=torch.float32, device=x.device
    ) / max(half - 1, 1))
    angles = x.float()[..., None] * freq
    return nn.functional.pad(torch.cat((angles.sin(), angles.cos()), -1), (0, width % 2))


def masked_pool(text, mask):
    return text.float().masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


@dataclass(frozen=True)
class TextBatch:
    hidden: torch.Tensor
    mask: torch.Tensor
    truncated: torch.Tensor


class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.model = None
        if cfg["encoder"] == "mock":
            self.dim = cfg["mock_width"]
        else:
            if not cfg["revision"]:
                raise ValueError("A pinned text.revision is required for the real text encoder")
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg["encoder"], revision=cfg["revision"], src_lang=cfg["source_lang"]
            )
            full = AutoModel.from_pretrained(cfg["encoder"], revision=cfg["revision"])
            self.model = full.get_encoder()
            self.dim = full.config.d_model
            self.model.requires_grad_(False).eval()
        self.register_buffer("_device_indicator", torch.empty(0), persistent=False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, texts):
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Expected non-empty text strings")
        device = self._device_indicator.device
        limit = self.cfg["max_tokens"]
        if self.model is not None:
            sizes = self.tokenizer(list(texts), return_length=True, truncation=False)["length"]
            inputs = self.tokenizer(list(texts), truncation=True, max_length=limit,
                                    padding=True, return_tensors="pt")
            ids = inputs["input_ids"].to(device)
            mask = inputs["attention_mask"].to(device).bool()
            hidden = self.model(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
        else:
            tokens = [t.split() for t in texts]
            sizes = [len(row) for row in tokens]
            ids = torch.zeros(len(tokens), min(max(sizes), limit), dtype=torch.long, device=device)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            for i, row in enumerate(tokens):
                codes = [int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "big") % 65521
                         for t in row[:limit]]
                ids[i, :len(codes)] = torch.tensor(codes, device=device)
                mask[i, :len(codes)] = True
            position = torch.arange(ids.shape[1], device=device) * .07
            hidden = sinusoidal(ids.float() / 619 + position, self.dim)
        return TextBatch(hidden.float().masked_fill(~mask[..., None], 0), mask,
                         torch.tensor(sizes, device=device) > limit)


class LengthHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, text):
        return self.layers(masked_pool(text.hidden, text.mask)).squeeze(-1).float()

    def predict(self, text, minimum, maximum):
        log_t = self(text)
        # Bounding the exponent keeps extreme logits finite before counting rounding and clamps.
        rounded = log_t.clamp(-20, 30).exp().round()
        clamped = (rounded < minimum) | (rounded > maximum)
        return rounded.clamp(minimum, maximum).long(), clamped


class AnchorPlanner(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        self.k = cfg["count"]
        w = cfg["width"]
        self.queries = nn.Parameter(torch.randn(self.k, w) / math.sqrt(w))
        self.text = nn.Linear(dim, w)
        self.length = nn.Sequential(nn.Linear(1, w), nn.GELU(), nn.Linear(w, w))
        self.blocks = nn.ModuleList(nn.TransformerDecoderLayer(
            w, cfg["heads"], dim_feedforward=4*w, activation="gelu", dropout=.1,
            batch_first=True, norm_first=True
        ) for _ in range(cfg["blocks"]))
        self.norm = nn.LayerNorm(w)
        self.gaps = nn.Linear(w, self.k + 1)
        self.payload = nn.Linear(w, 48)

    def forward(self, text: TextBatch, lengths):
        """The only planner inputs are frozen text states and explicit duration."""
        valid = torch.arange(self.k, device=lengths.device) < lengths[:, None]
        x = self.queries + self.length(lengths[:, None].float().log())[:, None]
        memory = self.text(text.hidden)
        for block in self.blocks:
            x = block(x, memory, tgt_key_padding_mask=~valid, memory_key_padding_mask=~text.mask)
        hidden = self.norm(x).float()
        logits = self.gaps(masked_pool(hidden, valid)).float()
        g = logits.softmax(-1).clamp_min(1e-6)
        time = (g / g.sum(-1, keepdim=True)).cumsum(-1)[:, :-1]
        payload = self.payload(hidden).float().unflatten(-1, (2, 24))
        mask = valid[..., None, None].expand_as(payload)
        return Anchors(time.masked_fill(~valid, 0), payload.masked_fill(~mask, 0), valid, mask,
                       valid[..., None].expand(-1, -1, 2).float(), hidden)


def plan_anchors(planner, text, lengths, confidence=None):
    output = planner(text, lengths)
    if confidence is None:
        return output
    q = confidence(output.hidden.detach()).float().sigmoid() * output.slot[..., None]
    return replace(output, confidence=q)


@dataclass(frozen=True)
class Condition:
    text: TextBatch
    times: torch.Tensor
    frame_mask: torch.Tensor
    coarse: torch.Tensor | None = None
    coarse_times: torch.Tensor | None = None
    coarse_mask: torch.Tensor | None = None
    full_body: torch.Tensor | None = None
    anchors: Anchors | None = None


def _valid(x, mask):
    return x.masked_fill(~mask[..., None], 0)


class AnchorAttention(nn.Module):
    def __init__(self, width, heads, cfg):
        super().__init__()
        self.heads = heads
        self.sigma = cfg["sigma_frames"]
        self.use_bias = cfg["use_time_bias"]
        self.hand = nn.ModuleList(nn.Linear(48, width, bias=False) for _ in range(2))
        self.time = nn.Linear(width, width, bias=False)
        self.attend = nn.MultiheadAttention(width, heads, batch_first=True, bias=False)

    def forward(self, query, anchors, frame_mask):
        batch, t, w = query.shape
        if anchors is None:
            return torch.zeros_like(query)
        valid = anchors.slot & anchors.feature.any(dim=(-1, -2))
        if not valid.any():
            return torch.zeros_like(query)
        seen = anchors.feature & valid[..., None, None]
        hand = anchors.payload.float().masked_fill(~seen, 0)
        features = torch.cat((hand, seen.float()), -1)
        confidence = anchors.confidence.masked_fill(~seen.any(-1), 0)
        content = sum(proj(features[:, :, h]) * confidence[..., h, None]
                      for h, proj in enumerate(self.hand))
        positions = anchors.time.float().masked_fill(~valid, 0) * (
            frame_mask.sum(-1, keepdim=True) - 1
        )
        key = content + self.time(sinusoidal(positions, w))
        zero = content.new_zeros(batch, 1, w)
        key = torch.cat((key, zero), 1)
        value = torch.cat((content, zero), 1)
        n = torch.arange(t, device=query.device).float()
        delta = n[:, None] - positions[:, None, :]
        bias = -(delta.square() / (2 * self.sigma**2)).clamp(max=30) if self.use_bias else (
            torch.zeros_like(delta)
        )
        bias = bias.masked_fill(~valid[:, None], -torch.inf)
        bias = nn.functional.pad(bias, (0, 1)).repeat_interleave(self.heads, 0).to(query.dtype)
        output = self.attend(query, key, value, attn_mask=bias, need_weights=False)[0]
        return _valid(output, valid.any(-1)[:, None] & frame_mask)


class EndpointBlock(nn.Module):
    def __init__(self, width, heads, dropout, fine=False, anchors=None):
        super().__init__()
        self.self_norm = nn.LayerNorm(width)
        self.self_attn = nn.MultiheadAttention(width, heads, dropout, batch_first=True)
        self.text_norm = nn.LayerNorm(width)
        self.text_attn = nn.MultiheadAttention(width, heads, dropout, batch_first=True)
        self.coarse_norm = nn.LayerNorm(width) if fine else None
        self.coarse_attn = nn.MultiheadAttention(width, heads, dropout, batch_first=True) if fine else None
        self.anchor_norm = nn.LayerNorm(width) if anchors is not None else None
        self.anchor_attn = AnchorAttention(width, heads, anchors) if anchors is not None else None
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * width, width))
        self.dropout = nn.Dropout(dropout)
        self.heads = heads

    def forward(self, x, tau, frame_mask, text, text_mask, coarse, coarse_mask, local, anchors):
        x = _valid(x + tau, frame_mask)
        q = self.self_norm(x)
        # Give padded query rows one safe key. Its output is removed below.
        forbidden = local[None, :, :] | ~frame_mask[:, None, :]
        forbidden = forbidden & frame_mask[:, :, None]
        self_mask = forbidden.repeat_interleave(self.heads, 0)
        v = self.self_attn(q, q, q, attn_mask=self_mask, need_weights=False)[0]
        x = _valid(x + self.dropout(v), frame_mask)
        v = self.text_attn(self.text_norm(x), text, text, key_padding_mask=~text_mask,
                           need_weights=False)[0]
        x = _valid(x + self.dropout(v), frame_mask)
        if self.coarse_attn is not None:
            v = self.coarse_attn(self.coarse_norm(x), coarse, coarse,
                                 key_padding_mask=~coarse_mask, need_weights=False)[0]
            x = _valid(x + self.dropout(v), frame_mask)
        if self.anchor_attn is not None:
            x = x + self.dropout(self.anchor_attn(self.anchor_norm(x), anchors, frame_mask))
        return _valid(x + self.dropout(self.ffn(self.ffn_norm(x))), frame_mask)


class EndpointFlow(nn.Module):
    def __init__(self, text_dim, cfg, anchors=None):
        super().__init__()
        self.fine = anchors is not None
        self.width = w = cfg["width"]
        self.radius = cfg.get("local_radius")
        self.state_dim = 204 if self.fine else 72
        self.input = nn.Linear(self.state_dim, w)
        self.output = nn.Linear(w, self.state_dim)
        self.text = nn.Linear(text_dim, w)
        self.flow_time = nn.Sequential(nn.Linear(128, w), nn.GELU(), nn.Linear(w, w))
        self.full_body = nn.Linear(72, w) if self.fine else None
        self.coarse = nn.Linear(72, w) if self.fine else None
        self.blocks = nn.ModuleList(
            EndpointBlock(w, cfg["heads"], cfg["dropout"], self.fine,
                          anchors if (i + 1) % 2 == 0 or cfg["blocks"] == 1 else None)
            for i in range(cfg["blocks"])
        )

    def predict_clean(self, x, tau, condition):
        m = condition.frame_mask
        x = self.input(_valid(x.float(), m)) + sinusoidal(condition.times.masked_fill(~m, 0),
                                                          self.width)
        text = self.text(_valid(condition.text.hidden, condition.text.mask))
        ticks = torch.arange(x.shape[1], device=x.device)
        local = ((ticks[:, None] - ticks).abs() > self.radius) if self.fine else (
            torch.zeros((len(ticks), len(ticks)), device=x.device, dtype=torch.bool)
        )
        coarse = None
        if self.fine:
            x = x + self.full_body(_valid(condition.full_body, m))
            cm = condition.coarse_mask
            coarse = self.coarse(_valid(condition.coarse, cm)) + sinusoidal(
                condition.coarse_times.masked_fill(~cm, 0), self.width
            )
        flow = self.flow_time(sinusoidal(tau, 128))[:, None]
        for block in self.blocks:
            x = block(x, flow, m, text, condition.text.mask, coarse, condition.coarse_mask,
                      local, condition.anchors)
        return _valid(self.output(x).float(), m)

    forward = predict_clean


def make_model(stage, dim, cfg):
    if stage == "length":
        return LengthHead(dim)
    if stage == "anchors":
        return AnchorPlanner(dim, cfg["anchors"])
    if stage == "body":
        return EndpointFlow(dim, cfg["body"])
    if stage == "fine":
        return EndpointFlow(dim, cfg["fine"], cfg["anchors"])
    raise ValueError(f"Unknown stage {stage}")
