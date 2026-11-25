"""Frozen mBART planner, direct head, and time-modulated endpoint transformer."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .data import phase_grid
from .relations import project_relations
from .rotations import identity_like, normalize, rotation6d

SIGNS = {"CSL": 0, "DGS": 1, "ASL": 2}
SOURCE_LANGS = ("zh_CN", "de_DE", "en_XX")


def sinusoid(x, dim):
    n = torch.arange(dim // 2, device=x.device, dtype=x.dtype)
    freq = torch.exp(-math.log(10000) * n / (dim // 2))
    phase = x[..., None] * freq
    return torch.cat((phase.sin(), phase.cos()), -1)


class TextEncoder(nn.Module):
    def __init__(self, cfg, frozen):
        super().__init__()
        self.frozen = frozen
        self.fixture = cfg.get("backend") == "fixture"
        if self.fixture:
            self.dim = cfg.get("fixture_width", 32)
            self.encoder = nn.Embedding(260, self.dim)
            self.max_position = 32
        else:
            from transformers import MBartModel, MBartTokenizer

            revision = cfg.get("revision")
            if not revision or revision in ("main", "master"):
                raise ValueError("Pin text.revision to an immutable mBART revision")
            source = cfg.get("local_snapshot") or cfg["encoder"]
            self.tokenizer = MBartTokenizer.from_pretrained(source, revision=revision)
            model = MBartModel.from_pretrained(source, revision=revision)
            self.encoder = model.encoder
            self.dim = model.config.d_model
            self.max_position = model.config.max_position_embeddings
            if cfg.get("gradient_checkpointing") and not frozen:
                self.encoder.gradient_checkpointing_enable()
        if frozen:
            self.encoder.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode and not self.frozen)
        return self

    def tokenize(self, requests):
        windows, counts = [], []
        for req in requests:
            if req.source_language not in SOURCE_LANGS:
                raise ValueError(f"Unsupported source language {req.source_language!r}")
            if self.fixture:
                # 0 is padding. Fixture token IDs use 1..256.
                raw = [v + 1 for v in req.text.encode()]
                special = [257 + SOURCE_LANGS.index(req.source_language)]
            else:
                raw = self.tokenizer(req.text, add_special_tokens=False, verbose=False)["input_ids"]
                # mBART-cc25 source format is tokens, EOS, then the language ID.
                # Append the suffix to each window, including on fast v5 tokenizers.
                special = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.lang_code_to_id[req.source_language],
                ]
            room = self.max_position - len(special)
            chunks = [raw[i : i + room] for i in range(0, len(raw), room)] or [[]]
            windows.extend(c + special for c in chunks)
            counts.append(len(chunks))
        device = next(self.encoder.parameters()).device
        pad_id = 0 if self.fixture else self.tokenizer.pad_token_id
        ids = pad_sequence(
            [torch.tensor(w, device=device) for w in windows],
            batch_first=True,
            padding_value=pad_id,
        )
        valid = (
            torch.arange(ids.shape[1], device=device)[None]
            < torch.tensor([len(w) for w in windows], device=device)[:, None]
        )
        return ids, valid, counts

    def forward(self, requests):
        ids, valid, counts = self.tokenize(requests)
        if self.fixture:
            states = self.encoder(ids)
        else:
            states = self.encoder(input_ids=ids, attention_mask=valid.long()).last_hidden_state
        sequences = [h[m] for h, m in zip(states.split(counts), valid.split(counts))]
        states = pad_sequence(sequences, batch_first=True)
        valid = (
            torch.arange(states.shape[1], device=states.device)[None]
            < torch.tensor([len(s) for s in sequences], device=states.device)[:, None]
        )
        return states, valid


def sign_ids(requests, device):
    return torch.tensor([SIGNS[r.target_sign_language] for r in requests], device=device)


class DecoderBlock(nn.Module):
    def __init__(self, cfg, temporal_dim=None):
        super().__init__()
        width = cfg["width"]
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(3))
        self.self_attn = nn.MultiheadAttention(
            width, cfg["heads"], cfg["dropout"], batch_first=True
        )
        self.cross = nn.MultiheadAttention(width, cfg["heads"], cfg["dropout"], batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(width, cfg["ffn"]), nn.GELU(), nn.Linear(cfg["ffn"], width)
        )
        self.drop = nn.Dropout(cfg["dropout"])
        self.mod = nn.ModuleList()
        if temporal_dim is not None:
            for _ in range(3):
                layer = nn.Linear(temporal_dim, 2 * width)
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                self.mod.append(layer)

    def residual(self, x, y, time, branch):
        if time is not None:
            gamma, beta = self.mod[branch](time).chunk(2, dim=-1)
            y = (1 + gamma[:, None]) * y + beta[:, None]
        return x + self.drop(y)

    def forward(self, x, valid, text, text_valid, time=None):
        h = self.norms[0](x)
        y = self.self_attn(h, h, h, key_padding_mask=~valid, need_weights=False)[0]
        x = self.residual(x, y, time, 0)
        y = self.cross(
            self.norms[1](x), text, text, key_padding_mask=~text_valid, need_weights=False
        )[0]
        x = self.residual(x, y, time, 1)
        x = self.residual(x, self.ffn(self.norms[2](x)), time, 2)
        return x.masked_fill(~valid[..., None], 0)


class Planner(nn.Module):
    def __init__(self, text, cfg):
        super().__init__()
        self.text = text
        self.stride = cfg["stride_frames"]
        self.width = cfg["width"]
        self.language = nn.Embedding(len(SIGNS), cfg["sign_lang_dim"])
        self.duration = nn.Sequential(
            nn.Linear(text.dim + cfg["sign_lang_dim"], 256), nn.GELU(), nn.Linear(256, 1)
        )
        self.sign_query = nn.Linear(cfg["sign_lang_dim"], self.width)
        self.duration_query = nn.Linear(1, self.width)
        self.memory = nn.Linear(text.dim, self.width)
        self.blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg["layers"]))
        self.relations = nn.Linear(self.width, 39)

    def forward(self, requests, cap, lengths=None):
        h, text_valid = self.text(requests)
        sign = self.language(sign_ids(requests, h.device))
        pooled = (h * text_valid[..., None]).sum(1) / text_valid.sum(1, keepdim=True)
        log_length = self.duration(torch.cat((pooled, sign), -1)).squeeze(-1)
        predicted = log_length.clamp(0, math.log(cap)).exp().round().long().clamp(1, cap)
        lengths = predicted if lengths is None else lengths
        log_grid_length = lengths.to(h).log()
        grids = [
            phase_grid(n, self.stride, device=h.device, dtype=h.dtype) for n in lengths.tolist()
        ]
        phase = pad_sequence(grids, batch_first=True)
        valid = (
            torch.arange(phase.shape[1], device=h.device)[None]
            < torch.tensor([len(g) for g in grids], device=h.device)[:, None]
        )
        x = (
            sinusoid(100 * phase, self.width)
            + self.sign_query(sign)[:, None]
            + self.duration_query(log_grid_length[:, None])[:, None]
        )
        memory = self.memory(h)
        for block in self.blocks:
            x = block(x, valid, memory, text_valid)
        return {
            "length": lengths,
            "predicted": predicted,
            "log_length": log_length,
            "relation": project_relations(self.relations(x), from_logits=True),
            "hidden": x,
            "valid": valid,
        }

    @torch.no_grad()
    def predict(self, request, cap):
        out = self([request], cap)
        return int(out["length"][0]), out["relation"][0, out["valid"][0]]


class DirectHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.output = nn.Linear(width, 246)
        with torch.no_grad():
            self.output.bias.copy_(torch.tensor([1, 0, 0, 0, 1, 0]).repeat(41))

    def forward(self, hidden):
        return rotation6d(self.output(hidden).unflatten(-1, (41, 6)))


class Generator(nn.Module):
    def __init__(self, text, cfg):
        super().__init__()
        self.text = text
        self.width = cfg["width"]
        self.last_degenerate = None
        self.motion = nn.Linear(174, self.width)
        self.relation = nn.Linear(39, self.width)
        self.language = nn.Embedding(len(SIGNS), self.width)
        self.memory = nn.Linear(text.dim, self.width)
        time_dim = cfg["time_embedding"]
        self.time = nn.Sequential(
            nn.Linear(self.width, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.blocks = nn.ModuleList(DecoderBlock(cfg, time_dim) for _ in range(cfg["layers"]))
        self.output = nn.Sequential(nn.LayerNorm(self.width), nn.Linear(self.width, 174))
        with torch.no_grad():
            self.output[-1].bias.zero_()
            self.output[-1].bias[:164:4] = 1.0

    def encode(self, requests):
        states, valid = self.text(requests)
        return self.memory(states), valid, sign_ids(requests, states.device)

    def predict_target(self, q, e, time, context, relation, frame_valid, sign_enabled=True):
        memory, text_valid, sign = context
        t = self.time(sinusoid(time.reshape(len(q)), self.width))
        frames = sinusoid(torch.arange(q.shape[1], device=q.device, dtype=q.dtype), self.width)
        x = self.motion(torch.cat((q.flatten(-2), e), -1)) + frames
        if sign_enabled:
            x = x + self.language(sign)[:, None]
        if relation is not None:
            x = x + self.relation(relation)
        for block in self.blocks:
            x = block(x, frame_valid, memory, text_valid, t)
        y = self.output(x)
        raw = y[..., :164].unflatten(-1, (41, 4))
        self.last_degenerate = (
            (raw.detach().square().sum(-1) < 1e-16) & frame_valid[..., None]
        ).sum()
        qhat = normalize(raw)
        return (
            torch.where(frame_valid[..., None, None], qhat, identity_like(qhat)),
            y[..., 164:].masked_fill(~frame_valid[..., None], 0),
        )


def length_term(out, lengths):
    log_length = out["log_length"]
    target = lengths.to(log_length).log()
    return (
        F.smooth_l1_loss(log_length, target, beta=0.2, reduction="sum"),
        log_length.new_tensor(len(lengths)),
    )
