"""Typed-stream DiT with real gathered cross-part windows (no dense PL x PL mask)."""

from dataclasses import dataclass

import torch
from torch import nn

from cprflow.models.attention import Attention, relation_key_bias
from cprflow.models.embeddings import sinusoidal


@dataclass
class Prediction:
    z: torch.Tensor
    r: torch.Tensor | None
    hidden: torch.Tensor


class Block(nn.Module):
    def __init__(self, hidden, heads, ffn, radius, global_context):
        super().__init__()
        self.radius = radius
        self.global_context = global_context
        self.temporal = Attention(hidden, heads)
        self.local = Attention(hidden, heads)
        self.text = Attention(hidden, heads)
        if global_context:
            self.global_attn = Attention(hidden, heads)
        self.ffn = nn.Sequential(nn.Linear(hidden, ffn), nn.GELU(), nn.Linear(ffn, hidden))
        names = ["temporal", "local"] + (["global"] if global_context else []) + ["text", "ffn"]
        self.norms = nn.ModuleDict(
            {name: nn.LayerNorm(hidden, elementwise_affine=False) for name in names}
        )
        self.mods = nn.ModuleDict({name: nn.Linear(hidden, 3 * hidden) for name in names})
        for mod in self.mods.values():
            nn.init.zeros_(mod.weight)
            nn.init.zeros_(mod.bias)

    def forward(self, x, condition, text, text_mask, valid, alpha):
        b, p, length, h = x.shape
        query_mask = valid[:, None, :, None]

        def prepare(name):
            shift, scale, gate = self.mods[name](condition).chunk(3, -1)
            hidden = self.norms[name](x) * (1 + scale[:, None, None]) + shift[:, None, None]
            return hidden, gate[:, None, None]

        def add(y, gate):
            return (x + gate * y) * query_mask

        y, gate = prepare("temporal")
        y = y.reshape(b * p, length, h)
        vm = valid[:, None].expand(b, p, length).reshape(b * p, length)
        y = self.temporal(y, y, vm[:, None].expand(-1, length, -1), vm).reshape(b, p, length, h)
        x = add(y, gate)
        y, gate = prepare("local")
        offsets = torch.arange(-self.radius, self.radius + 1, device=x.device)
        positions = torch.arange(length, device=x.device)[:, None] + offsets
        in_range = (positions >= 0) & (positions < length)
        positions = positions.clamp(0, length - 1)
        # [B,L,W,P,H] -> [B*L,W*P,H], stream axis remains explicit for gating.
        context = y.permute(0, 2, 1, 3)[:, positions].reshape(b * length, -1, h)
        queries = y.permute(0, 2, 1, 3).reshape(b * length, p, h)
        key_valid = (
            (valid[:, positions] & in_range[None]).unsqueeze(-1).expand(b, length, len(offsets), p)
        )
        key_valid = key_valid.reshape(b * length, -1)
        part_ids = torch.arange(p, device=x.device)
        key_parts = part_ids.repeat(len(offsets))
        allowed, bias = relation_key_bias(part_ids, key_parts, alpha, x.device)
        mask = key_valid[:, None, :] & allowed[None]
        y = self.local(
            queries,
            context,
            mask,
            valid.reshape(b * length, 1).expand(-1, p),
            bias[None].expand(b * length, -1, -1),
        )
        x = add(y.reshape(b, length, p, h).permute(0, 2, 1, 3), gate)
        if self.global_context:
            y, gate = prepare("global")
            context = (y * query_mask).sum(2) / valid.sum(1).clamp_min(1)[:, None, None]
            queries = y.reshape(b, p * length, h)
            query_parts = part_ids.repeat_interleave(length)
            allowed, bias = relation_key_bias(query_parts, part_ids, alpha, x.device)
            mask = allowed[None].expand(b, -1, -1) & valid.any(1)[:, None, None]
            qv = valid[:, None].expand(b, p, length).reshape(b, p * length)
            y = self.global_attn(queries, context, mask, qv, bias[None].expand(b, -1, -1))
            x = add(y.reshape(b, p, length, h), gate)
        y, gate = prepare("text")
        qv = valid[:, None].expand(b, p, length).reshape(b, p * length)
        y = self.text(
            y.reshape(b, p * length, h), text, text_mask[:, None].expand(-1, p * length, -1), qv
        )
        x = add(y.reshape(b, p, length, h), gate)
        y, gate = prepare("ffn")
        return add(self.ffn(y), gate)


class TypedDiT(nn.Module):
    def __init__(
        self,
        hidden=384,
        blocks=8,
        heads=6,
        ffn=1536,
        radius=4,
        global_every=2,
        relational=False,
        text_dim=768,
        condition_dropout=0.1,
    ):
        super().__init__()
        self.relational = relational
        self.hidden_dim = hidden
        self.condition_dropout = condition_dropout
        dims = [128] * 4 + ([33] if relational else [])
        self.inputs = nn.ModuleList([nn.Linear(d, hidden) for d in dims])
        self.outputs = nn.ModuleList([nn.Linear(hidden, d) for d in dims])
        # ParameterList permits shape-compatible copying of the first four type embeddings.
        self.part_embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(hidden) * 0.02) for _ in dims]
        )
        self.text_projection = nn.Linear(text_dim, hidden)
        self.null = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.language = nn.Embedding(3, hidden)
        self.time = nn.Sequential(nn.Linear(256, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.length = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList(
            [Block(hidden, heads, ffn, radius, (i + 1) % global_every == 0) for i in range(blocks)]
        )
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        for head in self.outputs:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        z,
        r,
        t,
        text,
        text_mask,
        language,
        lengths,
        valid,
        token_times,
        drop_text=None,
        relation_alpha=1.0,
    ):
        b, _, length, _ = z.shape
        streams = [z[:, i] for i in range(4)]
        if self.relational:
            if r is None or r.shape != (b, length, 33):
                raise ValueError("Relational model requires [B,L,33] relation state")
            streams.append(r)
        elif r is not None:
            raise ValueError("P4 model must not receive relation state")
        x = torch.stack(
            [proj(s) + part for proj, s, part in zip(self.inputs, streams, self.part_embeddings)], 1
        )
        x = (x + sinusoidal(token_times * 30, self.hidden_dim)[:, None]) * valid[:, None, :, None]
        if drop_text is None:
            drop_text = (
                (torch.rand(b, device=z.device) < self.condition_dropout)
                if self.training
                else torch.zeros(b, device=z.device, dtype=torch.bool)
            )
        context = self.text_projection(text)
        context = torch.cat((self.null.expand(b, -1, -1), context), 1)
        # Dropped examples see exactly one learnable null token; no all-false text rows.
        context_mask = torch.cat((drop_text[:, None], text_mask & ~drop_text[:, None]), 1)
        lengths = lengths.float()
        length_features = torch.stack((lengths.log(), (lengths / 30).log()), -1)
        condition = torch.nn.functional.silu(
            self.time(sinusoidal(t, 256)) + self.language(language) + self.length(length_features)
        )
        for block in self.blocks:
            x = block(x, condition, context, context_mask, valid, relation_alpha)
        hidden = x
        x = self.final_norm(x)
        out = [head(x[:, i]) * valid[..., None] for i, head in enumerate(self.outputs)]
        return Prediction(torch.stack(out[:4], 1), out[4] if self.relational else None, hidden)

    def warm_start(self, state):
        current = self.state_dict()
        unexpected = [
            key for key in state if key not in current or current[key].shape != state[key].shape
        ]
        if unexpected:
            raise ValueError(f"Incompatible warm start: {unexpected}")
        missing = [key for key in current if key not in state]
        allowed = ("inputs.4.", "outputs.4.", "part_embeddings.4")
        if any(not key.startswith(allowed) for key in missing):
            raise ValueError(f"Missing common weights: {missing}")
        self.load_state_dict(state, strict=False)
