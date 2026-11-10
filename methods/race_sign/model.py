import math
import torch
from torch import nn
from .config import RaceConfig
from .schema import PART_SLICES


def sinusoidal(values, width):
    half = (width + 1) // 2
    freq = torch.exp(
        -math.log(10000)
        * torch.arange(half, device=values.device, dtype=values.dtype)
        / max(half - 1, 1)
    )
    angle = values[..., None] * freq
    return torch.cat((angle.sin(), angle.cos()), -1)[..., :width]


class CoupledBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        w = cfg.width
        self.time_attn = nn.MultiheadAttention(w, cfg.heads, cfg.dropout, batch_first=True)
        self.part_attn = nn.MultiheadAttention(w, cfg.heads, cfg.dropout, batch_first=True)
        self.text_attn = nn.MultiheadAttention(w, cfg.heads, cfg.dropout, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(w) for _ in range(4)])
        self.ff = nn.Sequential(
            nn.Linear(w, 4 * w), nn.SiLU(), nn.Dropout(cfg.dropout), nn.Linear(4 * w, w)
        )
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(w, 2 * w))

    def forward(self, x, time_cond, text, frame_mask, text_mask, coupled=True):
        b, t, p, w = x.shape
        mask = frame_mask[:, :, None, None]
        scale, shift = self.mod(time_cond).chunk(2, -1)
        q = self.norms[0](x) * (1 + scale[:, None, None]) + shift[:, None, None]
        q = q.permute(0, 2, 1, 3).reshape(b * p, t, w)
        pad = ~frame_mask[:, None].expand(b, p, t).reshape(b * p, t)
        a = self.time_attn(q, q, q, key_padding_mask=pad, need_weights=False)[0]
        x = (x + a.reshape(b, p, t, w).permute(0, 2, 1, 3)) * mask
        if coupled:
            q = self.norms[1](x).reshape(b * t, p, w)
            x = (x + self.part_attn(q, q, q, need_weights=False)[0].reshape(b, t, p, w)) * mask
        q = self.norms[2](x).reshape(b, t * p, w)
        x = (
            x
            + self.text_attn(q, text, text, key_padding_mask=~text_mask, need_weights=False)[
                0
            ].reshape(b, t, p, w)
        ) * mask
        return (x + self.ff(self.norms[3](x))) * mask


class FlowDecoder(nn.Module):
    """Raw-manual-motion velocity model. Codebooks stay frozen in the base adapter."""

    def __init__(self, cfg: RaceConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.pose_in = nn.ModuleList(nn.Linear(s.stop - s.start, w) for s in PART_SLICES)
        self.pose_out = nn.ModuleList(nn.Linear(w, s.stop - s.start) for s in PART_SLICES)
        # Each token produces distinct frame features; no repeat-identical-frame shortcut.
        self.code_projection = nn.ModuleList(
            nn.Linear(cfg.code_dim, cfg.factor * w) for _ in range(3)
        )
        self.text_in = nn.Linear(cfg.text_dim, w)
        self.null_text = nn.Parameter(torch.zeros(1, 1, w))
        self.language = nn.Embedding(cfg.languages, w)
        self.part = nn.Parameter(torch.randn(3, w) * 0.02)
        self.relation_in = nn.Linear(15, w)
        self.time_in = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w))
        self.blocks = nn.ModuleList(CoupledBlock(cfg) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(w)

    def forward(
        self,
        state,
        time,
        codes,
        token_lengths,
        text,
        text_mask,
        lengths,
        language,
        relations,
        *,
        offsets=None,
        use_text=True,
        coupled=True,
    ):
        b, t, d = state.shape
        if d != 240 or codes.shape[0] != b or codes.shape[2:] != (3, self.cfg.code_dim):
            raise ValueError("Expected state[B,T,240], codes[B,K,3,C]")
        if (lengths <= 0).any() or (lengths > t).any():
            raise ValueError("Invalid motion lengths")
        offsets = torch.zeros_like(lengths) if offsets is None else offsets
        if (offsets < 0).any() or (offsets + lengths > token_lengths * self.cfg.factor).any():
            raise ValueError("Frame window exceeds available token timeline")
        if text.shape[:2] != text_mask.shape or text.shape[0] != b or relations.shape != (b, t, 15):
            raise ValueError("Invalid text or relational features")
        if time.shape != (b,) or (time < 0).any() or (time > 1).any():
            raise ValueError("Flow times must have shape [B] in [0,1]")
        if (token_lengths <= 0).any() or (token_lengths > codes.shape[1]).any():
            raise ValueError("Invalid token lengths")
        frame_mask = torch.arange(t, device=state.device)[None] < lengths[:, None]
        token_mask = (
            torch.arange(codes.shape[1], device=state.device)[None] < token_lengths[:, None]
        )
        # Padding must not enter layer biases/attention/conditioning.
        state = state.masked_fill(~frame_mask[..., None], 0)
        codes = codes.masked_fill(~token_mask[..., None, None], 0)
        text = text.masked_fill(~text_mask[..., None], 0)
        frames = offsets[:, None] + torch.arange(t, device=state.device)[None]
        features = []
        for p, sl in enumerate(PART_SLICES):
            expanded = self.code_projection[p](codes[:, :, p]).reshape(b, -1, self.cfg.width)
            gather = frames.clamp_max(expanded.shape[1] - 1)[..., None].expand(
                -1, -1, self.cfg.width
            )
            features.append(self.pose_in[p](state[..., sl]) + expanded.gather(1, gather))
        x = torch.stack(features, 2)
        x = (
            x
            + self.part[None, None]
            + sinusoidal(frames.to(state.dtype), self.cfg.width)[:, :, None]
        )
        if coupled:
            rel = relations.masked_fill(~frame_mask[..., None], 0)
            x = x + self.relation_in(rel)[:, :, None]
        x = x * frame_mask[..., None, None]
        text = self.text_in(text)
        # A valid NULL token also handles all-padding captions without NaNs.
        drop = torch.zeros(b, dtype=torch.bool, device=state.device)
        if not use_text:
            drop[:] = True
        elif self.training:
            drop = torch.rand(b, device=state.device) < self.cfg.text_dropout
        text_mask = text_mask & ~drop[:, None]
        text = torch.cat((self.null_text.expand(b, -1, -1), text), 1)
        text_mask = torch.cat(
            (torch.ones(b, 1, dtype=torch.bool, device=state.device), text_mask), 1
        )
        tc = self.time_in(sinusoidal(time * 1000, self.cfg.width)) + self.language(language)
        for block in self.blocks:
            x = block(x, tc, text, frame_mask, text_mask, coupled)
        x = self.norm(x)
        return torch.cat(
            [head(x[:, :, p]) for p, head in enumerate(self.pose_out)], -1
        ).masked_fill(~frame_mask[..., None], 0)
