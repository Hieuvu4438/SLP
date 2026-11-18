"""Backbone shared by Stage A (anchor regressor) and Stage B (residual flow).

Same decoder as the SignNMFlow replication (8 layers, latent 512, ff 2048, 8 heads, separate
text cross-attention, MotionDiffuse Stylization blocks after every sub-layer, time embedding
2048) and the same text encoder (mBART-large-cc25 encoder with SOKE's reduced vocabulary,
fine-tuned), so Stage B with a uniform source is a matched SignNMFlow-GE baseline.

Differences, all switchable:
* ``mode='anchor'``: no noisy state input; the per-frame input is a learned query + absolute
  and *relative* (frame / length) position, and the timestep embedding is a constant (t = 0).
* ``cond_anchor``: Stage B additionally reads the anchor mu (and its expression) per frame.
* ``null_text``: learned unconditional text token for classifier-free guidance; samples whose
  text is dropped attend to this single token instead of the sentence.
* Length head on the pooled, detached text features (as in the replication).
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .representation import EXPR_DIM, N_JOINTS

LANG_CODE = {"phoenix": "de_DE", "csl": "zh_CN", "how2sign": "en_XX"}


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of a scalar in [0,1] (scaled by 1000 like diffusion timesteps)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = (t.float() * 1000.0)[..., None] * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


class StylizationBlock(nn.Module):
    def __init__(self, latent: int, time_dim: int, dropout: float):
        super().__init__()
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2 * latent))
        self.norm = nn.LayerNorm(latent)
        self.out = nn.Sequential(nn.SiLU(), nn.Dropout(dropout), nn.Linear(latent, latent))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, h, emb):
        scale, shift = self.emb(emb)[:, None].chunk(2, -1)
        return self.out(self.norm(h) * (1 + scale) + shift)


class DecoderLayer(nn.Module):
    def __init__(self, latent, heads, ff, time_dim, text_dim, dropout):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(latent), nn.LayerNorm(latent), nn.LayerNorm(latent)
        self.self_attn = nn.MultiheadAttention(latent, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(latent, heads, dropout=dropout, batch_first=True,
                                                kdim=text_dim, vdim=text_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.ff = nn.Sequential(nn.Linear(latent, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, latent))
        self.s1 = StylizationBlock(latent, time_dim, dropout)
        self.s2 = StylizationBlock(latent, time_dim, dropout)
        self.s3 = StylizationBlock(latent, time_dim, dropout)

    def forward(self, x, emb, pad, text, text_pad):
        h = self.n1(x)
        x = x + self.s1(self.self_attn(h, h, h, key_padding_mask=pad, need_weights=False)[0], emb)
        tx = self.text_norm(text)
        x = x + self.s2(self.cross_attn(self.n2(x), tx, tx, key_padding_mask=text_pad,
                                        need_weights=False)[0], emb)
        return x + self.s3(self.ff(self.n3(x)), emb)


class TextEncoder(nn.Module):
    """mBART encoder with SOKE's reduced vocabulary (token id -> embedding id map)."""

    def __init__(self, path: str, max_len: int = 128):
        super().__init__()
        from transformers import MBartForConditionalGeneration, MBartTokenizer

        self.tok = MBartTokenizer.from_pretrained(path, legacy=True)
        self.encoder = MBartForConditionalGeneration.from_pretrained(path).get_encoder()
        mp = Path(path) / "map_ids.pkl"
        self.id_map = pickle.load(open(mp, "rb")) if mp.exists() else None
        self.unk = self.tok.convert_tokens_to_ids("<unk>")
        self.max_len = max_len
        self.dim = self.encoder.config.d_model

    def tokenize(self, texts, srcs, device):
        ids = []
        for text, src in zip(texts, srcs):
            self.tok.src_lang = LANG_CODE[src]
            ids.append(self.tok(text, truncation=True, max_length=self.max_len).input_ids)
        if self.id_map is not None:
            ids = [[self.id_map.get(i, self.unk) for i in seq] for seq in ids]
        L = max(len(s) for s in ids)
        out = torch.full((len(ids), L), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(ids), L), dtype=torch.long)
        for i, s in enumerate(ids):
            out[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        return out.to(device), mask.to(device)

    def forward(self, texts, srcs, device):
        ids, mask = self.tokenize(texts, srcs, device)
        return self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state, mask == 0


class PoseNet(nn.Module):
    def __init__(self, mode: str = "flow", cond_anchor: bool = False, latent: int = 512,
                 layers: int = 8, heads: int = 8, ff: int = 2048, time_dim: int = 2048,
                 dropout: float = 0.1, max_frames: int = 400, text_model: str | None = None,
                 text_max_len: int = 128, text_dim: int = 1024):
        super().__init__()
        assert mode in ("flow", "anchor")
        self.mode, self.cond_anchor = mode, bool(cond_anchor and mode == "flow")
        self.pose_dim = N_JOINTS * 4
        frame_dim = self.pose_dim + EXPR_DIM
        self.text = TextEncoder(text_model, text_max_len) if text_model else None
        text_dim = self.text.dim if self.text is not None else text_dim
        self.null_text = nn.Parameter(torch.zeros(1, 1, text_dim))
        nn.init.normal_(self.null_text, std=0.02)
        if mode == "flow":
            self.inp = nn.Linear(frame_dim * (2 if self.cond_anchor else 1), latent)
        else:
            self.query = nn.Parameter(torch.zeros(1, 1, latent))
            self.rel_pos = nn.Sequential(nn.Linear(latent, latent), nn.SiLU(), nn.Linear(latent, latent))
        self.pos = nn.Parameter(torch.zeros(1, max_frames, latent))
        nn.init.normal_(self.pos, std=0.02)
        self.latent = latent
        self.time_mlp = nn.Sequential(nn.Linear(latent, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.layers = nn.ModuleList([DecoderLayer(latent, heads, ff, time_dim, text_dim, dropout)
                                     for _ in range(layers)])
        self.out_norm = nn.LayerNorm(latent)
        self.out = nn.Linear(latent, frame_dim)
        self.len_head = nn.Sequential(nn.Linear(text_dim, 512), nn.SiLU(), nn.Linear(512, 1))
        self.max_frames = max_frames

    # ------------------------------------------------------------------ text
    def encode_text(self, texts, srcs, device):
        return self.text(texts, srcs, device)

    def drop_text(self, text, text_pad, drop: torch.Tensor):
        """Replace the text of samples with drop[b] = True by the learned null token."""
        if not bool(drop.any()):
            return text, text_pad
        B, L, C = text.shape
        null = self.null_text.to(text.dtype).expand(B, L, C)
        d = drop[:, None, None]
        text = torch.where(d, null, text)
        null_pad = torch.ones_like(text_pad)
        null_pad[:, 0] = False
        return text, torch.where(drop[:, None], null_pad, text_pad)

    def null_like(self, text, text_pad):
        return self.drop_text(text, text_pad, torch.ones(text.shape[0], dtype=torch.bool, device=text.device))

    # ------------------------------------------------------------------ network
    def forward(self, x_t, e_t, t, text, text_pad, pad, anchor=None):
        """x_t [B,T,41,4] (ignored in anchor mode, may be None), e_t [B,T,10], t [B];
        anchor [B,T,174] (flat mu) when cond_anchor; pad/text_pad: True = padding.
        Returns raw pose [B,T,41,4] and expression [B,T,10]."""
        B, T = pad.shape
        if self.mode == "flow":
            inp = torch.cat([x_t.flatten(-2), e_t], -1)
            if self.cond_anchor:
                inp = torch.cat([inp, anchor], -1)
            h = self.inp(inp) + self.pos[:, :T]
        else:
            length = (~pad).sum(1).clamp_min(1).float()
            rel = torch.arange(T, device=pad.device)[None].float() / length[:, None]   # [B,T]
            h = self.query + self.pos[:, :T] + self.rel_pos(timestep_embedding(rel.clamp(0, 1), self.latent))
            t = torch.zeros(B, device=pad.device)
        emb = self.time_mlp(timestep_embedding(t, self.latent))
        for layer in self.layers:
            h = layer(h, emb, pad, text, text_pad)
        o = self.out(self.out_norm(h)).float()   # losses / geometry always in fp32
        return o[..., :self.pose_dim].unflatten(-1, (N_JOINTS, 4)), o[..., self.pose_dim:]

    def predict_log_length(self, text, text_pad):
        keep = (~text_pad).float()[..., None]
        pooled = (text.detach() * keep).sum(1) / keep.sum(1).clamp_min(1)
        return self.len_head(pooled)[:, 0]

    @staticmethod
    def length_loss(pred_log_len, lengths):
        return F.l1_loss(pred_log_len, torch.log(lengths.float()))
