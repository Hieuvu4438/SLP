"""Sign motion generation network phi(x_t, e_t, t, c) (paper Sec. 4.1, App. D.1, Table 6).

* Text encoder: pre-trained mBART-large-cc25 encoder (SOKE's reduced-vocabulary release
  ``mbart-h2s-csl-phoenix``), fine-tuned jointly with phi; the mBART source-language token
  is set per sample from the dataset of origin (de_DE / zh_CN / en_XX).
* Decoder: 8 layers, latent 512, feed-forward 2048, 8 heads, dropout 0.1, a separate
  cross-attention to the text features, and the timestep injected through a Stylization
  block (MotionDiffuse, Zhang et al. 2024) after every self-attention, cross-attention and
  feed-forward layer; time-embedding dimension 2048.
* Output: raw per-frame pose prediction (41 x 4 for quaternions, 41 x 6 for the 6D baseline)
  and the 10-D expression prediction.
* Length head (not specified in the paper, see README): log-length regression from the
  pooled, *detached* text features, so it does not alter the paper's training objective.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

LANG_CODE = {"phoenix": "de_DE", "csl": "zh_CN", "how2sign": "en_XX"}


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of t in [0,1] (scaled to [0,1000] like diffusion timesteps)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


class StylizationBlock(nn.Module):
    """MotionDiffuse stylization: h = Linear0(Drop(SiLU(LN(h) * (1 + scale) + shift)))."""

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
        x = x + self.s3(self.ff(self.n3(x)), emb)
        return x


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
        pad_id = self.tok.pad_token_id
        out = torch.full((len(ids), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(ids), L), dtype=torch.long)
        for i, s in enumerate(ids):
            out[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        return out.to(device), mask.to(device)

    def forward(self, texts, srcs, device):
        ids, mask = self.tokenize(texts, srcs, device)
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return h, mask == 0


class SignGenerator(nn.Module):
    def __init__(self, rot_dim: int = 4, n_joints: int = 41, expr_dim: int = 10, latent: int = 512,
                 layers: int = 8, heads: int = 8, ff: int = 2048, time_dim: int = 2048,
                 dropout: float = 0.1, max_frames: int = 400, text_model: str | None = None,
                 text_max_len: int = 128, text_dim: int = 1024, null_text: bool = False):
        super().__init__()
        self.pose_dim = n_joints * rot_dim
        self.n_joints, self.rot_dim, self.expr_dim = n_joints, rot_dim, expr_dim
        self.text = TextEncoder(text_model, text_max_len) if text_model else None
        text_dim = self.text.dim if self.text is not None else text_dim
        self.inp = nn.Linear(self.pose_dim + expr_dim, latent)
        self.pos = nn.Parameter(torch.zeros(1, max_frames, latent))
        nn.init.normal_(self.pos, std=0.02)
        self.latent = latent
        self.time_mlp = nn.Sequential(nn.Linear(latent, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.layers = nn.ModuleList([DecoderLayer(latent, heads, ff, time_dim, text_dim, dropout)
                                     for _ in range(layers)])
        self.out_norm = nn.LayerNorm(latent)
        self.out = nn.Linear(latent, self.pose_dim + expr_dim)
        self.len_head = nn.Sequential(nn.Linear(text_dim, 512), nn.SiLU(), nn.Linear(512, 1))
        self.max_frames = max_frames
        # C7: learned "no text" token for classifier-free guidance (only when trained with text dropout,
        # so checkpoints of the paper configuration keep their exact parameter set)
        self.null_text = nn.Parameter(torch.zeros(1, 1, text_dim)) if null_text else None

    def drop_text(self, text, text_pad, drop):
        """Replace the text of samples with drop[b] = True by the single null token."""
        if self.null_text is None:
            raise RuntimeError("the generator was built without a null-text token (generator.p_uncond = 0)")
        drop = drop.to(text.device)
        null = self.null_text.to(text.dtype).expand(text.shape[0], text.shape[1], -1)
        text = torch.where(drop[:, None, None], null, text)
        null_pad = torch.ones_like(text_pad)
        null_pad[:, 0] = False  # one valid key: the null token
        return text, torch.where(drop[:, None], null_pad, text_pad)

    def encode_text(self, texts, srcs, device):
        return self.text(texts, srcs, device)

    def forward(self, x_t, e_t, t, text, text_pad, pad):
        """x_t [B,T,J,D], e_t [B,T,E], t [B], text [B,L,C]; pad/text_pad: True = padding."""
        B, T = x_t.shape[:2]
        h = self.inp(torch.cat([x_t.flatten(-2), e_t], -1)) + self.pos[:, :T]
        emb = self.time_mlp(timestep_embedding(t, self.latent))
        for layer in self.layers:
            h = layer(h, emb, pad, text, text_pad)
        o = self.out(self.out_norm(h))
        pose = o[..., :self.pose_dim].reshape(B, T, self.n_joints, self.rot_dim)
        return pose, o[..., self.pose_dim:]

    def predict_log_length(self, text, text_pad):
        keep = (~text_pad).float()[..., None]
        pooled = (text.detach() * keep).sum(1) / keep.sum(1).clamp_min(1)
        return self.len_head(pooled)[:, 0]

    @staticmethod
    def length_loss(pred_log_len, lengths):
        return F.l1_loss(pred_log_len, torch.log(lengths.float()))
