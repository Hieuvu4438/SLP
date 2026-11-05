"""FLARE R5: fine-grained text<->motion aligner (FILIP / SiCLIP-style), trained on the TRAIN split only.

sim(text, motion) = mean_{text tokens t} max_{frames f} cos(T_t, M_f)   (and the symmetric term)
Loss: symmetric InfoNCE over the batch. Text: frozen mBART-large encoder (SOKE's pruned
en/zh/de checkpoint) + trainable projection. Motion: part-aware conv/transformer encoder
over the 133-D SOKE features (x2 temporal down-sampling).

Use: `locate(model, motion, text)` returns, per text token, the best-matching frame. The
span of the content tokens gives a learned segment for composition (replaces the
positional heuristic that failed in F39).

  third_party/MaDiS/.venv/bin/python methods/d2c_sign/aligner.py train --source how2sign --out outputs/aligner/h2s
  third_party/MaDiS/.venv/bin/python methods/d2c_sign/aligner.py eval  --source how2sign --ckpt outputs/aligner/h2s/model.pt
"""

import argparse
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/haipd/SLP/src")
from slp.data.sign_smplx import LOADERS, feats133, load_stats, madis_crop  # noqa: E402

MBART = "/home/haipd/SLP/third_party/SOKE/deps/mbart-h2s-csl-phoenix"
ROOTS = {"phoenix": "Phoenix_2014T", "csl": "CSL-Daily", "how2sign": "How2Sign"}
LANG = {"phoenix": "de_DE", "csl": "zh_CN", "how2sign": "en_XX"}


class MotionEnc(nn.Module):
    def __init__(self, d=512, layers=4):
        super().__init__()
        self.body, self.lh, self.rh = nn.Linear(43, d // 4), nn.Linear(45, 3 * d // 8), nn.Linear(45, 3 * d // 8)
        self.conv = nn.Sequential(nn.Conv1d(d, d, 5, 2, 2), nn.GELU(), nn.Conv1d(d, d, 3, 1, 1))
        self.pos = nn.Parameter(torch.zeros(1, 256, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True), layers)
        self.proj = nn.Linear(d, 256)

    def forward(self, x, m):
        h = torch.cat([self.body(torch.cat([x[..., :30], x[..., 120:]], -1)), self.lh(x[..., 30:75]), self.rh(x[..., 75:120])], -1)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        m = m[:, ::2][:, : h.shape[1]]
        h = self.enc(h + self.pos[:, : h.shape[1]], src_key_padding_mask=~m)
        return F.normalize(self.proj(h), dim=-1), m


class Aligner(nn.Module):
    def __init__(self, lang):
        super().__init__()
        from transformers import MBartModel, MBartTokenizer
        self.tok = MBartTokenizer.from_pretrained(MBART, legacy=True)
        self.map = pickle.load(open(f"{MBART}/map_ids.pkl", "rb"))
        self.text = MBartModel.from_pretrained(MBART).encoder.eval().requires_grad_(False)
        self.tproj = nn.Sequential(nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, 256))
        self.menc = MotionEnc()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.lang = lang

    def encode_text(self, texts, dev):
        enc = self.tok(texts, padding=True, truncation=True, max_length=48, return_tensors="pt")
        ids = enc.input_ids.apply_(lambda t: self.map.get(t, self.map[self.tok.unk_token_id])).to(dev)
        am = enc.attention_mask.to(dev)
        special = (enc.input_ids <= 3).to(dev) | (enc.input_ids >= self.tok.convert_tokens_to_ids("ar_AR")).to(dev)
        with torch.no_grad():
            h = self.text(input_ids=ids, attention_mask=am).last_hidden_state
        return F.normalize(self.tproj(h), dim=-1), (am.bool() & ~special), enc.input_ids

    def token_frame_sim(self, T, M):  # T [Bt,L,d], M [Bm,F,d] -> [Bt,Bm,L,F]
        return torch.einsum("tld,mfd->tmlf", T, M)

    def pair_logits(self, T, tm, M, mm):
        s = self.token_frame_sim(T, M).masked_fill(~mm[None, :, None, :], -1e4)
        t2m = (s.max(-1).values * tm[:, None]).sum(-1) / tm.sum(-1, keepdim=True).clamp_min(1)
        s2 = self.token_frame_sim(T, M).masked_fill(~tm[:, None, :, None], -1e4)
        m2t = (s2.max(-2).values * mm[None]).sum(-1) / mm.sum(-1)[None].clamp_min(1)
        return self.logit_scale.exp().clamp(max=100) * (t2m + m2t) / 2


def load(source, split, mean, std, limit=-1):
    out = []
    for info, path, fps in LOADERS[source](Path(f"/home/haipd/SLP/data/processed/madis/{ROOTS[source]}"), split):
        f = feats133(pickle.load(open(path, "rb")))
        if fps and fps > 24:
            f = f[np.linspace(0, len(f) - 1, int(24 * len(f) / fps), dtype=int)]
        out.append((info["text"], madis_crop((f - mean) / std).astype(np.float32)))
        if 0 < limit <= len(out):
            break
    return out


def batchify(items, dev):
    T = max(len(f) for _, f in items)
    x = torch.zeros(len(items), T, 133); m = torch.zeros(len(items), T, dtype=torch.bool)
    for i, (_, f) in enumerate(items):
        x[i, : len(f)] = torch.from_numpy(f); m[i, : len(f)] = True
    return [t for t, _ in items], x.to(dev), m.to(dev)


def train(a):
    dev = "cuda"; torch.manual_seed(a.seed); random.seed(a.seed)
    torch.backends.cuda.enable_cudnn_sdp(False)
    mean, std = load_stats("/home/haipd/SLP/third_party/SOKE/deps/stats")
    tr = load(a.source, "train", mean, std, a.limit)
    model = Aligner(LANG[a.source]).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for ep in range(a.epochs):
        random.shuffle(tr); tot = 0
        for s in range(0, len(tr) - a.bs + 1, a.bs):
            texts, x, m = batchify(tr[s:s + a.bs], dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                T, tm, _ = model.encode_text(texts, dev)
                M, mm = model.menc(x, m)
                logits = model.pair_logits(T.float(), tm, M.float(), mm)
            lab = torch.arange(len(texts), device=dev)
            loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab)) / 2
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            tot += loss.item()
        print(json.dumps({"epoch": ep + 1, "loss": round(tot / max(1, len(tr) // a.bs), 4), "min": round((time.time() - t0) / 60, 1)}), flush=True)
        torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("text.")}, out / "model.pt")


@torch.no_grad()
def evaluate(a):
    """Text->motion retrieval R@1/5 on val within random pools of 100 (sanity check of the aligner)."""
    dev = "cuda"; torch.backends.cuda.enable_cudnn_sdp(False)
    mean, std = load_stats("/home/haipd/SLP/third_party/SOKE/deps/stats")
    va = load(a.source, "val", mean, std)
    model = Aligner(LANG[a.source]).to(dev).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location=dev), strict=False)
    rng = np.random.default_rng(0); r1 = r5 = n = 0
    for _ in range(10):
        idx = rng.choice(len(va), 100, replace=False); items = [va[i] for i in idx]
        texts, x, m = batchify(items, dev)
        T, tm, _ = model.encode_text(texts, dev); M, mm = model.menc(x, m)
        L = model.pair_logits(T.float(), tm, M.float(), mm)
        rank = (L > L.diag()[:, None]).sum(1)
        r1 += (rank < 1).sum().item(); r5 += (rank < 5).sum().item(); n += len(items)
    print(json.dumps({"t2m_R@1_pool100": 100 * r1 / n, "t2m_R@5_pool100": 100 * r5 / n, "chance_R@1": 1.0}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--source", default="how2sign")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/aligner/h2s")
    ap.add_argument("--ckpt", default="outputs/aligner/h2s/model.pt")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()
