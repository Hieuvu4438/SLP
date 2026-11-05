"""FLARE C1 prototype: residual rectified-flow refiner  x~ (decoded sign tokens) -> x (continuous SMPL-X).

v0: no text conditioning (pure articulation refiner); text conditioning is an ablation
for later. Two modes:
  --mode flow     : rectified flow on the normalised residual  r = (x - x~) / s
  --mode regress  : deterministic L1 regression of the residual (over-smoothing baseline)

Training data: outputs/refiner_data/phoenix_train.npz from scripts/decode_sign_tokens.py
(GT-token decodes + corrupted-token decodes). Only the train split is used for fitting.
Inference writes refined dev sequences to an npz for scripts/madis_oracle_eval.py --pred_npz.

  python methods/d2c_sign/refiner.py train --mode flow --out outputs/refiner/flow_v0
  python methods/d2c_sign/refiner.py infer --ckpt outputs/refiner/flow_v0/model.pt \
         --coarse_npz outputs/refiner_data/phoenix_dev.npz --out outputs/refiner/flow_v0/dev_gt_tokens.npz
  python methods/d2c_sign/refiner.py infer --ckpt ... --madis_dir outputs/madis/<run>/val_rank_0 --out ...
"""

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Refiner(nn.Module):
    def __init__(self, d=384, layers=6, heads=6, max_len=512, nfeat=133):
        super().__init__()
        self.inp = nn.Linear(2 * nfeat, d)
        self.conv = nn.Conv1d(d, d, 5, padding=2)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        self.temb = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, nfeat))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    @staticmethod
    def tfeat(t):
        f = torch.exp(torch.linspace(0, math.log(1000), 32, device=t.device))
        a = t[:, None] * f[None] * 2 * math.pi
        return torch.cat([a.sin(), a.cos()], -1)

    def forward(self, xt, coarse, t, mask):
        h = self.inp(torch.cat([xt, coarse], -1))
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = h + self.pos[:, : h.shape[1]] + self.temb(self.tfeat(t))[:, None]
        h = self.enc(h, src_key_padding_mask=~mask)
        return self.out(h)


def load_pairs(path):
    z = np.load(path)
    gt, tok = {}, {}
    for k in z.files:
        if k.startswith("gt::"):
            gt[k[4:]] = z[k]
        else:
            _, name, v = k.split("::")
            tok.setdefault(name, {})[int(v)] = z[k]
    return gt, tok


def collate(items, dev):
    T = max(len(c) for c, _ in items)
    B = len(items)
    c = torch.zeros(B, T, 133); x = torch.zeros(B, T, 133); m = torch.zeros(B, T, dtype=torch.bool)
    for i, (ci, xi) in enumerate(items):
        c[i, : len(ci)] = torch.from_numpy(ci); x[i, : len(xi)] = torch.from_numpy(xi); m[i, : len(ci)] = True
    return c.to(dev), x.to(dev), m.to(dev)


def train(a):
    dev = "cuda"
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gt, tok = load_pairs(a.train_npz)
    names = sorted(gt)
    sg_gt, sg_tok, sg_names = {}, {}, []
    if a.selfgen_npz:  # input mix (c): MaDiS self-generated coarse + DTW-warped GT targets
        sg_gt, sg_tok = load_pairs(a.selfgen_npz)
        sg_names = sorted(sg_gt)
    res = np.concatenate([gt[n] - tok[n][0] for n in names])
    scale = res.std(0).astype(np.float32) + 1e-3  # per-dim residual scale (GT-token decodes)
    s = torch.from_numpy(scale).to(dev)
    model = Refiner().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.iters, pct_start=0.03)
    t0 = time.time()
    for it in range(1, a.iters + 1):
        batch = []
        for i in np.random.randint(0, len(names), a.bs):
            if sg_names and np.random.rand() < a.p_selfgen:
                n = sg_names[np.random.randint(len(sg_names))]
                c, x = sg_tok[n][0], sg_gt[n]
            else:
                n = names[i]
                v = 0 if np.random.rand() < a.p_clean else np.random.randint(1, len(tok[n]))
                c, x = tok[n][v], gt[n]
            if len(c) > a.win:
                st = np.random.randint(0, len(c) - a.win + 1)
                c, x = c[st:st + a.win], x[st:st + a.win]
            batch.append((c, x))
        c, x, m = collate(batch, dev)
        r1 = (x - c) / s
        if a.mode == "flow":
            t = torch.rand(len(c), device=dev)
            r0 = torch.randn_like(r1)
            rt = (1 - t[:, None, None]) * r0 + t[:, None, None] * r1
            pred = model(rt, c, t, m)
            loss = ((pred - (r1 - r0)) ** 2)[m].mean()
        else:
            pred = model(torch.zeros_like(r1), c, torch.zeros(len(c), device=dev), m)
            loss = (pred - r1).abs()[m].mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if it % 1000 == 0:
            print(json.dumps({"it": it, "loss": round(loss.item(), 4), "min": round((time.time() - t0) / 60, 1)}), flush=True)
    torch.save({"model": model.state_dict(), "scale": scale, "mode": a.mode}, out / "model.pt")
    json.dump(vars(a), open(out / "args.json", "w"), indent=2)


@torch.no_grad()
def infer(a):
    dev = "cuda"
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.manual_seed(a.seed)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model = Refiner().to(dev); model.load_state_dict(ck["model"]); model.eval()
    s = torch.from_numpy(ck["scale"]).to(dev)
    coarse = {}
    if a.coarse_npz:
        _, tok = load_pairs(a.coarse_npz)
        coarse = {n: v[0] for n, v in tok.items()}
    if a.plain_npz:  # {split__name: feats} (e.g. FLARE-R1 compositions)
        z = np.load(a.plain_npz)
        coarse.update({k: z[k].astype(np.float32) for k in z.files})
    if a.madis_dir:
        for p in sorted(Path(a.madis_dir).glob("*.pkl")):
            d = pickle.load(open(p, "rb"))
            coarse["dev__" + p.stem] = d["feats_rst"].astype(np.float32)
    outd = {}
    for n, c in coarse.items():
        ct = torch.from_numpy(c)[None].to(dev)
        m = torch.ones(1, len(c), dtype=torch.bool, device=dev)
        if ck["mode"] == "flow":
            r = torch.randn_like(ct) * a.temp
            for k in range(a.steps):
                t = torch.full((1,), k / a.steps, device=dev)
                r = r + model(r, ct, t, m) / a.steps
        else:
            r = model(torch.zeros_like(ct), ct, torch.zeros(1, device=dev), m)
        outd[n] = (ct + r * s)[0].cpu().numpy()
        outd["coarse::" + n] = c
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **outd)
    print("wrote", len(coarse), "sequences to", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "infer"])
    ap.add_argument("--mode", choices=["flow", "regress"], default="flow")
    ap.add_argument("--train_npz", default="outputs/refiner_data/phoenix_train.npz")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--win", type=int, default=160)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--p_clean", type=float, default=0.25)
    ap.add_argument("--selfgen_npz", default=None)
    ap.add_argument("--p_selfgen", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--coarse_npz")
    ap.add_argument("--madis_dir")
    ap.add_argument("--plain_npz")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else infer(a)


if __name__ == "__main__":
    main()
