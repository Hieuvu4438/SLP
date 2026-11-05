"""Pilot: how much of the hand quantization floor can a residual tokenizer remove?

Trains a right-hand tokenizer on Phoenix-2014T train (SOKE SMPL-X fittings,
SOKE normalisation) with either a single EMA VQ (SOKE-like: 192 codes, x4
temporal down-sampling) or a residual VQ (RVQ, MoMask-style, quantizer dropout),
and dumps dev reconstructions for the shared DTW-JPE evaluation
(`scripts/madis_oracle_eval.py --recon_npz`). Uses train for fitting and dev
only for reporting; never touches test.

  python methods/d2c_sign/tokenizer_pilot.py --quant vq  --out outputs/d2c_pilot/vq192
  python methods/d2c_sign/tokenizer_pilot.py --quant rvq --nq 4 --out outputs/d2c_pilot/rvq4
"""

import argparse
import gzip
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KEYS = ("smplx_root_pose", "smplx_body_pose", "smplx_lhand_pose",
        "smplx_rhand_pose", "smplx_jaw_pose", "smplx_shape", "smplx_expr")
SL = slice(75, 120)  # right hand in the 133-D SOKE feature


def feats133(p):
    f = np.concatenate([p[k] for k in KEYS], 1)[:, 36:]
    return np.concatenate([f[:, :-20], f[:, -10:]], 1).astype(np.float32)


def load_split(root, split, mean, std):
    ann = pickle.load(gzip.open(root / f"phoenix14t.{split}", "rb"))
    out = []
    for a in ann:
        p = root / "poses_video_level" / f"{a['name']}.pkl"
        if p.exists():
            f = feats133(pickle.load(open(p, "rb")))
            if 40 <= len(f) <= 400:
                out.append((a["name"], ((f - mean) / std)[:, SL]))
    return out


# ---------------------------------------------------------------- model
class ResBlock(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(), nn.Conv1d(c, c, 3, padding=dil, dilation=dil),
                                 nn.ReLU(), nn.Conv1d(c, c, 1))

    def forward(self, x):
        return x + self.net(x)


def stack(c, depth=3, rate=3):
    return nn.Sequential(*[ResBlock(c, rate ** d) for d in reversed(range(depth))])


class Enc(nn.Module):
    def __init__(self, nin, w=512, down=2):
        super().__init__()
        layers = [nn.Conv1d(nin, w, 3, padding=1), nn.ReLU()]
        for _ in range(down):
            layers += [nn.Conv1d(w, w, 4, 2, 1), stack(w)]
        layers += [nn.Conv1d(w, w, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Dec(nn.Module):
    def __init__(self, nout, w=512, down=2):
        super().__init__()
        layers = [nn.Conv1d(w, w, 3, padding=1), nn.ReLU()]
        for _ in range(down):
            layers += [stack(w), nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv1d(w, w, 3, padding=1)]
        layers += [nn.Conv1d(w, w, 3, padding=1), nn.ReLU(), nn.Conv1d(w, nout, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EMAVQ(nn.Module):
    """EMA codebook with dead-code reset (as in T2M-GPT / SOKE)."""

    def __init__(self, n, d, mu=0.99):
        super().__init__()
        self.n, self.d, self.mu = n, d, mu
        self.register_buffer("cb", torch.zeros(n, d))
        self.register_buffer("sum", torch.zeros(n, d))
        self.register_buffer("cnt", torch.ones(n))
        self.register_buffer("init", torch.zeros(()))

    def forward(self, z):  # z [N,d]
        if self.training and self.init == 0:
            idx = torch.randint(0, z.shape[0], (self.n,), device=z.device)
            self.cb.copy_(z[idx].detach()); self.sum.copy_(self.cb); self.init.fill_(1)
        d = (z.pow(2).sum(1, keepdim=True) - 2 * z @ self.cb.t() + self.cb.pow(2).sum(1)[None])
        code = d.argmin(1)
        q = self.cb[code]
        if self.training:
            with torch.no_grad():
                oh = F.one_hot(code, self.n).float()
                c = oh.sum(0)
                s = oh.t() @ z
                self.sum.mul_(self.mu).add_(s, alpha=1 - self.mu)
                self.cnt.mul_(self.mu).add_(c, alpha=1 - self.mu)
                usage = (self.cnt >= 1.0).float()[:, None]
                new = self.sum / self.cnt[:, None].clamp_min(1e-6)
                rnd = z[torch.randint(0, z.shape[0], (self.n,), device=z.device)]
                self.cb.copy_(usage * new + (1 - usage) * rnd)
        commit = F.mse_loss(z, q.detach())
        q = z + (q - z).detach()
        return q, code, commit


class Tokenizer(nn.Module):
    def __init__(self, nin=45, n_codes=192, nq=1, qdrop=0.2):
        super().__init__()
        self.enc, self.dec = Enc(nin), Dec(nin)
        self.qs = nn.ModuleList([EMAVQ(n_codes, 512) for _ in range(nq)])
        self.qdrop = qdrop

    def quantize(self, z, nq_use=None):
        B, C, T = z.shape
        r = z.permute(0, 2, 1).reshape(-1, C)
        nq = len(self.qs) if nq_use is None else nq_use
        if self.training and len(self.qs) > 1 and np.random.rand() < self.qdrop:
            nq = np.random.randint(1, len(self.qs) + 1)
        out, codes, commit = torch.zeros_like(r), [], 0.0
        for q in self.qs[:nq]:
            qq, c, cm = q(r - out)
            out = out + qq
            codes.append(c)
            commit = commit + cm
        return out.reshape(B, T, C).permute(0, 2, 1), codes, commit

    def forward(self, x, nq_use=None):  # x [B,T,45]
        z = self.enc(x.transpose(1, 2))
        q, codes, commit = self.quantize(z, nq_use)
        return self.dec(q).transpose(1, 2), codes, commit


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/madis/Phoenix_2014T")
    ap.add_argument("--stats", default="third_party/SOKE/deps/stats")
    ap.add_argument("--quant", choices=["vq", "rvq"], default="vq")
    ap.add_argument("--nq", type=int, default=1)
    ap.add_argument("--codes", type=int, default=192)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--win", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    mean = torch.load(f"{a.stats}/mean.pt", weights_only=True)[36:]
    std = torch.load(f"{a.stats}/std.pt", weights_only=True)[36:]
    mean = torch.cat([mean[:-20], mean[-10:]]).float().numpy()
    std = torch.cat([std[:-20], std[-10:]]).float().numpy()
    root = Path(a.root)
    train = load_split(root, "train", mean, std)
    val = load_split(root, "dev", mean, std)
    print(f"train {len(train)} dev {len(val)}", flush=True)
    seqs = [torch.from_numpy(x) for _, x in train if len(x) >= a.win]
    lens = np.array([len(s) for s in seqs])

    nq = 1 if a.quant == "vq" else a.nq
    model = Tokenizer(n_codes=a.codes, nq=nq).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters, eta_min=1e-6)
    t0 = time.time()
    for it in range(1, a.iters + 1):
        idx = np.random.randint(0, len(seqs), a.bs)
        batch = []
        for i in idx:
            st = np.random.randint(0, lens[i] - a.win + 1)
            batch.append(seqs[i][st:st + a.win])
        x = torch.stack(batch).to(dev)
        model.train()
        rec, _, commit = model(x)
        vel = F.smooth_l1_loss(rec[:, 1:] - rec[:, :-1], x[:, 1:] - x[:, :-1])
        loss = F.smooth_l1_loss(rec, x) + 0.02 * commit + 0.5 * vel
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 1000 == 0:
            print(f"it {it} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)

    model.eval()
    recs, usage = {}, [set() for _ in range(nq)]
    with torch.no_grad():
        for name, x in val:
            T = (len(x) // 4) * 4
            xx = torch.from_numpy(x[:T])[None].to(dev)
            r, codes, _ = model(xx)
            for k, c in enumerate(codes):
                usage[k].update(c.tolist())
            recs[name] = r[0].cpu().numpy()
    np.savez_compressed(out / "dev_rhand_recon.npz", **{k.replace("/", "__"): v for k, v in recs.items()})
    meta = vars(a) | {"usage_per_layer": [len(u) for u in usage],
                      "bits_per_token": float(nq * np.log2(a.codes)),
                      "train_minutes": (time.time() - t0) / 60}
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    torch.save(model.state_dict(), out / "tokenizer.pt")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
