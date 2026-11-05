"""Decode (optionally corrupted) SOKE/MaDiS part tokens into coarse 133-D motion.

Produces the coarse inputs x~ = D(c) for the FLARE C1 refiner, aligned with the GT
features using MaDiS's own crop rule (resample to [40,400], centre-crop to a multiple
of 4). Output npz per split: {name: coarse [T,133]} and {name: gt [T,133]} (normalised).

  cd third_party/MaDiS && .venv/bin/python /home/haipd/SLP/scripts/decode_sign_tokens.py \
      --split train --variants 3 --corrupt 0.3 --out /home/haipd/SLP/outputs/refiner_data/phoenix_train.npz
"""

import argparse
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "/home/haipd/SLP/scripts")
from madis_oracle_eval import feats133, load_vae  # noqa: E402

TOK = "/home/haipd/SLP/third_party/MaDiS/hf_extract/tokens/TOKENS_h2s_csl_phoenix/phoenix"


def madis_crop(f, lo=40, hi=400, unit=4):
    n = len(f)
    if n < lo:
        return f[np.linspace(0, n - 1, lo, dtype=int)]
    if n > hi:
        return f[np.linspace(0, n - 1, hi, dtype=int)]
    u = (n // unit) * unit
    s = (n - u) // 2
    return f[s:s + u]


def corrupt(codes, p, sizes, rng):
    c = codes.copy()
    for k, n in enumerate(sizes):
        m = rng.random(len(c)) < p
        c[m, k] = rng.integers(0, n, m.sum())
    # part-wise temporal jitter of +-1 token on the hands
    for k in (1, 2):
        if rng.random() < 0.5:
            sh = rng.choice([-1, 1])
            c[:, k] = np.roll(c[:, k], sh)
    return c


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/haipd/SLP/data/processed/madis/Phoenix_2014T")
    ap.add_argument("--split", default="train")
    ap.add_argument("--tokenizer", default="/home/haipd/SLP/third_party/SOKE/checkpoints/tokenizer.ckpt")
    ap.add_argument("--stats", default="/home/haipd/SLP/third_party/SOKE/deps/stats")
    ap.add_argument("--variants", type=int, default=0, help="extra corrupted variants per clip")
    ap.add_argument("--corrupt", type=float, default=0.3, help="max per-token replacement prob")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reencode", action="store_true",
                    help="encode GT locally instead of using released tokens (released Phoenix-dev tokens "
                         "do not match the SOKE dev fittings; train/CSL match 100%%)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    mean = torch.load(f"{a.stats}/mean.pt", weights_only=True)[36:]
    std = torch.load(f"{a.stats}/std.pt", weights_only=True)[36:]
    mean = torch.cat([mean[:-20], mean[-10:]]).float().numpy()
    std = torch.cat([std[:-20], std[-10:]]).float().numpy()
    body, lh, rh = [m.to(dev) for m in load_vae(a.tokenizer)]

    def dec(c):
        c = torch.from_numpy(c).to(dev)
        b = body.decode(c[None, :, 0])[0]
        l_ = lh.decode(c[None, :, 1])[0]
        r = rh.decode(c[None, :, 2])[0]
        return torch.cat([b[:, :30], l_, r, b[:, 30:]], -1).cpu().numpy().astype(np.float32)

    ann = pickle.load(gzip.open(Path(a.root) / f"phoenix14t.{a.split}", "rb"))
    out, n = {}, 0
    for x in ann:
        name = x["name"]
        p = Path(a.root) / "poses_video_level" / f"{name}.pkl"
        tf = Path(TOK) / a.split / f"{Path(name).name}.npy"
        if not (p.exists() and tf.exists()):
            continue
        gt = madis_crop((feats133(pickle.load(open(p, "rb"))) - mean) / std).astype(np.float32)
        if a.reencode:
            x = torch.from_numpy(gt)[None].to(dev)
            cb, _ = body.encode(torch.cat([x[..., :30], x[..., 120:]], -1))
            cl, _ = lh.encode(x[..., 30:75]); cr, _ = rh.encode(x[..., 75:120])
            codes = torch.stack([cb[0], cl[0], cr[0]], -1).cpu().numpy()
        else:
            codes = np.load(tf)[0]
        if len(codes) * 4 != len(gt):
            continue
        key = name.replace("/", "__")
        out[f"gt::{key}"] = gt
        out[f"tok::{key}::0"] = dec(codes)
        for v in range(1, a.variants + 1):
            out[f"tok::{key}::{v}"] = dec(corrupt(codes, rng.uniform(0, a.corrupt), (96, 192, 192), rng))
        n += 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(a.split, "clips", n, "of", len(ann))


if __name__ == "__main__":
    main()
