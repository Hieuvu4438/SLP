"""R7 v1 training data (content-consistent): input = the sentence's OWN motion with composition-style
corruptions (random split, per-part time-warp, signer-style mean-pose offset from another train clip,
global retime); target = original motion DTW-warped onto the corrupted timeline. Refiner npz format."""
import argparse, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/haipd/SLP/src"); sys.path.insert(0, "/home/haipd/SLP/scripts")
from slp.data.sign_smplx import LOADERS, feats133, load_stats, madis_crop
from build_selfgen_targets import dtw_path
ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=12000); ap.add_argument("--out"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); rng = np.random.default_rng(a.seed)
root = Path("/home/haipd/SLP/data/processed/madis/How2Sign"); mean, std = load_stats("/home/haipd/SLP/third_party/SOKE/deps/stats")
tr = LOADERS["how2sign"](root, "train"); idx = rng.choice(len(tr), min(a.limit, len(tr)), replace=False)
def motion(j):
    t = tr[j]; f = feats133(pickle.load(open(t[1], "rb")))
    if t[2] and t[2] > 24: f = f[np.linspace(0, len(f) - 1, int(24 * len(f) / t[2]), dtype=int)]
    return madis_crop((f - mean) / std).astype(np.float32)
def warp(x, r):
    L = max(4, int(len(x) * r)); return x[np.linspace(0, len(x) - 1, L).round().astype(int)]
w = np.ones(133, dtype=np.float32); w[30:120] = 2.0; out = {}; n = 0
for i in idx:
    gt = motion(i)
    if len(gt) < 16: continue
    other = motion(int(rng.integers(len(tr))))
    off = (other.mean(0) - gt.mean(0)) * rng.uniform(0.3, 1.0)          # signer-style offset
    cut = int(rng.integers(len(gt) // 4, 3 * len(gt) // 4))
    c = np.concatenate([warp(gt[:cut], rng.uniform(0.7, 1.4)), warp(gt[cut:], rng.uniform(0.7, 1.4))]) + off
    c = warp(c, rng.uniform(0.8, 1.25)).astype(np.float32)
    cost = np.sqrt((((c[:, None] - gt[None]) ** 2) * w).sum(-1))
    tgt = np.zeros_like(c); cnt = np.zeros(len(c))
    for p, q in dtw_path(cost): tgt[p] += gt[q]; cnt[p] += 1
    key = f"train__{Path(tr[i][0]['name']).name}"
    out[f"gt::{key}"] = (tgt / cnt[:, None]).astype(np.float32); out[f"tok::{key}::0"] = c; n += 1
np.savez(a.out, **out); print("clips", n)
