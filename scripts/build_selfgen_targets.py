"""Timing-preserving refiner targets from MaDiS self-generated outputs (FLARE C1, input mix (c)).

For each TRAIN clip: coarse = MaDiS feats_rst [Tg,133]; the GT [Tr,133] is DTW-aligned
(L2 in normalised feature space, body+hands weighted) onto the generated time axis,
and target[t] = mean of the GT frames matched to generated frame t. Output is in the
refiner npz format: gt::<key> (warped target, length Tg) and tok::<key>::0 (coarse).

  python scripts/build_selfgen_targets.py --gen outputs/madis_gen/phoenix_train \
      --out outputs/refiner_data/phoenix_train_selfgen.npz
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/haipd/SLP/src")
from slp.data.sign_smplx import LOADERS, feats133, load_stats, madis_crop  # noqa: E402


def dtw_path(cost):
    r, c = cost.shape
    acc = np.full((r + 1, c + 1), np.inf)
    acc[0, 0] = 0.0
    for d in range(2, r + c + 1):
        i = np.arange(max(1, d - c), min(r, d - 1) + 1)
        j = d - i
        acc[i, j] = cost[i - 1, j - 1] + np.minimum(np.minimum(acc[i - 1, j - 1], acc[i - 1, j]), acc[i, j - 1])
    i, j, path = r, c, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else ((i - 1, j) if k == 1 else (i, j - 1))
    return path[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--source", default="phoenix")
    ap.add_argument("--root", default="data/processed/madis/Phoenix_2014T")
    ap.add_argument("--stats", default="third_party/SOKE/deps/stats")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mean, std = load_stats(a.stats)
    rows = {Path(r[0]["name"]).name: r for r in LOADERS[a.source](Path(a.root), "train")}
    w = np.ones(133, dtype=np.float32); w[30:120] = 2.0  # emphasise hands in the alignment cost
    out, n = {}, 0
    for f in sorted(Path(a.gen).glob("*.pkl")):
        if f.stem not in rows:
            continue
        g = pickle.load(open(f, "rb"))["feats_rst"].astype(np.float32)
        gt = madis_crop((feats133(pickle.load(open(rows[f.stem][1], "rb"))) - mean) / std).astype(np.float32)
        cost = np.sqrt((((g[:, None] - gt[None]) ** 2) * w).sum(-1))
        tgt = np.zeros_like(g); cnt = np.zeros(len(g))
        for i, j in dtw_path(cost):
            tgt[i] += gt[j]; cnt[i] += 1
        key = f"train__{f.stem}"
        out[f"gt::{key}"] = (tgt / cnt[:, None]).astype(np.float32)
        out[f"tok::{key}::0"] = g
        n += 1
    np.savez(a.out, **out)
    print("clips", n)


if __name__ == "__main__":
    main()
