"""Score predicted 133-D (normalised) sequences against GT with the MaDiS crop rule.

Prediction sources (any number, each `label=path`):
  npz  : {<split>__<name>: [T,133]}            (e.g. refiner outputs; keys 'coarse::*' ignored)
  dir  : MaDiS SAVE_PREDICTIONS dir with <name>.pkl containing feats_rst
Reports accumulated DTW-JPE (paper unit) and per-step DTW (cm) for body and hands,
mean over clips present in *all* sources (paired comparison), plus hand-velocity
ratio vs GT (under-articulation indicator).

  cd third_party/MaDiS && .venv/bin/python /home/haipd/SLP/scripts/score_predictions.py --split dev \
      madis=/home/haipd/SLP/outputs/madis/.../val_rank_0 refined=/home/haipd/SLP/outputs/refiner/flow_v0/dev_madis.npz
"""

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "/home/haipd/SLP/scripts")
from madis_oracle_eval import Joints, feats133, scores  # noqa: E402
from decode_sign_tokens import madis_crop  # noqa: E402


def load_source(path, split):
    """Keys are bare clip names (no split prefix)."""
    p = Path(path)
    out = {}
    if p.is_dir():
        for f in p.glob("*.pkl"):
            out[f.stem] = pickle.load(open(f, "rb"))["feats_rst"].astype(np.float32)
    else:
        z = np.load(p)
        out = {k.split("__")[-1]: z[k] for k in z.files if not k.startswith("coarse::")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--source", default="phoenix", choices=["phoenix", "csl", "how2sign"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--stats", default="/home/haipd/SLP/third_party/SOKE/deps/stats")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mean = torch.load(f"{a.stats}/mean.pt", weights_only=True)[36:]
    std = torch.load(f"{a.stats}/std.pt", weights_only=True)[36:]
    mean = torch.cat([mean[:-20], mean[-10:]]).float()
    std = torch.cat([std[:-20], std[-10:]]).float()
    J = Joints(mean, std, "cuda")
    srcs = {s.split("=", 1)[0]: load_source(s.split("=", 1)[1], a.split) for s in a.sources if "=" in s}
    builtins = [s for s in a.sources if s.startswith("@")]  # @mean_pose, @gt_reversed (computed per clip)
    sys.path.insert(0, "/home/haipd/SLP/src")
    from slp.data.sign_smplx import LOADERS
    roots = {"phoenix": "Phoenix_2014T", "csl": "CSL-Daily", "how2sign": "How2Sign"}
    root = Path(a.root or f"/home/haipd/SLP/data/processed/madis/{roots[a.source]}")
    split = {"dev": "val"}.get(a.split, a.split)
    rows = LOADERS[a.source](root, split)
    files = {Path(r[0]["name"]).name: (r[1], r[2]) for r in rows}
    keys = list(files)
    common = [k for k in keys if all(k in s for s in srcs.values())]
    for b in builtins:
        srcs[b[1:]] = {}
    print({k: len(v) for k, v in srcs.items()}, "common", len(common), flush=True)
    agg = {lab: {} for lab in srcs}
    vel = {lab: [] for lab in srcs}
    for k in common:
        fpath, fps = files[k]
        raw = feats133(pickle.load(open(fpath, "rb")))
        if fps is not None and fps > 24:  # How2Sign: MaDiS subsamples to 24 fps
            raw = raw[np.linspace(0, len(raw) - 1, int(24 * len(raw) / fps), dtype=int)]
        gt = madis_crop((raw - mean.numpy()) / std.numpy()).astype(np.float32)
        ref = J(torch.from_numpy(gt))
        gv = (ref[2][1:] - ref[2][:-1]).norm(dim=-1).mean().item()
        if "mean_pose" in srcs:
            srcs["mean_pose"][k] = np.zeros_like(gt)
        if "gt_reversed" in srcs:
            srcs["gt_reversed"][k] = gt[::-1].copy()
        for lab, s in srcs.items():
            pred = J(torch.from_numpy(s[k]))
            for m, v in scores(pred, ref).items():
                agg[lab].setdefault(m, []).append(float(v))
            pv = (pred[2][1:] - pred[2][:-1]).norm(dim=-1).mean().item()
            vel[lab].append(pv / max(gv, 1e-6))
    res = {}
    for lab in srcs:
        r = {m: float(np.mean(v)) for m, v in agg[lab].items()}
        r["rhand_speed_ratio_vs_gt"] = float(np.median(vel[lab]))
        res[lab] = r
        print(f"{lab:14s} body {r['body']:.3f} hands {r['hands']:.3f} | per-step body {100*r['body_perframe']:.2f} "
              f"hands {100*r['hands_perframe']:.2f} cm | rh speed ratio {r['rhand_speed_ratio_vs_gt']:.2f}", flush=True)
    res["_meta"] = {"n": len(common), "split": a.split}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
