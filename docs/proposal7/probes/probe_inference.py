"""Inference-only experiments (docs/proposal7 E0.1-E0.3) on the trained v2 SignNMFlow model.

No training, CPU only, Phoenix dev (val) only, never the test split. Every variant uses the same
fixed output lengths (taken from the v2 run's saved predictions), the same dev sequences and the
same sampler noise (paired design); differences are tested with a paired per-sequence bootstrap.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONPATH=.:src python docs/proposal7/probes/probe_inference.py \
        --n 100 --variants base_s0,base_s1,ep100,ep150,nfe1,nfe10,geoshift,grid_p0.7,grid_p1.5

Variants (all modify only the sampler / checkpoint choice, i.e. methods/signmflow flow.sample):
  base_s{k} : epoch-300 generator, uniform 5-step grid, naturalness path (= flow.sample), sampler seed k
  ep{E}     : generator_ep{E}.pt instead of epoch 300 (checkpoint selection, E0.1)
  nfe{N}    : N uniform steps (E0.2)
  geoshift  : at inference replace the naturalness path by the geodesic with SD3 shift s=2.55 (E0.3)
  grid_p{p} : non-uniform time grid t_k = (k/N)^p (p<1: fewer steps at high noise; p>1: more)
  shrink{a} : text-free control, base predictions scaled by (1-a) in normalised feature space (towards the
              mean pose); used to ask whether a DTW gain beats plain mean-regression at equal motion speed
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from methods.signmflow import geometry as G
from methods.signmflow.model import SignGenerator
from methods.signmflow.paths import CorrectionNet, path_point
from methods.signmflow.representation import MotionCodec, load_hand_mean
from slp.data.sign_smplx import load_stats
from slp.evaluation.metrics import dtw_cost, pairwise_part_cost
from slp.evaluation.smplx_fk import UPPER_BODY_IDX, default_fk

torch.set_grad_enabled(False)
ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "outputs/runs/signmflow_v2_main_phoenix_smplx/signmflow/seed0_20261116-185137-965824"
KEYS = ("jpe_body", "jpe_hand", "pa_body_ua", "pa_hand")


def load_gen(epoch: int) -> SignGenerator:
    gen = SignGenerator(rot_dim=4, latent=512, layers=8, heads=8, ff=2048, time_dim=2048, dropout=0.1,
                        max_frames=400, text_model=str(ROOT / "third_party/SOKE/deps/mbart-h2s-csl-phoenix"),
                        text_max_len=128)
    st = torch.load(RUN / f"checkpoints/generator_ep{epoch}.pt", map_location="cpu", weights_only=False)
    gen.load_state_dict(st["model"])
    return gen.eval()


def geoshift_point(x0, x1, t, s):
    n = s * (1 - t) / (1 + (s - 1) * (1 - t))
    return G.expmap(x0, (1 - n) * G.logmap(x0, x1))


def sample(gen, corr, text, pad, lengths, grid, path, s, seed):
    """Re-planned sampler of Alg. 2 / flow.sample with an arbitrary time grid and path."""
    B, T = len(lengths), int(max(lengths))
    g = torch.Generator().manual_seed(seed)
    mask = torch.arange(T)[None] < torch.as_tensor(lengths)[:, None]
    x0 = G.uniform_hemisphere((B, T, 41), generator=g)
    e = torch.randn(B, T, 10, generator=g)
    x = x0
    for k in range(len(grid) - 1):
        t, tn = float(grid[k]), float(grid[k + 1])
        pose, expr = gen(x, e, torch.full((B,), t), text, pad, ~mask)
        x1h = G.normalize(pose)
        tt = torch.full((B, 1, 1, 1), tn)
        x = path_point("naturalness", x0, x1h, tt, corr) if path == "naturalness" else geoshift_point(x0, x1h, tt, s)
        e = e + (tn - t) / (1 - t) * (expr - e)
        x = G.normalize(x)
    return x, e


def scores(preds, gts, fk, cache):
    out = {k: [] for k in KEYS}
    ub = list(UPPER_BODY_IDX)
    for i, (p, t) in enumerate(zip(preds, gts)):
        gj, gl, gr = (a.cpu().numpy() for a in fk(p))
        if i not in cache:
            cache[i] = tuple(a.cpu().numpy() for a in fk(t))
        rj, rl, rr = cache[i]
        out["jpe_body"].append(dtw_cost(pairwise_part_cost(gj, rj, UPPER_BODY_IDX, "root"))[0])
        out["jpe_hand"].append((dtw_cost(pairwise_part_cost(gl, rl, None, "root"))[0]
                                + dtw_cost(pairwise_part_cost(gr, rr, None, "root"))[0]) / 2)
        out["pa_hand"].append((dtw_cost(pairwise_part_cost(gl, rl, None, "procrustes"))[0]
                               + dtw_cost(pairwise_part_cost(gr, rr, None, "procrustes"))[0]) / 2)
        out["pa_body_ua"].append(dtw_cost(pairwise_part_cost(gj[:, ub], rj[:, ub], None, "procrustes"))[0])
    return {k: np.asarray(v) for k, v in out.items()}


def lowpass_speed(seqs, fk, sigma=2.0):
    """mean per-frame speed (mm/frame) of pelvis-relative body joints and wrist-relative fingers,
    after Gaussian temporal low-pass (removes fitting jitter, see P6)."""
    b, h = [], []
    for x in seqs:
        j, l, r = (a.cpu().numpy() for a in fk(x))
        body = gaussian_filter1d(j[:, :22] - j[:, :1], sigma, axis=0, mode="nearest")
        hh = gaussian_filter1d(np.concatenate([l - l[:, :1], r - r[:, :1]], 1), sigma, axis=0, mode="nearest")
        b.append(np.linalg.norm(np.diff(body, axis=0), axis=-1).mean())
        h.append(np.linalg.norm(np.diff(hh, axis=0), axis=-1).mean())
    return 1000 * float(np.mean(b)), 1000 * float(np.mean(h))


def paired_bootstrap(a, b, n_boot=4000, seed=0):
    """mean(a - b) and 95% CI over sequences (a, b per-sequence scores; lower is better)."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def parse(v):
    cfg = {"epoch": 300, "steps": 5, "path": "naturalness", "s": 2.55, "p": 1.0, "seed": 0, "shrink": 0.0}
    if "+" in v:  # combination, e.g. ep150+grid_p1.5
        for part in v.split("+"):
            sub = parse(part)
            cfg.update({k: sub[k] for k in sub if sub[k] != parse("base_s0")[k]})
    elif v.startswith("base_s"):
        cfg["seed"] = int(v[6:])
    elif v.startswith("ep"):
        cfg["epoch"] = int(v[2:])
    elif v.startswith("nfe"):
        cfg["steps"] = int(v[3:])
    elif v == "geoshift":
        cfg["path"] = "geoshift"
    elif v.startswith("shrink"):  # text-free control: pull base predictions towards the mean pose
        cfg["shrink"] = float(v[6:])
    elif v.startswith("grid_p"):
        cfg["p"] = float(v[6:])
    else:
        raise ValueError(v)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--variants", default="base_s0,base_s1,ep100,ep150,nfe1,nfe10,geoshift,grid_p0.7,grid_p1.5")
    ap.add_argument("--bs", type=int, default=20)
    ap.add_argument("--out", default=str(Path(__file__).with_name("probe_inference_phoenix_smplx.json")))
    a = ap.parse_args()

    f = sorted(glob.glob(str(ROOT / "outputs/signmflow/cache/phoenix_smplx_val_*.npy")))[0]
    arr = np.load(f, mmap_mode="r")
    meta = json.load(open(f.replace(".npy", ".json")))
    offs, inputs = meta["offsets"], meta["inputs"][: a.n]
    gts = [np.array(arr[offs[i]:offs[i + 1]]) for i in range(len(inputs))]
    saved = np.load(RUN / "predictions.npz")
    lengths = [len(saved[x["name"].replace("/", "__")]) for x in inputs]   # fixed lengths (E0.0)

    mean, std = load_stats(str(ROOT / "third_party/SOKE/deps/stats"))
    codec = MotionCodec(mean, std, load_hand_mean(ROOT / "third_party/SOKE/deps/smpl_models"), "quat")
    corr = CorrectionNet()
    corr.load_state_dict(torch.load(RUN / "checkpoints/correction_final.pt", map_location="cpu", weights_only=False)["model"])
    corr.eval()
    fk = default_fk("cpu")
    cache = {}

    out_path = Path(a.out)
    res = json.load(open(out_path)) if out_path.exists() else {"n_sequences": len(inputs), "variants": {}}
    if "reference" not in res:
        mp = scores([np.zeros((L, 133), np.float32) for L in lengths], gts, fk, cache)
        res["reference"] = {"mean_pose_fixed_len": {k: float(v.mean()) for k, v in mp.items()},
                            "mean_pose_per_seq": {k: v.tolist() for k, v in mp.items()},
                            "gt_lowpass_speed_mm": lowpass_speed(gts, fk)}
    gens = {}
    for v in a.variants.split(","):
        if v in res["variants"]:
            print("skip", v); continue
        cfg = parse(v)
        t0 = time.time()
        if cfg["epoch"] not in gens:
            gens = {cfg["epoch"]: load_gen(cfg["epoch"])}
        gen = gens[cfg["epoch"]]
        grid = [(k / cfg["steps"]) ** cfg["p"] for k in range(cfg["steps"] + 1)]
        preds = [None] * len(inputs)
        order = np.argsort(lengths)
        for s0 in range(0, len(order), a.bs):
            idx = order[s0:s0 + a.bs].tolist()
            text, pad = gen.encode_text([inputs[i]["text"] for i in idx], [inputs[i]["src"] for i in idx], "cpu")
            L = [lengths[i] for i in idx]
            rot, expr = sample(gen, corr, text, pad, L, grid, cfg["path"], cfg["s"], cfg["seed"] * 1000 + s0)
            feats = codec.decode(rot, expr).float().numpy() * (1.0 - cfg["shrink"])
            for j, i in enumerate(idx):
                preds[i] = feats[j, :L[j]]
        t1 = time.time()
        sc = scores(preds, gts, fk, cache)
        res["variants"][v] = {"cfg": cfg, "mean": {k: float(x.mean()) for k, x in sc.items()},
                              "per_seq": {k: x.tolist() for k, x in sc.items()},
                              "lowpass_speed_mm": lowpass_speed(preds, fk),
                              "sec_sample": t1 - t0, "sec_eval": time.time() - t1}
        print(v, res["variants"][v]["mean"], res["variants"][v]["lowpass_speed_mm"], flush=True)
        json.dump(res, open(out_path, "w"))
    # paired comparisons against base_s0
    if "base_s0" in res["variants"]:
        base = res["variants"]["base_s0"]["per_seq"]
        comp = {}
        for v, r in res["variants"].items():
            if v == "base_s0":
                continue
            comp[v] = {k: paired_bootstrap(r["per_seq"][k], base[k]) for k in KEYS}
        comp["mean_pose"] = {k: paired_bootstrap(res["reference"]["mean_pose_per_seq"][k], base[k]) for k in KEYS}
        res["paired_vs_base_s0"] = comp
        json.dump(res, open(out_path, "w"))
        for v, c in comp.items():
            print(v, {k: tuple(round(x, 3) for x in c[k]) for k in KEYS})


if __name__ == "__main__":
    main()
