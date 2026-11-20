"""CPU probe of how the SOKE DTW metrics react to simple, text-independent edits (docs/proposal7, P6).

Reads the saved dev predictions of a trained v2 run and the signmflow val cache (= the reference
targets used by scripts/run.py). Never touches the test split.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src python docs/proposal7/probes/probe_metrics.py \
        --dataset phoenix_smplx --run outputs/runs/signmflow_v2_main_phoenix_smplx/signmflow/seed0_20261116-185137-965824 --n 150

Variants (all on normalised 133-D SOKE features):
  pred            : the model's predictions
  mean_pose       : a constant zero feature (= SOKE stats mean pose), predicted length
  shrink_a        : (1-a) * pred  (pull every frame towards the mean pose)
  smooth_s        : Gaussian temporal smoothing of pred (sigma s frames)
  gt_shuffled     : ground truth of another dev sentence (a real, natural, wrong motion)
Also reports joint-speed statistics (FK positions) of pred vs GT.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from slp.evaluation.metrics import dtw_cost, pairwise_part_cost
from slp.evaluation.smplx_fk import UPPER_BODY_IDX, default_fk

ROOT = Path(__file__).resolve().parents[3]


def scores(preds, gts, fk, cache):
    """Same per-part DTW scores as slp.evaluation.metrics.soke_protocol_scores, without the
    expensive 144-joint 'all' part (not used in Tables 1-2)."""
    out = {k: [] for k in ("jpe_body", "jpe_lhand", "jpe_rhand", "pa_lhand", "pa_rhand", "pa_body_ua")}
    ub = list(UPPER_BODY_IDX)
    for i, (p, t) in enumerate(zip(preds, gts)):
        gj, gl, gr = (x.cpu().numpy() for x in fk(p))
        if id(t) not in cache:
            cache[id(t)] = tuple(x.cpu().numpy() for x in fk(t))
        rj, rl, rr = cache[id(t)]
        out["jpe_body"].append(dtw_cost(pairwise_part_cost(gj, rj, UPPER_BODY_IDX, "root"))[0])
        out["jpe_lhand"].append(dtw_cost(pairwise_part_cost(gl, rl, None, "root"))[0])
        out["jpe_rhand"].append(dtw_cost(pairwise_part_cost(gr, rr, None, "root"))[0])
        out["pa_lhand"].append(dtw_cost(pairwise_part_cost(gl, rl, None, "procrustes"))[0])
        out["pa_rhand"].append(dtw_cost(pairwise_part_cost(gr, rr, None, "procrustes"))[0])
        out["pa_body_ua"].append(dtw_cost(pairwise_part_cost(gj[:, ub], rj[:, ub], None, "procrustes"))[0])
    return {k: np.asarray(v) for k, v in out.items()}


def summarise(s):
    return {"jpe_body": float(np.mean(s["jpe_body"])),
            "jpe_hand": float(np.mean((s["jpe_lhand"] + s["jpe_rhand"]) / 2)),
            "pa_body_ua": float(np.mean(s["pa_body_ua"])),
            "pa_hand": float(np.mean((s["pa_lhand"] + s["pa_rhand"]) / 2))}


def speeds(seqs, fk):
    """mean per-frame speed (m/frame) of body joints (pelvis-relative) and wrist-relative fingers."""
    b, h = [], []
    for x in seqs:
        j, l, r = fk(x)
        j, l, r = j.cpu().numpy(), l.cpu().numpy(), r.cpu().numpy()
        body = j[:, :22] - j[:, :1]
        b.append(np.linalg.norm(np.diff(body, axis=0), axis=-1).mean())
        hh = np.concatenate([l - l[:, :1], r - r[:, :1]], 1)
        h.append(np.linalg.norm(np.diff(hh, axis=0), axis=-1).mean())
    return {"body_speed_mm_per_frame": 1000 * float(np.mean(b)), "finger_speed_mm_per_frame": 1000 * float(np.mean(h))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="phoenix_smplx")
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    f = sorted(glob.glob(str(ROOT / f"outputs/signmflow/cache/{a.dataset}_val_*.npy")))[0]
    arr = np.load(f, mmap_mode="r")
    meta = json.load(open(f.replace(".npy", ".json")))
    offs, inputs = meta["offsets"], meta["inputs"]
    preds = np.load(ROOT / a.run / "predictions.npz")
    idx = list(range(min(a.n, len(inputs))))
    gts = [np.asarray(arr[offs[i]:offs[i + 1]]) for i in idx]
    pr = [preds[inputs[i]["name"].replace("/", "__")] for i in idx]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(idx))
    variants = {
        "pred": pr,
        "mean_pose": [np.zeros_like(p) for p in pr],
        "shrink_0.25": [0.75 * p for p in pr],
        "shrink_0.5": [0.5 * p for p in pr],
        "smooth_1": [gaussian_filter1d(p, 1.0, axis=0, mode="nearest") for p in pr],
        "smooth_3": [gaussian_filter1d(p, 3.0, axis=0, mode="nearest") for p in pr],
        "gt_shuffled": [gts[j] for j in perm],
    }
    fk = default_fk("cpu")
    cache = {}
    res = {"dataset": a.dataset, "run": a.run, "n_sequences": len(idx), "metrics": {}}
    for k, v in variants.items():
        res["metrics"][k] = summarise(scores(v, gts, fk, cache))
        print(k, res["metrics"][k], flush=True)
    res["speed"] = {"gt": speeds(gts, fk), "pred": speeds(pr, fk), "smooth_3": speeds(variants["smooth_3"], fk)}
    res["length"] = {"gt_mean": float(np.mean([len(g) for g in gts])),
                     "pred_mean": float(np.mean([len(p) for p in pr])),
                     "abs_len_err_mean": float(np.mean([abs(len(g) - len(p)) for g, p in zip(gts, pr)]))}
    print(res["speed"], res["length"])
    out = Path(__file__).with_name(f"probe_metrics_{a.dataset}.json")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
