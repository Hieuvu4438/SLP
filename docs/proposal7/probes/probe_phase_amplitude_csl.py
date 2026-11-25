"""P0.2-lite (docs/proposal7/08 §6): phase vs amplitude share of the difference between two
signers performing the same CSL-Daily dev sentence. Ground truth only; no model is run.

For each pair (A, B) and each evaluated part (12 upper-body joints relative to the pelvis; each
hand's 21 points relative to the wrist; the SOKE per-frame cost = mean joint L2 distance):
  D_lock = mean per-frame cost after linearly resampling B to len(A)   (frame-locked)
  D_dtw  = accumulated DTW cost / warping-path length                  (aligned)
  f      = 1 - D_dtw / D_lock          share of the difference that time alignment removes
Null: DTW always absorbs some purely spatial difference. The frame-locked residual R = B~ - A is
phase-randomised (per channel, same power spectrum, no timing structure) and added to A; the
same f is computed for (A, A + R_rand). f - f_null is the timing share beyond that artefact.

    OMP_NUM_THREADS=4 python docs/proposal7/probes/probe_phase_amplitude_csl.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from slp.data.sign_smplx import LazyPoses, _csl, load_stats  # noqa: E402
from slp.evaluation.metrics import dtw_cost  # noqa: E402
from slp.evaluation.smplx_fk import UPPER_BODY_IDX, default_fk  # noqa: E402


def rel(p):  # [T,J,3] -> root-relative (joint 0 = pelvis for body, wrist for hands)
    return p - p[:, :1]


def cost_matrix(a, b):  # SOKE per-frame cost, [Ta,Tb]
    return np.linalg.norm(a[:, None] - b[None], axis=-1).mean(-1)


def resample(x, n):
    t = np.linspace(0, len(x) - 1, n)
    i = np.floor(t).astype(int).clip(0, len(x) - 2) if len(x) > 1 else np.zeros(n, int)
    w = (t - i)[:, None, None]
    return x[i] * (1 - w) + x[np.minimum(i + 1, len(x) - 1)] * w


def shares(a, b, rng):
    bl = resample(b, len(a))
    d_lock = np.linalg.norm(a - bl, axis=-1).mean()
    acc, n = dtw_cost(cost_matrix(a, b))
    f = 1 - (acc / n) / d_lock
    r = (bl - a).reshape(len(a), -1)
    F = np.fft.rfft(r, axis=0)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, F.shape))
    ph[0] = 1.0
    if len(a) % 2 == 0:
        ph[-1] = 1.0
    rr = np.fft.irfft(F * ph, n=len(a), axis=0).reshape(a.shape)
    an = a + rr
    d_lock0 = np.linalg.norm(a - an, axis=-1).mean()
    acc0, n0 = dtw_cost(cost_matrix(a, an))
    f0 = 1 - (acc0 / n0) / d_lock0
    return f, f0, d_lock, acc / n


def main():
    root = ROOT / "data/processed/madis/CSL-Daily"
    rows = _csl(root, "val")
    mean, std = load_stats(str(ROOT / "third_party/SOKE/deps/stats"))
    poses = LazyPoses([r[1] for r in rows], mean, std, fps=None, subsample="madis")
    meta = [r[0] for r in rows]
    by_text = collections.defaultdict(list)
    for i, m in enumerate(meta):
        by_text[m["text"]].append(i)
    pairs = []
    for _, idx in sorted(by_text.items()):
        s = {}
        for i in sorted(idx, key=lambda i: meta[i]["name"]):
            s.setdefault(meta[i]["name"].split("_")[1], i)
        if len(s) >= 2:
            pairs.append(tuple(list(s.values())[:2]))
    fk = default_fk()
    rng = np.random.default_rng(0)
    ub = list(UPPER_BODY_IDX)
    out = {k: [] for k in ("body", "lhand", "rhand")}
    for i, j in pairs:
        ja, la, ra = (np.asarray(x.cpu() if hasattr(x, "cpu") else x, np.float64) for x in fk(np.asarray(poses[i])))
        jb, lb, rb = (np.asarray(x.cpu() if hasattr(x, "cpu") else x, np.float64) for x in fk(np.asarray(poses[j])))
        for k, a, b in (("body", ja[:, ub], jb[:, ub]), ("lhand", la, lb), ("rhand", ra, rb)):
            out[k].append(shares(rel(a), rel(b), rng))
    res = {"n_pairs": len(pairs)}
    for k, v in out.items():
        v = np.asarray(v)  # [P, 4]: f, f_null, d_lock, d_dtw_per_step
        d = v[:, 0] - v[:, 1]
        boot = np.random.default_rng(1).integers(0, len(d), (4000, len(d)))
        res[k] = {
            "f_timing_median": float(np.median(v[:, 0]).round(3)),
            "f_null_median": float(np.median(v[:, 1]).round(3)),
            "f_minus_null_mean": float(d.mean().round(3)),
            "f_minus_null_ci95": np.percentile(d[boot].mean(1), [2.5, 97.5]).round(3).tolist(),
            "frac_pairs_f_gt_null": float((d > 0).mean().round(3)),
            "f_timing_mean": float(v[:, 0].mean().round(3)),
            "f_null_mean": float(v[:, 1].mean().round(3)),
            "f_null_frac_exact_diagonal": float((np.abs(v[:, 1]) < 1e-9).mean().round(3)),
            "f_null_p90": float(np.percentile(v[:, 1], 90).round(3)),
            "d_lock_median_m": float(np.median(v[:, 2]).round(4)),
            "d_dtw_per_step_median_m": float(np.median(v[:, 3]).round(4)),
        }
    p = ROOT / "docs/proposal7/probes/probe_phase_amplitude_csl.json"
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
