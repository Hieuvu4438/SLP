"""Gate P0.S (docs/proposal7/08 §6): does the predictive variance of the v2 model contain *useful*
timing variance? Inference only (no training), val split only, never test. Written for the user to
run; it loads a trained v2 checkpoint and draws K samples per sentence.

For each val sentence: K samples (different sampler seeds of one model, or one sample from each of
several models = seed ensemble, ``--runs a,b,c``), all with the same fixed output length (the v2
run's saved predictions). Estimators per evaluated part (body: 12 upper-body joints, pelvis-
relative; each hand: 21 wrist-relative points; SOKE per-frame cost = mean joint L2):
  single         sample 0
  frame_mean     per-frame mean of the K samples
  frame_median   per-frame, per-joint geometric median (Weiszfeld)
  aligned_median DTW-registered median: samples aligned to the current estimate (start: medoid),
                 matched frames averaged, per-joint geometric median; 3 iterations
  medoid         the sample with the smallest summed DTW to the others
Gate statistic (per part, per sequence, relative to frame_median):
  Gain = [DTW(frame_median) - DTW(aligned_median*)] / DTW(frame_median)
where * means: the estimator with the larger low-pass path length is shrunk towards the mean
pose until both path lengths match (path length scales exactly by (1 - a) under p -> (1-a) p +
a m for a static m). Null: same Gain on samples whose frame-locked residuals around the frame
mean are phase-randomised (same spectrum, no timing structure). The gate passes for a dataset
when mean(Gain - Gain_null) >= 0.02 with a 95% bootstrap CI excluding 0 on the selection half
(sentences split by a hash of the text), and the T1 consistency check is negative-correlated.

    PYTHONPATH=.:src OMP_NUM_THREADS=8 python docs/proposal7/probes/probe_gate_ps.py \
        --data phoenix_smplx --runs outputs/runs/signmflow_v2_main_phoenix_smplx/signmflow/seed0_20261116-185137-965824 \
        --k 8 --device cuda
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[3]
PARTS = ("body", "lhand", "rhand")


# ----------------------------------------------------------------------------- estimators (numpy)
def cost(a, b):
    """SOKE per-frame cost on root-relative points: [Ta,J,3] x [Tb,J,3] -> [Ta,Tb]."""
    return np.linalg.norm(a[:, None] - b[None], axis=-1).mean(-1)


def dtw_path(C):
    """Exact DTW (accumulated cost, path) with the evaluator's step set."""
    r, c = C.shape
    D = np.full((r + 1, c + 1), np.inf)
    D[0, 0] = 0.0
    for d in range(2, r + c + 1):  # anti-diagonals, vectorised (same recursion as metrics.dtw_cost)
        i = np.arange(max(1, d - c), min(r, d - 1) + 1)
        j = d - i
        D[i, j] = C[i - 1, j - 1] + np.minimum(np.minimum(D[i - 1, j - 1], D[i - 1, j]), D[i, j - 1])
    i, j, path = r, c, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = int(np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
        i, j = (i - 1, j - 1) if k == 0 else ((i - 1, j) if k == 1 else (i, j - 1))
    return float(D[r, c]), path[::-1]


def geomedian(X, iters=20, eps=1e-9):
    """Per-joint geometric median over axis 0: X [K,...,3] -> [...,3] (Weiszfeld)."""
    m = X.mean(0)
    for _ in range(iters):
        d = np.linalg.norm(X - m, axis=-1, keepdims=True).clip(eps)
        w = 1.0 / d
        m = (w * X).sum(0) / w.sum(0)
    return m


def medoid(S):
    K = len(S)
    M = np.zeros((K, K))
    for a in range(K):
        for b in range(a + 1, K):
            M[a, b] = M[b, a] = dtw_path(cost(S[a], S[b]))[0]
    return int(M.sum(1).argmin())


def aligned_median(S, iters=3):
    est = S[medoid(S)].copy()
    for _ in range(iters):
        stacks = []
        for s in S:
            _, path = dtw_path(cost(est, s))
            acc = np.zeros_like(est)
            n = np.zeros(len(est))
            for i, j in path:
                acc[i] += s[j]
                n[i] += 1
            stacks.append(acc / n[:, None, None])
        est = geomedian(np.stack(stacks))
    return est


def path_length(p, sigma=2.0):
    q = gaussian_filter1d(p, sigma, axis=0, mode="nearest")
    return float(np.linalg.norm(np.diff(q, axis=0), axis=-1).sum())


def matched_pair(a, b, mean_pose):
    """Shrink the estimator with the larger path length towards the static mean pose until equal."""
    la, lb = path_length(a), path_length(b)
    if la > lb and la > 0:
        a = (1 - (1 - lb / la)) * a + (1 - lb / la) * mean_pose
    elif lb > la and lb > 0:
        b = (1 - (1 - la / lb)) * b + (1 - la / lb) * mean_pose
    return a, b


def phase_null(S, rng):
    m = S.mean(0)
    out = []
    for s in S:
        r = (s - m).reshape(len(s), -1)
        F = np.fft.rfft(r, axis=0)
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, F.shape))
        ph[0] = 1.0
        if len(s) % 2 == 0:
            ph[-1] = 1.0
        out.append(m + np.fft.irfft(F * ph, n=len(s), axis=0).reshape(s.shape))
    return np.stack(out)


def warp_dispersion(S):
    """Mean over sample pairs of the std of (i - j) along the DTW path, in frames."""
    v = []
    for a in range(len(S)):
        for b in range(a + 1, len(S)):
            _, path = dtw_path(cost(S[a], S[b]))
            d = np.array([i - j for i, j in path], float)
            v.append(d.std())
    return float(np.mean(v))


def evaluate(S, gt, mean_pose):
    """All estimators for one part. S [K,T,J,3], gt [Tg,J,3]. Returns dict of DTW / path lengths."""
    est = {"single": S[0], "frame_mean": S.mean(0), "frame_median": geomedian(S),
           "aligned_median": aligned_median(S)}
    est["medoid"] = S[medoid(S)]
    out = {k: dtw_path(cost(v, gt))[0] for k, v in est.items()}
    out.update({f"pl_{k}": path_length(v) / max(path_length(gt), 1e-9) for k, v in est.items()})
    # Gain of aligned over frame-wise median. If the aligned median dominates (lower DTW and at least
    # the same path length), the raw gain counts: articulation was not traded for DTW. Otherwise
    # compare at matched path length (shrink the more articulated one towards the mean pose).
    raw = (out["frame_median"] - out["aligned_median"]) / max(out["frame_median"], 1e-9)
    fm, am = matched_pair(est["frame_median"], est["aligned_median"], mean_pose)
    d_fm, d_am = dtw_path(cost(fm, gt))[0], dtw_path(cost(am, gt))[0]
    matched = (d_fm - d_am) / max(d_fm, 1e-9)
    dominates = out["aligned_median"] <= out["frame_median"] and out["pl_aligned_median"] >= out["pl_frame_median"]
    out["gain_raw"], out["gain_matched"], out["dominates"] = raw, matched, bool(dominates)
    out["gain"] = raw if dominates else matched
    out["frame_mean_speed_ratio"] = path_length(est["frame_mean"]) / max(np.mean([path_length(s) for s in S]), 1e-9)
    return out


# ----------------------------------------------------------------------------- sampling (torch)
def load_model(run, device, epoch):
    from methods.signmflow.model import SignGenerator
    from methods.signmflow.paths import CorrectionNet

    gen = SignGenerator(rot_dim=4, latent=512, layers=8, heads=8, ff=2048, time_dim=2048, dropout=0.1,
                        max_frames=400, text_model=str(ROOT / "third_party/SOKE/deps/mbart-h2s-csl-phoenix"),
                        text_max_len=128)
    st = torch.load(Path(run) / f"checkpoints/generator_ep{epoch}.pt", map_location="cpu", weights_only=False)
    gen.load_state_dict(st["model"])
    corr = CorrectionNet()
    corr.load_state_dict(torch.load(Path(run) / "checkpoints/correction_final.pt", map_location="cpu",
                                    weights_only=False)["model"])
    return gen.eval().to(device), corr.eval().to(device)


@torch.no_grad()
def sample(gen, corr, text, pad, lengths, steps, seed, device):
    from methods.signmflow import geometry as G
    from methods.signmflow.paths import path_point

    B, T = len(lengths), int(max(lengths))
    g = torch.Generator(device="cpu").manual_seed(seed)
    mask = (torch.arange(T)[None] < torch.as_tensor(lengths)[:, None]).to(device)
    x0 = G.uniform_hemisphere((B, T, 41), generator=g).to(device)
    e = torch.randn(B, T, 10, generator=g).to(device)
    x = x0
    for k in range(steps):
        t, tn = k / steps, (k + 1) / steps
        pose, expr = gen(x, e, torch.full((B,), t, device=device), text, pad, ~mask)
        x1h = G.normalize(pose)
        x = G.normalize(path_point("naturalness", x0, x1h, torch.full((B, 1, 1, 1), tn, device=device), corr))
        e = e + (tn - t) / (1 - t) * (expr - e)
    return x, e


def main():
    from methods.signmflow.representation import MotionCodec, load_hand_mean
    from slp.data.sign_smplx import load_stats
    from slp.evaluation.smplx_fk import UPPER_BODY_IDX, default_fk

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="phoenix_smplx")
    ap.add_argument("--runs", required=True, help="one run (K sampler seeds) or a comma list (seed ensemble)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--epoch", type=int, default=300)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--n", type=int, default=0, help="first n sentences (0 = all)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    runs = [r for r in a.runs.split(",") if r]

    f = sorted(glob.glob(str(ROOT / f"outputs/signmflow/cache/{a.data}_val_*.npy")))[0]
    arr = np.load(f, mmap_mode="r")
    meta = json.load(open(f.replace(".npy", ".json")))
    offs, inputs = meta["offsets"], meta["inputs"]
    if a.n:
        inputs = inputs[: a.n]
    gts = [np.array(arr[offs[i]:offs[i + 1]]) for i in range(len(inputs))]
    saved = np.load(Path(runs[0]) / "predictions.npz")
    lengths = [len(saved[x["name"].replace("/", "__")]) for x in inputs]

    mean, std = load_stats(str(ROOT / "third_party/SOKE/deps/stats"))
    codec = MotionCodec(mean, std, load_hand_mean(ROOT / "third_party/SOKE/deps/smpl_models"), "quat")
    fk = default_fk("cpu")
    ub = list(UPPER_BODY_IDX)

    def parts(feats):
        j, l, r = (x.cpu().numpy().astype(np.float64) for x in fk(feats))
        return {"body": j[:, ub] - j[:, :1], "lhand": l - l[:, :1], "rhand": r - r[:, :1]}

    zero = parts(np.zeros((1, 133), np.float32))  # normalised 0 = training mean pose
    mean_pose = {p: zero[p][0] for p in PARTS}

    # ---- draw samples: samples[i] = list of K [T,133] arrays
    samples = [[] for _ in inputs]
    plan = [(runs[0], s) for s in range(a.k)] if len(runs) == 1 else [(r, 0) for r in runs]
    order = np.argsort(lengths)
    t0 = time.time()
    for run, seed in plan:
        gen, corr = load_model(run, a.device, a.epoch)
        for s0 in range(0, len(order), a.bs):
            idx = order[s0:s0 + a.bs].tolist()
            text, pad = gen.encode_text([inputs[i]["text"] for i in idx], [inputs[i]["src"] for i in idx], a.device)
            L = [lengths[i] for i in idx]
            rot, expr = sample(gen, corr, text, pad, L, a.steps, seed * 100003 + s0, a.device)
            feats = codec.decode(rot.cpu(), expr.cpu()).float().numpy()
            for j, i in enumerate(idx):
                samples[i].append(feats[j, :L[j]])
        del gen, corr
        torch.cuda.empty_cache() if a.device.startswith("cuda") else None
    print(f"sampling done in {time.time() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(0)
    rows = []
    for i, (S, g) in enumerate(zip(samples, gts)):
        P = [parts(s) for s in S]
        G = parts(g)
        half = int(hashlib.md5(inputs[i]["text"].encode()).hexdigest(), 16) % 2  # 0 = selection half
        row = {"name": inputs[i]["name"], "half": half}
        for p in PARTS:
            Sp = np.stack([x[p] for x in P])
            r = evaluate(Sp, G[p], mean_pose[p])
            rn = evaluate(phase_null(Sp, rng), G[p], mean_pose[p])
            row[p] = r
            row[p + "_null_gain"] = rn["gain"]
            row[p + "_warp_disp"] = warp_dispersion(Sp)
        rows.append(row)
        if i % 25 == 0:
            print(i, {p: round(row[p]["gain"] - row[p + "_null_gain"], 4) for p in PARTS}, flush=True)

    summary = {}
    for p in PARTS:
        for h in (0, 1):
            R = [r for r in rows if r["half"] == h]
            d = np.array([r[p]["gain"] - r[p + "_null_gain"] for r in R])
            bs = d[np.random.default_rng(1).integers(0, len(d), (4000, len(d)))].mean(1)
            wd = np.array([r[p + "_warp_disp"] for r in R])
            sr = np.array([r[p]["frame_mean_speed_ratio"] for r in R])
            summary[f"{p}_half{h}"] = {
                "n": len(R),
                "gain_minus_null_mean": float(d.mean()), "ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
                "t1_corr_warpdisp_vs_framemean_speed": float(np.corrcoef(wd, sr)[0, 1]),
                "mean_dtw": {k: float(np.mean([r[p][k] for r in R]))
                             for k in ("single", "frame_mean", "frame_median", "aligned_median", "medoid")},
                "mean_path_ratio": {k: float(np.mean([r[p][f"pl_{k}"] for r in R]))
                                    for k in ("single", "frame_mean", "frame_median", "aligned_median", "medoid")},
                "gate_pass": bool(d.mean() >= 0.02 and np.percentile(bs, 2.5) > 0),
            }
    res = {"data": a.data, "runs": runs, "k": a.k, "steps": a.steps, "epoch": a.epoch, "summary": summary,
           "rows": rows}
    out = Path(a.out or Path(__file__).with_name(f"probe_gate_ps_{a.data}.json"))
    out.write_text(json.dumps(res))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
