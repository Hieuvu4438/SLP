"""E1-0: does the kernel bandwidth sigma change the learned path only through an equivalent time shift?

Trains only the correction network eps_theta (Alg. 1) for a fixed small number of steps, with the
v2 Phoenix prototypes and sigma in {0.25, 0.5, 0.75} (same steps/seed for all), then measures the
per-joint progress kappa(t) along Log_{x0}x1, the distance to the geodesic curve, and the best SD3
shift s. Train split for training (as in fit), val split for measurement. CPU only.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 PYTHONPATH=.:src python docs/proposal7/probes/probe_sigma.py --steps 3000
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from methods.signmflow import geometry as G
from methods.signmflow.naturalness import NaturalnessMeasure
from methods.signmflow.paths import CorrectionNet, path_and_velocity, path_point
from methods.signmflow.representation import MotionCodec, load_hand_mean
from slp.data.sign_smplx import load_stats

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "outputs/runs/signmflow_v2_main_phoenix_smplx/signmflow/seed0_20261116-185137-965824"


def frames(split):
    f = sorted(glob.glob(str(ROOT / f"outputs/signmflow/cache/phoenix_smplx_{split}_*.npy")))[0]
    return np.load(f, mmap_mode="r"), np.asarray(json.load(open(f.replace(".npy", ".json")))["offsets"])


def fit_s(ts, kappa):
    best = None
    for s in np.linspace(0.5, 8.0, 751):
        n = s * (1 - ts) / (1 + (s - 1) * (1 - ts))
        err = float(np.sqrt(np.mean((1 - n - kappa) ** 2)))
        if best is None or err < best[1]:
            best = (float(s), err)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--sigmas", default="0.25,0.5,0.75")
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    mean, std = load_stats(str(ROOT / "third_party/SOKE/deps/stats"))
    codec = MotionCodec(mean, std, load_hand_mean(ROOT / "third_party/SOKE/deps/smpl_models"), "quat")
    st = torch.load(RUN / "checkpoints/prototypes.pt", map_location="cpu", weights_only=False)
    tr, toff = frames("train")
    va, _ = frames("val")
    rng = np.random.default_rng(0)
    vidx = np.sort(rng.choice(len(va), 2000, replace=False))
    with torch.no_grad():
        x1v = codec.encode(torch.from_numpy(np.array(va[vidx])))[0]
        torch.manual_seed(123)
        x0v = G.uniform_hemisphere(x1v.shape[:2])
        xiv = G.logmap(x0v, x1v)
    out_path = Path(__file__).with_name("probe_sigma_phoenix_smplx.json")
    res = json.load(open(out_path)) if out_path.exists() else {"steps": a.steps, "sigmas": {}}
    ts = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    for sig in [float(x) for x in a.sigmas.split(",")]:
        key = f"{sig:g}"
        if key in res["sigmas"]:
            continue
        meas = NaturalnessMeasure(st["protos"], st["weights"], sig, 1e-6)
        torch.manual_seed(0)
        corr = CorrectionNet()
        opt = torch.optim.AdamW(corr.parameters(), lr=2e-4, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + np.cos(np.pi * min(1.0, s / a.steps))))
        g = np.random.default_rng(0)
        hist = []
        for step in range(a.steps):
            seqs = g.integers(0, len(toff) - 1, a.bs)
            fr = np.concatenate([np.array(tr[toff[i]:toff[i + 1]]) for i in seqs], 0)
            x1 = codec.encode(torch.from_numpy(fr))[0]
            lens = [toff[i + 1] - toff[i] for i in seqs]
            t = torch.cat([torch.full((L, 1, 1), float(torch.rand(1))) for L in lens])  # one t per sequence
            x0 = G.uniform_hemisphere(x1.shape[:2])
            xt, v = path_and_velocity("naturalness", x0, x1, t, corr)
            loss = meas.quad_form(xt, v).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            hist.append(float(loss))
            if (step + 1) % 500 == 0:
                print(f"sigma={sig} step {step + 1} energy={np.mean(hist[-500:]):.3f}", flush=True)
        corr.eval()
        with torch.no_grad():
            kap, dev = [], []
            for t in ts:
                tt = torch.full((x1v.shape[0], 1, 1), float(t))
                xn = path_point("naturalness", x0v, x1v, tt, corr)
                w = G.logmap(x0v, xn)
                k = (w * xiv).sum(-1) / (xiv.norm(dim=-1) ** 2).clamp_min(1e-9)
                kap.append(float(k.median()))
                perp = (w - k[..., None] * xiv).norm(dim=-1)
                dev.append(float(perp.median()))
            geo_e, nat_e = 0.0, 0.0
            for _ in range(5):
                tt = torch.rand(x1v.shape[0], 1, 1)
                with torch.enable_grad():
                    xg, vg = path_and_velocity("geodesic", x0v, x1v, tt)
                    xn_, vn = path_and_velocity("naturalness", x0v, x1v, tt, corr)
                geo_e += float(meas.quad_form(xg.detach(), vg.detach()).sum(-1).mean()) / 5
                nat_e += float(meas.quad_form(xn_.detach(), vn.detach()).sum(-1).mean()) / 5
        s_fit, rmse = fit_s(ts, np.array(kap))
        res["sigmas"][key] = {"kappa_median": dict(zip(map(str, ts), kap)), "perp_tangent_median": dict(zip(map(str, ts), dev)),
                              "shift_s": s_fit, "rmse": rmse, "val_energy_geodesic": geo_e, "val_energy_learned": nat_e,
                              "train_energy_last500": float(np.mean(hist[-500:]))}
        print(key, res["sigmas"][key], flush=True)
        json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
