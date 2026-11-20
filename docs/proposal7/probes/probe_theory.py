"""CPU probes of SignNMFlow's naturalness measure / learned path (docs/proposal7, P1-P5).

Uses only methods/signmflow code + a trained v2 run (prototypes.pt, correction_final.pt) and
the signmflow feature cache. No GPU, no test split.

    PYTHONPATH=.:src python docs/proposal7/probes/probe_theory.py --dataset phoenix_smplx \
        --run outputs/runs/signmflow_v2_main_phoenix_smplx/signmflow/seed0_20261116-185137-965824

P1  energy decomposition: kernel term sum_k pi_k <delta_k,v>^2  vs  density term rho|v|^2/m
P2  anisotropy of G^-1 at x_t: where does the direction towards x1 sit in [lambda_min, lambda_max]
P3  learned correction: size of t(1-t)eps vs t*xi, extra arc length, log m along the path
P4  data: quaternion scalar part near the hemisphere boundary, temporal sign flips
P5  "noise schedule" implied by the paths: geodesic distance of x_t to x1 (rotation angle) vs t
    for geodesic / linear-quat(+renorm) / linear-6D(Gaussian source)
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
from methods.signmflow.representation import HAND_SLICES, MotionCodec, load_hand_mean
from slp.data.sign_smplx import load_stats

torch.set_grad_enabled(False)
ROOT = Path(__file__).resolve().parents[3]
PARTS = {"body": slice(0, 10), "lhand": HAND_SLICES[0], "rhand": HAND_SLICES[1], "jaw": slice(40, 41)}


def q(x, ps=(5, 25, 50, 75, 95)):
    x = np.asarray(x).ravel()
    return {f"p{p}": float(np.percentile(x, p)) for p in ps} | {"mean": float(x.mean())}


def load_frames(dataset: str, split: str, n_frames: int, seed: int):
    fs = sorted(glob.glob(str(ROOT / f"outputs/signmflow/cache/{dataset}_{split}_*.npy")))
    if not fs:
        raise FileNotFoundError(f"no cache for {dataset} {split}")
    arr = np.load(fs[0], mmap_mode="r")
    meta = json.load(open(fs[0].replace(".npy", ".json")))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(arr), size=min(n_frames, len(arr)), replace=False))
    return torch.from_numpy(np.asarray(arr[idx])), arr, np.asarray(meta["offsets"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="phoenix_smplx")
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    run = ROOT / a.run
    mean, std = load_stats(str(ROOT / "third_party/SOKE/deps/stats"))
    codec = MotionCodec(mean, std, load_hand_mean(ROOT / "third_party/SOKE/deps/smpl_models"), "quat")
    st = torch.load(run / "checkpoints/prototypes.pt", map_location="cpu", weights_only=False)
    meas = NaturalnessMeasure(st["protos"], st["weights"], 0.5, 1e-6)
    corr = CorrectionNet()
    corr.load_state_dict(torch.load(run / "checkpoints/correction_final.pt", map_location="cpu",
                                    weights_only=False)["model"])
    corr.eval()
    res = {"dataset": a.dataset, "run": a.run, "n_frames": a.n}

    feats, arr, offs = load_frames(a.dataset, "val", a.n, a.seed)
    x1 = codec.encode(feats)[0]                       # [N,41,4]
    N = x1.shape[0]
    x0 = G.uniform_hemisphere((N, 41))
    t = torch.rand(N, 1, 1)

    # ---------------- P1 energy decomposition along geodesic and learned paths
    def decompose(x, v):
        a_, theta, logits = meas._stats(x)
        log_m = torch.logsumexp(logits, -1)
        pi = torch.softmax(logits, -1)
        cv = torch.einsum("...jd,jkd->...jk", v, meas.protos)
        xv = (x * v).sum(-1, keepdim=True)
        dv = theta / torch.sin(theta) * (cv - a_ * xv)
        kern = (pi * dv ** 2).sum(-1)
        dens = meas.rho * (v * v).sum(-1) * torch.exp(-log_m)
        return kern, dens, log_m

    p1 = {}
    for kind in ("geodesic", "naturalness"):
        with torch.enable_grad():
            xt, v = path_and_velocity(kind, x0, x1, t, corr if kind == "naturalness" else None)
        xt, v = xt.detach(), v.detach()
        kern, dens, log_m = decompose(xt, v)
        frac = (dens / (kern + dens)).numpy()
        p1[kind] = {"density_term_fraction": q(frac), "log10_m": q(log_m.numpy() / np.log(10)),
                    "energy_per_frame": q((kern + dens).sum(-1).numpy()),
                    "euclid_speed2_per_frame": q((v * v).sum((-1, -2)).numpy())}
        # energy if the density factor were removed entirely (rho -> 0)
        p1[kind]["energy_without_density_term"] = q(kern.sum(-1).numpy())
    res["P1_energy_decomposition"] = p1

    # ---------------- P2 anisotropy: Rayleigh quotient of the direction towards x1
    xt = path_point("geodesic", x0, x1, t)
    Ginv = meas.metric_inverse(xt)                    # [N,41,4,4]
    # restrict to the tangent space: eigen-decompose P Ginv P, drop the normal direction (eig ~0)
    P = torch.eye(4) - xt[..., :, None] * xt[..., None, :]
    M = P @ Ginv @ P
    ev = torch.linalg.eigvalsh(M)[..., 1:]           # 3 tangent eigenvalues (ascending)
    u = G.logmap(xt, x1)
    un = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    rq = torch.einsum("...d,...de,...e->...", un, M, un)
    pos = ((rq - ev[..., 0]) / (ev[..., 2] - ev[..., 0]).clamp_min(1e-12)).numpy()
    cond = (ev[..., 2] / ev[..., 0].clamp_min(1e-30)).numpy()
    tt = t.expand(-1, 41, 1)[..., 0].numpy()
    p2 = {}
    for name, sl in PARTS.items():
        p2[name] = {"towards_x1_position_in_[lmin,lmax]": q(pos[:, sl]),
                    "log10_condition_number": q(np.log10(cond[:, sl])),
                    "towards_x1_position_t<0.3": q(pos[:, sl][tt[:, sl] < 0.3]),
                    "towards_x1_position_t>0.7": q(pos[:, sl][tt[:, sl] > 0.7])}
    res["P2_anisotropy"] = p2

    # ---------------- P3 learned correction: magnitude, extra length, density along the path
    ts = torch.linspace(0.0, 1.0, 21)
    nsub = min(N, 1000)
    xa, xb = x0[:nsub], x1[:nsub]
    xi = G.logmap(xa, xb)
    rel, logm_g, logm_n, angdev = [], [], [], []
    pts_g, pts_n = [], []
    for s in ts:
        tt_ = torch.full((nsub, 1, 1), float(s))
        eps = corr(tt_, xa, xb)
        if 0 < s < 1:
            rel.append(((s * (1 - s) * eps).norm(dim=-1) / (s * xi).norm(dim=-1).clamp_min(1e-9)).numpy())
        pg = path_point("geodesic", xa, xb, tt_)
        pn = path_point("naturalness", xa, xb, tt_, corr)
        pts_g.append(pg); pts_n.append(pn)
        logm_g.append(meas.log_mass(pg).numpy()); logm_n.append(meas.log_mass(pn).numpy())
        angdev.append(G.geodesic_distance(pg, pn).numpy())
    def arclen(pts):
        return sum(G.geodesic_distance(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    Lg, Ln = arclen(pts_g), arclen(pts_n)
    rel = np.stack(rel)
    p3 = {"rel_correction_norm_over_t": {f"t={float(s):.2f}": float(np.median(r)) for s, r in zip(ts[1:-1], rel)},
          "rel_correction_norm_all": q(rel),
          "max_deviation_from_geodesic_rad(per joint quat arc)": q(np.stack(angdev).max(0)),
          "arc_length_ratio_learned/geodesic": q((Ln / Lg.clamp_min(1e-9)).numpy()),
          "delta_log_m_learned_minus_geodesic_t=0.5": q((logm_n[10] - logm_g[10])),
          "mean_log_m_over_path_geodesic": float(np.mean(logm_g)),
          "mean_log_m_over_path_learned": float(np.mean(logm_n))}
    res["P3_correction"] = p3

    # ---------------- P4 data: hemisphere boundary / sign flips (full val sequences)
    allq = codec.encode(torch.from_numpy(np.asarray(arr[: min(len(arr), 200_000)])))[0]
    w = allq[..., 0].numpy()
    p4 = {}
    for name, sl in PARTS.items():
        p4[name] = {"frac_w<0.1": float((w[:, sl] < 0.1).mean()), "frac_w<0.3": float((w[:, sl] < 0.3).mean()),
                    "rotation_angle_deg": q(2 * np.degrees(np.arccos(np.clip(w[:, sl], -1, 1))))}
    # temporal sign flips inside sequences: <q_t, q_{t+1}> < 0 although the rotation changed little
    flips, n_pairs = np.zeros(41), 0
    for i in range(len(offs) - 1):
        a0, b0 = offs[i], offs[i + 1]
        if b0 > 200_000:
            break
        qs = allq[a0:b0]
        d = (qs[1:] * qs[:-1]).sum(-1).numpy()
        flips += (d < 0).sum(0); n_pairs += len(d)
    p4["temporal_sign_flip_rate"] = {name: float(flips[sl].sum() / (n_pairs * (sl.stop - sl.start)))
                                     for name, sl in PARTS.items()}
    res["P4_data"] = p4

    # ---------------- P5 implied noise schedule: rotation angle between x_t and x1 vs t
    def rot_angle_quat(qa, qb):
        c = (G.normalize(qa) * G.normalize(qb)).sum(-1).abs().clamp(max=1.0)
        return 2 * torch.acos(c)
    xa, xb = x0[:nsub], x1[:nsub]
    six1 = G.matrix_to_6d(G.quat_to_matrix(xb))
    g6 = torch.randn_like(six1)
    sched = {}
    for s in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        tt_ = torch.full((nsub, 1, 1), s)
        geo = rot_angle_quat(path_point("geodesic", xa, xb, tt_), xb)
        nat = rot_angle_quat(path_point("naturalness", xa, xb, tt_, corr), xb)
        lin = rot_angle_quat((1 - s) * xa + s * xb, xb)
        x6 = (1 - s) * g6 + s * six1
        q6 = G.matrix_to_quat(G.sixd_to_matrix(x6))
        l6 = rot_angle_quat(q6, xb)
        sched[f"t={s:.2f}"] = {k: float(np.degrees(v.mean())) for k, v in
                               (("geodesic_quat", geo), ("naturalness_quat", nat),
                                ("linear_quat_renorm", lin), ("linear_6d_gauss", l6))}
    res["P5_schedule_mean_rotation_angle_deg_to_x1"] = sched

    out = Path(a.out or Path(__file__).with_name(f"probe_theory_{a.dataset}.json"))
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()


def fit_time_warp(json_path: str):
    """Fit the SD3-style shift s so that the geodesic at kappa(t) reproduces the learned path's
    mean rotation angle to x1: noise level n(t) = s(1-t) / (1 + (s-1)(1-t)),  kappa = 1 - n."""
    r = json.load(open(json_path))["P5_schedule_mean_rotation_angle_deg_to_x1"]
    ts = np.array([float(k[2:]) for k in r])
    a0 = r["t=0.00"]["geodesic_quat"]
    n_nat = np.array([v["naturalness_quat"] for v in r.values()]) / a0
    best = None
    for s in np.linspace(1.0, 6.0, 501):
        n = s * (1 - ts) / (1 + (s - 1) * (1 - ts))
        err = float(np.sqrt(np.mean((n - n_nat) ** 2)))
        if best is None or err < best[1]:
            best = (float(s), err)
    return {"shift_s": best[0], "rmse_noise_level": best[1],
            "naturalness_noise_level": dict(zip(map(str, ts), n_nat.round(3).tolist()))}
