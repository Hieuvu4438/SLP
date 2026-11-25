"""Representation diagnostics for the plan's gates G0/G1 (analysis only; uses train + DEV).

    # G0: how much of each joint group the other groups predict (no model needed)
    python -m methods.morae_slp.diagnostics --data phoenix_smplx --raw-cp --out g0.json
    # G1: spectrum / kappa / C-P loadings, reconstruction ceiling, gloss probe of a latent
    python -m methods.morae_slp.diagnostics --data phoenix_smplx --ae <ae.pt> \
        --spectrum --recon 256 --probe --out g1.json

* raw-cp   ridge readout (MoRAE Sec. 4.1 / B) of each of the 42 groups from the other groups in
           the same frame: R^2 per region, and for hand joints with vs without the other joints
           of the same finger (the "local shortcut" the masking units remove, PLAN.md 3.2).
* spectrum PCA of the standardised flow coordinates: eigenvalues, kappa, directions below
           1e-3 lambda_1, cross-joint loadings L_C / L_P (Eq. S1) and their Spearman correlation
           with the eigenvalues (paper Table 1 / S7).
* recon    decode(encode(x)) of dev clips scored with the shared SOKE evaluator (latent ceiling).
* probe    linear multi-label probe of the top-K glosses from clip-mean features
           (teacher H, latent mean, raw pose); macro AP on dev (analogue of the action probe).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

from slp import config as config_lib
from slp.data import load_dataset
from slp.evaluation import evaluate
from slp.evaluation.metrics import (directional_loadings, latent_spectrum,
                                    multilabel_average_precision)

from .data import GROUPS, N_GROUPS, UNITS, load_pose
from .stages import encode_split, load_space, resolve

REGION_OF_GROUP = (["body"] * 10 + ["lhand"] * 15 + ["rhand"] * 15 + ["face"] * 2)


def _chain_of(group: int) -> tuple[int, ...]:
    return next(u for u in UNITS["chain"] if group in u)


def sample_frames(split, indices: list[int], frames: int, rng: np.random.Generator
                  ) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    """Up to ``frames`` interior frames per clip; returns stacked poses and (clip, frame ids)."""
    rows, where = [], []
    for i in indices:
        pose = load_pose(split, i)
        lo, hi = len(pose) // 8, len(pose) - len(pose) // 8
        pick = np.sort(rng.choice(np.arange(lo, max(hi, lo + 1)), min(frames, max(1, hi - lo)),
                                  replace=False))
        rows.append(pose[pick])
        where.append((i, pick))
    return np.concatenate(rows), where


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Least squares with an intercept column; returns W [(p + 1), k]."""
    xa = np.c_[x, np.ones(len(x))]
    reg = alpha * np.eye(xa.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(xa.T @ xa + reg * len(x), xa.T @ y)


def ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.c_[x, np.ones(len(x))] @ w


class CrossJoint:
    """C(x): each group predicted from all other groups of the same frame (standardised features)."""

    def __init__(self, fit_frames: np.ndarray, alpha: float = 1e-3,
                 exclude_chain: bool = False):
        self.mu, self.sd = fit_frames.mean(0), fit_frames.std(0) + 1e-6
        x = (fit_frames - self.mu) / self.sd
        self.weights = []
        for g in range(N_GROUPS):
            drop = set(_chain_of(g)) if exclude_chain else {g}
            inputs = [c for h in range(N_GROUPS) if h not in drop for c in GROUPS[h]]
            self.weights.append((inputs, ridge_fit(x[:, inputs], x[:, list(GROUPS[g])], alpha)))

    def standardise(self, frames: np.ndarray) -> np.ndarray:
        return (frames - self.mu) / self.sd

    def predict(self, frames: np.ndarray) -> np.ndarray:
        x = self.standardise(frames)
        out = np.zeros_like(x)
        for g, (inputs, w) in enumerate(self.weights):
            out[:, list(GROUPS[g])] = ridge_predict(x[:, inputs], w)
        return out


def group_r2(target: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    r2 = []
    for g in range(N_GROUPS):
        cols = list(GROUPS[g])
        resid = ((target[:, cols] - predicted[:, cols]) ** 2).sum()
        total = ((target[:, cols] - target[:, cols].mean(0)) ** 2).sum()
        r2.append(1 - resid / max(total, 1e-12))
    return np.asarray(r2)


def raw_cp(train, rng, clips: int, frames: int) -> dict:
    idx = rng.permutation(len(train))[:2 * clips].tolist()
    fit, _ = sample_frames(train, idx[:clips], frames, rng)
    test, _ = sample_frames(train, idx[clips:], frames, rng)
    out = {}
    for name, exclude in (("all_other_groups", False), ("without_own_chain", True)):
        model = CrossJoint(fit, exclude_chain=exclude)
        r2 = group_r2(model.standardise(test), model.predict(test))
        out[name] = {"per_group": r2.round(4).tolist(),
                     **{f"mean_{region}": float(np.mean([r for r, reg in zip(r2, REGION_OF_GROUP)
                                                         if reg == region]))
                        for region in ("body", "lhand", "rhand", "face")}}
    hand = list(range(10, 40))
    out["hand_shortcut_r2_drop"] = float(np.mean(
        np.asarray(out["all_other_groups"]["per_group"])[hand]
        - np.asarray(out["without_own_chain"]["per_group"])[hand]))
    return out


def spectrum(space, train, device, rng, clips: int, frames: int, workers: int) -> dict:
    idx = rng.permutation(len(train))[:2 * clips].tolist()
    fit, _ = sample_frames(train, idx[:clips], frames, rng)
    cross = CrossJoint(fit)
    probe_idx = idx[clips:]
    latents = encode_split(space, train, probe_idx, device, workers, standardised=True)
    z_rows, x_rows = [], []
    for i in probe_idx:
        pose = load_pose(train, i)
        n = len(pose)
        lo, hi = n // 8, n - n // 8
        pick = np.sort(rng.choice(np.arange(lo, max(hi, lo + 1)), min(frames, max(1, hi - lo)),
                                  replace=False))
        z_rows.append(latents[i].numpy()[pick])
        x_rows.append(pose[pick])
    z, x = np.concatenate(z_rows), np.concatenate(x_rows)
    c = cross.predict(x)
    p = cross.standardise(x) - c
    spec = latent_spectrum(z)
    lam, lc = directional_loadings(z, c)
    _, lp = directional_loadings(z, p)
    d = len(lam)
    quarter = max(1, d // 4)
    return {
        "frames": int(len(z)), "dim": d, "kappa": spec["kappa"], "n_below_1e-3": spec["n_below"],
        "eigenvalues": spec["eigenvalues"].round(6).tolist(),
        "rho_lambda_LC": float(spearmanr(lam, lc).statistic),
        "rho_lambda_LP": float(spearmanr(lam, lp).statistic),
        "rho_lambda_LC_over_LP": float(spearmanr(lam, lc / np.maximum(lp, 1e-12)).statistic),
        "C_share_leading_quarter": float(lc[:quarter].sum() / max(lc.sum(), 1e-12)),
        "P_share_trailing_quarter": float(lp[-quarter:].sum() / max(lp.sum(), 1e-12)),
        "C_linear_r2": float(lc.sum()), "P_linear_r2": float(lp.sum()),
    }


@torch.no_grad()
def reconstruction(space, val, device, limit: int, metrics: list[str]) -> dict:
    idx = np.linspace(0, len(val) - 1, min(limit, len(val)), dtype=int).tolist()
    preds, targets = [], []
    for i in idx:
        pose = torch.from_numpy(load_pose(val, i))[None].to(device)
        n = torch.tensor([pose.shape[1]], device=device)
        mean, _ = space.posterior(pose, n)
        preds.append(space.decode(mean, n)[0].cpu().numpy())
        targets.append(val.targets[i])
    return {"clips": len(idx), **evaluate(preds, targets, metrics)}


def _clip_features(space, split, indices, device) -> dict[str, np.ndarray]:
    feats = {"raw": [], "latent": []}
    if space.kind != "raw":
        feats["teacher"] = []
    for i in indices:
        pose = torch.from_numpy(load_pose(split, i))[None].to(device)
        n = torch.tensor([pose.shape[1]], device=device)
        feats["raw"].append(pose[0].mean(0).cpu().numpy())
        mean, _ = space.posterior(pose, n)
        feats["latent"].append(mean[0].mean(0).cpu().numpy())
        if "teacher" in feats:
            feats["teacher"].append(space.features(pose, n)[0].mean(0).cpu().numpy())
    return {k: np.stack(v) for k, v in feats.items()}


def _labels(split, indices, vocab: dict[str, int]) -> np.ndarray:
    y = np.zeros((len(indices), len(vocab)), dtype=bool)
    for row, i in enumerate(indices):
        for g in (split.inputs[i].get("gloss") or "").split():
            if g in vocab:
                y[row, vocab[g]] = True
    return y


def _linear_probe(xtr, ytr, xva, yva, device, epochs: int = 300) -> dict:
    """BCE logistic probe; weight decay chosen on a 10 % hold-out of train."""
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xva = (xtr - mu) / sd, (xva - mu) / sd
    hold = np.arange(len(xtr)) % 10 == 0

    def fit(x, y, wd):
        xt = torch.tensor(x, dtype=torch.float32, device=device)
        yt = torch.tensor(y, dtype=torch.float32, device=device)
        layer = torch.nn.Linear(x.shape[1], y.shape[1]).to(device)
        opt = torch.optim.Adam(layer.parameters(), lr=1e-2, weight_decay=wd)
        for _ in range(epochs):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(layer(xt), yt).backward()
            opt.step()
        return layer

    def score(layer, x, y):
        with torch.no_grad():
            s = layer(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
        return multilabel_average_precision(s, y)

    choices = {}
    for wd in (1e-4, 1e-3, 1e-2):
        choices[wd] = score(fit(xtr[~hold], ytr[~hold], wd), xtr[hold], ytr[hold])
    best = max(choices, key=choices.get)
    return {"weight_decay": best, "dev_map": score(fit(xtr, ytr, best), xva, yva)}


def gloss_probe(space, train, val, device, rng, clips: int, top: int) -> dict:
    if not train.inputs[0].get("gloss"):
        return {"skipped": "dataset has no glosses"}
    counts: dict[str, int] = {}
    for item in train.inputs:
        for g in set((item.get("gloss") or "").split()):
            counts[g] = counts.get(g, 0) + 1
    vocab = {g: j for j, g in enumerate(sorted(counts, key=counts.get, reverse=True)[:top])}
    tr = sorted(rng.permutation(len(train))[:clips].tolist())
    va = list(range(len(val)))
    ytr, yva = _labels(train, tr, vocab), _labels(val, va, vocab)
    ftr, fva = _clip_features(space, train, tr, device), _clip_features(space, val, va, device)
    return {"labels": len(vocab), "train_clips": len(tr), "dev_clips": len(va),
            **{name: _linear_probe(ftr[name], ytr, fva[name], yva, device) for name in ftr}}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--ae", default=None, help="ae.pt (needed for --spectrum/--recon/--probe)")
    p.add_argument("--raw-cp", action="store_true")
    p.add_argument("--spectrum", action="store_true")
    p.add_argument("--recon", type=int, default=0, help="number of dev clips to reconstruct")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--clips", type=int, default=1024)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--probe-clips", type=int, default=4000)
    p.add_argument("--top-glosses", type=int, default=300)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=20261124)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if (args.spectrum or args.recon or args.probe) and not args.ae:
        sys.exit("--spectrum/--recon/--probe need --ae")

    cfg = config_lib.compose("morae_slp", args.data, [])
    data = load_dataset(cfg)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    result: dict = {"data": args.data, "ae": args.ae, "seed": args.seed}
    if args.raw_cp:
        result["raw_cp"] = raw_cp(data.train, np.random.default_rng(args.seed), args.clips // 2,
                                  args.frames)
    if args.ae:
        space, spec = load_space(args.ae, device)
        result["space"] = {"kind": spec["kind"], "latent_dim": space.latent_dim}
        if args.spectrum:
            result["spectrum"] = spectrum(space, data.train, device,
                                          np.random.default_rng(args.seed + 1), args.clips,
                                          args.frames, workers=0)
        if args.recon:
            result["reconstruction"] = reconstruction(space, data.val, device, args.recon,
                                                      cfg["eval"]["metrics"])
        if args.probe:
            result["gloss_probe"] = gloss_probe(space, data.train, data.val, device,
                                                np.random.default_rng(args.seed + 2),
                                                args.probe_clips, args.top_glosses)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    summary = {k: v for k, v in result.items() if k not in ("raw_cp",)}
    print(json.dumps(summary, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
