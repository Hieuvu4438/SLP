"""Signed S³ endpoint bridge and the prototype naturalness energy."""

import math

import numpy as np
import torch
from torch import nn
from slp.geometry import project

from .rotations import normalize, sphere_exp, sphere_log_signed


class Correction(nn.Module):
    def __init__(self, config, joints=41):
        super().__init__()
        self.harmonics = config["time_harmonics"]
        first, h1, h2, last = config["correction_dims"]
        if first != joints * 8 + 1 + 2 * self.harmonics or last != joints * 4:
            raise ValueError("Incompatible correction input/output width")
        self.net = nn.Sequential(
            nn.Linear(first, h1), nn.SiLU(), nn.Linear(h1, h2), nn.SiLU(), nn.Linear(h2, last)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, q0, q1, time):
        t = time.expand(*q0.shape[:-2], 1, 1)[..., 0]
        frequency = math.pi * torch.arange(1, 1 + self.harmonics, device=q0.device, dtype=q0.dtype)
        phase = t * frequency
        inputs = torch.cat((q0.flatten(-2), q1.flatten(-2), t, phase.sin(), phase.cos()), -1)
        return self.net(inputs).reshape_as(q0)


def bridge(q0, q1, time, correction=None, *, interior=False):
    tangent = time * sphere_log_signed(q0, q1)
    if correction is not None:
        tangent = tangent + time * (1 - time) * project(q0, correction(q0, q1, time))
    x = sphere_exp(q0, tangent)
    return x if interior else torch.where(time == 0, q0, torch.where(time == 1, q1, x))


def kmeans(points, count, restarts, max_iter, seed):
    """K-means++ and full Lloyd passes, with bounded assignment memory."""
    rng = np.random.default_rng(seed)
    n, width = points.shape
    if n == 0:
        raise ValueError("Prototype fitting requires valid train frames at every joint")
    effective = min(count, n)
    best = None

    def assign(centers):
        assignments, errors = [], []
        for start in range(0, n, 4096):
            batch = points[start : start + 4096]
            distances = (
                np.sum(batch**2, axis=1, keepdims=True)
                + np.sum(centers**2, axis=1)
                - 2 * (batch @ centers.T)
            )
            idx = distances.argmin(1)
            assignments.append(idx)
            errors.append(distances[np.arange(len(batch)), idx].clip(0))
        return np.concatenate(assignments), np.concatenate(errors)

    for _ in range(restarts):
        centers = np.empty((effective, width), dtype=points.dtype)
        centers[0] = points[rng.integers(n)]
        nearest = np.full(n, np.inf)
        for k in range(1, effective):
            nearest = np.minimum(nearest, np.sum((points - centers[k - 1]) ** 2, axis=-1))
            mass = nearest.sum(dtype=np.float64)
            idx = (
                rng.choice(n, p=nearest.astype(np.float64) / mass) if mass > 0 else rng.integers(n)
            )
            centers[k] = points[idx]
        for iteration in range(1, max_iter + 1):
            assignment, errors = assign(centers)
            counts = np.bincount(assignment, minlength=effective)
            sums = np.zeros((effective, width), dtype=np.float64)
            np.add.at(sums, assignment, points)
            updated = (sums / counts.clip(1)[:, None]).astype(points.dtype)
            empty = counts == 0
            updated[empty] = points[np.argsort(-errors, kind="stable")[: empty.sum()]]
            if np.array_equal(updated, centers):
                break
            centers = updated
        assignment, errors = assign(centers)
        counts = np.bincount(assignment, minlength=effective)
        score = float(errors.sum(dtype=np.float64))
        if best is None or score < best[0]:
            best = (score, centers, counts, iteration)
    _, centers, counts, iteration = best
    padded = np.zeros((count, width), dtype=points.dtype)
    padded[:, 0] = 1
    padded[:effective] = centers
    return padded, np.pad(counts, (0, count - effective)), iteration


def build_bank(clips, cfg, seed, *, checkpoint_dir=None, resume=False):
    clips = sorted(clips, key=lambda c: c.id)

    def fit_joint(j):
        train = torch.cat([c.q[c.q_valid[:, j], j] for c in clips]).cpu().numpy()
        c, n, i = kmeans(
            train,
            cfg["prototypes_per_joint"],
            cfg["kmeans_restarts"],
            cfg["kmeans_max_iter"],
            seed + j,
        )
        return torch.from_numpy(c), torch.from_numpy(n), i

    joints = list(range(41))
    if checkpoint_dir is None:
        results = [fit_joint(j) for j in joints]
    else:
        from .runtime import checkpointed_records

        results = checkpointed_records(
            checkpoint_dir, joints, fit_joint, batch_size=1, resume=resume, label="prototype bank"
        ).values()
    centers, counts, iterations = zip(*results)
    return dict(
        centers=normalize(torch.stack(centers)),
        counts=torch.stack(counts),
        iterations=torch.tensor(iterations),
        seed=torch.tensor(seed),
    )


def naturalness_energy(x, v, bank, sigma, rho, valid, tile=512):
    """Return a masked sum and diagnostics with gradients through position and velocity."""
    x, v = x[valid], v[valid]
    if not len(x):
        return x.sum(), {"valid": 0}
    joints = valid.nonzero()[:, -1]
    centers = bank["centers"].to(x)
    counts = bank["counts"].to(x)
    weights = counts / counts.sum(-1, keepdim=True)
    log_weight = torch.where(counts > 0, weights.clamp_min(1e-30).log(), -torch.inf)

    def energy(xt, vt, c, weight):
        delta = sphere_log_signed(xt[:, None], c)
        log_kernel = weight - delta.square().sum(-1) / (2 * sigma**2)
        log_mass = log_kernel.logsumexp(-1)
        p = (log_kernel - log_mass[:, None]).exp()
        ridge = rho * (-log_mass).exp() * vt.square().sum(-1)
        return (ridge + (p * (delta * vt[:, None]).sum(-1).square()).sum(-1)).sum(), log_mass

    total = x.new_zeros(())
    masses = []
    antipodal = x.new_zeros((), dtype=torch.long)
    for a, b, ind in zip(x.split(tile), v.split(tile), joints.split(tile)):
        # Recompute prototype intermediates in backward to bound retained activation memory.
        if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
            from torch.utils.checkpoint import checkpoint

            value, mass = checkpoint(
                lambda p, w, j: energy(p, w, centers[j], log_weight[j]),
                a,
                b,
                ind,
                use_reentrant=False,
            )
        else:
            value, mass = energy(a, b, centers[ind], log_weight[ind])
        total = total + value
        masses.append(mass.detach())
        antipodal += (a.detach()[:, None] == -centers[ind]).all(-1).sum()
    mass = torch.cat(masses)
    return total, {
        "valid": len(x),
        "log_mass_quantiles": mass.quantile(mass.new_tensor([0, 0.5, 1])).tolist(),
        "speed": float(v.detach().norm(dim=-1).mean()),
        "unit_error": float((x.detach().norm(dim=-1) - 1).abs().max()),
        "antipodal_pairs": int(antipodal),
        "tangent_error": float((x.detach() * v.detach()).sum(-1).abs().max()),
    }


def correction_loss(q0, q1, time, model, bank, config, valid):
    def path(t):
        return bridge(q0, q1, t, model, interior=True)

    x, velocity = torch.func.jvp(path, (time,), (torch.ones_like(time),))
    return naturalness_energy(x, velocity, bank, config["kernel_sigma"], config["ridge_rho"], valid)
