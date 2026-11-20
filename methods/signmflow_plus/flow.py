"""Training objective and samplers of the generation network (Sec. 3.3, App. C.5, Alg. 2).

Variants (used by the ablations of Tables 3 and 5):
    representation: quat (S^3, p0 = uniform hemisphere) | 6d (Euclidean, p0 = N(0, I))
    path          : naturalness (Eq. 11) | geodesic (eps = 0, "GE") | linear (CFM)
    prediction    : target (Eq. 13, default) | velocity (regress dx_t/dt)
The expression channel always uses e0 ~ N(0, I) and e_t = (1-t) e0 + t e1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import geometry as G
from .paths import CorrectionNet, path_and_velocity, path_point


@dataclass(frozen=True)
class FlowSpec:
    representation: str = "quat"
    path: str = "naturalness"
    prediction: str = "target"
    w_pose: float = 10.0
    w_expr: float = 0.1
    reduction: str = "sum"   # per-frame ||.||^2 summed over channels (Alg. 2) | "mean" over channels

    def __post_init__(self):
        assert self.reduction in ("sum", "mean")
        assert self.representation in ("quat", "6d")
        assert self.path in ("naturalness", "geodesic", "linear")
        assert self.prediction in ("target", "velocity")
        if self.representation == "6d" and self.path != "linear":
            raise ValueError("6D rotations are Euclidean: only the linear path is defined")

    @property
    def on_manifold(self) -> bool:
        return self.representation == "quat"


def sample_source(spec: FlowSpec, shape, device, generator=None) -> torch.Tensor:
    """x0 ~ p0 with shape [..., J] + rotation dim."""
    if spec.representation == "quat":
        return G.uniform_hemisphere(shape, device=device, generator=generator)
    return torch.randn(*shape, 6, device=device, generator=generator)


def _masked_mse(a, b, mask, reduction="sum", denom=None):
    """Per-frame squared error (summed or averaged over channels), averaged over valid frames.

    mask [B,T] True = valid. ``denom`` (number of valid frames) defaults to mask.sum(); passing the
    frame count of a larger batch makes micro-batch losses add up to that batch's loss exactly."""
    d = ((a - b) ** 2).flatten(2)
    se = d.sum(-1) if reduction == "sum" else d.mean(-1)
    n = mask.sum().clamp_min(1) if denom is None else denom
    return (se * mask).sum() / n


def flow_loss(gen, corr: CorrectionNet | None, spec: FlowSpec, x1, e1, mask, text, text_pad, denom=None):
    """Phase 1 of Alg. 2: 10 |x1 - x^1|^2 + 0.1 |e1 - e^1|^2 (target) or velocity regression.

    ``denom``: valid-frame count to normalise by (default: this batch's)."""
    B, T = x1.shape[:2]
    dev = x1.device
    x0 = sample_source(spec, (B, T, x1.shape[2]), dev)
    e0 = torch.randn_like(e1)
    t = torch.rand(B, device=dev)
    tt = t[:, None, None, None]
    with torch.no_grad():  # the correction network is frozen
        if spec.prediction == "target":
            x_t = path_point(spec.path, x0, x1, tt, corr)
            u_t = None
        else:
            x_t, u_t = path_and_velocity(spec.path, x0, x1, tt, corr)
        if spec.on_manifold and spec.path != "linear":
            x_t = G.normalize(x_t)
    e_t = (1 - t[:, None, None]) * e0 + t[:, None, None] * e1
    pose, expr = gen(x_t, e_t, t, text, text_pad, ~mask)
    if spec.prediction == "target":
        x_hat = G.normalize(pose) if spec.on_manifold else pose
        lp = _masked_mse(x_hat, x1, mask, spec.reduction, denom)
        le = _masked_mse(expr, e1, mask, spec.reduction, denom)
    else:
        v = G.project(x_t, pose) if spec.on_manifold and spec.path != "linear" else pose
        lp = _masked_mse(v, u_t, mask, spec.reduction, denom)
        le = _masked_mse(expr, e1 - e0, mask, spec.reduction, denom)
    return spec.w_pose * lp + spec.w_expr * le, {"loss_pose": lp.detach(), "loss_expr": le.detach()}


@torch.no_grad()
def sample(gen, corr: CorrectionNet | None, spec: FlowSpec, text, text_pad, lengths,
           steps: int = 5, n_joints: int = 41, expr_dim: int = 10, generator=None):
    """Phase 2 of Alg. 2 (re-planned path-consistent transport) / Euler for velocity variants.

    Returns rotations [B,T,J,D] and expressions [B,T,E] padded to max(lengths)."""
    B, T = len(lengths), int(max(lengths))
    dev = text.device
    mask = torch.arange(T, device=dev)[None] < torch.as_tensor(lengths, device=dev)[:, None]
    x0 = sample_source(spec, (B, T, n_joints), dev, generator)
    e = torch.randn(B, T, expr_dim, device=dev, generator=generator)
    x = x0
    dt = 1.0 / steps
    for k in range(steps):
        t = k * dt
        tb = torch.full((B,), t, device=dev)
        pose, expr = gen(x, e, tb, text, text_pad, ~mask)
        s = torch.full((B, 1, 1, 1), t + dt, device=dev)
        if spec.prediction == "target":
            x1_hat = G.normalize(pose) if spec.on_manifold else pose
            if spec.path == "linear":
                x = x + dt / (1 - t) * (x1_hat - x)
            else:  # Eq. 53: place the state on the path of Eq. 11 for the pair (x0, x^1)
                x = path_point(spec.path, x0, x1_hat, s, corr)
            e = e + dt / (1 - t) * (expr - e)
        else:
            if spec.on_manifold and spec.path != "linear":
                x = G.expmap(x, dt * G.project(x, pose))
            else:
                x = x + dt * pose
            e = e + dt * expr
        if spec.on_manifold and spec.path != "linear":
            x = G.normalize(x)  # numerical safeguard (Alg. 2, line 14)
    if spec.on_manifold:
        x = G.normalize(x)  # linear (MR) variant: project the final state onto M
    return x, e, mask
