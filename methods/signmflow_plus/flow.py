"""Training objective and samplers of the generation network (Sec. 3.3, App. C.5, Alg. 2).

Variants (used by the ablations of Tables 3 and 5):
    representation: quat (S^3, p0 = uniform hemisphere) | 6d (Euclidean, p0 = N(0, I))
    path          : naturalness (Eq. 11) | geodesic (eps = 0, "GE") | linear (CFM)
                    + geoshift | ng_parallel | linear_renorm (docs/proposal7, see paths.py)
    prediction    : target (Eq. 13, default) | velocity (regress dx_t/dt)
The expression channel always uses e0 ~ N(0, I) and e_t = (1-t) e0 + t e1.

Noise-level schedules (docs/proposal7 E2-F), all identity by default:
    train_shift : training times t = kappa_s(u), u ~ U[0, 1] (SD3 shift, paths.time_shift)
    sampler grid: t_k = kappa_shift((k / N) ** grid_power), see ``time_grid``
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import geometry as G
from .paths import NEEDS_CORRECTION, PATHS, CorrectionNet, path_and_velocity, path_point, time_shift


@dataclass(frozen=True)
class FlowSpec:
    representation: str = "quat"
    path: str = "naturalness"
    prediction: str = "target"
    w_pose: float = 10.0
    w_expr: float = 0.1
    reduction: str = "sum"   # per-frame ||.||^2 summed over channels (Alg. 2) | "mean" over channels
    source: str = "hemisphere"  # p0: hemisphere (paper) | wrapped_normal (E6, around the mean pose)
    source_sigma: float = 0.5   # sigma0 of the wrapped normal (radians in the tangent space)
    shift: float = 1.0       # s of the geoshift path
    train_shift: float = 1.0  # SD3 shift of the training-time distribution (1 = uniform, paper)

    def __post_init__(self):
        assert self.reduction in ("sum", "mean")
        assert self.representation in ("quat", "6d")
        assert self.path in PATHS, self.path
        assert self.prediction in ("target", "velocity")
        assert self.shift > 0 and self.train_shift > 0
        assert self.source in ("hemisphere", "wrapped_normal")
        if self.source != "hemisphere" and self.representation != "quat":
            raise ValueError("the wrapped-normal source is defined on (S^3)^J only")
        if self.representation == "6d" and self.path != "linear":
            raise ValueError("6D rotations are Euclidean: only the linear path is defined")

    @property
    def on_manifold(self) -> bool:
        return self.representation == "quat"

    @property
    def needs_correction(self) -> bool:
        return self.path in NEEDS_CORRECTION

    @property
    def renormalise(self) -> bool:
        """x_t is kept on (S^3)^J (every path except the paper's linear MR/CFM path)."""
        return self.on_manifold and self.path != "linear"


def time_grid(steps: int, grid_power: float = 1.0, shift: float = 1.0) -> list[float]:
    """Sampler times t_0 = 0 < ... < t_N = 1. Defaults give the paper's uniform grid k * (1 / N)."""
    dt = 1.0 / steps
    out = []
    for k in range(steps + 1):
        u = k * dt
        if grid_power != 1.0:
            u = u ** grid_power
        out.append(float(time_shift(u, shift)) if shift != 1.0 else u)
    return out


def sample_source(spec: FlowSpec, shape, device, generator=None, mean=None) -> torch.Tensor:
    """x0 ~ p0 with shape [..., J] + rotation dim. ``mean`` [J,4]: centre of the wrapped normal (C8)."""
    if spec.representation == "quat":
        if spec.source == "wrapped_normal":
            if mean is None:
                raise ValueError("the wrapped-normal source needs the mean pose (source_mean)")
            return G.wrapped_normal(mean, spec.source_sigma, shape, device=device, generator=generator)
        return G.uniform_hemisphere(shape, device=device, generator=generator)
    return torch.randn(*shape, 6, device=device, generator=generator)


def _masked_mse(a, b, mask, reduction="sum", denom=None, joint_weights=None):
    """Per-frame squared error (summed or averaged over channels), averaged over valid frames.

    mask [B,T] True = valid. ``denom`` (number of valid frames) defaults to mask.sum(); passing the
    frame count of a larger batch makes micro-batch losses add up to that batch's loss exactly.
    ``joint_weights`` [J] (mean 1, C5) multiplies each joint's squared error (a, b: [B,T,J,D])."""
    d = (a - b) ** 2
    if joint_weights is not None:
        d = d * joint_weights.to(d)[:, None]
    d = d.flatten(2)
    se = d.sum(-1) if reduction == "sum" else d.mean(-1)
    n = mask.sum().clamp_min(1) if denom is None else denom
    return (se * mask).sum() / n


def flow_loss(gen, corr: CorrectionNet | None, spec: FlowSpec, x1, e1, mask, text, text_pad, denom=None,
              joint_weights=None, position_loss=None, source_mean=None):
    """Phase 1 of Alg. 2: 10 |x1 - x^1|^2 + 0.1 |e1 - e^1|^2 (target) or velocity regression.

    ``denom``: valid-frame count to normalise by (default: this batch's).
    ``joint_weights``: optional [J] weights of the pose term (C5, levers.py).
    ``position_loss``: optional fk.PositionLoss on the predicted clean sample (C6, target only)."""
    B, T = x1.shape[:2]
    dev = x1.device
    x0 = sample_source(spec, (B, T, x1.shape[2]), dev, mean=source_mean)
    e0 = torch.randn_like(e1)
    t = time_shift(torch.rand(B, device=dev), spec.train_shift)
    tt = t[:, None, None, None]
    with torch.no_grad():  # the correction network is frozen
        if spec.prediction == "target":
            x_t = path_point(spec.path, x0, x1, tt, corr, spec.shift)
            u_t = None
        else:
            x_t, u_t = path_and_velocity(spec.path, x0, x1, tt, corr, spec.shift)
        if spec.renormalise:
            x_t = G.normalize(x_t)
    e_t = (1 - t[:, None, None]) * e0 + t[:, None, None] * e1
    pose, expr = gen(x_t, e_t, t, text, text_pad, ~mask)
    if spec.prediction == "target":
        x_hat = G.normalize(pose) if spec.on_manifold else pose
        lp = _masked_mse(x_hat, x1, mask, spec.reduction, denom, joint_weights)
        le = _masked_mse(expr, e1, mask, spec.reduction, denom)
        if position_loss is not None:
            lfk, fk_parts = position_loss(x_hat, expr, x1, e1, mask, denom)
            return (spec.w_pose * lp + spec.w_expr * le + lfk,
                    {"loss_pose": lp.detach(), "loss_expr": le.detach(), **fk_parts})
    else:
        if position_loss is not None:
            raise ValueError("position (FK) losses need the target-prediction parametrisation")
        v = G.project(x_t, pose) if spec.renormalise else pose
        lp = _masked_mse(v, u_t, mask, spec.reduction, denom, joint_weights)
        le = _masked_mse(expr, e1 - e0, mask, spec.reduction, denom)
    return spec.w_pose * lp + spec.w_expr * le, {"loss_pose": lp.detach(), "loss_expr": le.detach()}


@torch.no_grad()
def guided(gen, x, e, tb, text, text_pad, pad, spec: FlowSpec, cfg_w: float):
    """Network outputs with classifier-free guidance (C7 / E5).

    Target prediction on the manifold: x^ = Exp_{x^u}(w Log_{x^u} x^c) per joint (geodesic
    extrapolation from the unconditional towards the conditional clean estimate); Euclidean
    outputs (6D, velocities, expression): u + w (c - u). w = 1 is the plain conditional model."""
    pose, expr = gen(x, e, tb, text, text_pad, pad)
    if cfg_w == 1.0:
        return pose, expr
    B = x.shape[0]
    null, null_pad = gen.drop_text(text, text_pad, torch.ones(B, dtype=torch.bool))
    pose_u, expr_u = gen(x, e, tb, null, null_pad, pad)
    if spec.prediction == "target" and spec.on_manifold:
        xc, xu = G.normalize(pose), G.normalize(pose_u)
        pose = G.expmap(xu, cfg_w * G.logmap(xu, xc))
    else:
        pose = pose_u + cfg_w * (pose - pose_u)
    return pose, expr_u + cfg_w * (expr - expr_u)


def sample(gen, corr: CorrectionNet | None, spec: FlowSpec, text, text_pad, lengths,
           steps: int = 5, n_joints: int = 41, expr_dim: int = 10, generator=None,
           grid_power: float = 1.0, grid_shift: float = 1.0, cfg_w: float = 1.0, source_mean=None):
    """Phase 2 of Alg. 2 (re-planned path-consistent transport) / Euler for velocity variants.

    ``grid_power`` / ``grid_shift`` change the time grid (``time_grid``); defaults = paper.
    ``cfg_w``: classifier-free guidance weight (1 = none; > 1 doubles the network calls).
    Returns rotations [B,T,J,D] and expressions [B,T,E] padded to max(lengths)."""
    B, T = len(lengths), int(max(lengths))
    dev = text.device
    mask = torch.arange(T, device=dev)[None] < torch.as_tensor(lengths, device=dev)[:, None]
    x0 = sample_source(spec, (B, T, n_joints), dev, generator, mean=source_mean)
    e = torch.randn(B, T, expr_dim, device=dev, generator=generator)
    x = x0
    grid = time_grid(steps, grid_power, grid_shift)
    uniform = grid_power == 1.0 and grid_shift == 1.0
    for k in range(steps):
        t = grid[k]
        if uniform:  # exactly the original loop's floating-point times (t = k dt, s = t + dt)
            dt = 1.0 / steps
            tn = t + dt
        else:
            tn = grid[k + 1]
            dt = tn - t
        tb = torch.full((B,), t, device=dev)
        pose, expr = guided(gen, x, e, tb, text, text_pad, ~mask, spec, cfg_w)
        s = torch.full((B, 1, 1, 1), tn, device=dev)
        if spec.prediction == "target":
            x1_hat = G.normalize(pose) if spec.on_manifold else pose
            if spec.path == "linear":
                x = x + dt / (1 - t) * (x1_hat - x)
            else:  # Eq. 53: place the state on the path of Eq. 11 for the pair (x0, x^1)
                x = path_point(spec.path, x0, x1_hat, s, corr, spec.shift)
            e = e + dt / (1 - t) * (expr - e)
        else:
            if spec.renormalise:
                x = G.expmap(x, dt * G.project(x, pose))
            else:
                x = x + dt * pose
            e = e + dt * expr
        if spec.renormalise:
            x = G.normalize(x)  # numerical safeguard (Alg. 2, line 14)
    if spec.on_manifold:
        x = G.normalize(x)  # linear (MR) variant: project the final state onto M
    return x, e, mask
