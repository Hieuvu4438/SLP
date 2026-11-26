"""Rectified-flow training pairs and ODE samplers (MoRAE Eq. 3, RK4 on a uniform grid)."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

SOLVERS = ("euler", "heun", "rk4")
EVALS_PER_STEP = {"euler": 1, "heun": 2, "rk4": 4}


def sample_time(n: int, mode: str, device, generator: torch.Generator | None = None) -> Tensor:
    if mode == "uniform":
        return torch.rand(n, device=device, generator=generator)
    if mode == "logit_normal":
        return torch.sigmoid(torch.randn(n, device=device, generator=generator))
    raise ValueError("flow.time_sampling must be uniform or logit_normal")


def training_pair(q: Tensor, noise: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """z_t = (1 - t) eps + t z and the velocity target z - eps."""
    tt = t.view(-1, *([1] * (q.dim() - 1))).to(q.dtype)
    return (1 - tt) * noise + tt * q, q - noise


def cfg_combine(conditional: Tensor, unconditional: Tensor, scale: float) -> Tensor:
    """Classifier-free guidance v_u + w (v_c - v_u)."""
    return unconditional + scale * (conditional - unconditional)


def integrate(field: Callable[[Tensor, Tensor], Tensor], x: Tensor, steps: int, solver: str,
              valid: Tensor | None = None) -> Tensor:
    """Integrate dx/dt = field(x, t) from t = 0 to 1 on a uniform grid of ``steps`` intervals."""
    if solver not in SOLVERS:
        raise ValueError(f"sampler.solver must be one of {SOLVERS}")
    if steps < 1:
        raise ValueError("sampler.steps must be positive")
    keep = None if valid is None else valid.unsqueeze(-1).to(x.dtype)

    def f(state: Tensor, time: float) -> Tensor:
        v = field(state, torch.full((len(state),), time, device=state.device))
        return v if keep is None else v * keep

    grid = torch.linspace(0.0, 1.0, steps + 1).tolist()
    for t0, t1 in zip(grid[:-1], grid[1:]):
        h = t1 - t0
        if solver == "euler":
            x = x + h * f(x, t0)
        elif solver == "heun":
            k1 = f(x, t0)
            k2 = f(x + h * k1, t1)
            x = x + 0.5 * h * (k1 + k2)
        else:
            k1 = f(x, t0)
            k2 = f(x + 0.5 * h * k1, t0 + 0.5 * h)
            k3 = f(x + 0.5 * h * k2, t0 + 0.5 * h)
            k4 = f(x + h * k3, t1)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x if keep is None else x * keep


# --------------------------------------------------------------------------------------------
# v3 (PLAN_V3 C1): target (x1) prediction and the re-planned samplers of SignNMFlow (App. C.5)

PREDICTIONS = ("velocity", "target")
REPLAN = ("replan_implied", "replan_anchored")


def to_target(output: Tensor, x: Tensor, t: Tensor, prediction: str) -> Tensor:
    """Network output -> x1 estimate. velocity: x1 = x_t + (1 - t) v (exact on the linear path)."""
    if prediction == "target":
        return output
    if prediction == "velocity":
        return x + (1 - t.view(-1, *([1] * (x.dim() - 1))).to(x.dtype)) * output
    raise ValueError(f"flow.prediction must be one of {PREDICTIONS}")


def to_velocity(output: Tensor, x: Tensor, t: Tensor, prediction: str,
                min_gap: float = 1e-2) -> Tensor:
    """Network output -> velocity. target: v = (x1 - x_t) / (1 - t), with 1 - t >= ``min_gap``
    (an ODE solver that evaluates t = 1 would otherwise divide by zero)."""
    if prediction == "velocity":
        return output
    if prediction == "target":
        gap = (1 - t.view(-1, *([1] * (x.dim() - 1))).to(x.dtype)).clamp_min(min_gap)
        return (output - x) / gap
    raise ValueError(f"flow.prediction must be one of {PREDICTIONS}")


def target_weight(t: Tensor, weighting: str, cap: float = 100.0) -> Tensor:
    """Per-sample weight of the x1 loss. 'uniform' = SignNMFlow Eq. 13; 'velocity' = 1/(1-t)^2,
    which makes ||x1 - x1_hat||^2 equal to the velocity loss (capped near t = 1)."""
    if weighting == "uniform":
        return torch.ones_like(t)
    if weighting == "velocity":
        return (1.0 / (1 - t).square().clamp_min(1.0 / cap))
    raise ValueError("flow.target_weighting must be uniform or velocity")


def replan(predict_x1: Callable[[Tensor, Tensor], Tensor], x0: Tensor, steps: int, mode: str,
           valid: Tensor | None = None) -> Tensor:
    """Re-planned sampling on the uniform grid t_k = k / steps (never evaluates t = 1).

    replan_implied  : x_{t+D} = x_t + D (x1_hat - x_t) / (1 - t)   (SignNMFlow expression channel,
                      Alg. 2 l.13; Euler on the implied velocity, a DDIM-type update)
    replan_anchored : x_{t+D} = (1 - t - D) x0 + (t + D) x1_hat     (SignNMFlow pose channel, Eq. 53,
                      linear-path analogue: the state is re-placed on the path from the kept noise)
    The first step of both rules is the same, (1 - D) x0 + D x1_hat, and the last step of both
    returns the last x1_hat; they differ only from a 3-step grid on.
    """
    if mode not in REPLAN:
        raise ValueError(f"replan mode must be one of {REPLAN}")
    if steps < 1:
        raise ValueError("sampler.steps must be positive")
    keep = None if valid is None else valid.unsqueeze(-1).to(x0.dtype)
    x = x0
    grid = torch.linspace(0.0, 1.0, steps + 1).tolist()
    for t0, t1 in zip(grid[:-1], grid[1:]):
        x1 = predict_x1(x, torch.full((len(x),), t0, device=x.device))
        if keep is not None:
            x1 = x1 * keep
        if mode == "replan_implied":
            x = x + (t1 - t0) * (x1 - x) / (1 - t0)
        else:
            x = (1 - t1) * x0 + t1 * x1
    return x if keep is None else x * keep
