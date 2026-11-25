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
