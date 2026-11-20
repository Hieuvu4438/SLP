"""Conditional paths and the correction network epsilon_theta (paper Sec. 3.3, App. C.4).

Naturalness-guided interpolation (Eq. 11):
    x_t = Exp_{x0}( t Log_{x0}(x1) + t(1-t) P_{x0} eps~_theta(t, x0, x1) )
with the orthogonal tangent projection P_{x0} of Eq. 48.  ``geodesic`` is the same path
with eps = 0 (ablation GE), ``linear`` is (1-t) x0 + t x1 (CFM baseline / ablation MR).

All paths act frame-wise: x0, x1 [..., J, D]; t broadcastable to [..., 1, 1].
"""

from __future__ import annotations

import torch
from torch import nn

from . import geometry as G


class CorrectionNet(nn.Module):
    """epsilon_theta(t, x0, x1): 3-layer MLP, hidden 512 (Sec. 4.1, Table 6), frame-wise.

    The last layer is zero-initialised so training starts from the geodesic (eps = 0)."""

    def __init__(self, n_joints: int = 41, hidden: int = 512, layers: int = 3):
        super().__init__()
        d = n_joints * 4
        dims = [1 + 2 * d] + [hidden] * (layers - 1) + [d]
        mods = []
        for i in range(layers):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < layers - 1:
                mods.append(nn.SiLU())
        self.net = nn.Sequential(*mods)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.n_joints = n_joints

    def forward(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Projected correction eps_theta = P_{x0} eps~_theta, shape like x0."""
        lead = x0.shape[:-2]
        tt = t.expand(*lead, 1, 1).reshape(*lead, 1)
        h = torch.cat([tt, x0.flatten(-2), x1.flatten(-2)], -1)
        raw = self.net(h).reshape(*lead, self.n_joints, 4)
        return G.project(x0, raw)


def path_point(kind: str, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor,
               corr: CorrectionNet | None = None) -> torch.Tensor:
    """x_t on the chosen conditional path. t has shape [..., 1, 1] (broadcast over joints)."""
    if kind == "linear":
        return (1 - t) * x0 + t * x1
    xi = G.logmap(x0, x1)
    if kind == "geodesic" or corr is None:
        if kind == "naturalness":
            raise ValueError("naturalness path needs a correction network")
        return G.expmap(x0, t * xi)
    if kind != "naturalness":
        raise ValueError(kind)
    return G.expmap(x0, t * xi + t * (1 - t) * corr(t, x0, x1))


def path_and_velocity(kind: str, x0, x1, t, corr: CorrectionNet | None = None):
    """(x_t, dx_t/dt) via forward-mode automatic differentiation (Sec. 3.3)."""
    if kind == "linear":
        return (1 - t) * x0 + t * x1, (x1 - x0).expand_as(x0)
    return torch.func.jvp(lambda tt: path_point(kind, x0, x1, tt, corr), (t,), (torch.ones_like(t),))


def velocity_chain_rule(x0, x1, t, corr: CorrectionNet):
    """Analytic velocity of Alg. 1, line 5: DExp_{x0}(omega)[xi + (1-2t) zeta + t(1-t) d zeta/dt].

    Provided as an independent check of the autodiff velocity (used in the unit tests)."""
    xi = G.logmap(x0, x1)
    zeta, dzeta = torch.func.jvp(lambda tt: corr(tt, x0, x1), (t,), (torch.ones_like(t),))
    omega = t * xi + t * (1 - t) * zeta
    domega = xi + (1 - 2 * t) * zeta + t * (1 - t) * dzeta
    # differential of Exp_{x0} at omega applied to domega (sphere closed form)
    n = omega.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    u = omega / n
    par = (u * domega).sum(-1, keepdim=True)
    perp = domega - par * u
    return (-torch.sin(n) * par) * x0 + torch.cos(n) * par * u + torch.sin(n) / n * perp
