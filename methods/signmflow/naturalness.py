"""Motion naturalness measure (paper Sec. 3.2, App. C.2-C.3).

Prototypes (Eq. 38-39): for every joint j independently, K-means over the sign-aligned
training quaternions q_n^j in R^4, centroids renormalised onto S^3, frequency weights
w_k^j = |I_k^j| / N (every cluster kept non-empty, so w_k > 0).

Measure (Eq. 7-8), instantiated joint-wise on the product manifold (App. C.2):
    w~_k(x) = w_k exp(-|delta_k|^2 / 2 sigma^2),  delta_k = Log_x(c_k)
    S(x)    = sum_k w~_k(x) delta_k delta_k^T,     m(x) = sum_k w~_k(x)
    G(x)    = m(x) (S(x) + rho I)^-1
Kinetic-energy integrand (Eq. 10/12) through the closed form of Eq. 42:
    v^T G(x)^-1 v = (1/m(x)) (rho |v|^2 + sum_k w~_k(x) <delta_k, v>^2)
computed in log-space (softmax weights) so that no 4x4 inverse and no [.., K, 4] tensor is
materialised: with a_k = <x, c_k> = cos(theta_k), |delta_k| = theta_k and
<delta_k, v> = theta_k / sin(theta_k) (<c_k, v> - a_k <x, v>).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from . import geometry as G


@torch.no_grad()
def kmeans_joint(q: torch.Tensor, K: int, iters: int = 300, tol: float = 1e-4,
                 init_sample: int = 200_000, chunk: int = 1 << 18, seed: int = 0, return_iters: bool = False):
    """Euclidean K-means (Eq. 38) on one joint's quaternions q [N,4] -> (centroids [K,4], counts [K])."""
    g = torch.Generator(device=q.device).manual_seed(seed)
    N = q.shape[0]
    if N < K:
        raise ValueError(f"need at least K={K} frames, got {N}")
    # k-means++ initialisation on a subsample
    sub = q[torch.randperm(N, generator=g, device=q.device)[:min(N, init_sample)]]
    cent = sub[torch.randint(len(sub), (1,), generator=g, device=q.device)]
    d2 = ((sub - cent) ** 2).sum(-1)
    for _ in range(1, K):
        p = d2 / d2.sum().clamp_min(1e-20)
        nxt = sub[torch.multinomial(p, 1, generator=g)]
        cent = torch.cat([cent, nxt], 0)
        d2 = torch.minimum(d2, ((sub - nxt) ** 2).sum(-1))

    def assign(c):
        lab = torch.empty(N, dtype=torch.long, device=q.device)
        dmin = torch.empty(N, device=q.device)
        cn = (c * c).sum(-1)
        for s in range(0, N, chunk):
            x = q[s:s + chunk]
            d = cn[None] - 2 * x @ c.T + (x * x).sum(-1, keepdim=True)
            dmin[s:s + chunk], lab[s:s + chunk] = d.min(-1)
        return lab, dmin.clamp_min(0)

    prev = math.inf
    n_it = 0
    for _ in range(iters):
        n_it += 1
        lab, dmin = assign(cent)
        counts = torch.bincount(lab, minlength=K)
        sums = torch.zeros(K, 4, device=q.device).index_add_(0, lab, q)
        empty = counts == 0
        new = sums / counts.clamp_min(1)[:, None]
        if empty.any():  # re-seed empty clusters with the worst-fit points
            far = dmin.topk(int(empty.sum())).indices
            new[empty] = q[far]
        cent = new
        inertia = float(dmin.mean())  # inertia of the assignment to the previous centroids
        # Lloyd converged: relative change of the inertia below tol (never on the first pass,
        # where prev = inf would give inf <= inf)
        if not empty.any() and math.isfinite(prev) and abs(prev - inertia) <= tol * max(prev, 1e-12):
            break
        prev = inertia
    lab, _ = assign(cent)
    counts = torch.bincount(lab, minlength=K)
    while (counts == 0).any():  # guarantee non-empty clusters (w_k > 0, App. C.3)
        _, dmin = assign(cent)
        e = (counts == 0).nonzero()[:, 0]
        cent[e] = q[dmin.topk(len(e)).indices]
        lab, _ = assign(cent)
        counts = torch.bincount(lab, minlength=K)
    return (cent, counts, n_it) if return_iters else (cent, counts)


@torch.no_grad()
def extract_prototypes(frames: torch.Tensor, K: int, iters: int = 300, seed: int = 0,
                       device: str = "cuda", log=None, tol: float = 1e-4):
    """frames [N,J,4] sign-aligned quaternions -> prototypes c [J,K,4] (unit), weights w [J,K]."""
    N, J, _ = frames.shape
    protos, weights = [], []
    for j in range(J):
        q = frames[:, j].to(device, torch.float32)
        cent, counts, n_it = kmeans_joint(q, K, iters=iters, tol=tol, seed=seed + j, return_iters=True)
        protos.append(G.normalize(cent).cpu())  # c_k^j = c~_k^j / |c~_k^j|
        weights.append((counts.float() / N).cpu())  # w_k^j = |I_k^j| / N
        if log:
            log(f"prototypes joint {j + 1}/{J}: {n_it} Lloyd iterations, min w={weights[-1].min():.2e}")
    return torch.stack(protos), torch.stack(weights)


class NaturalnessMeasure(nn.Module):
    """Joint-wise naturalness measure G on (S^3)^J with K prototypes per joint."""

    def __init__(self, protos: torch.Tensor, weights: torch.Tensor, sigma: float = 0.5,
                 rho: float = 1e-6):
        super().__init__()
        if (weights <= 0).any():
            raise ValueError("frequency weights must be strictly positive (non-empty clusters)")
        dt = torch.get_default_dtype()
        self.register_buffer("protos", protos.to(dt))  # [J,K,4]
        self.register_buffer("log_w", weights.to(dt).log())  # [J,K]
        self.sigma, self.rho = float(sigma), float(rho)

    def _stats(self, x: torch.Tensor):
        a = torch.einsum("...jd,jkd->...jk", x, self.protos)  # cos(theta_k)
        a = a.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(a)  # |delta_k| = d_M(x, c_k)
        logits = self.log_w - theta ** 2 / (2 * self.sigma ** 2)  # log w~_k(x)
        return a, theta, logits

    def log_mass(self, x: torch.Tensor) -> torch.Tensor:
        """log m(x) per joint [..., J]."""
        return torch.logsumexp(self._stats(x)[2], -1)

    def quad_form(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """v^T G(x)^-1 v per joint [..., J] (Eq. 42). x on (S^3)^J, v in T_x."""
        a, theta, logits = self._stats(x)
        log_m = torch.logsumexp(logits, -1)
        pi = torch.softmax(logits, -1)  # w~_k(x) / m(x)
        cv = torch.einsum("...jd,jkd->...jk", v, self.protos)
        xv = (x * v).sum(-1, keepdim=True)
        dv = theta / torch.sin(theta) * (cv - a * xv)  # <delta_k, v>
        return (pi * dv ** 2).sum(-1) + self.rho * (v * v).sum(-1) * torch.exp(-log_m)

    def metric_inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Dense G(x)^-1 = (S(x) + rho I) / m(x) per joint [..., J, 4, 4] (tests / analysis)."""
        _, _, logits = self._stats(x)
        pi = torch.softmax(logits, -1)
        log_m = torch.logsumexp(logits, -1)
        delta = G.logmap(x[..., None, :], self.protos.expand(*x.shape[:-1], -1, -1))
        S = torch.einsum("...k,...kd,...ke->...de", pi, delta, delta)
        eye = torch.eye(4, device=x.device, dtype=x.dtype)
        return S + self.rho * torch.exp(-log_m)[..., None, None] * eye
