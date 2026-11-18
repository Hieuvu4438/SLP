"""SO(3) primitives with finite derivatives at identity and a stable pi branch.

Exp/Log use radians. Log selects the principal branch; it is necessarily
discontinuous at the cut locus. Geometry should run in float32 or float64.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def hat(v):
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).unflatten(-1, (3, 3))


def vee(m):
    return torch.stack((m[..., 2, 1], m[..., 0, 2], m[..., 1, 0]), -1)


def exp(v):
    q = v.square().sum(-1)
    t = q.clamp_min(1e-12).sqrt()
    a = torch.where(q < 1e-6, 1 - q / 6 + q.square() / 120, t.sin() / t)
    b = torch.where(
        q < 1e-6, 0.5 - q / 24 + q.square() / 720, 2 * (t / 2).sin().square() / t.square()
    )
    k = hat(v)
    eye = torch.eye(3, dtype=v.dtype, device=v.device)
    return eye + a[..., None, None] * k + b[..., None, None] * (k @ k)


def log(r):
    """Stable matrix -> wxyz quaternion -> principal rotation vector."""
    a, b, c = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    sq = torch.stack((1 + a + b + c, 1 + a - b - c, 1 - a + b - c, 1 - a - b + c), -1)
    # Inactive candidates also need finite derivatives for backward / double backward.
    roots = sq.clamp_min(1e-12).sqrt()
    cand = torch.stack(
        (
            torch.stack(
                (
                    sq[..., 0],
                    r[..., 2, 1] - r[..., 1, 2],
                    r[..., 0, 2] - r[..., 2, 0],
                    r[..., 1, 0] - r[..., 0, 1],
                ),
                -1,
            ),
            torch.stack(
                (
                    r[..., 2, 1] - r[..., 1, 2],
                    sq[..., 1],
                    r[..., 1, 0] + r[..., 0, 1],
                    r[..., 0, 2] + r[..., 2, 0],
                ),
                -1,
            ),
            torch.stack(
                (
                    r[..., 0, 2] - r[..., 2, 0],
                    r[..., 1, 0] + r[..., 0, 1],
                    sq[..., 2],
                    r[..., 2, 1] + r[..., 1, 2],
                ),
                -1,
            ),
            torch.stack(
                (
                    r[..., 1, 0] - r[..., 0, 1],
                    r[..., 2, 0] + r[..., 0, 2],
                    r[..., 2, 1] + r[..., 1, 2],
                    sq[..., 3],
                ),
                -1,
            ),
        ),
        -2,
    )
    cand = cand / (2 * roots[..., :, None])
    idx = sq.argmax(-1)
    q = (cand * F.one_hot(idx, 4)[..., :, None]).sum(-2)
    q = F.normalize(q, dim=-1)
    q = torch.where(q[..., :1] < 0, -q, q)
    v, w = q[..., 1:], q[..., :1]
    n2 = v.square().sum(-1, keepdim=True)
    n = n2.clamp_min(1e-12).sqrt()
    ratio = 2 * torch.atan2(n, w) / n
    ratio = torch.where(n2 < 1e-8, 2 + n2 / 3 + 3 * n2.square() / 20, ratio)
    return ratio * v


def to_6d(r):
    return r[..., :, :2].transpose(-1, -2).flatten(-2)


def from_6d(x, eps=1e-6, return_degenerate=False):
    """Gram-Schmidt columns with deterministic fallback for zero/collinear input."""
    a, b = x[..., :3], x[..., 3:]
    bad_a = a.norm(dim=-1, keepdim=True) < eps
    e1 = torch.zeros_like(a)
    e1[..., 0] = 1
    u = F.normalize(torch.where(bad_a, e1, a), dim=-1, eps=eps)
    v = b - (u * b).sum(-1, keepdim=True) * u
    bad_b = v.norm(dim=-1, keepdim=True) < eps
    axis = F.one_hot(u.abs().argmin(-1), 3).to(u)
    fallback = axis - (axis * u).sum(-1, keepdim=True) * u
    v = F.normalize(torch.where(bad_b, fallback, v), dim=-1, eps=eps)
    r = torch.stack((u, v, torch.linalg.cross(u, v)), -1)
    return (r, (bad_a | bad_b).squeeze(-1)) if return_degenerate else r


def distance(a, b):
    return log(a.transpose(-1, -2) @ b).norm(dim=-1)


def assert_valid(r, atol=2e-4):
    eye = torch.eye(3, device=r.device, dtype=r.dtype)
    if (
        not torch.isfinite(r).all()
        or not torch.allclose(r.transpose(-1, -2) @ r, eye.expand_as(r), atol=atol)
        or not torch.allclose(torch.linalg.det(r), torch.ones_like(r[..., 0, 0]), atol=atol)
    ):
        raise ValueError("Expected finite SO(3) rotation matrices")
