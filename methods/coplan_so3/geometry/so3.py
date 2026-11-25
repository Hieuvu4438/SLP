"""SO(3) primitives with finite derivatives at identity and a stable pi branch.

Exp/Log use radians. Log selects the principal branch; it is necessarily
discontinuous at the cut locus. Geometry should run in float32 or float64.
"""

from __future__ import annotations

import torch
from slp.geometry import matrix_to_6d as to_6d
from slp.geometry import matrix_to_aa as log
from slp.geometry import rotation_6d_to_matrix as from_6d


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
