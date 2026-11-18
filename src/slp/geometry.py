"""Unit-quaternion geometry on S^3 and (S^3)^J, shared by rotation-manifold methods.

Quaternions are scalar-first (w, x, y, z). Every function works on arbitrary leading
dimensions; the last dimension is the per-joint coordinate (3 = axis-angle, 4 = quaternion,
3x3 = rotation matrix). Exp/Log are the Riemannian maps of the unit sphere S^3 embedded in
R^4 (a quaternion geodesic distance d is a rotation angle of 2d).
"""

from __future__ import annotations

import torch

EPS = 1e-8


# --------------------------------------------------------------------------- conversions
def aa_to_quat(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle (rotation vector) -> unit quaternion (w >= 0 for |angle| <= pi)."""
    angle = aa.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    small = angle < 1e-4  # sin(angle/2)/angle ~ 1/2 - angle^2/48
    k = torch.where(small, 0.5 - angle * angle / 48.0, torch.sin(half) / angle.clamp_min(1e-12))
    return torch.cat([torch.cos(half), k * aa], -1)


def quat_to_aa(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion -> axis-angle in the principal branch (angle in [0, pi])."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(EPS)
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = v.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s, q[..., :1])
    small = s < 1e-6
    k = torch.where(small, 2.0 / q[..., :1].clamp_min(EPS), angle / s.clamp_min(1e-12))
    return k * v


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (any sign; normalised here) -> rotation matrix [..., 3, 3]."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(EPS)
    w, x, y, z = q.unbind(-1)
    m = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
    return m.reshape(*q.shape[:-1], 3, 3)


def aa_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    return quat_to_matrix(aa_to_quat(aa))


def sign_convention(q: torch.Tensor) -> torch.Tensor:
    """Canonical representative of {q, -q}: the first non-zero component is positive."""
    sign = torch.zeros_like(q[..., 0])
    for i in range(4):
        c = q[..., i]
        sign = torch.where((sign == 0) & (c != 0), torch.sign(c), sign)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return q * sign[..., None]


def align_sign(q: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Representative of {q, -q} in the hemisphere of ``ref`` (<q, ref> >= 0).

    q and -q are the same rotation; after alignment Log_ref(q) is the shortest rotation."""
    s = torch.where(inner(q, ref) < 0, -1.0, 1.0)
    return q * s


# --------------------------------------------------------------------------- S^3
def inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1, keepdim=True)


def project(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Orthogonal projection of an ambient vector u onto the tangent space T_x S^3."""
    return u - inner(x, u) * x


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Radial projection onto S^3 (per joint)."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(EPS)


def expmap(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Exp_x(v) = cos|v| x + sin|v|/|v| v for a tangent vector v at x."""
    n = v.norm(dim=-1, keepdim=True)
    small = n < 1e-4
    sinc = torch.where(small, 1.0 - n * n / 6.0, torch.sin(n) / n.clamp_min(1e-12))
    return torch.cos(n) * x + sinc * v


def logmap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Log_x(y) in T_x S^3 (y not antipodal to x)."""
    c = inner(x, y).clamp(-1.0, 1.0)
    u = y - c * x
    s = u.norm(dim=-1, keepdim=True)
    theta = torch.atan2(s, c)
    small = s < 1e-6
    k = torch.where(small, 1.0 + theta * theta / 6.0, theta / s.clamp_min(1e-12))
    return k * u


def geodesic(x: torch.Tensor, y: torch.Tensor, t) -> torch.Tensor:
    """Point at fraction t of the geodesic x -> y: Exp_x(t Log_x y) (t may extrapolate)."""
    return expmap(x, t * logmap(x, y))


def geodesic_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    c = inner(x, y).clamp(-1.0, 1.0)[..., 0]
    s = (y - c[..., None] * x).norm(dim=-1)
    return torch.atan2(s, c)


def uniform_hemisphere(shape, device=None, generator=None) -> torch.Tensor:
    """Uniform on S^3 restricted to the sign-convention hemisphere; shape [..., 4]."""
    g = torch.randn(*shape, 4, device=device, generator=generator)
    return sign_convention(normalize(g))


def tangent_gaussian(x: torch.Tensor, sigma, generator=None) -> torch.Tensor:
    """xi ~ N(0, sigma^2 I_3) in T_x S^3 (isotropic in the 3-D tangent space).

    ``sigma`` broadcasts against x[..., :1] (e.g. a per-joint [J, 1] tensor or a float)."""
    g = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return project(x, g) * sigma
