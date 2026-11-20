"""Rotation representations and the motion manifold M = (S^3)^J (paper Sec. 3.1, App. C).

Quaternions are scalar-first (w, x, y, z). Every function works on arbitrary leading
dimensions; the last dimension is the per-joint coordinate (3 = axis-angle, 4 =
quaternion, 6 = 6D rotation, 3x3 = matrix).
"""

from __future__ import annotations

import torch

EPS = 1e-8


# --------------------------------------------------------------------------- quaternions
def aa_to_quat(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle (rotation vector) -> unit quaternion (w>=0 for |angle|<=pi)."""
    squared = aa.square().sum(-1, keepdim=True)
    angle = squared.clamp_min(1e-8).sqrt()
    half = 0.5 * angle
    small = squared < 1e-8
    scalar = torch.where(small, 1 - squared / 8 + squared.square() / 384, torch.cos(half))
    factor = torch.where(small, 0.5 - squared / 48 + squared.square() / 3840, torch.sin(half) / angle)
    return torch.cat((scalar, factor * aa), -1)


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


def sign_convention(q: torch.Tensor) -> torch.Tensor:
    """App. C.1, Eq. 37: choose the representative whose first non-zero component is positive.

    q_w < 0 -> -q ; q_w > 0 -> q ; q_w == 0 -> sigma * q with sigma making the first non-zero
    entry of (q_x, q_y, q_z) positive. All representatives lie in a closed hemisphere of S^3 and
    no two are antipodal."""
    sign = torch.zeros_like(q[..., 0])
    for i in range(4):  # first non-zero component decides
        c = q[..., i]
        sign = torch.where((sign == 0) & (c != 0), torch.sign(c), sign)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return q * sign[..., None]


# --------------------------------------------------------------------------- 6D (baseline)
def aa_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    return quat_to_matrix(aa_to_quat(aa))


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(EPS)
    w, x, y, z = q.unbind(-1)
    m = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
    return m.reshape(*q.shape[:-1], 3, 3)


def matrix_to_quat(m: torch.Tensor) -> torch.Tensor:
    """Robust (Shepperd) rotation matrix -> unit quaternion."""
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    cands = torch.stack([
        torch.stack([1 + m00 + m11 + m22, m[..., 2, 1] - m[..., 1, 2],
                     m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]], -1),
        torch.stack([m[..., 2, 1] - m[..., 1, 2], 1 + m00 - m11 - m22,
                     m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0]], -1),
        torch.stack([m[..., 0, 2] - m[..., 2, 0], m[..., 0, 1] + m[..., 1, 0],
                     1 - m00 + m11 - m22, m[..., 1, 2] + m[..., 2, 1]], -1),
        torch.stack([m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0],
                     m[..., 1, 2] + m[..., 2, 1], 1 - m00 - m11 + m22], -1),
    ], -2)  # [...,4,4]; row i has 4*q_i^2 on its i-th entry
    diag = torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                        1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1)
    best = diag.argmax(-1)
    q = torch.gather(cands, -2, best[..., None, None].expand(*best.shape, 1, 4))[..., 0, :]
    return q / q.norm(dim=-1, keepdim=True).clamp_min(EPS)


def matrix_to_6d(m: torch.Tensor) -> torch.Tensor:
    """psi(R) = (a, b): the first two columns of R (App. B.2)."""
    return torch.cat([m[..., :, 0], m[..., :, 1]], -1)


def sixd_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt decoding of (a, b) (Zhou et al., 2019)."""
    a, b = d6[..., :3], d6[..., 3:]
    c0 = a / a.norm(dim=-1, keepdim=True).clamp_min(EPS)
    b = b - (c0 * b).sum(-1, keepdim=True) * c0
    c1 = b / b.norm(dim=-1, keepdim=True).clamp_min(EPS)
    c2 = torch.cross(c0, c1, dim=-1)
    return torch.stack([c0, c1, c2], -1)


def aa_to_6d(aa: torch.Tensor) -> torch.Tensor:
    return matrix_to_6d(aa_to_matrix(aa))


def sixd_to_aa(d6: torch.Tensor) -> torch.Tensor:
    return quat_to_aa(matrix_to_quat(sixd_to_matrix(d6)))


# --------------------------------------------------------------------------- S^3 manifold
def inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1, keepdim=True)


def project(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Orthogonal projection onto T_x S^3 (App. C.4.2, Eq. 48), applied per joint."""
    return u - inner(x, u) * x


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Radial projection onto S^3 per joint (Eq. 52)."""
    norm = x.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(x)
    fallback[..., 0] = 1
    return torch.where(norm > EPS, x / norm.clamp_min(EPS), fallback)


def expmap(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Exp_x(v) = cos(|v|) x + sin(|v|)/|v| v (Eq. 47)."""
    squared = v.square().sum(-1, keepdim=True)
    n = squared.clamp_min(1e-8).sqrt()
    small = squared < 1e-8
    cosine = torch.where(small, 1 - squared / 2 + squared.square() / 24, torch.cos(n))
    sinc = torch.where(small, 1 - squared / 6 + squared.square() / 120, torch.sin(n) / n)
    return cosine * x + sinc * v


def logmap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Log_x(y) in T_x S^3 (local inverse of Eq. 47; y not antipodal to x)."""
    c = inner(x, y).clamp(-1.0, 1.0)
    u = y - c * x
    s = u.norm(dim=-1, keepdim=True)
    theta = torch.atan2(s, c)
    small_same = (s < 1e-6) & (c >= 0)
    k = torch.where(small_same, 1.0 + theta * theta / 6.0, theta / s.clamp_min(1e-12))
    # The antipode has multiple geodesics. Pick a deterministic unit tangent.
    index = x.abs().argmin(dim=-1, keepdim=True)
    axis = torch.zeros_like(x).scatter(-1, index, 1)
    antipodal = (s < 1e-12) & (c < 0)
    return torch.where(antipodal, torch.pi * normalize(project(x, axis)), k * u)


def geodesic_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    c = inner(x, y).clamp(-1.0, 1.0)[..., 0]
    s = (y - c[..., None] * x).norm(dim=-1)
    return torch.atan2(s, c)


def uniform_hemisphere(shape, device=None, generator=None) -> torch.Tensor:
    """p0: uniform on S^3 restricted to the sign-convention hemisphere (App. D.1, Table 6)."""
    g = torch.randn(*shape, 4, device=device, generator=generator)
    return sign_convention(normalize(g))


def quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def quat_multiply(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.linalg.cross(av, bv)), -1)


def relative_rotvec(base, target):
    """Principal SO(3) log expressed in the body frame of base.

    The sign convention is applied to the relative quaternion. This makes the
    exact pi tie deterministic under independent changes of endpoint lifts.
    """
    relative = sign_convention(quat_multiply(quat_conjugate(base), target))
    return quat_to_aa(relative)


def rotate_by(base, body_rotvec):
    return quat_multiply(base, aa_to_quat(body_rotvec))


def body_velocity(q, dq):
    """Convert an S³ lift derivative to angular velocity in radians."""
    return 2 * quat_multiply(quat_conjugate(q), dq)[..., 1:]


def projected_6d_path(x0, x1, t):
    """Decode frame-wise 6D interpolation with a deterministic singular fallback."""
    a0, a1 = (matrix_to_6d(quat_to_matrix(x)) for x in (x0, x1))
    a, b = torch.lerp(a0, a1, t).split(3, dim=-1)
    norm_a = a.norm(dim=-1, keepdim=True)
    c0 = torch.where(norm_a > 1e-6, a / norm_a.clamp_min(1e-6), a0[..., :3])
    b = b - inner(c0, b) * c0
    norm_b = b.norm(dim=-1, keepdim=True)
    index = c0.abs().argmin(dim=-1, keepdim=True)
    axis = torch.zeros_like(c0).scatter(-1, index, 1)
    axis = axis - inner(c0, axis) * c0
    c1 = torch.where(norm_b > 1e-6, b / norm_b.clamp_min(1e-6), normalize(axis))
    rotation = torch.stack((c0, c1, torch.linalg.cross(c0, c1)), dim=-1)
    return matrix_to_quat(rotation)


def infinitesimal_rotation(vector):
    """Skew-symmetric matrix giving the SO(3) exponential derivative at zero."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).unflatten(-1, (3, 3))
