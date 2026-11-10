"""Differentiable SO(3) operations. 6D = first two ROWS, consistently throughout."""

import torch
import torch.nn.functional as F


def axis_angle_to_matrix(a):
    if a.shape[-1] != 3:
        raise ValueError("Expected axis-angle (...,3)")
    x, y, z = a.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*a.shape[:-1], 3, 3)
    theta = torch.linalg.vector_norm(a, dim=-1)
    s = torch.sinc(theta / torch.pi)
    c = 0.5 * torch.sinc(theta / (2 * torch.pi)).square()
    eye = torch.eye(3, device=a.device, dtype=a.dtype)
    return eye + s[..., None, None] * k + c[..., None, None] * (k @ k)


def matrix_to_quaternion(m):
    """Largest-diagonal branch avoids instability at rotations near pi."""
    if m.shape[-2:] != (3, 3):
        raise ValueError("Expected (...,3,3)")
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    vals = torch.stack(
        (1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1
    )
    # Clamp before sqrt: unselected zero candidates must not produce infinite gradients.
    qabs = vals.clamp_min(1e-12).sqrt()
    candidates = torch.stack(
        (
            torch.stack(
                (
                    vals[..., 0],
                    m[..., 2, 1] - m[..., 1, 2],
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] - m[..., 0, 1],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 2, 1] - m[..., 1, 2],
                    vals[..., 1],
                    m[..., 1, 0] + m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] + m[..., 0, 1],
                    vals[..., 2],
                    m[..., 2, 1] + m[..., 1, 2],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 1, 0] - m[..., 0, 1],
                    m[..., 2, 0] + m[..., 0, 2],
                    m[..., 2, 1] + m[..., 1, 2],
                    vals[..., 3],
                ),
                -1,
            ),
        ),
        -2,
    )
    candidates = candidates / (2 * qabs.clamp_min(0.1)[..., :, None])
    idx = qabs.argmax(-1)
    q = candidates.gather(-2, idx[..., None, None].expand(*idx.shape, 1, 4)).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return torch.where(q[..., :1] < 0, -q, q)


def quaternion_to_matrix(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


def matrix_to_axis_angle(m):
    q = matrix_to_quaternion(m)
    half = torch.atan2(torch.linalg.vector_norm(q[..., 1:], dim=-1), q[..., 0])
    return q[..., 1:] * (2 / torch.sinc(half / torch.pi))[..., None]


def matrix_to_rotation_6d(m):
    return m[..., :2, :].reshape(*m.shape[:-2], 6)


def rotation_6d_to_matrix(d):
    if d.shape[-1] != 6:
        raise ValueError("Expected (...,6)")
    a, b = d[..., :3], d[..., 3:]
    norm = a.norm(dim=-1, keepdim=True)
    e1 = torch.zeros_like(a)
    e1[..., 0] = 1
    u = torch.where(norm > 1e-6, a / norm.clamp_min(1e-6), e1)
    orth = b - (u * b).sum(-1, keepdim=True) * u
    # Least-aligned basis yields a stable valid rotation even for zero/parallel inputs.
    basis = F.one_hot(u.abs().argmin(-1), 3).to(d)
    fallback = F.normalize(basis - (u * basis).sum(-1, keepdim=True) * u, dim=-1)
    onorm = orth.norm(dim=-1, keepdim=True)
    v = torch.where(onorm > 1e-6, orth / onorm.clamp_min(1e-6), fallback)
    return torch.stack((u, v, torch.linalg.cross(u, v, dim=-1)), -2)


def geodesic_distance(a, b):
    r = a @ b.transpose(-1, -2)
    skew = (
        torch.stack(
            (r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]),
            -1,
        )
        / 2
    )
    sin = torch.linalg.vector_norm(skew, dim=-1)
    cos = ((r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.atan2(sin, cos)


def slerp_matrix(a, b, weight):
    qa, qb = matrix_to_quaternion(a), matrix_to_quaternion(b)
    dot = (qa * qb).sum(-1, keepdim=True)
    qb = torch.where(dot < 0, -qb, qb)
    dot = dot.abs().clamp(max=1 - 1e-7)
    angle = torch.acos(dot)
    w = weight[..., None]
    mixed = (torch.sin((1 - w) * angle) * qa + torch.sin(w * angle) * qb) / torch.sin(angle)
    linear = (1 - w) * qa + w * qb
    q = torch.where(dot > 0.9995, linear, mixed)
    return quaternion_to_matrix(q)
