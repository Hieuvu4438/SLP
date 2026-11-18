"""Native row-based rot6D and stable, differentiable SO(3) operations."""

import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6, return_degenerate=False):
    d6 = d6.float()
    a1, a2 = d6[..., :3], d6[..., 3:]
    n1 = a1.norm(dim=-1, keepdim=True)
    e1 = torch.zeros_like(a1)
    e1[..., 0] = 1
    b1 = torch.where(n1 < 1e-6, e1, a1 / n1.clamp_min(1e-6))
    u2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    n2 = u2.norm(dim=-1, keepdim=True)
    basis = F.one_hot(b1.abs().argmin(-1), 3).to(b1)
    fallback = F.normalize(basis - (basis * b1).sum(-1, keepdim=True) * b1, dim=-1)
    b2 = torch.where(n2 < 1e-6, fallback, u2 / n2.clamp_min(1e-6))
    matrix = torch.stack((b1, b2, torch.cross(b1, b2, dim=-1)), dim=-2)
    degenerate = (n1[..., 0] < 1e-6) | (n2[..., 0] < 1e-6)
    return (matrix, degenerate) if return_degenerate else matrix


def matrix_to_rotation_6d(matrix):
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def geodesic_distance(a, b):
    r = a.float() @ b.float().transpose(-1, -2)
    skew = torch.stack(
        (r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]), -1
    )
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    cosine = (r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5
    return torch.atan2(sine, cosine)


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


def matrix_to_quaternion(r):
    """Offline conversion; branch on largest quaternion component, scalar first."""
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = r.reshape(*r.shape[:-2], 9).unbind(-1)
    qa = (
        torch.stack(
            (1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1
        )
        .clamp_min(0)
        .sqrt()
    )
    candidates = torch.stack(
        (
            torch.stack((qa[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), -1),
            torch.stack((m21 - m12, qa[..., 1] ** 2, m10 + m01, m02 + m20), -1),
            torch.stack((m02 - m20, m10 + m01, qa[..., 2] ** 2, m12 + m21), -1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, qa[..., 3] ** 2), -1),
        ),
        -2,
    )
    candidates = candidates / (2 * qa[..., :, None].clamp_min(0.1))
    index = qa.argmax(-1)
    q = candidates.gather(-2, index[..., None, None].expand(*index.shape, 1, 4)).squeeze(-2)
    return F.normalize(q, dim=-1)


def slerp(q0, q1, fraction):
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1)
    theta = torch.acos(dot.clamp(max=1 - 1e-7))
    f = fraction
    while f.ndim < q0.ndim:
        f = f.unsqueeze(-1)
    spherical = (torch.sin((1 - f) * theta) * q0 + torch.sin(f * theta) * q1) / theta.sin()
    linear = (1 - f) * q0 + f * q1
    return F.normalize(torch.where(dot > 0.9995, linear, spherical), dim=-1)
