"""Versioned endpoint-preserving grid and shortest-arc rotation interpolation."""

import numpy as np
import torch

from cprflow.geometry.rotations import (
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    slerp,
)


def resample_motion(x, source_fps, target_fps=30.0, verified=False, allow_assumed=False):
    if (not verified and not allow_assumed) or not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Verified feature sampling FPS required")
    x = torch.as_tensor(x, dtype=torch.float32)
    if len(x) < 2 or x.shape[-1] != 296:
        raise ValueError("Expected T>=2 canonical296")
    duration = (len(x) - 1) / source_fps
    count = round(duration * target_fps) + 1
    if count < 2:
        raise ValueError("T_out<=1 under endpoint_linspace_v1; quarantine production clip")
    times = torch.linspace(0, duration, count, dtype=torch.float64)
    u = (times * source_fps).clamp(0, len(x) - 1)
    lo = u.floor().long().clamp(max=len(x) - 2)
    hi = lo + 1
    fraction = (u - lo).float()
    q = matrix_to_quaternion(rotation_6d_to_matrix(x[:, :246].reshape(-1, 41, 6)))
    # Sign continuity is explicit, in addition to the pairwise shortest-arc rule.
    signs = torch.where((q[1:] * q[:-1]).sum(-1) < 0, -1.0, 1.0)
    signs = torch.cat((torch.ones_like(signs[:1]), signs), 0).cumprod(0)
    q = q * signs[..., None]
    coefficients = x[:, 246:]
    if source_fps > target_fps:
        # quaternion_fir_v1: filter sign-continuous quaternion components offline,
        # renormalize on S3, then SLERP. Endpoints are retained exactly.
        from scipy.ndimage import convolve1d
        from scipy.signal import firwin

        taps = 2 * int(np.ceil(4 * source_fps / target_fps)) + 1
        kernel = firwin(taps, 0.45 * target_fps, fs=source_fps, window="hamming")
        filtered = torch.from_numpy(convolve1d(q.numpy(), kernel, axis=0, mode="nearest"))
        norms = filtered.norm(dim=-1, keepdim=True)
        if (norms < 1e-6).any():
            raise ValueError("Degenerate quaternion after quaternion_fir_v1 anti-alias filter")
        filtered = filtered / norms
        filtered[0], filtered[-1] = q[0], q[-1]
        q = filtered
        filtered_coeff = torch.from_numpy(
            convolve1d(coefficients.numpy(), kernel, axis=0, mode="nearest")
        )
        filtered_coeff[0], filtered_coeff[-1] = coefficients[0], coefficients[-1]
        coefficients = filtered_coeff
    rotations = matrix_to_rotation_6d(quaternion_to_matrix(slerp(q[lo], q[hi], fraction)))
    coeff = torch.lerp(coefficients[lo], coefficients[hi], fraction[:, None])
    result = torch.cat((rotations.flatten(1), coeff), -1)
    return result.numpy(), times.numpy(), (count - 1) / duration
