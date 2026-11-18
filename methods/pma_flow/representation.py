"""SOKE 133-D features <-> (S^3)^41 joint quaternions + 10-D expression.

SOKE 133-D (axis-angle, normalised by the SOKE stats):
    upper body 0:30 (10 joints) | lhand 30:75 (15) | rhand 75:120 (15) | jaw 120:123 (1) | expr 123:133

Rotations are the *physical* SMPL-X rotations (SOKE's SMPL-X layer uses flat_hand_mean=False, so
the applied hand rotation is feature + hands_mean in axis-angle); ``decode`` subtracts the mean
again, so encode/decode is an FK-equivalent round trip. Quaternions use the sign convention
(first non-zero component positive). The expression stays in the normalised Euclidean space.
"""

from __future__ import annotations

import numpy as np
import torch

from slp import geometry as G

N_JOINTS = 41
EXPR_DIM = 10
POSE_DIM = N_JOINTS * 4
FEAT_DIM = POSE_DIM + EXPR_DIM   # flat (quaternions, expression) layout used by the anchor caches


class MotionCodec:
    def __init__(self, mean: np.ndarray, std: np.ndarray, hand_mean: np.ndarray):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.hand_mean = torch.as_tensor(hand_mean, dtype=torch.float32)   # [41,3]

    def _dev(self, x):
        return self.mean.to(x.device), self.std.to(x.device), self.hand_mean.to(x.device)

    def encode(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """normalised [...,133] -> (quats [...,41,4], expr [...,10] normalised)."""
        mean, std, hm = self._dev(feats)
        raw = feats[..., :123] * std[:123] + mean[:123]
        aa = raw.reshape(*raw.shape[:-1], N_JOINTS, 3) + hm
        return G.sign_convention(G.aa_to_quat(aa)), feats[..., 123:]

    def decode(self, quats: torch.Tensor, expr: torch.Tensor) -> torch.Tensor:
        """(quats, expr) -> normalised [...,133]."""
        mean, std, hm = self._dev(quats)
        aa = G.quat_to_aa(quats) - hm
        raw = aa.reshape(*aa.shape[:-2], N_JOINTS * 3)
        return torch.cat([(raw - mean[:123]) / std[:123], expr], -1)

    def expr_raw(self, expr: torch.Tensor) -> torch.Tensor:
        """Normalised expression -> raw SMPL-X expression coefficients."""
        mean, std, _ = self._dev(expr)
        return expr * std[123:] + mean[123:]


def flatten(q: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """(quats [...,41,4], expr [...,10]) -> [...,174]."""
    return torch.cat([q.flatten(-2), e], -1)


def unflatten(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x[..., :POSE_DIM].unflatten(-1, (N_JOINTS, 4)), x[..., POSE_DIM:]
