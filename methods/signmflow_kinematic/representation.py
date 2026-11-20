"""SOKE 133-D features <-> SignNMFlow motion representation (paper Sec. 3.1, App. C.1, D.1).

SOKE 133-D (axis-angle, normalised by the SOKE stats):
    upper body 0:30 (10 joints) | lhand 30:75 (15) | rhand 75:120 (15) | jaw 120:123 (1) | expr 123:133
-> J = 41 joint rotations + a 10-D facial-expression channel (Table 6: J = 41, expression dim 10).

Joint rotations are the *physical* SMPL-X rotations: SOKE's SMPL-X layer uses
``flat_hand_mean=False`` so the hand rotation applied by FK is ``pose + hands_mean`` (in
axis-angle); with ``add_hand_mean`` we add it before exponentiation and subtract it again
when exporting (FK-equivalent round trip). The expression channel stays in the normalised
Euclidean space (its source is N(0, I), App. C.5).

Representations:
    quat : x in (S^3)^41, sign convention of Eq. 37            -> [T, 41, 4]
    6d   : first two columns of each rotation matrix (baseline)  -> [T, 41, 6]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import geometry as G

N_JOINTS = 41
EXPR_DIM = 10
HAND_SLICES = (slice(10, 25), slice(25, 40))  # joint index ranges of lhand / rhand


def neutral_asset(smplx_dir):
    path = Path(smplx_dir)
    if not path.is_dir():
        return path
    neutral = path / "SMPLX_NEUTRAL.npz"
    return neutral if neutral.exists() else path / "smplx" / neutral.name


def load_hand_mean(smplx_dir: str | Path) -> np.ndarray:
    """[41,3] additive axis-angle offsets (zeros except the 30 hand joints)."""
    offset = np.zeros((N_JOINTS, 3), np.float32)
    with np.load(neutral_asset(smplx_dir), allow_pickle=False) as data:
        offset[HAND_SLICES[0]] = np.asarray(data["hands_meanl"], np.float32).reshape(15, 3)
        offset[HAND_SLICES[1]] = np.asarray(data["hands_meanr"], np.float32).reshape(15, 3)
    return offset


class MotionCodec:
    """Bidirectional map between normalised 133-D frames and (rotations, expression)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, hand_mean: np.ndarray | None,
                 representation: str = "quat"):
        if representation not in ("quat", "6d"):
            raise ValueError(representation)
        self.rep = representation
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        hm = np.zeros((N_JOINTS, 3), np.float32) if hand_mean is None else hand_mean
        self.hand_mean = torch.as_tensor(hm, dtype=torch.float32)

    @property
    def rot_dim(self) -> int:
        return 4 if self.rep == "quat" else 6

    def _dev(self, x):
        return self.mean.to(x.device), self.std.to(x.device), self.hand_mean.to(x.device)

    def encode(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """normalised [...,133] -> (rot [...,41,4|6], expr [...,10] normalised)."""
        mean, std, hm = self._dev(feats)
        raw = feats[..., :123] * std[:123] + mean[:123]
        aa = raw.reshape(*raw.shape[:-1], N_JOINTS, 3) + hm
        rot = G.sign_convention(G.aa_to_quat(aa)) if self.rep == "quat" else G.aa_to_6d(aa)
        return rot, feats[..., 123:]

    def decode(self, rot: torch.Tensor, expr: torch.Tensor) -> torch.Tensor:
        """(rot, expr) -> normalised [...,133]."""
        mean, std, hm = self._dev(rot)
        aa = (G.quat_to_aa(rot) if self.rep == "quat" else G.sixd_to_aa(rot)) - hm
        raw = aa.reshape(*aa.shape[:-2], N_JOINTS * 3)
        return torch.cat([(raw - mean[:123]) / std[:123], expr], -1)
