"""SMPL-X forward kinematics for the SOKE text-to-sign evaluation protocol.

Converts SOKE 133-D feature sequences (upper body 0:30 | lhand 30:75 | rhand 75:120 |
jaw 120:123 | expr 123:133, axis-angle, *normalised* with the SOKE ``deps/stats``) into
3-D joint positions exactly as SOKE's ``mGPT/utils/human_models.py`` + ``metrics/t2m.py``:

* neutral SMPL-X, ``use_pca=False``, ``use_face_contour=True`` (144 output joints),
  root and lower body set to zero, eyes zero, fixed shape coefficients;
* body joints = ``output.joints``; hand joints (21 per hand) from SOKE's
  ``orig_hand_regressor`` applied to the mesh vertices.

Paths default to the SOKE release inside ``third_party/SOKE/deps`` and can be
overridden with ``SLP_SMPLX_DIR`` (folder that contains ``smplx/SMPLX_NEUTRAL.npz``)
and ``SLP_SOKE_STATS`` (folder with ``mean.pt``/``std.pt``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from slp.paths import PROJECT_ROOT

DEFAULT_SMPLX_DIR = PROJECT_ROOT / "third_party" / "SOKE" / "deps" / "smpl_models"
DEFAULT_STATS_DIR = PROJECT_ROOT / "third_party" / "SOKE" / "deps" / "stats"

# Fixed shape used to decode SOKE features (shape is not part of the 133-D feature).
SOKE_SHAPE = [-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
              0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]

# SOKE human_models.SMPLX.joint_idx restricted to its 'upper_body' part:
# joints_name[Neck..R_Wrist] + joints_name[L_Ear..Nose] mapped into output.joints.
UPPER_BODY_IDX = [12, 16, 17, 18, 19, 20, 21, 59, 58, 57, 56, 55]
_LHAND = ([20, 37, 38, 39], 5361, [25, 26, 27], 4933, [28, 29, 30], 5058,
          [34, 35, 36], 5169, [31, 32, 33], 5286)
_RHAND = ([21, 52, 53, 54], 8079, [40, 41, 42], 7669, [43, 44, 45], 7794,
          [49, 50, 51], 7905, [46, 47, 48], 8022)


def smplx_dir() -> Path:
    return Path(os.environ.get("SLP_SMPLX_DIR", DEFAULT_SMPLX_DIR))


def stats_dir() -> Path:
    return Path(os.environ.get("SLP_SOKE_STATS", DEFAULT_STATS_DIR))


@lru_cache(maxsize=4)
def soke_stats(path: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """133-D mean/std of the SOKE release (179-D stats reduced with SOKE's index rule)."""
    import torch

    root = Path(path) if path else stats_dir()
    out = []
    for n in ("mean", "std"):
        t = torch.load(root / f"{n}.pt", weights_only=True).float()
        if t.numel() == 179:
            t = t[36:]
            t = torch.cat([t[:-20], t[-10:]])
        out.append(t.numpy().astype(np.float32))
    return out[0], out[1]


def _hand_regressor(J: np.ndarray, spec, n_verts: int) -> np.ndarray:
    rows = []
    for item in spec:
        if isinstance(item, list):
            rows.append(J[item])
        else:
            e = np.zeros((1, n_verts), dtype=J.dtype)
            e[0, item] = 1.0
            rows.append(e)
    return np.concatenate(rows, 0)


class SMPLXJoints:
    """Batched FK: normalised [T,133] -> (joints [T,144,3], lhand [T,21,3], rhand [T,21,3])."""

    def __init__(self, device: str | None = None, model_dir: str | Path | None = None,
                 stats: str | Path | None = None, *, load_stats: bool = True):
        import smplx
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        args = {k: False for k in ("create_global_orient", "create_body_pose",
                                   "create_left_hand_pose", "create_right_hand_pose",
                                   "create_jaw_pose", "create_leye_pose", "create_reye_pose",
                                   "create_betas", "create_expression", "create_transl")}
        path = Path(model_dir or smplx_dir())
        ext = path.suffix[1:] if path.is_file() else "npz"
        if path.is_dir() and not (path / "smplx/SMPLX_NEUTRAL.npz").is_file():
            ext = "pkl"
        self.layer = smplx.create(str(path), "smplx", gender="NEUTRAL",
                                  use_pca=False, use_face_contour=True, num_betas=10,
                                  num_expression_coeffs=10, ext=ext, **args).to(self.device)
        self.layer.eval()
        J = self.layer.J_regressor.detach().cpu().numpy()
        nv = J.shape[1]
        self.lreg = torch.from_numpy(_hand_regressor(J, _LHAND, nv)).float().to(self.device)
        self.rreg = torch.from_numpy(_hand_regressor(J, _RHAND, nv)).float().to(self.device)
        self.stats_path = str(stats) if stats else None
        self.mean = self.std = None
        if load_stats:
            self._load_stats()
        self.shape = torch.tensor(SOKE_SHAPE, device=self.device)

    def _load_stats(self):
        mean, std = soke_stats(self.stats_path)
        self.mean = self.torch.from_numpy(mean).to(self.device)
        self.std = self.torch.from_numpy(std).to(self.device)

    def __call__(self, feats_norm, chunk: int = 512, normalised: bool = True):
        torch = self.torch
        with torch.no_grad():
            f = torch.as_tensor(np.asarray(feats_norm), dtype=torch.float32, device=self.device)
            if normalised:
                if self.mean is None:
                    self._load_stats()
                f = f * self.std + self.mean
            outs = ([], [], [])
            for s in range(0, f.shape[0], chunk):
                x = f[s:s + chunk]
                T = x.shape[0]
                full = torch.cat([torch.zeros(T, 36, device=self.device), x], -1)
                z3 = torch.zeros(T, 3, device=self.device)
                o = self.layer(betas=self.shape.expand(T, -1), global_orient=full[:, :3],
                               body_pose=full[:, 3:66], left_hand_pose=full[:, 66:111],
                               right_hand_pose=full[:, 111:156], jaw_pose=full[:, 156:159],
                               leye_pose=z3, reye_pose=z3, expression=full[:, 159:169])
                outs[0].append(o.joints)
                outs[1].append(torch.einsum("jv,tvc->tjc", self.lreg, o.vertices))
                outs[2].append(torch.einsum("jv,tvc->tjc", self.rreg, o.vertices))
            return tuple(torch.cat(o, 0) for o in outs)


@lru_cache(maxsize=2)
def default_fk(device: str | None = None) -> SMPLXJoints:
    return SMPLXJoints(device=device)
