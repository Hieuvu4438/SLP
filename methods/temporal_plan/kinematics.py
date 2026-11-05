"""Official SMPL-X forward adapter. Asset loading is explicit and CPU by default."""
from __future__ import annotations

from pathlib import Path
import hashlib

import torch
from torch import Tensor, nn

from .geometry import unpack_smplx

# Native SMPL-X joints corresponding to retained rotations (no lower-body dilution).
RETAINED_JOINTS = tuple(range(12, 22)) + tuple(range(25, 55)) + (22,)
SOKE_FIXED_BETAS = (-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
                    0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842)


class SMPLXForward(nn.Module):
    """Frozen official body model, differentiable with respect to input poses.

    This is the FK TRAINING LOSS backend, not SOKE's mesh-regressed hand evaluator.
    The official model applies its hand-pose mean (flat_hand_mean=False), matching
    the public SOKE construction. Raw input parameters retain that convention.
    """
    def __init__(self, body_model: nn.Module, betas=SOKE_FIXED_BETAS,
                 joint_indices=RETAINED_JOINTS):
        super().__init__()
        self.body_model = body_model.requires_grad_(False).eval()
        if len(betas) != 10 or not joint_indices:
            raise ValueError("expected ten fixed shape coefficients and a joint selection")
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32)[None])
        self.register_buffer("joint_indices", torch.tensor(joint_indices, dtype=torch.long))
        self.asset_info: dict | None = None

    def provenance(self) -> dict:
        return {"backend": type(self.body_model).__module__ + "." + type(self.body_model).__name__,
                "betas": self.betas.detach().cpu().flatten().tolist(),
                "native_joint_indices": self.joint_indices.detach().cpu().tolist(),
                "flat_hand_mean": getattr(self.body_model, "flat_hand_mean", None),
                "assets": self.asset_info}

    @property
    def rotation_offsets(self) -> Tensor:
        """Effective rotations include the official additive hand pose mean."""
        return self.body_model.pose_mean.reshape(55, 3)[list(RETAINED_JOINTS)]

    @classmethod
    def from_model_dir(cls, model_dir: str | Path, *, gender: str = "neutral",
                       ext: str = "npz", betas=SOKE_FIXED_BETAS):
        if not Path(model_dir).exists():
            raise FileNotFoundError("supply licensed SMPL-X model assets explicitly")
        import smplx

        model = smplx.create(str(model_dir), model_type="smplx", gender=gender,
                             ext=ext, use_pca=False, flat_hand_mean=False,
                             num_betas=10, num_expression_coeffs=10,
                             use_face_contour=False)
        result = cls(model, betas)
        path = Path(model_dir)
        if path.is_dir():
            path = path / "smplx" / f"SMPLX_{gender.upper()}.{ext}"
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024*1024), b""):
                    digest.update(block)
            result.asset_info = {"path": str(path.resolve()), "sha256": digest.hexdigest()}
        return result

    def train(self, mode: bool = True):
        super().train(mode)
        self.body_model.eval()
        return self

    def forward(self, raw133: Tensor) -> Tensor:
        kwargs = unpack_smplx(raw133)
        # Do not use no_grad: decoder/pose gradients must pass through the frozen model.
        out = self.body_model(**kwargs, betas=self.betas.to(raw133).expand(len(raw133), -1),
                              return_verts=False)
        return out.joints.index_select(1, self.joint_indices)
