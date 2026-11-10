"""Geometry providers: actual SMPL-X for production, explicit skeletons for controlled checks."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import torch
from torch import nn
from .schema import BODY_FULL_JOINTS, LEFT_FULL_JOINTS, RIGHT_FULL_JOINTS
from .rotations import rotation_6d_to_matrix


@dataclass
class GeometryOutput:
    body: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    relations: torch.Tensor


def geometry_signature(geometry):
    result = {"provider": type(geometry).__name__, "codec": "physical_rotations_mean_before_exp_v2"}
    for name in ("manual_mean", "parents", "rest", "betas"):
        if hasattr(geometry, name):
            value = getattr(geometry, name).detach().cpu().contiguous()
            if value.is_floating_point():
                value = value.float()
            result[name] = hashlib.sha256(value.numpy().tobytes()).hexdigest()
    if hasattr(geometry, "asset_sha256"):
        result["asset_sha256"] = geometry.asset_sha256
    return result


def pack_geometry(joints):
    body = joints[..., list(BODY_FULL_JOINTS), :]
    left = joints[..., list(LEFT_FULL_JOINTS), :]
    right = joints[..., list(RIGHT_FULL_JOINTS), :]
    if joints.shape[-2] >= 76:
        left = torch.cat((left, joints[..., 66:71, :]), -2)
        right = torch.cat((right, joints[..., 71:76, :]), -2)
    # SMPL-X indices: head=15, shoulders=16/17, wrists=20/21.
    rel = torch.cat(
        (
            joints[..., 20, :] - joints[..., 15, :],
            joints[..., 21, :] - joints[..., 15, :],
            joints[..., 20, :] - joints[..., 16, :],
            joints[..., 21, :] - joints[..., 17, :],
            joints[..., 20, :] - joints[..., 21, :],
        ),
        -1,
    )
    return GeometryOutput(body, left, right, rel)


class SkeletonGeometry(nn.Module):
    """Exact FK for a supplied 55-joint SMPL-X tree/rest pose (no fake default skeleton).
    This yields skeletal joints only; use SMPLXGeometry for mesh-derived fingertip losses.
    """

    def __init__(self, parents, rest_joints):
        super().__init__()
        parents = torch.as_tensor(parents, dtype=torch.long)
        rest = torch.as_tensor(rest_joints).float()
        if parents.shape != (55,) or rest.shape != (55, 3) or int(parents[0]) != -1:
            raise ValueError("Expected root-first 55-joint SMPL-X skeleton")
        if any(not 0 <= int(parents[i]) < i for i in range(1, 55)):
            raise ValueError("Parents must precede children")
        self.register_buffer("parents", parents)
        self.register_buffer("rest", rest)
        self.register_buffer("manual_mean", torch.zeros(120))

    def forward(self, manual):
        rot = rotation_6d_to_matrix(manual.reshape(*manual.shape[:-1], 40, 6))
        identity = torch.eye(3, device=manual.device, dtype=manual.dtype)
        lookup = {
            j: i for i, j in enumerate(BODY_FULL_JOINTS + LEFT_FULL_JOINTS + RIGHT_FULL_JOINTS)
        }
        global_rot, positions = [], []
        for j in range(55):
            local = (
                rot[..., lookup[j], :, :]
                if j in lookup
                else identity.expand(*manual.shape[:-1], 3, 3)
            )
            if j == 0:
                global_rot.append(local)
                positions.append(self.rest[0].to(manual).expand(*manual.shape[:-1], 3))
            else:
                parent = int(self.parents[j])
                global_rot.append(global_rot[parent] @ local)
                offset = self.rest[j] - self.rest[parent]
                positions.append(
                    positions[parent] + (global_rot[parent] @ offset[..., None]).squeeze(-1)
                )
        return pack_geometry(torch.stack(positions, -2))


class SMPLXGeometry(nn.Module):
    def __init__(self, model_path, betas=None, chunk_size=32):
        super().__init__()
        import smplx

        asset = Path(model_path)
        if asset.is_dir():
            asset = asset / "SMPLX_NEUTRAL.npz"
        digest = hashlib.sha256()
        with asset.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.asset_sha256 = digest.hexdigest()

        self.layer = smplx.SMPLXLayer(
            model_path,
            gender="neutral",
            ext="npz",
            use_pca=False,
            flat_hand_mean=False,
            num_betas=10,
            num_expression_coeffs=10,
        )
        self.layer.requires_grad_(False)
        self.register_buffer(
            "manual_mean",
            torch.cat(
                (torch.zeros(30), self.layer.left_hand_mean, self.layer.right_hand_mean)
            ).float(),
        )
        default = [
            -0.07284723,
            0.1795129,
            -0.27608207,
            0.135155,
            0.10748172,
            0.16037364,
            -0.01616933,
            -0.03450319,
            0.01369138,
            0.01108842,
        ]
        self.register_buffer("betas", torch.tensor(default if betas is None else betas).float())
        if self.betas.shape != (10,) or chunk_size < 1:
            raise ValueError("Expected 10 shape coefficients and positive chunk_size")
        self.chunk_size = chunk_size

    def forward(self, manual):
        prefix = manual.shape[:-1]
        rotations = rotation_6d_to_matrix(manual.reshape(-1, 40, 6))
        out = []
        for r in rotations.split(self.chunk_size):
            n = len(r)
            eye = torch.eye(3, device=r.device, dtype=r.dtype).expand(n, 1, 3, 3)
            body = torch.cat((eye.expand(n, 11, 3, 3), r[:, :10]), 1)
            # Physical rotations already include the author's axis-angle hand mean.
            result = self.layer(
                betas=self.betas.to(r).expand(n, -1),
                global_orient=eye,
                body_pose=body,
                left_hand_pose=r[:, 10:25],
                right_hand_pose=r[:, 25:40],
                jaw_pose=eye,
                leye_pose=eye,
                reye_pose=eye,
                expression=torch.zeros(n, 10, device=r.device, dtype=r.dtype),
                return_verts=False,
            )
            out.append(result.joints)
        joints = torch.cat(out).reshape(*prefix, -1, 3)
        return pack_geometry(joints)
