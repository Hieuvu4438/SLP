"""Matrix-input SMPLXLayer. All predictions remain in the Torch autograd graph."""

from pathlib import Path

import torch
from torch import nn

from cprflow.data.cache import digest, file_hash
from cprflow.geometry.joint_maps import BODY_NAMES, hand_names, select
from cprflow.geometry.rotations import rotation_6d_to_matrix


def assemble_parameters(x):
    shape = x.shape[:-1]
    x = x.float().reshape(-1, 296)
    r, degenerate = rotation_6d_to_matrix(x[:, :246].reshape(-1, 41, 6), True)
    n = len(x)
    identity = torch.eye(3, device=x.device).expand(n, 1, 3, 3)
    body = torch.cat((identity.expand(n, 11, 3, 3), r[:, :10]), 1)
    return dict(
        body_pose=body,
        left_hand_pose=r[:, 10:25],
        right_hand_pose=r[:, 25:40],
        jaw_pose=r[:, 40:41],
        global_orient=identity,
        leye_pose=identity,
        reye_pose=identity,
        expression=x[:, 246:],
        betas=x.new_zeros(n, 10),
        transl=x.new_zeros(n, 3),
    ), degenerate.reshape(*shape, 41)


class BodyModel(nn.Module):
    def __init__(self, path, chunk_frames=128):
        super().__init__()
        path = Path(path)
        self.asset_path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing licensed expression50 asset: {path}; obtain from https://smpl-x.is.tue.mpg.de/"
            )
        import smplx
        from smplx.joint_names import JOINT_NAMES

        self.layer = smplx.SMPLXLayer(
            str(path),
            gender="neutral",
            ext="npz",
            num_betas=10,
            num_expression_coeffs=50,
            use_pca=False,
            flat_hand_mean=True,
            use_face_contour=False,
            joint_mapper=None,
        )
        if self.layer.expr_dirs.shape[-1] != 50:
            raise ValueError("Asset does not support 50 expression coefficients")
        self.layer.requires_grad_(False)
        self.chunk_frames = chunk_frames
        self.names = list(JOINT_NAMES)
        self.profile_hash = digest(
            dict(
                asset=file_hash(path),
                expression=50,
                betas=10,
                face_contour=False,
                joint_mapper=None,
            )
        )

    def forward(self, x, return_vertices=False):
        # Caller trims true lengths, so padding cannot enter geometry or derivatives.
        shape = x.shape[:-1]
        flat = x.float().reshape(-1, 296)
        pieces = []
        vertices = []
        degenerates = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            for chunk in flat.split(self.chunk_frames):
                params, deg = assemble_parameters(chunk)
                out = self.layer(**params, return_verts=return_vertices)
                pieces.append(out.joints)
                if return_vertices:
                    vertices.append(out.vertices)
                degenerates.append(deg)
        joints = torch.cat(pieces).reshape(*shape, -1, 3)
        if joints.shape[-2] > len(self.names):
            raise ValueError("SMPL-X joint-name mapping does not match output")
        names = self.names[: joints.shape[-2]]
        body = select(joints, BODY_NAMES, names)
        lh = select(joints, hand_names("left"), names)
        rh = select(joints, hand_names("right"), names)
        return {
            "joints": joints,
            "names": names,
            "body": body,
            "left": lh,
            "right": rh,
            "left_local": lh[..., 1:, :] - lh[..., :1, :],
            "right_local": rh[..., 1:, :] - rh[..., :1, :],
            "degenerate": torch.cat(degenerates).reshape(*shape, 41),
            "vertices": torch.cat(vertices).reshape(*shape, -1, 3) if vertices else None,
        }
