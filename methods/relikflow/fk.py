"""Physical SOKE decoding and differentiable landmark FK."""

import torch
from torch import nn

from slp.evaluation.smplx_fk import SOKE_SHAPE, _LHAND, _RHAND, _hand_regressor
from slp.geometry import quat_to_matrix

from .data import pack133

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PACKED_FINGERS = ("index", "middle", "pinky", "ring", "thumb")
SIDES = ("left", "right")
LANDMARKS = ("pelvis", "spine3", "nose", "left_shoulder", "right_shoulder") + tuple(
    name
    for side in SIDES
    for name in (
        f"{side}_wrist",
        *(f"{side}_{finger}{j}" for finger in FINGERS for j in (1, 2, 3, "")),
    )
)


class SokeFK(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        import smplx
        from smplx.joint_names import JOINT_NAMES
        from smplx.lbs import blend_shapes, vertices2joints

        flags = {
            k: False
            for k in (
                "create_global_orient",
                "create_body_pose",
                "create_left_hand_pose",
                "create_right_hand_pose",
                "create_jaw_pose",
                "create_leye_pose",
                "create_reye_pose",
                "create_betas",
                "create_expression",
                "create_transl",
            )
        }
        from pathlib import Path

        path = Path(model_path)
        ext = path.suffix[1:] if path.is_file() else "npz"
        if path.is_dir() and not (path / "smplx/SMPLX_NEUTRAL.npz").is_file():
            ext = "pkl"
        self.layer = smplx.create(
            str(path),
            "smplx",
            gender="NEUTRAL",
            use_pca=False,
            flat_hand_mean=False,
            use_face_contour=True,
            num_betas=10,
            num_expression_coeffs=10,
            ext=ext,
            **flags,
        )
        self.layer.eval().requires_grad_(False)
        self.register_buffer("betas", torch.tensor(SOKE_SHAPE))
        self.register_buffer("indices", torch.tensor([JOINT_NAMES.index(n) for n in LANDMARKS]))
        j = self.layer.J_regressor.detach().cpu().numpy()
        for name, spec in (("left", _LHAND), ("right", _RHAND)):
            self.register_buffer(
                name + "_reg", torch.from_numpy(_hand_regressor(j, spec, j.shape[1]))
            )

        # Relation FK uses only native joints and the extra nose/fingertip vertices.
        layer = self.layer
        native = layer.J_regressor.shape[0]
        extra = self.indices[self.indices >= native] - native
        selected = layer.vertex_joint_selector.extra_joints_idxs[extra]
        rest = layer.v_template + blend_shapes(self.betas[None], layer.shapedirs)
        self.register_buffer("rest_joints", vertices2joints(layer.J_regressor, rest))
        self.register_buffer("rest_vertices", rest[:, selected])
        self.register_buffer(
            "pose_directions",
            layer.posedirs.reshape(layer.posedirs.shape[0], -1, 3)[:, selected].flatten(1),
        )
        self.register_buffer("vertex_weights", layer.lbs_weights[selected])
        mapping = self.indices.clone()
        mapping[mapping >= native] = native + torch.arange(len(selected))
        self.register_buffer("landmark_order", mapping)

    def decode(self, q, e):
        pose = pack133(q, e).reshape(-1, 133)
        batches = []
        for p in pose.split(128):
            z3 = p.new_zeros(len(p), 3)
            batches.append(
                self.layer(
                    betas=self.betas.expand(len(p), -1),
                    global_orient=z3,
                    body_pose=torch.cat((p.new_zeros(len(p), 33), p[:, :30]), -1),
                    left_hand_pose=p[:, 30:75],
                    right_hand_pose=p[:, 75:120],
                    jaw_pose=p[:, 120:123],
                    leye_pose=z3,
                    reye_pose=z3,
                    expression=p[:, 123:],
                )
            )
        return torch.cat([x.joints for x in batches]), torch.cat([x.vertices for x in batches])

    def forward(self, q):
        from smplx.lbs import batch_rigid_transform, batch_rodrigues
        from torch.nn.functional import pad

        pose = pack133(q, q.new_zeros(*q.shape[:-2], 10)).reshape(-1, 133)
        output = []
        for p in pose.split(128):
            full_pose = torch.cat(
                (
                    p.new_zeros(len(p), 36),
                    p[:, :30],
                    p[:, 120:123],
                    p.new_zeros(len(p), 6),
                    p[:, 30:120],
                ),
                -1,
            )
            full_pose = full_pose + self.layer.pose_mean
            rot = batch_rodrigues(full_pose.reshape(-1, 3)).reshape(len(p), -1, 3, 3)
            features = (rot[:, 1:] - torch.eye(3, device=p.device, dtype=p.dtype)).flatten(1)
            vertices = (features @ self.pose_directions).reshape(len(p), -1, 3) + self.rest_vertices
            joints, transform = batch_rigid_transform(
                rot, self.rest_joints.expand(len(p), -1, -1), self.layer.parents, dtype=p.dtype
            )
            skinning = (self.vertex_weights @ transform.flatten(-2)).unflatten(-1, (4, 4))
            landmarks = (skinning @ pad(vertices, (0, 1), value=1)[..., None])[..., :3, 0]
            output.append(torch.cat((joints, landmarks), dim=1)[:, self.landmark_order])
        return torch.cat(output).reshape(*q.shape[:-2], len(LANDMARKS), 3)

    def score_joints(self, q, e):
        joints, vertices = self.decode(q, e)
        return (joints, self.left_reg @ vertices, self.right_reg @ vertices)


def fk_soke(q, e, adapter):
    return adapter.score_joints(q, e)


class FixtureFK(nn.Module):
    """An articulated skeleton used only for synthetic verification."""

    def forward(self, q):
        matrix = quat_to_matrix(q)

        def constant(point):
            return q.new_tensor(point).expand(*q.shape[:-2], 3)

        pelvis = constant((0, 0, 0))
        chest = constant((0, 0.32, 0))
        nose = constant((0, 0.65, 0.11))
        shoulders = [constant((-0.18, 0.41, 0)), constant((0.18, 0.41, 0))]
        landmarks = [pelvis, chest, nose, *shoulders]
        for side, sign in enumerate((-1, 1)):

            def move(p, r, offset):
                return p + (r @ q.new_tensor(offset)).squeeze(-1)

            arm = matrix[..., 4 + side, :, :]
            elbow = move(shoulders[side], arm, [[0.19 * sign], [-0.13], [0.04]])
            forearm = arm @ matrix[..., 6 + side, :, :]
            wrist = move(elbow, forearm, [[0.13 * sign], [-0.08], [0.14]])
            palm = forearm @ matrix[..., 8 + side, :, :]
            landmarks.append(wrist)
            for f, finger in enumerate(FINGERS):
                base = 10 + 15 * side + 3 * PACKED_FINGERS.index(finger)
                point = move(wrist, palm, [[(0.042 + 0.011 * f) * sign], [0.014 * (2 - f)], [0.01]])
                rotation = palm
                landmarks.append(point)
                for i in range(3):
                    rotation = rotation @ matrix[..., base + i, :, :]
                    point = move(point, rotation, [[(0.029 - 0.006 * i) * sign], [0], [0.003]])
                    landmarks.append(point)
        return torch.stack(landmarks, dim=-2)

    def score_joints(self, q, e):
        p = self(q)
        # Match the application scorer's landmark indexing for synthetic verification.
        body = p.new_zeros(len(p), 60, 3)
        body[:, [0, 9, 55, 16, 17, 20, 21]] = p[:, [0, 1, 2, 3, 4, 5, 26]]
        return body, p[:, 5:26], p[:, 26:47]
