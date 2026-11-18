"""Differentiable, sparse SMPL-X forward kinematics for the joints of the SOKE protocol.

``slp.evaluation.smplx_fk.SMPLXJoints`` evaluates the full mesh (10,475 vertices) under
``no_grad``. A training loss in joint space only needs the points the SOKE metrics read:

* body : SOKE's 12 upper-body joints (``UPPER_BODY_IDX``): 7 skeleton joints + nose/eyes/ears
         (mesh vertices selected by SMPL-X's vertex_joint_selector);
* hands: 21 points per hand from SOKE's hand regressor (rows of ``J_regressor`` applied to the
         posed mesh, plus one fingertip vertex per finger).

Skeleton joints depend only on the rest joints and the kinematic chain; the regressed points
depend on ~1.5k vertices (the support of the regressor rows). This module runs the exact SMPL-X
linear blend skinning (shape + expression blend shapes, pose blend shapes, skinning) on those
vertices only, so its output equals the full model's up to float rounding, at a fraction of the
cost, and it is differentiable w.r.t. the joint rotations and the expression.

Input layout follows the SOKE 133-D feature: 41 joint rotations = upper body (SMPL-X joints
12..21) | left hand (25..39) | right hand (40..54) | jaw (22); expression 10-D. Root, lower body
and eyes stay at identity, shape is SOKE's fixed ``SOKE_SHAPE``, exactly as in the evaluator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from slp import geometry as G
from slp.evaluation.smplx_fk import (_LHAND, _RHAND, SOKE_SHAPE, UPPER_BODY_IDX, smplx_dir,
                                     soke_stats)

N_FEAT_JOINTS = 41
# feature joint k -> SMPL-X joint index
FEAT_TO_SMPLX = list(range(12, 22)) + list(range(25, 40)) + list(range(40, 55)) + [22]
N_SMPLX_JOINTS = 55
# order of SMPL-X's vertex_joint_selector extras: output.joints[55 + k]
_EXTRA_NAMES = ("nose", "reye", "leye", "rear", "lear")


def _regressor_rows(J: np.ndarray, spec) -> list[np.ndarray]:
    """SOKE hand regressor rows as dense vertex weights (same construction as smplx_fk)."""
    rows = []
    for item in spec:
        if isinstance(item, list):
            rows.extend(J[i] for i in item)
        else:
            e = np.zeros(J.shape[1], dtype=J.dtype)
            e[item] = 1.0
            rows.append(e)
    return rows


class SkeletonFK(nn.Module):
    """Rotations [..., 41, 3, 3] (+ expression) -> body [..., 12, 3], lhand/rhand [..., 21, 3]."""

    def __init__(self, model_dir: str | Path | None = None, stats: str | Path | None = None):
        super().__init__()
        import smplx
        from smplx.vertex_ids import vertex_ids

        layer = smplx.create(str(model_dir or smplx_dir()), "smplx", gender="NEUTRAL",
                             use_pca=False, use_face_contour=True, num_betas=10,
                             num_expression_coeffs=10)
        Jreg = layer.J_regressor.detach().double().numpy()            # [55, V]
        v_t = layer.v_template.detach().double()                        # [V, 3]
        dirs = torch.cat([layer.shapedirs, layer.expr_dirs], -1).detach().double()  # [V,3,20]
        V = v_t.shape[0]

        lrows, rrows = _regressor_rows(Jreg, _LHAND), _regressor_rows(Jreg, _RHAND)
        extra = [vertex_ids["smplx"][n] for n in _EXTRA_NAMES]
        support = set(extra)
        for r in lrows + rrows:
            support.update(np.nonzero(r)[0].tolist())
        S = sorted(support)
        col = {v: i for i, v in enumerate(S)}

        betas = torch.tensor(SOKE_SHAPE, dtype=torch.float64)
        J_full = torch.from_numpy(Jreg)
        self.register_buffer("J_base", (J_full @ (v_t + dirs[..., :10] @ betas)).float())        # [55,3]
        self.register_buffer("J_exprdirs", torch.einsum("jv,vcd->jcd", J_full, dirs[..., 10:]).float())
        self.register_buffer("v_base", (v_t + dirs[..., :10] @ betas)[S].float())               # [S,3]
        self.register_buffer("v_exprdirs", dirs[S, :, 10:].float())                              # [S,3,10]
        pd = layer.posedirs.detach().reshape(-1, V, 3)[:, S]                                     # [486,S,3]
        self.register_buffer("posedirs", pd.reshape(pd.shape[0], -1).float())
        self.register_buffer("lbs_weights", layer.lbs_weights.detach()[S].float())               # [S,55]
        self.register_buffer("lreg", torch.from_numpy(np.stack(lrows)[:, S]).float())            # [21,S]
        self.register_buffer("rreg", torch.from_numpy(np.stack(rrows)[:, S]).float())
        self.parents = [int(p) for p in layer.parents.tolist()]
        # body points: skeleton joint (index < 55) or extra vertex (index 55 + k)
        self.body_joint_pos = [i for i, j in enumerate(UPPER_BODY_IDX) if j < N_SMPLX_JOINTS]
        self.body_joint_idx = [j for j in UPPER_BODY_IDX if j < N_SMPLX_JOINTS]
        self.body_vert_pos = [i for i, j in enumerate(UPPER_BODY_IDX) if j >= N_SMPLX_JOINTS]
        self.body_vert_idx = [col[extra[j - N_SMPLX_JOINTS]] for j in UPPER_BODY_IDX
                              if j >= N_SMPLX_JOINTS]
        self.register_buffer("feat_to_smplx", torch.tensor(FEAT_TO_SMPLX))
        # flat_hand_mean=False: the physical hand rotation is (feature + hands_mean) in axis-angle
        hm = np.zeros((N_FEAT_JOINTS, 3), np.float32)
        hm[10:25] = layer.left_hand_mean.detach().numpy().reshape(15, 3)
        hm[25:40] = layer.right_hand_mean.detach().numpy().reshape(15, 3)
        self.register_buffer("hand_mean", torch.from_numpy(hm))
        mean, std = soke_stats(str(stats) if stats else None)
        self.register_buffer("feat_mean", torch.from_numpy(mean))
        self.register_buffer("feat_std", torch.from_numpy(std))
        self.n_support = len(S)

    # ------------------------------------------------------------------ core
    def forward(self, rotmats: torch.Tensor, expr: torch.Tensor | None = None) -> dict:
        """rotmats [...,41,3,3] physical joint rotations; expr [...,10] raw SMPL-X coefficients."""
        lead = rotmats.shape[:-3]
        R = rotmats.reshape(-1, N_FEAT_JOINTS, 3, 3)
        N, dev, dt = R.shape[0], R.device, R.dtype
        rot = torch.eye(3, device=dev, dtype=dt).expand(N, N_SMPLX_JOINTS, 3, 3).clone()
        rot[:, self.feat_to_smplx] = R
        if expr is None:
            J = self.J_base.expand(N, -1, -1)
            v = self.v_base.expand(N, -1, -1)
        else:
            e = expr.reshape(N, -1).to(dt)
            J = self.J_base + torch.einsum("jcd,nd->njc", self.J_exprdirs, e)
            v = self.v_base + torch.einsum("scd,nd->nsc", self.v_exprdirs, e)
        eye = torch.eye(3, device=dev, dtype=dt)
        pose_feat = (rot[:, 1:] - eye).reshape(N, -1)
        v = v + (pose_feat @ self.posedirs).reshape(N, -1, 3)
        # kinematic chain (parents[i] < i)
        Rg, tg = [rot[:, 0]], [J[:, 0]]
        for i in range(1, N_SMPLX_JOINTS):
            p = self.parents[i]
            Rg.append(Rg[p] @ rot[:, i])
            tg.append(tg[p] + (Rg[p] @ (J[:, i] - J[:, p])[..., None])[..., 0])
        Rg, tg = torch.stack(Rg, 1), torch.stack(tg, 1)                 # [N,55,3,3], [N,55,3]
        trans = tg - (Rg @ J[..., None])[..., 0]                         # rest-relative translation
        Rv = torch.einsum("sj,njab->nsab", self.lbs_weights, Rg)
        tv = torch.einsum("sj,nja->nsa", self.lbs_weights, trans)
        verts = (Rv @ v[..., None])[..., 0] + tv                         # [N,S,3]
        body = torch.empty(N, len(UPPER_BODY_IDX), 3, device=dev, dtype=dt)
        body[:, self.body_joint_pos] = tg[:, self.body_joint_idx]
        body[:, self.body_vert_pos] = verts[:, self.body_vert_idx]
        out = {"pelvis": tg[:, 0], "body": body,
               "lhand": torch.einsum("ks,nsc->nkc", self.lreg, verts),
               "rhand": torch.einsum("ks,nsc->nkc", self.rreg, verts)}
        return {k: o.reshape(*lead, *o.shape[1:]) for k, o in out.items()}

    # ------------------------------------------------------------------ helpers
    def from_quats(self, quats: torch.Tensor, expr: torch.Tensor | None = None) -> dict:
        """Physical joint quaternions [...,41,4] (either sign) -> points."""
        return self(G.quat_to_matrix(quats), expr)

    def from_features(self, feats: torch.Tensor, normalised: bool = True) -> dict:
        """SOKE 133-D features [...,133] -> points (same input as SMPLXJoints)."""
        raw = feats * self.feat_std + self.feat_mean if normalised else feats
        aa = raw[..., :123].reshape(*raw.shape[:-1], N_FEAT_JOINTS, 3) + self.hand_mean
        return self(G.aa_to_matrix(aa), raw[..., 123:133])
