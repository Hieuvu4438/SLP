"""Differentiable forward kinematics for the position losses of E4-a / E4-b (task C6).

Reproduces the SOKE evaluation skeleton (``slp.evaluation.smplx_fk.SMPLXJoints``) from the 41
joint rotations (unit quaternions, *physical* rotations: hands include ``hands_mean``) and the
10 expression coefficients, with gradients, through a sparse linear-blend skinning:

* only the vertices the evaluated points depend on are skinned (support of SOKE's hand
  regressors: J_regressor rows + fingertip vertices, plus the five face landmarks of the upper
  body set) - about 1.5k of the 10,475 SMPL-X vertices, including pose-corrective blend shapes;
* body points  = SMPL-X output joints UPPER_BODY_IDX (12 points; 55..59 are vertex landmarks);
* hand points  = SOKE ``orig_hand_regressor`` (21 per hand) applied to the skinned vertices.

Everything else is fixed exactly as in the evaluation: neutral SMPL-X, SOKE shape coefficients,
zero root / lower body / eyes, no translation. ``SparseSMPLX(...)(quats, expr)`` returns
(body [N,12,3], lhand [N,21,3], rhand [N,21,3]) in metres.

Losses (``position_losses``): squared distances in cm^2, averaged over points and valid frames.
    body : pelvis-aligned upper-body points (the pelvis is fixed, as in DTW-JPE body)
    hand : wrist-relative hand points (as DTW-JPE hand)
    vel  : temporal differences of both (E4-b)
"""

from __future__ import annotations

import torch
from torch import nn

from . import geometry as G
from .levers import SMPLX_OF_ROT

N_SMPLX_JOINTS = 55


class SparseSMPLX(nn.Module):
    def __init__(self, device: str | torch.device = "cpu", model_dir=None):
        super().__init__()
        from slp.evaluation.smplx_fk import UPPER_BODY_IDX, SMPLXJoints, SOKE_SHAPE

        ref = SMPLXJoints(device="cpu", model_dir=model_dir)
        L = ref.layer
        lreg, rreg = ref.lreg.double(), ref.rreg.double()
        extra = L.vertex_joint_selector.extra_joints_idxs.long()
        body_idx = list(UPPER_BODY_IDX)
        land_vids = [int(extra[k - N_SMPLX_JOINTS]) for k in body_idx if k >= N_SMPLX_JOINTS]
        support = torch.unique(torch.cat([(lreg != 0).any(0).nonzero()[:, 0], (rreg != 0).any(0).nonzero()[:, 0],
                                          torch.tensor(land_vids)]))
        pos = {int(v): i for i, v in enumerate(support.tolist())}
        S = len(support)
        betas = torch.tensor(SOKE_SHAPE, dtype=torch.float64)
        v_t = L.v_template.double()
        sd, ed = L.shapedirs.double(), L.expr_dirs.double()
        Jr = L.J_regressor.double()
        v_shaped = v_t + torch.einsum("vcb,b->vc", sd, betas)  # betas fixed: constant part
        # joints of the shaped (rest) skeleton: J = Jr (v_shaped + expr_dirs psi), affine in psi
        self.register_buffer("J0", (Jr @ v_shaped).float())                       # [55,3]
        self.register_buffer("J_expr", torch.einsum("jv,vce->jce", Jr, ed).float())  # [55,3,10]
        self.register_buffer("v0", v_shaped[support].float())                     # [S,3]
        self.register_buffer("v_expr", ed[support].float())                       # [S,3,10]
        posedirs = L.posedirs.double().reshape(L.posedirs.shape[0], -1, 3)        # [486,V,3]
        self.register_buffer("posedirs", posedirs[:, support].reshape(posedirs.shape[0], -1).float())
        self.register_buffer("W", L.lbs_weights.double()[support].float())        # [S,55]
        self.register_buffer("lreg", lreg[:, support].float())                    # [21,S]
        self.register_buffer("rreg", rreg[:, support].float())
        self.register_buffer("parents", L.parents.long().clone())
        # body points: skeleton joints, or landmark vertices (index into the support)
        self.body_joint = [k if k < N_SMPLX_JOINTS else -1 for k in body_idx]
        self.body_vertex = [pos[int(extra[k - N_SMPLX_JOINTS])] if k >= N_SMPLX_JOINTS else -1 for k in body_idx]
        self.register_buffer("rot_index", torch.tensor(SMPLX_OF_ROT))
        self.n_support = S
        self.to(device)

    def forward(self, quats: torch.Tensor, expr: torch.Tensor):
        """quats [N,41,4] physical rotations, expr [N,10] raw expression coefficients."""
        from smplx.lbs import batch_rigid_transform

        N = quats.shape[0]
        dt = self.J0.dtype
        R41 = G.quat_to_matrix(quats.to(dt))
        rot = torch.eye(3, device=quats.device, dtype=dt).expand(N, N_SMPLX_JOINTS, 3, 3).clone()
        rot[:, self.rot_index] = R41
        J = self.J0 + torch.einsum("jce,ne->njc", self.J_expr, expr.to(dt))
        posed_J, A = batch_rigid_transform(rot, J, self.parents, dtype=dt)
        eye = torch.eye(3, device=quats.device, dtype=dt)
        pose_feat = (rot[:, 1:] - eye).reshape(N, -1)                              # [N,486]
        v = self.v0 + torch.einsum("sce,ne->nsc", self.v_expr, expr.to(dt))
        v = v + (pose_feat @ self.posedirs).reshape(N, self.n_support, 3)
        T = torch.einsum("sj,njab->nsab", self.W, A)                               # [N,S,4,4]
        verts = torch.einsum("nsab,nsb->nsa", T[..., :3, :3], v) + T[..., :3, 3]
        body = torch.stack([posed_J[:, j] if j >= 0 else verts[:, s]
                            for j, s in zip(self.body_joint, self.body_vertex)], 1)
        lh = torch.einsum("ks,nsc->nkc", self.lreg, verts)
        rh = torch.einsum("ks,nsc->nkc", self.rreg, verts)
        return body, lh, rh


def _points(fk: SparseSMPLX, quats, expr):
    """[N, 54, 3] in cm: 12 body points, then both hands wrist-relative (21 + 21)."""
    body, lh, rh = fk(quats, expr)
    hands = torch.cat([lh - lh[:, :1], rh - rh[:, :1]], 1)
    return 100.0 * body, 100.0 * hands


def position_losses(fk: SparseSMPLX, q_hat, e_hat, q_gt, e_gt, mask, denom=None, velocity: bool = False):
    """FK losses of a predicted clean sample. q_* [B,T,41,4] physical rotations, e_* [B,T,10] raw
    expression, mask [B,T] valid frames. Returns dict of scalar losses (cm^2):
        body, hand : mean over points of the squared distance, mean over valid frames
                     (``denom`` = valid-frame count of the full batch, as in flow._masked_mse)
        vel        : the same for the frame-to-frame differences (body + hand), when ``velocity``."""
    B, T = mask.shape
    m = mask.reshape(-1)
    bh, hh = _points(fk, q_hat.reshape(B * T, 41, 4)[m], e_hat.reshape(B * T, -1)[m])
    with torch.no_grad():
        bg, hg = _points(fk, q_gt.reshape(B * T, 41, 4)[m], e_gt.reshape(B * T, -1)[m])
    n = mask.sum().clamp_min(1) if denom is None else denom
    out = {"body": ((bh - bg) ** 2).sum(-1).mean(-1).sum() / n,
           "hand": ((hh - hg) ** 2).sum(-1).mean(-1).sum() / n}
    if velocity:
        def full(x):  # scatter the valid frames back to [B,T,P,3]
            z = x.new_zeros(B * T, *x.shape[1:])
            z[m] = x
            return z.reshape(B, T, *x.shape[1:])
        P = torch.cat([full(bh), full(hh)], 2)
        Pg = torch.cat([full(bg), full(hg)], 2)
        pair = mask[:, 1:] & mask[:, :-1]
        d = ((P[:, 1:] - P[:, :-1]) - (Pg[:, 1:] - Pg[:, :-1])) ** 2
        out["vel"] = (d.sum(-1).mean(-1) * pair).sum() / (pair.sum().clamp_min(1) if denom is None else denom)
    return out


class PositionLoss(nn.Module):
    """lambda_body * body + lambda_hand * hand (+ lambda_vel * vel) on the predicted clean sample.

    ``expr_mean`` / ``expr_std``: SOKE statistics of the 10 expression channels (the generator works
    on normalised expressions)."""

    def __init__(self, fk: SparseSMPLX, expr_mean, expr_std, lambda_body=0.0, lambda_hand=0.0, lambda_vel=0.0):
        super().__init__()
        self.fk = fk
        self.register_buffer("e_mean", torch.as_tensor(expr_mean, dtype=torch.float32))
        self.register_buffer("e_std", torch.as_tensor(expr_std, dtype=torch.float32))
        self.lb, self.lh, self.lv = float(lambda_body), float(lambda_hand), float(lambda_vel)

    def forward(self, q_hat, e_hat, q_gt, e_gt, mask, denom=None):
        raw = lambda e: e * self.e_std + self.e_mean
        parts = position_losses(self.fk, q_hat, raw(e_hat), q_gt, raw(e_gt), mask, denom,
                                velocity=self.lv > 0)
        total = self.lb * parts["body"] + self.lh * parts["hand"] + self.lv * parts.get("vel", 0.0)
        return total, {f"loss_fk_{k}": v.detach() for k, v in parts.items()}
