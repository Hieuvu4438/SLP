"""Reconstruction losses of an x1-prediction (shared by Stage A and Stage B).

    L = w_quat  * sum_j |q^_j - q_j|^2                  chordal, per frame (SignNMFlow Alg. 2)
      + w_pos   * Huber_d( |p^ - p| )                     joint positions in metres (SOKE points)
      + w_vel   * Huber_d( |dp^ - dp| )                   frame-to-frame displacement of the points
      + w_expr  * sum_k (e^_k - e_k)^2                    normalised expression

Positions come from the exact sparse SMPL-X FK (``slp.skeleton_fk``) on the points the SOKE
metrics read: 12 upper-body points (pelvis-relative = absolute, root is fixed) and 21 points per
hand *relative to the wrist* (the hand metrics are wrist-aligned). Per frame, the position term
is mean(body) + mean(hands) so body and hands weigh equally, as in the reported columns. The
Huber penalty acts on the per-point Euclidean distance: quadratic below ``delta`` metres and
linear (= the unsquared L2 the JPE metrics average) above it.

Every term is a sum over frames divided by ``denom`` (number of valid frames of the full batch),
so micro-batch losses add up exactly to the batch loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from slp.skeleton_fk import SkeletonFK


@dataclass(frozen=True)
class LossWeights:
    quat: float = 10.0
    pos: float = 0.0
    vel: float = 0.0
    expr: float = 0.1
    delta_pos: float = 0.01   # metres
    delta_vel: float = 0.002  # metres / frame


def huber(d: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber of a non-negative distance, scaled so the linear part is d - delta/2."""
    return torch.where(d < delta, 0.5 * d * d / delta, d - 0.5 * delta)


def _dist(a, b):
    return ((a - b).pow(2).sum(-1) + 1e-12).sqrt()


def points(fk: SkeletonFK, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """quats [...,41,4] -> body [...,12,3], wrist-relative hands [...,2,20,3] (always fp32)."""
    with torch.autocast(device_type=q.device.type, enabled=False):
        o = fk.from_quats(q.float())
    lh = o["lhand"][..., 1:, :] - o["lhand"][..., :1, :]
    rh = o["rhand"][..., 1:, :] - o["rhand"][..., :1, :]
    return o["body"], torch.stack([lh, rh], -3)


def reconstruction_loss(w: LossWeights, fk: SkeletonFK | None, q_hat, e_hat, q1, e1, mask,
                        denom=None, q_target=None) -> tuple[torch.Tensor, dict]:
    """q_hat/q1 [B,T,41,4] (q_hat on S^3), e [B,T,10], mask [B,T] (True = valid).

    ``q_target``: representative of q1 used by the chordal term (defaults to q1); positions are
    sign-invariant. Returns (weighted loss, detached parts)."""
    m = mask.float()
    n = m.sum().clamp_min(1) if denom is None else denom
    qt = q1 if q_target is None else q_target
    lq = (((q_hat - qt) ** 2).flatten(2).sum(-1) * m).sum() / n
    le = (((e_hat - e1) ** 2).sum(-1) * m).sum() / n
    loss = w.quat * lq + w.expr * le
    parts = {"loss_quat": lq.detach(), "loss_expr": le.detach()}
    if fk is not None and (w.pos > 0 or w.vel > 0):
        bh, hh = points(fk, q_hat)
        with torch.no_grad():
            b1, h1 = points(fk, q1)
        db, dh = _dist(bh, b1), _dist(hh, h1)                      # [B,T,12], [B,T,2,20]
        per_frame = huber(db, w.delta_pos).mean(-1) + huber(dh, w.delta_pos).flatten(2).mean(-1)
        lp = (per_frame * m).sum() / n
        parts["loss_pos"] = lp.detach()
        parts["err_body_m"] = ((db.mean(-1) * m).sum() / m.sum().clamp_min(1)).detach()
        parts["err_hand_m"] = ((dh.flatten(2).mean(-1) * m).sum() / m.sum().clamp_min(1)).detach()
        loss = loss + w.pos * lp
        if w.vel > 0 and mask.shape[1] > 1:
            mv = (mask[:, 1:] & mask[:, :-1]).float()
            vb = _dist(bh[:, 1:] - bh[:, :-1], b1[:, 1:] - b1[:, :-1])
            vh = _dist(hh[:, 1:] - hh[:, :-1], h1[:, 1:] - h1[:, :-1])
            pf = huber(vb, w.delta_vel).mean(-1) + huber(vh, w.delta_vel).flatten(2).mean(-1)
            lv = (pf * mv).sum() / n
            parts["loss_vel"] = lv.detach()
            loss = loss + w.vel * lv
    return loss, parts


@torch.no_grad()
def point_error(fk: SkeletonFK, q_hat, q1, mask) -> dict:
    """Frame-aligned mean point distance in metres (monitoring / checkpoint selection only)."""
    bh, hh = points(fk, q_hat)
    b1, h1 = points(fk, q1)
    m = mask.float()
    k = m.sum().clamp_min(1)
    return {"body": float((_dist(bh, b1).mean(-1) * m).sum() / k),
            "hand": float((_dist(hh, h1).flatten(2).mean(-1) * m).sum() / k)}
