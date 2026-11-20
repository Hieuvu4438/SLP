"""Lever-arm joint weights for the quaternion loss (docs/proposal7 E4-c, task C5).

The paper's loss weighs the 41 joint rotations equally, but the reported metrics are joint
*positions* after forward kinematics. To first order, a small rotation error theta about joint j
moves a measured point k by |theta x r_jk|, so the squared position error it causes is
proportional to |r_jk|^2 (the weighting that makes the quaternion MSE match a position MSE).

One lever term per reported DTW column, each over the points that column measures:
    jpe_body : 12 upper-body points, pelvis-aligned; r_jk = p_k - p_j
    pa_body  : same points (Procrustes removes a global rotation; approximated by jpe_body)
    jpe_hand : 21 points per hand, wrist-aligned; r_jk = p_k - p_wrist when j is the wrist or one of
               its ancestors (rotating the arm turns the whole hand about the wrist), else p_k - p_j
    pa_hand  : same points, but the hand's global rotation is removed by Procrustes, so only the
               finger joints below the wrist count
W_c(j) = mean over the column's points k moved by j of |r_jk|^2 (hands: mean of left and right).
Summing raw position errors would let the 50 cm arm dominate the 3 cm fingers, so each column is
normalised to mean 1 first and the columns are averaged: w = mean_c W_c / mean(W_c). Positions
are those of the SOKE evaluation skeleton at the rest pose (SMPL-X neutral, SOKE shape, relaxed
hands = hands_mean). Weights are floored at ``floor`` x mean, so that joints no reported point
depends on (the jaw) keep some supervision, and normalised to mean 1 (sum 41), so the loss
scale matches the uniform weighting.
"""

from __future__ import annotations

import numpy as np

# SMPL-X joint index of each of the 41 rotations (representation.py order):
# upper body 12..21 | left hand 25..39 | right hand 40..54 | jaw 22
SMPLX_OF_ROT = list(range(12, 22)) + list(range(25, 40)) + list(range(40, 55)) + [22]
HEAD = 15
WRISTS = (20, 21)


def _ancestors(j: int, parents) -> list[int]:
    out = []
    while j >= 0:
        out.append(j)
        j = int(parents[j])
    return out  # j itself first


def _hand_points(spec) -> list[tuple[int, bool]]:
    """SOKE hand regressor spec -> (attached SMPL-X joint, is_vertex) for its 21 points."""
    out, last = [], None
    for item in spec:
        if isinstance(item, list):
            for j in item:
                out.append((int(j), False))
                last = int(j)
        else:  # fingertip vertex, carried by the last joint of its finger
            out.append((last, True))
    return out


def column_levers(joints: np.ndarray, lhand: np.ndarray, rhand: np.ndarray, parents,
                  body_idx, lhand_spec, rhand_spec) -> dict[str, np.ndarray]:
    """Raw lever terms W_c [41] per reported column (see module docstring)."""
    body_attach = [(int(k), False) if k < 55 else (HEAD, True) for k in body_idx]  # 55..59: face landmarks

    def moves(j, a, is_vertex):  # does rotating joint j move a point carried by joint a?
        return j in _ancestors(a, parents)[1:] or (is_vertex and j == a)

    out = {c: np.zeros(len(SMPLX_OF_ROT)) for c in ("jpe_body", "jpe_hand", "pa_hand")}
    for i, j in enumerate(SMPLX_OF_ROT):
        acc = sum(float(((joints[k] - joints[j]) ** 2).sum())
                  for (a, v), k in zip(body_attach, body_idx) if moves(j, a, v))
        out["jpe_body"][i] = acc / len(body_idx)
        for pts, spec, wrist in ((lhand, lhand_spec, WRISTS[0]), (rhand, rhand_spec, WRISTS[1])):
            above = j in _ancestors(wrist, parents)  # wrist itself or an arm / torso joint
            hp = [(a, v, p) for (a, v), p in zip(_hand_points(spec), pts) if moves(j, a, v)]
            pivot = joints[wrist] if above else joints[j]
            r2 = sum(float(((p - pivot) ** 2).sum()) for _, _, p in hp) / len(pts)
            out["jpe_hand"][i] += 0.5 * r2
            if not above:
                out["pa_hand"][i] += 0.5 * r2
    out["pa_body"] = out["jpe_body"]
    return out


def lever_arm_weights(joints: np.ndarray, lhand: np.ndarray, rhand: np.ndarray, parents,
                      body_idx, lhand_spec, rhand_spec, floor: float = 0.1) -> np.ndarray:
    """joints [>=60,3] SMPL-X output joints, lhand/rhand [21,3] regressed hand points (rest pose).

    Returns float32 [41] weights (mean 1) in the order of the 41 rotations."""
    cols = column_levers(joints, lhand, rhand, parents, body_idx, lhand_spec, rhand_spec)
    w = np.mean([c / c.mean() for c in cols.values()], 0)
    w = np.maximum(w, floor * w.mean())
    return (w / w.mean()).astype(np.float32)


def soke_lever_arm_weights(floor: float = 0.1, device: str = "cpu") -> np.ndarray:
    """Weights from the SOKE evaluation skeleton (needs the licensed SMPL-X model files)."""
    from slp.evaluation.smplx_fk import _LHAND, _RHAND, UPPER_BODY_IDX, SMPLXJoints

    fk = SMPLXJoints(device=device)
    j, l, r = fk(np.zeros((1, 133), np.float32), normalised=False)  # raw zero pose = rest + hands_mean
    parents = fk.layer.parents.cpu().numpy()
    return lever_arm_weights(j[0].cpu().numpy(), l[0].cpu().numpy(), r[0].cpu().numpy(), parents,
                             UPPER_BODY_IDX, _LHAND, _RHAND, floor=floor)
