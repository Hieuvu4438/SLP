"""Named 39D descriptor, body frame, validity and group-wise reductions."""

import torch
import torch.nn.functional as F

from .data import resample
from .fk import FINGERS, LANDMARKS, PACKED_FINGERS
from .rotations import axes

GROUPS = {
    "wrist": [*range(3), *range(18, 21)],
    "axes": [*range(3, 9), *range(21, 27)],
    "flex": [*range(9, 14), *range(27, 32)],
    "aperture": [14, 15, 32, 33],
    "face_chest": [16, 17, 34, 35],
    "inter": [36, 37, 38],
}


def unit(v):
    norm = v.norm(dim=-1, keepdim=True)
    return v / norm.clamp_min(1e-6), norm[..., 0] >= 1e-6


def body_frame(points):
    pelvis, _, _, left, right = points[..., :5, :].unbind(-2)
    origin = (left + right) / 2
    across = right - left
    up = origin - pelvis
    a, b = axes(across, up)
    width = across.norm(dim=-1)
    orthogonal_up = up - (up * a).sum(-1, keepdim=True) * a
    valid = (
        (width >= 1e-6)
        & (orthogonal_up.norm(dim=-1) >= 1e-6)
        & torch.isfinite(points[..., [0, 3, 4], :]).all(dim=(-2, -1))
    )
    return origin, width, torch.stack((a, b, torch.cross(a, b, dim=-1)), -1), valid


def kinematic_mask(q_valid):
    shape = q_valid.shape[:-1]
    mask = torch.ones(*shape, 39, device=q_valid.device, dtype=torch.bool)
    # Collar motion moves each shoulder and therefore the shared coordinate frame.
    frame = q_valid[..., 1] & q_valid[..., 2]
    for side in range(2):
        offset = side * 18
        arm = q_valid[..., 4 + side] & q_valid[..., 6 + side]
        palm = arm & q_valid[..., 8 + side]
        start = 10 + side * 15
        finger = {
            name: q_valid[..., start + 3 * i : start + 3 * i + 3].all(-1)
            for i, name in enumerate(PACKED_FINGERS)
        }
        mask[..., offset : offset + 3] &= arm[..., None]
        mask[..., offset + 3 : offset + 9] &= palm[..., None]
        for i, name in enumerate(FINGERS):
            mask[..., offset + 9 + i] &= finger[name]
        mask[..., offset + 14] &= finger["thumb"] & finger["index"]
        mask[..., offset + 15] &= finger["index"] & finger["middle"]
        mask[..., offset + 16 : offset + 18] &= palm[..., None]
        mask[..., offset + 16] &= q_valid[..., [0, 3, 40]].all(-1)
    mask[..., 36:] &= mask[..., :3] & mask[..., 18:21]
    return mask & frame[..., None]


def describe(points, q_valid=None, rest_points=None):
    finite = points.isfinite().all(dim=-1)
    _, _, _, torso_valid = body_frame(points)
    points = torch.where(points.isfinite(), points, torch.zeros_like(points))
    origin, width, basis, _ = body_frame(points)
    if rest_points is not None:
        rest_origin, rest_width, rest_basis, _ = body_frame(rest_points)
        count = len(points)
        time = torch.arange(count, device=points.device)
        previous = torch.where(torso_valid, time + 1, 0).cummax(0).values
        origin = torch.cat((rest_origin.reshape(1, 3), origin))[previous]
        width = torch.cat((rest_width.reshape(1), width))[previous]
        basis = torch.cat((rest_basis.reshape(1, 3, 3), basis))[previous]
    width = width.clamp_min(1e-6)
    blocks, masks = [], []
    for side, offset in enumerate((5, 26)):
        wrist = points[..., offset, :]
        f = points[..., offset + 1 : offset + 21, :].unflatten(-2, (5, 4))
        tip = f[..., -1, :]
        first = f[..., 0, :]
        d, valid_d = unit(first[..., 2, :] - wrist)
        n, valid_n = unit(torch.cross(d, first[..., 1, :] - first[..., 4, :], dim=-1))
        n = n * (-1 if side == 0 else 1)
        segments, segment_valid = unit(f.diff(dim=-2))
        u, v = segments[..., :-1, :], segments[..., 1:, :]
        pairs = segment_valid[..., :-1] & segment_valid[..., 1:]
        cross = torch.cross(u, v, dim=-1).norm(dim=-1)
        dot = (u * v).sum(-1).clamp(-1, 1)
        flex = torch.atan2(cross, torch.where(pairs, dot, torch.ones_like(dot)))
        flex_valid = segment_valid.all(dim=-1)
        palm = (wrist + first[..., 1:, :].sum(-2)) / 5
        distances = (
            torch.stack(
                (
                    (tip[..., 0, :] - tip[..., 1, :]).norm(dim=-1),
                    (tip[..., 1, :] - tip[..., 2, :]).norm(dim=-1),
                    (palm - points[..., 2, :]).norm(dim=-1),
                    (palm - points[..., 1, :]).norm(dim=-1),
                ),
                -1,
            )
            / width[..., None]
        )

        def local(point):
            return (point[..., None, :] @ basis).squeeze(-2)

        blocks.append(
            torch.cat(
                (
                    local(wrist - origin) / width[..., None],
                    local(d),
                    local(n),
                    flex.mean(-1),
                    distances,
                ),
                -1,
            )
        )
        mask = torso_valid[..., None].expand_as(blocks[-1]).clone()
        mask[..., 3:6] &= valid_d[..., None]
        mask[..., 6:9] &= (valid_d & valid_n)[..., None]
        mask[..., 9:14] &= flex_valid
        masks.append(mask)
    r = torch.cat((*blocks, blocks[1][..., :3] - blocks[0][..., :3]), -1)
    mask = torch.cat((*masks, masks[0][..., :3] & masks[1][..., :3]), -1)
    mask &= landmark_mask(finite)
    if q_valid is not None:
        mask &= kinematic_mask(q_valid)
    return r, mask


def extract_relations(q, valid, fk, rest=None):
    points = fk(q)
    return describe(points, valid, fk(rest[None])[0] if rest is not None else None)


def landmark_mask(finite):
    """Limit invalidity to descriptors using missing landmarks."""
    torso = finite[..., [0, 3, 4]].all(-1)
    mask = torso[..., None].expand(*torso.shape, 39).clone()
    for side, off in enumerate((5, 26)):
        desc = mask[..., 18 * side : 18 * (side + 1)]
        wrist = finite[..., off]
        hand = finite[..., off + 1 : off + 21].unflatten(-1, (5, 4))
        first, tip = hand[..., 0], hand[..., 3]
        desc[..., :3] &= wrist[..., None]
        desc[..., 3:6] &= (wrist & first[..., 2])[..., None]
        desc[..., 6:9] &= (wrist & first[..., [1, 2, 4]].all(-1))[..., None]
        desc[..., 9:14] &= hand.all(-1)
        desc[..., 14] &= tip[..., :2].all(-1)
        desc[..., 15] &= tip[..., 1:3].all(-1)
        palm = wrist & first[..., 1:].all(-1)
        desc[..., 16] &= palm & finite[..., 2]
        desc[..., 17] &= palm & finite[..., 1]
    mask[..., 36:] &= mask[..., :3] & mask[..., 18:21]
    return mask


def project_relations(raw, from_logits=False):
    blocks = []
    for start in (0, 18):
        b = raw[..., start : start + 18]
        d, n = axes(b[..., 3:6], b[..., 6:9])
        flex = torch.pi * b[..., 9:14].sigmoid() if from_logits else b[..., 9:14]
        distance = F.softplus(b[..., 14:18]) if from_logits else b[..., 14:18]
        blocks.append(torch.cat((b[..., :3], d, n, flex, distance), -1))
    return torch.cat((*blocks, raw[..., 36:]), -1)


def dense_relations(r, phase):
    return project_relations(resample(r, phase))


def scaler(relations, weight):
    count = weight.sum(0).clamp_min(1)
    mean = (relations * weight).sum(0) / count
    scale = (((relations - mean).square() * weight).sum(0) / count).sqrt().clamp_min(0.05)
    mean[GROUPS["axes"]] = 0
    scale[GROUPS["axes"]] = 1
    return mean, scale


def relation_terms(pred, target, weight, scale):
    terms = {}
    for name, ids in GROUPS.items():
        p, r, w = pred[..., ids], target[..., ids], weight[..., ids]
        if name == "axes":
            error = 1 - (p.unflatten(-1, (4, 3)) * r.unflatten(-1, (4, 3))).sum(-1)
            w = w.unflatten(-1, (4, 3)).amin(-1)
        else:
            error = F.smooth_l1_loss((p - r) / scale[ids], torch.zeros_like(p), reduction="none")
        terms[name] = ((error * w).sum(), w.sum())
    return terms


def ik_weights(weights, scale, ablate=()):
    result = torch.zeros_like(scale)
    for name, ids in GROUPS.items():
        if name not in ablate:
            result[ids] = (weights[name] / len(ids)) ** 0.5 / scale[ids]
    return result


def wrist_tips(points):
    origin, width, basis, valid = body_frame(points)
    indices = [5, 26] + [LANDMARKS.index(f"{s}_{f}") for s in ("left", "right") for f in FINGERS]
    p = (points[..., indices, :] - origin[..., None, :]) @ basis
    return p / width.clamp_min(1e-6)[..., None, None], valid
