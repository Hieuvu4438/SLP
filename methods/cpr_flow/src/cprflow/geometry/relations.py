import torch
import torch.nn.functional as F

from cprflow.geometry.joint_maps import TIP_ORDER
from cprflow.geometry.temporal import pool2


def relation33(joints, names, threshold=0.08, softness=0.02):
    def get(name):
        return joints[..., names.index(name), :]

    ls, rs = get("left_shoulder"), get("right_shoulder")
    neck, pelvis = get("neck"), get("pelvis")
    origin = (neck + pelvis) * 0.5
    width = (rs - ls).norm(dim=-1, keepdim=True)
    if not torch.isfinite(width).all() or (width <= 0.05).any():
        raise ValueError("Invalid neutral shoulder width (metres)")
    ex = F.normalize(rs - ls, dim=-1)
    v = neck - pelvis
    ey = F.normalize(v - (v * ex).sum(-1, keepdim=True) * ex, dim=-1)
    ez = torch.cross(ex, ey, dim=-1)
    frame = torch.stack((ex, ey, ez), -1)
    if (ey.norm(dim=-1) < 0.99).any():
        raise ValueError("Degenerate body coordinate frame")

    def local(vector):
        return (frame.transpose(-1, -2) @ vector.unsqueeze(-1)).squeeze(-1)

    lw, rw = get("left_wrist"), get("right_wrist")

    def normal(side):
        wrist = get(f"{side}_wrist")
        world = F.normalize(
            torch.cross(get(f"{side}_index1") - wrist, get(f"{side}_pinky1") - wrist, dim=-1),
            dim=-1,
        )
        return local(world)

    anchors = torch.stack((get("jaw"), origin, ls, rs), -2)
    pointsets = [
        torch.stack([get(f"{s}_wrist")] + [get(f"{s}_{f}") for f in TIP_ORDER], -2)
        for s in ("left", "right")
    ]
    anchor_dist = [torch.cdist(points, anchors).amin(-2) / width for points in pointsets]
    between = torch.cdist(*pointsets).amin(dim=(-2, -1))[..., None] / width
    distances = torch.cat((*anchor_dist, between), -1)
    proximity = torch.sigmoid((threshold - distances) / softness)
    return torch.cat(
        (
            local(lw - origin) / width,
            local(rw - origin) / width,
            local(rw - lw) / width,
            normal("left"),
            normal("right"),
            distances,
            proximity,
        ),
        -1,
    )


def relation_operator(x, lengths, body_model, normalizer, threshold=0.08, softness=0.02):
    """A(X) = normalize(pool2(relation(FK(X)))); never relation(mean pose)."""
    from torch.nn.utils.rnn import pad_sequence

    features = []
    for clip, n in zip(x, lengths):
        geo = body_model(clip[: int(n)])
        features.append(relation33(geo["joints"], geo["names"], threshold, softness))
    pooled = pool2(pad_sequence(features, batch_first=True), lengths)
    out = normalizer(pooled)
    from cprflow.geometry.temporal import lengths_mask

    return out * lengths_mask((lengths + 1) // 2, out.shape[1])[..., None]


def relation_invalid_rates(physical_relation):
    """Plan diagnostics only: averaged normals may have norm<1 and are valid."""
    r = physical_relation.float()
    return {
        "relation_plan_nonfinite_rate": float((~torch.isfinite(r)).float().mean()),
        "relation_plan_negative_distance_rate": float((r[..., 15:24] < 0).float().mean()),
        "relation_plan_invalid_proximity_rate": float(
            ((r[..., 24:] < 0) | (r[..., 24:] > 1)).float().mean()
        ),
        "relation_plan_invalid_normal_rate": float(
            (r[..., 9:15].reshape(*r.shape[:-1], 2, 3).norm(dim=-1) > 1 + 1e-5).float().mean()
        ),
    }
