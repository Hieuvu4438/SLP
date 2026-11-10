import torch
from .rotations import rotation_6d_to_matrix, geodesic_distance
from .schema import PART_SLICES, manual_to_6d
from .flow import flow_path, endpoint


def sequence_mask(lengths, t):
    return torch.arange(t, device=lengths.device)[None] < lengths[:, None]


def masked_mean(values, mask):
    if values.shape[:2] != mask.shape or (mask.sum(-1) == 0).any():
        raise ValueError("Every sample needs supervised frames")
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    weight = mask.expand_as(values).to(values.dtype)
    dims = tuple(range(1, values.ndim))
    return (values.masked_fill(~mask, 0).sum(dims) / weight.sum(dims)).mean()


def frame_errors(pred, target, geometry, cfg):
    """Training surrogate only, NOT final DTW/BT benchmark metrics."""
    a, b = geometry(pred), geometry(target)
    joint, velocity = [], []
    for p, q in ((a.body, b.body), (a.left, b.left), (a.right, b.right)):
        joint.append((p - q).norm(dim=-1).mean(-1))
        diff = (p[:, 1:] - p[:, :-1]) - (q[:, 1:] - q[:, :-1])
        vel = diff.norm(dim=-1).mean(-1)
        velocity.append(
            torch.cat((torch.zeros_like(vel[:, :1]), vel), 1)
            if pred.shape[1] > 1
            else pred.new_zeros(pred.shape[:2])
        )
    relation = (a.relations - b.relations).abs().mean(-1)
    rotation = []
    for sl in PART_SLICES:
        pm = rotation_6d_to_matrix(pred[..., sl].reshape(*pred.shape[:2], -1, 6))
        qm = rotation_6d_to_matrix(target[..., sl].reshape(*target.shape[:2], -1, 6))
        rotation.append(geodesic_distance(pm, qm).square().mean(-1))
    return {
        "joint": torch.stack(joint).mean(0),
        "relation": relation,
        "rotation": torch.stack(rotation).mean(0),
        "velocity": torch.stack(velocity).mean(0),
    }


def surrogate_frames(pred, target, geometry, cfg):
    e = frame_errors(pred, target, geometry, cfg)
    return sum(getattr(cfg, f"{key}_weight") * value for key, value in e.items())


def decoder_objective(
    model, normalizer, geometry, batch, cfg, *, generator=None, use_text=True, coupled=True
):
    target_manual = (
        manual_to_6d(batch["target_pose"], geometry.manual_mean)
        if "target_pose" in batch
        else batch["target_manual"]
    )
    target = normalizer(target_manual)
    noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
    time = torch.rand(len(target), device=target.device, dtype=target.dtype, generator=generator)
    state, truth = flow_path(target, time, noise)
    condition = {
        k: batch[k]
        for k in (
            "codes",
            "token_lengths",
            "text",
            "text_mask",
            "lengths",
            "language",
            "relations",
            "offsets",
        )
    }
    velocity = model(state, time, **condition, use_text=use_text, coupled=coupled)
    valid = sequence_mask(batch["lengths"], target.shape[1])
    supervised = batch.get("loss_mask", valid) & valid
    flow = masked_mean((velocity - truth).square(), supervised)
    pred = normalizer.inverse(endpoint(state, time, velocity))
    errors = frame_errors(pred, target_manual, geometry, cfg)
    terms = {"flow": flow}
    for key, frames in errors.items():
        mask = supervised.clone()
        if key == "velocity":
            mask[:, 0] = False
            mask[:, 1:] &= valid[:, :-1]
            if not mask.any():
                terms[key] = velocity.sum() * 0
                continue
            # Single-frame items contribute zero, without invalid per-sample division.
            counts = mask.sum(-1).clamp_min(1)
            terms[key] = (frames.masked_fill(~mask, 0).sum(-1) / counts).mean()
        else:
            terms[key] = masked_mean(frames, mask)
    total = flow + sum(getattr(cfg, f"{k}_weight") * terms[k] for k in errors)
    return total, terms
