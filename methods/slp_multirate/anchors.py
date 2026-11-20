"""Oracle mining, matched-budget selection and planner supervision."""

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from .geometry import fk, matrix_to_rot6d, merge_local, rot6d_to_matrix


@dataclass(frozen=True)
class Anchors:
    time: torch.Tensor
    payload: torch.Tensor
    slot: torch.Tensor
    feature: torch.Tensor
    confidence: torch.Tensor
    hidden: torch.Tensor | None = None


def empty_anchors(batch, k, device=None):
    slot = torch.zeros((batch, k), dtype=torch.bool, device=device)
    payload = torch.zeros((batch, k, 2, 24), device=device)
    return Anchors(slot.float(), payload, slot, payload.bool(), payload[..., 0])


def stack_anchors(items, device=None):
    return Anchors(*(torch.cat([getattr(a, name) for a in items]).to(device)
                     for name in ("time", "payload", "slot", "feature", "confidence")))


def clip_pose(clip, rig):
    x = merge_local(rot6d_to_matrix(clip["body_rot6d"]), rot6d_to_matrix(clip["hands_rot6d"]))
    seen = torch.cat((clip["body_obs"], clip["hands_obs"].flatten(1, 2)), -1)
    return fk(x, rig, seen)


def _median3(x, valid):
    """Median within observed runs, replicating a run endpoint at its boundary."""
    previous = torch.cat((x[:1], x[:-1]))
    following = torch.cat((x[1:], x[-1:]))
    before = valid & torch.cat((valid[:1], valid[:-1]))
    after = valid & torch.cat((valid[1:], valid[-1:]))
    previous = torch.where(before[..., None], previous, x)
    following = torch.where(after[..., None], following, x)
    return torch.stack((previous, x, following)).median(0).values


def pose_signals(pose, min_speed=.005):
    """Shape/turn signals and their visibility, evaluated at real interior frames."""
    t = len(pose.wrist)
    result = pose.wrist.new_zeros((t, 2))
    valid = torch.zeros_like(result, dtype=torch.bool)
    if t < 3:
        return result, valid
    tip = _median3(pose.local_tips, pose.tip_mask)
    wrist = _median3(pose.wrist, pose.hand_mask)
    v = pose.tip_mask[2:] & pose.tip_mask[1:-1] & pose.tip_mask[:-2]
    diff = (tip[2:] - tip[:-2]).norm(dim=-1) * .5
    shape = (diff * v).sum(-1) / v.sum(-1).clamp_min(1)
    result[1:-1, 0] = shape.max(-1).values
    valid[1:-1, 0] = v.any(dim=(1, 2))
    minus, plus = wrist[1:-1] - wrist[:-2], wrist[2:] - wrist[1:-1]
    a, b = minus.norm(dim=-1), plus.norm(dim=-1)
    observed = pose.hand_mask[2:] & pose.hand_mask[1:-1] & pose.hand_mask[:-2]
    moving = observed & (a >= min_speed) & (b >= min_speed)
    cosine = (minus * plus).sum(-1) / (a * b).clamp_min(1e-12)
    turning = cosine.clamp(-1, 1).acos() / torch.pi
    result[1:-1, 1] = turning.masked_fill(~moving, 0).max(-1).values
    valid[1:-1, 1] = moving.any(-1)
    return result, valid


def event_signals(clip, rig, min_speed=.005, use_contact=False, pose=None):
    pose = clip_pose(clip, rig) if pose is None else pose
    base, valid = pose_signals(pose, min_speed)
    contact = torch.zeros(len(base), device=base.device)
    seen = torch.zeros_like(contact, dtype=torch.bool)
    if use_contact and len(base) > 2:
        labels = clip["contact"].to(base.device)
        covered = clip["contact_valid"].to(base.device)
        pairs = covered[1:] & covered[:-1]
        contact[1:-1] = ((labels[1:] != labels[:-1]) & pairs).any(-1)[:-1]
        seen[1:-1] = pairs.any(-1)[:-1]
    return torch.cat((base, contact[:, None]), -1), torch.cat((valid, seen[:, None]), -1)


def score_scales(clips, rig, cfg):
    chunks = [event_signals(c, rig, cfg["turn_min_speed_shoulder_widths"]) for c in clips]
    values = torch.cat([x[:, :2] for x, _ in chunks])
    masks = torch.cat([m[:, :2] for _, m in chunks])
    scales = []
    for i in range(2):
        positive = values[:, i][(values[:, i] > 0) & masks[:, i]]
        scales.append(max(float(positive.quantile(.9)), 1e-6) if len(positive) else 1e-6)
    return scales


def normalized_signals(signals, coverage, scales):
    denom = signals.new_tensor([*scales, 1.0]).clamp_min(1e-6)
    return (signals / denom).clamp(0, 1).masked_fill(~coverage, 0)


def choose_indices(t, k, policy, segments=(), score=None, gap=2):
    if min(t, k, gap) < 1:
        raise ValueError("Clip length, anchor budget and NMS gap must be positive")
    if policy == "none":
        return []
    k = min(t, k)
    uniforms = [min(t - 1, ((2 * i + 1) * t) // (2 * k)) for i in range(k)]
    if policy == "uniform":
        return uniforms
    selected = []
    if policy == "onset_mid_offset":
        candidates = {i for a, b in segments for i in (a, (a + b) // 2, b) if 0 <= i < t}
        if len(candidates) > k:
            for u in uniforms:
                j = min(candidates, key=lambda i: (abs(i - u), i))
                candidates.remove(j)
                selected.append(j)
        else:
            selected = sorted(candidates)
    elif policy == "event":
        if score is None or len(score) != t:
            raise ValueError("Event policy needs one score per frame")
        order = sorted((i for i in range(t) if score[i] > 0), key=lambda i: (-score[i], i))
        for j in order:
            if all(abs(i - j) >= gap for i in selected):
                selected.append(j)
                if len(selected) == k:
                    break
    else:
        raise ValueError(f"Unknown anchor policy {policy}")
    if not selected:
        selected.append(uniforms[0] if policy == "onset_mid_offset" else 0)
    remaining = set(range(t)) - set(selected)
    while len(selected) < k:
        j = max(remaining, key=lambda i: (min(abs(i - p) for p in selected), -i))
        selected.append(j)
        remaining.remove(j)
    return sorted(selected)


def build_oracle_anchors(clip, rig, cfg, scales, normalize=None):
    t = len(clip["timestamps_s"])
    k, policy = cfg["count"], cfg["policy"]
    result = empty_anchors(1, k)
    if policy == "none":
        return result
    pose = clip_pose(clip, rig)
    score = None
    if policy == "event":
        raw, m = event_signals(clip, rig, cfg["turn_min_speed_shoulder_widths"],
                               cfg["contact_mining"], pose)
        score = (normalized_signals(raw, m, scales) * raw.new_tensor(cfg["score_weights"])).sum(-1)
        score = score.tolist()
    indices = choose_indices(t, k, policy, clip["segments"], score, cfg["nms_min_gap_frames"])
    payload = torch.cat((pose.wrist, matrix_to_rot6d(pose.palm), pose.local_tips.flatten(-2)), -1)
    feature = torch.cat((pose.hand_mask[..., None].expand(-1, -1, 9),
                         pose.tip_mask.repeat_interleave(3, -1)), -1)
    count = len(indices)
    p, m = payload[indices], feature[indices]
    result.time[0, :count] = torch.tensor(indices) / max(1, t - 1)
    result.slot[0, :count] = True
    result.payload[0, :count] = p.masked_fill(~m, 0) if normalize is None else normalize(p, m)
    result.feature[0, :count] = m
    result.confidence[0, :count] = m.any(-1).float()
    return result


def jitter_anchors(anchors, lengths, jitter, noise, generator):
    jitter_shape = anchors.time.shape
    offset = (torch.rand(jitter_shape, generator=generator, device=anchors.time.device) * 2 - 1)
    times = (anchors.time + jitter * offset / (lengths[:, None] - 1).clamp_min(1)).clamp(0, 1)
    noise = torch.randn(anchors.payload.shape, generator=generator, device=anchors.time.device) * noise
    payload = anchors.payload + noise.masked_fill(~anchors.feature, 0)
    # Keep each time attached to its original hand payload through the stable reorder.
    order = times.masked_fill(~anchors.slot, 2).argsort(-1, stable=True)
    slot = anchors.slot.gather(1, order)
    return replace(anchors, time=times.gather(1, order).masked_fill(~slot, 0), slot=slot,
                   payload=payload.gather(1, order[..., None, None].expand_as(payload)),
                   feature=anchors.feature.gather(1, order[..., None, None].expand_as(payload)),
                   confidence=anchors.confidence.gather(1, order[..., None].expand_as(
                       anchors.confidence)))


def confidence_labels(pred, target, lengths, denormalize):
    """A timed hit with adequately observed positional payload, per hand."""
    p, q = (denormalize(a.payload).float() for a in (pred, target))
    pos = torch.cat(((p[..., :3] - q[..., :3]).unsqueeze(-2),
                     (p[..., 9:] - q[..., 9:]).unflatten(-1, (5, 3))), -2)
    m = torch.cat((target.feature[..., :3].all(-1, keepdim=True),
                   target.feature[..., 9:].unflatten(-1, (5, 3)).all(-1)), -1)
    observed = m.any(-1) & target.slot[..., None]
    error = (pos.norm(dim=-1) * m).sum(-1) / m.sum(-1).clamp_min(1)
    time = ((pred.time - target.time).abs() * (lengths[:, None] - 1)).unsqueeze(-1)
    return (time <= 3) & (error <= .05), observed


def reliability(probs, labels, bins=10):
    if not len(probs):
        return {"n": 0, "brier": None, "bins": []}
    index = (probs * bins).long().clamp(0, bins - 1)
    rows = []
    for i in range(bins):
        m = index == i
        rows.append({"lower": i / bins, "upper": (i + 1) / bins, "n": int(m.sum()),
                     "prediction": float(probs[m].mean()) if m.any() else None,
                     "frequency": float(labels[m].float().mean()) if m.any() else None})
    return {"n": len(probs), "brier": float(F.mse_loss(probs, labels.float())), "bins": rows}
