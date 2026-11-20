"""Endpoint flow matching, masked objectives, normalization and Euler sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .data import stable_seed
from .geometry import decode_pose, fk, merge_local, rot6d_to_matrix


@dataclass(frozen=True)
class Stats:
    mean: torch.Tensor
    std: torch.Tensor
    count: torch.Tensor

    def normalize(self, x, mask=None):
        y = (x.float() - self.mean.to(x.device)) / self.std.to(x.device)
        return y if mask is None else y.masked_fill(~mask, 0)

    def denormalize(self, x):
        return x.float() * self.std.to(x.device) + self.mean.to(x.device)

    def state(self):
        return dict(mean=self.mean.cpu(), std=self.std.cpu(), count=self.count.cpu())


class Moments:
    def __init__(self, shape):
        self.mean = torch.zeros(shape)
        self.m2 = torch.zeros(shape)
        self.count = torch.zeros(shape)

    def add(self, x, mask, weight=1.):
        """Merge a batch's centered moments in FP32, with optional mixture weights."""
        x = x.detach().float().cpu().reshape(-1, *self.mean.shape)
        mask = mask.detach().cpu().reshape_as(x)
        x = x.masked_fill(~mask, 0)
        n = mask.sum(0)
        batch_mean = x.sum(0) / n.clamp_min(1)
        batch_m2 = (x - batch_mean).masked_fill(~mask, 0).square().sum(0)
        weighted = n * weight
        count = self.count + weighted
        divisor = torch.where(count > 0, count, 1.)
        delta = batch_mean - self.mean
        self.m2 += weight * batch_m2 + delta.square() * self.count * weighted / divisor
        self.mean += delta * weighted / divisor
        self.count = count

    def finish(self, floor):
        divisor = torch.where(self.count > 0, self.count, 1.)
        std = (self.m2 / divisor).clamp_min(0).sqrt().clamp_min(floor)
        return Stats(self.mean, torch.where(self.count > 0, std, 1.), self.count)


def per_clip_mean(error, mask):
    """Scalar masked mean per clip with a count of eligible clips."""
    error = error.float()
    mask = torch.broadcast_to(mask, error.shape)
    dims = tuple(range(1, error.ndim))
    n = mask.sum(dims)
    score = error.masked_fill(~mask, 0).sum(dims) / n.clamp_min(1)
    present = n > 0
    return score, present


def masked_mean(error, mask):
    score, present = per_clip_mean(error, mask)
    return score.sum() / present.sum().clamp_min(1), present.sum()


def smooth(x, beta=1.):
    return F.smooth_l1_loss(x.float(), torch.zeros_like(x, dtype=torch.float32), beta=beta,
                            reduction="none")


def length_loss(model, text, lengths):
    log_t = model(text)
    relative = ((log_t.clamp(-20, 30).exp() - lengths).abs() / lengths).detach()
    value = smooth(log_t - lengths.float().log(), .10).mean()
    return value, {"length": value.detach(), "relative_errors": relative}


def anchor_loss(pred, target, lengths):
    slots = pred.slot & target.slot
    span = (lengths[:, None] - 1).float()
    time, time_n = masked_mean(smooth((pred.time - target.time) * span, .10), slots)
    spacing, spacing_n = masked_mean((1 - (pred.time.diff(dim=1) * span)).relu(),
                                     slots[:, 1:] & slots[:, :-1])
    difference = smooth(pred.payload - target.payload)
    groups = ((0, 3), (3, 9), (9, 24))
    scores, counts = zip(*(masked_mean(difference[..., p:q], target.feature[..., p:q] &
                                        slots[..., None, None]) for p, q in groups))
    groups_seen = torch.stack(counts) > 0
    payload = torch.stack(scores).sum() / groups_seen.sum().clamp_min(1)
    payload_n = (target.feature & slots[..., None, None]).flatten(1).any(-1).sum()
    return time + payload + .05 * spacing, {
        "time": time.detach(), "spacing": spacing.detach(), "payload": payload.detach(),
        "time_n": time_n, "spacing_n": spacing_n, "payload_n": payload_n,
    }


def linear_path(target, frame_mask, generator=None):
    t = torch.rand(len(target), generator=generator, device=target.device)
    noise = torch.randn(target.shape, generator=generator, device=target.device)
    x = torch.lerp(noise, target.float(), t[:, None, None])
    return x.masked_fill(~frame_mask[..., None], 0), t


def seeded_noise(clip_ids, seed, stage, sample_index, frame_mask, channels):
    """Generate only true-frame noise so batch layout cannot change a clip's draw."""
    device = frame_mask.device
    x = torch.zeros((*frame_mask.shape, channels), device=device)
    for i, name in enumerate(clip_ids):
        rng = torch.Generator(device=device).manual_seed(stable_seed(seed, stage, sample_index, name))
        x[i, frame_mask[i]] = torch.randn(int(frame_mask[i].sum()), channels, generator=rng,
                                          device=device)
    return x


@torch.no_grad()
def sample_cfm(model, x, condition, frame_mask, nfe):
    if type(nfe) is not int or nfe < 1:
        raise ValueError("NFE must be a positive integer")
    mask = frame_mask[..., None]
    x = x.float().masked_fill(~mask, 0)
    for i in range(nfe):
        tau = x.new_full((len(x),), i / nfe)
        clean = model(x, tau, condition).float().masked_fill(~mask, 0)
        if i == nfe - 1:
            return clean
        x = (x + (clean - x) / (nfe - i)).masked_fill(~mask, 0)


def _mse(pred, target, valid, weight):
    # Sanitize missing/padded inputs before taking squares to keep backward finite.
    delta = pred.float().masked_fill(~valid, 0) - target.float().masked_fill(~valid, 0)
    return delta.square() * weight


def time_weight(tau, cfg):
    return ((1 - tau.float()).clamp_min(.05).square().reciprocal()[:, None, None]
            if cfg["endpoint_time_weight"] == "velocity" else 1.)


def chordal(r, q):
    return (r.float() - q.float()).square().sum((-1, -2)) / 8


def _total(terms, cfg):
    value = sum(cfg[name] * term for name, (term, _) in terms.items())
    report = {name: v.detach() for name, (v, _) in terms.items()}
    report.update({f"{name}_n": n for name, (_, n) in terms.items()})
    return value, report


def body_objective(pred, norm_target, grid, stats, tau, cfg):
    obs = grid.observation & grid.mask[..., None]
    mask = obs.repeat_interleave(6, -1)
    mse = masked_mean(_mse(pred, norm_target, mask, time_weight(tau, cfg)), mask)
    raw = stats.denormalize(pred.float().masked_fill(~mask, 0)).unflatten(-1, (12, 6))
    rotations, fallback = rot6d_to_matrix(raw, return_fallback=True)
    rotation = masked_mean(chordal(rotations, grid.rotations), obs)
    value, terms = _total({"body_endpoint": mse, "body_chordal": rotation}, cfg)
    terms.update(gs_fallback=(fallback & obs).sum(),
                 gs_fallback_joints=obs.sum(), grid_fallback=pred.new_tensor(grid.fallback_count))
    return value, terms


def _contact_objective(pred, gt, clips, scale):
    total, count = [], []
    for i, clip in enumerate(clips):
        a, b = clip["contact_pairs"].to(pred.tips.device).unbind(-1)
        t = len(clip["timestamps_s"])
        p, q = (pose.tips[i, :t].flatten(-3, -2) for pose in (pred, gt))
        joint_valid = (gt.tip_mask[i, :t] & pred.tip_mask[i, :t]).flatten(-2)
        valid = (clip["contact_valid"].to(p.device) & clip["contact"].to(p.device)
                 & joint_valid[:, a] & joint_valid[:, b])
        diff = (p[:, a] - p[:, b]).norm(dim=-1) - (q[:, a] - q[:, b]).norm(dim=-1)
        error, valid_count = masked_mean(smooth(diff[None] / (.02 * scale)), valid[None])
        total.append(error)
        count.append(valid_count)
    n = torch.stack(count).sum()
    return torch.stack(total).sum() / n.clamp_min(1), n


def fine_objective(pred, endpoint, observation, batch, body, condition_obs, stats,
                   rig, tau, cfg, fine_cfg):
    """Use the same physical decode used by inference for every geometric term."""
    weights = time_weight(tau, cfg)
    err = _mse(pred, endpoint, observation, weights)
    hand_error = err[..., :192].unflatten(-1, (2, 16, 6))
    hand_obs = observation[..., :192].unflatten(-1, (2, 16, 6))
    wrist, wrist_n = masked_mean(hand_error[..., 0, :], hand_obs[..., 0, :])
    fingers, fingers_n = masked_mean(hand_error[..., 1:, :], hand_obs[..., 1:, :])
    hand_n = hand_obs.flatten(1).any(-1).sum()
    terms = {
        "hand_endpoint": (.5 * (wrist + fingers), hand_n),
        "correction": masked_mean(err[..., 192:], observation[..., 192:]),
    }
    endpoint_detail = {"wrist_endpoint": wrist.detach(), "wrist_endpoint_n": wrist_n,
                       "finger_endpoint": fingers.detach(), "finger_endpoint_n": fingers_n}
    geometry = ("fingertip_fk", "wrist_fk", "palm_chordal", "velocity_fk", "contact")
    if not any(cfg[k] for k in geometry):
        zero = pred.float().masked_fill(~batch["frame_mask"][..., None], 0).sum() * 0
        terms.update({k: (zero, zero.detach()) for k in geometry})
        loss, report = _total(terms, cfg)
        return loss, dict(report, **endpoint_detail)
    with torch.autocast(device_type=pred.device.type, enabled=False):
        seen = torch.cat((batch["body_obs"], batch["hands_obs"].flatten(-2)), -1)
        cond_seen = torch.cat((condition_obs, batch["hands_obs"].flatten(-2)), -1)
        physical = stats.denormalize(pred.float().masked_fill(~batch["frame_mask"][..., None], 0))
        prediction = decode_pose(body, physical, rig, fine_cfg["correction_max_rad"],
                                 fine_cfg["correction_enabled"], cond_seen)
        gt = fk(merge_local(rot6d_to_matrix(batch["body_rot6d"]),
                            rot6d_to_matrix(batch["hands_rot6d"])), rig, seen)
        tip_mask = prediction.tip_mask & gt.tip_mask
        wrist_mask = prediction.hand_mask & gt.hand_mask
        terms["fingertip_fk"] = masked_mean(smooth((prediction.local_tips - gt.local_tips) / .02),
                                           tip_mask[..., None])
        terms["wrist_fk"] = masked_mean(smooth((prediction.wrist - gt.wrist) / .05),
                                        wrist_mask[..., None])
        terms["palm_chordal"] = masked_mean(chordal(prediction.palm, gt.palm), wrist_mask)
        dt = batch["timestamps_s"].float().diff(dim=1).clamp_min(1e-6)
        velocity = (prediction.local_tips - gt.local_tips).diff(dim=1) / (
            .25 * dt[..., None, None, None]
        )
        terms["velocity_fk"] = masked_mean(smooth(velocity),
                                          (tip_mask[:, 1:] & tip_mask[:, :-1])[..., None])
        if cfg["contact"]:
            scaled = _contact_objective(prediction, gt, batch["clips"], rig.shoulder_width)
            terms["contact"] = scaled
        else:
            zero = pred.float().masked_fill(~batch["frame_mask"][..., None], 0).sum() * 0
            terms["contact"] = zero, zero.detach()
    loss, report = _total(terms, cfg)
    return loss, dict(report, **endpoint_detail)
