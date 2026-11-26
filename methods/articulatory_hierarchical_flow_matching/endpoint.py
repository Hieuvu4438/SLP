"""Physical wrist and finger supervision for the estimated coarse pose."""

import torch
from torch import Tensor
from torch.nn import functional as F

from slp.geometry import aa_to_quat


def quaternion_angle(a: Tensor, b: Tensor):
    # Chord lengths to the nearer and farther representatives of the target on S³.
    # atan2 avoids the poorly conditioned derivative of acos near identical rotations.
    minus = torch.linalg.vector_norm(a - b, dim=-1)
    plus = torch.linalg.vector_norm(a + b, dim=-1)
    return 4 * torch.atan2(torch.minimum(minus, plus), torch.maximum(minus, plus))


def rotation_loss(predicted, target, lengths, rig, samples):
    samples = min(samples, predicted.shape[1])
    grid = 2 * torch.arange(samples, device=predicted.device) + 1
    frames = grid[None] * lengths[:, None] // (2 * samples)
    rows = torch.arange(len(lengths), device=predicted.device)[:, None]

    def physical(poses):
        value = poses[rows, frames].float() * rig.std + rig.mean
        # SOKE's stored upper-body part starts at SMPL-X joint 12.
        wrists = value[..., 24:30] + rig.pose_mean[60:66]
        fingers = value[..., 30:120] + rig.pose_mean[75:165]
        return aa_to_quat(torch.cat((wrists, fingers), -1).unflatten(-1, (-1, 3)))

    angle = quaternion_angle(physical(predicted), physical(target))
    error = F.smooth_l1_loss(angle, torch.zeros_like(angle), beta=0.2, reduction="none")
    return 0.5 * (error[..., :2].mean() + error[..., 2:].mean())


def jitter_coarse(coarse, noise, projector, strength):
    delta = projector.coarse(noise)
    rms = (delta.square().sum((1, 2)) / projector.coarse_rank).sqrt()
    return coarse + strength * delta / rms.clamp_min(1e-8)[:, None, None]


FK_CHANNELS = ("coarse_palm", "coarse_face", "coarse_contact", "coarse_velocity")
COARSE_LOSSES = ("coarse_fk", "coarse_rotation", *FK_CHANNELS)


def extra_fk_losses(predicted, target, lengths, rig, samples, channels):
    """Optional physical channels on the same frame grid as the projected coarse target."""
    samples = min(samples, predicted.shape[1])
    grid = 2 * torch.arange(samples, device=predicted.device) + 1
    velocity = "coarse_velocity" in channels
    usable = (lengths - int(velocity)).clamp_min(1)
    frames = grid[None] * usable[:, None] // (2 * samples)
    if velocity:
        following = torch.minimum(frames + 1, lengths[:, None] - 1)
        frames = torch.cat((frames, following), dim=1)
    rows = torch.arange(len(lengths), device=predicted.device)[:, None]

    def descriptor(poses):
        joints, tips, nose = rig.landmarks(poses[rows, frames])
        wrists = joints[:, 20:22]
        out = {}
        if "coarse_palm" in channels:
            across = joints[:, [25, 40]] - joints[:, [31, 46]]
            along = joints[:, [28, 43]] - wrists
            normal = torch.linalg.cross(across, along)
            out["normal"] = F.normalize(normal, dim=-1, eps=1e-7)
            out["normal_valid"] = normal.norm(dim=-1) > 1e-7
        if "coarse_face" in channels:
            out["face"] = (tips - nose[:, None, None]).norm(dim=-1) * 100
        if "coarse_contact" in channels:
            out["opposite"] = (tips[:, 0, :, None] - tips[:, 1, None, :]).norm(dim=-1) * 100
        if velocity:
            relative = torch.cat((wrists - joints[:, 16:18],
                                  (tips - wrists[:, :, None]).flatten(1, 2)), dim=1)
            a, b = relative.reshape(len(lengths), 2, samples, -1, 3).unbind(1)
            out["speed"] = (b - a) * 100
        return out

    gen = descriptor(predicted)
    with torch.no_grad():
        ref = descriptor(target)
    result = {}
    if "coarse_palm" in channels:
        angle_proxy = 0.5 * (gen["normal"] - ref["normal"]).square().sum(-1)
        valid = ref["normal_valid"]
        result["coarse_palm"] = (angle_proxy * valid).sum() / valid.sum().clamp_min(1)
    if "coarse_face" in channels:
        result["coarse_face"] = F.smooth_l1_loss(gen["face"], ref["face"], beta=2.)
    if "coarse_contact" in channels:
        weight = (4. - ref["opposite"]).sigmoid()
        err = F.smooth_l1_loss(gen["opposite"], ref["opposite"], beta=1., reduction="none")
        result["coarse_contact"] = (err * weight).sum() / weight.sum().clamp_min(1.)
    if velocity:
        error = F.smooth_l1_loss(gen["speed"], ref["speed"], beta=.25, reduction="none")
        wrist = error[..., :2, :].mean((1, 2, 3))
        fingers = error[..., 2:, :].mean((1, 2, 3))
        valid = lengths > 1
        result["coarse_velocity"] = ((wrist + fingers) * valid).sum() / valid.sum().clamp_min(1)
    return result
