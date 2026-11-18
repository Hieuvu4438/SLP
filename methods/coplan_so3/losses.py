"""Auxiliary labels are confined to reconstruction; core KFM is endpoint-independent."""

import torch
import torch.nn.functional as F

from .geometry import so3
from .geometry.physical_time_features import orders


def valid_mean(value, mask):
    dims = tuple(range(1, value.ndim))
    expand = mask.reshape(*mask.shape, *([1] * (value.ndim - 2)))
    count = mask.sum(1).clamp_min(1) * (value[0, 0].numel())
    return ((value * expand).sum(dims) / count).mean()


def reconstruction(features, prediction, target, mask, times):
    r, e = prediction
    r1, e1 = target
    loss = valid_mean(so3.distance(r, r1).square(), mask) + valid_mean(
        F.smooth_l1_loss(e, e1, reduction="none"), mask
    )
    # Labeled physical velocities are matched; motion is not penalized toward rest.
    a, b = features.raw(r, e), features.raw(r1, e1)
    for name in a:
        for i, ((x, valid), (y, _)) in enumerate(
            zip(orders(a[name], times, mask), orders(b[name], times, mask))
        ):
            if i == 2 or x.shape[1] == 0:
                continue
            stat = features.stats[f"{name}_{i}"]
            if not stat.ready:
                raise RuntimeError("Reconstruction requires fixed training feature scales")
            rms = (stat.mean.square() + stat.m2 / stat.count).clamp_min(1e-8).sqrt().to(x)
            loss = loss + (1.0 if i == 0 else 0.1) * valid_mean(
                F.smooth_l1_loss(x / rms, y / rms, reduction="none"), valid
            )
    return loss


def contrastive(z, text, language, texts, temperature=0.07, positives=None, exclude=None):
    z, text = F.normalize(z, dim=-1), F.normalize(text, dim=-1)
    logits = z @ text.T / temperature
    same_language = language[:, None] == language[None]
    if positives is None:
        positives = (
            torch.tensor(
                [[x == y for y in texts] for x in texts], dtype=torch.bool, device=z.device
            )
            & same_language
        )
    if (
        positives.shape != logits.shape
        or not positives.diag().all()
        or (positives & ~same_language).any()
    ):
        raise ValueError(
            "Confirmed positives must include paired rows within the same sign language"
        )
    eligible = same_language if exclude is None else same_language & ~exclude
    eligible = eligible | positives

    def term(scores, pos, allowed):
        return (
            torch.logsumexp(scores.masked_fill(~allowed, -torch.inf), -1)
            - torch.logsumexp(scores.masked_fill(~pos, -torch.inf), -1)
        ).mean()

    return 0.5 * (term(logits, positives, eligible) + term(logits.T, positives.T, eligible.T))
