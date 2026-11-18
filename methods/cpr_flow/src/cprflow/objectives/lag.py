"""Finite-overlap Pearson distributions. Positive offset means delayed second signal."""

import torch

PAIRS = ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3))


def correlations(a, b, max_offset=15, variance_floor=1e-4):
    result = []
    for k in range(-max_offset, max_offset + 1):
        aa = a[: len(a) - k] if k >= 0 else a[-k:]
        bb = b[k:] if k >= 0 else b[: len(b) + k]
        aa = aa - aa.mean()
        bb = bb - bb.mean()
        covariance = (aa * bb).mean()
        denom = (
            aa.square().mean().clamp_min(variance_floor)
            * bb.square().mean().clamp_min(variance_floor)
        ).sqrt()
        result.append(covariance / denom)
    return torch.stack(result)


def lag_loss_reference(
    pred, target, window=32, hop=16, max_offset=15, temperature=0.1, variance_floor=1e-4
):
    if window <= max_offset + 1:
        raise ValueError("Lag window too short")
    terms = []
    for start in range(0, len(target) - window + 1, hop):
        tw = target[start : start + window]
        pw = pred[start : start + window]
        for a, b in PAIRS:
            if (
                tw[:, a].var(unbiased=False) <= variance_floor
                or tw[:, b].var(unbiased=False) <= variance_floor
            ):
                continue
            tq = torch.softmax(
                correlations(tw[:, a], tw[:, b], max_offset, variance_floor) / temperature, 0
            ).detach()
            pq = torch.softmax(
                correlations(pw[:, a], pw[:, b], max_offset, variance_floor) / temperature, 0
            )
            terms.append((tq * (tq.clamp_min(1e-8).log() - pq.clamp_min(1e-8).log())).sum())
    return (torch.stack(terms).mean() if terms else pred.sum() * 0), len(terms)


def correlation_windows(a, b, max_offset=15, variance_floor=1e-4):
    # [windows,pairs,samples] -> [windows,pairs,offsets].
    width = a.shape[-1]
    positions = torch.arange(width, device=a.device)[None]
    offsets = torch.arange(-max_offset, max_offset + 1, device=a.device)[:, None]
    shifted = positions + offsets
    valid = (shifted >= 0) & (shifted < width)
    count = valid.sum(-1).clamp_min(1).to(a.dtype)
    bb = b[..., shifted.clamp(0, width - 1)]
    aa = a.unsqueeze(-2).expand_as(bb)
    mask = valid.to(a.dtype)
    ac = aa - (aa * mask).sum(-1, keepdim=True) / count[:, None]
    bc = bb - (bb * mask).sum(-1, keepdim=True) / count[:, None]
    covariance = (ac * bc * mask).sum(-1) / count
    av = (ac.square() * mask).sum(-1) / count
    bv = (bc.square() * mask).sum(-1) / count
    return covariance / (av.clamp_min(variance_floor) * bv.clamp_min(variance_floor)).sqrt()


def lag_loss(pred, target, window=32, hop=16, max_offset=15, temperature=0.1, variance_floor=1e-4):
    if window <= max_offset + 1:
        raise ValueError("Lag window too short")
    if len(target) < window:
        return pred.sum() * 0, 0
    pw = pred.float().unfold(0, window, hop)
    tw = target.float().unfold(0, window, hop)
    left = [a for a, b in PAIRS]
    right = [b for a, b in PAIRS]
    valid = (tw[:, left].var(-1, unbiased=False) > variance_floor) & (
        tw[:, right].var(-1, unbiased=False) > variance_floor
    )
    tc = correlation_windows(tw[:, left], tw[:, right], max_offset, variance_floor)
    pc = correlation_windows(pw[:, left], pw[:, right], max_offset, variance_floor)
    tq = torch.softmax(tc / temperature, -1).detach()
    pq = torch.softmax(pc / temperature, -1)
    kl = (tq * (tq.clamp_min(1e-8).log() - pq.clamp_min(1e-8).log())).sum(-1)
    count = valid.sum()
    return (kl * valid).sum() / count.clamp_min(1), int(count)
