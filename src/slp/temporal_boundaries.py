"""Movement-derived boundary evidence on a common temporal grid."""

import torch
from torch import Tensor


@torch.no_grad()
def boundary_evidence(features: Tensor, radii: tuple[int, ...] = (1, 2, 4)) -> Tensor:
    """RMS mean-change per part/scale, then robust within-sequence calibration.

    Input contains TRAIN-standardized codec means [L,4,D]. Cut j separates
    features[:j] and features[j:]. Both endpoints have zero boundary reward.
    A constant signal has evidence .5 at interior cuts, not false certainty.
    This score is a heuristic, not calibrated linguistic boundary probability.
    """
    if features.ndim != 3 or features.shape[1] != 4 or len(features) < 1:
        raise ValueError("expected a non-empty [L,4,D] sequence")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("non-finite teacher input")
    length = len(features)
    evidence = features.new_zeros(length + 1)
    if length == 1:
        return evidence
    prefix = torch.cat((features.new_zeros(1, *features.shape[1:]), features.cumsum(0)))
    cuts = torch.arange(1, length, device=features.device)
    changes = []
    for radius in radii:
        start = (cuts - radius).clamp_min(0)
        stop = (cuts + radius).clamp_max(length)
        left = (prefix[cuts] - prefix[start]) / (cuts - start)[:, None, None]
        right = (prefix[stop] - prefix[cuts]) / (stop - cuts)[:, None, None]
        changes.append((left - right).square().mean(-1).sqrt())
    raw = torch.stack(changes, -1).mean((-1, -2))
    median = raw.median()
    mad = (raw - median).abs().median().clamp_min(1e-4)
    evidence[1:length] = torch.sigmoid(((raw - median) / (1.4826 * mad)).clamp(-6, 6))
    return evidence
