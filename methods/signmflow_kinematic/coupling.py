"""Optional exact-condition, whole-clip OT coupling."""

from collections import defaultdict

import torch


@torch.no_grad()
def pair_sources(source, expression, target, mask, texts, languages, representation):
    """Permute the source bank within exact (language,text,length) strata.

    The source multiset is preserved separately for every observed condition.
    Singleton conditions keep their independent draw. This routine is for
    training only. Inference samples from the same conditional source prior
    without access to a target.
    """
    strata = defaultdict(list)
    lengths = mask.sum(1).tolist()
    for i, (lang, text, length) in enumerate(zip(languages, texts, lengths, strict=True)):
        strata[lang, text, length].append(i)
    assignment = torch.arange(len(source), device=source.device)
    matched = 0
    for (_, _, length), ids in strata.items():
        if len(ids) < 2:
            continue
        from scipy.optimize import linear_sum_assignment

        xa, xb = source[ids, :length], target[ids, :length]
        if representation == "quat":
            cost = torch.stack([(1 - (xb * a).sum(-1).clamp(-1, 1).square()).mean((-1, -2)) for a in xa])
        else:
            cost = torch.cdist(xa.flatten(1), xb.flatten(1)).square() / xa[0].numel()
        rows, cols = linear_sum_assignment(cost.cpu().numpy())
        indices = torch.tensor(ids, device=source.device)
        assignment[indices[cols.tolist()]] = indices[rows.tolist()]
        matched += len(ids)
    return source[assignment], expression[assignment], matched
