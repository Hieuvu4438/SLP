"""Metric implementations. ONE implementation per metric, shared by all
methods. Each metric: fn(predictions, targets) -> float.
Placeholder metrics below - replace/extend for the actual domain, and add a
unit test for each new metric in tests/test_metrics.py."""

from __future__ import annotations

from typing import Callable

import numpy as np


def accuracy(preds, targets) -> float:
    preds, targets = np.asarray(preds), np.asarray(targets)
    return float((preds == targets).mean())


def macro_f1(preds, targets) -> float:
    preds, targets = np.asarray(preds), np.asarray(targets)
    f1s = []
    for c in np.unique(targets):
        tp = np.sum((preds == c) & (targets == c))
        fp = np.sum((preds == c) & (targets != c))
        fn = np.sum((preds != c) & (targets == c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


# ---------------------------------------------------------------------------
# Sign Language Production metrics (SOKE / MaDiS protocol + calibrated variants)
#
# Predictions/targets are lists of per-sequence joint arrays [T, J, 3] (already
# forward-kinematised and restricted to one body part). Lengths may differ
# between prediction and target. Joint 0 of each array is the alignment root
# (pelvis for body, wrist for hands), as in SOKE/MaDiS.
# ---------------------------------------------------------------------------


def _pairwise_jpe(gen: np.ndarray, ref: np.ndarray, align_idx: int = 0) -> np.ndarray:
    """[Tg, Tr] mean joint position error after root-translation alignment."""
    gen = np.asarray(gen, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    g = gen - gen[:, align_idx:align_idx + 1]
    r = ref - ref[:, align_idx:align_idx + 1]
    diff = g[:, None] - r[None]
    return np.sqrt((diff ** 2).sum(-1)).mean(-1)


def dtw_cost(cost: np.ndarray) -> tuple[float, int]:
    """Exact DTW on a cost matrix. Returns (accumulated cost, warping-path length)."""
    cost = np.asarray(cost, dtype=np.float64)
    r, c = cost.shape
    if r == 0 or c == 0:
        raise ValueError("DTW inputs must contain at least one frame")
    acc = np.full((r + 1, c + 1), np.inf)
    ln = np.zeros((r + 1, c + 1))
    acc[0, 0] = 0.0
    for d in range(2, r + c + 1):
        i = np.arange(max(1, d - c), min(r, d - 1) + 1)
        j = d - i
        cand = np.stack([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        lens = np.stack([ln[i - 1, j - 1], ln[i - 1, j], ln[i, j - 1]])
        best = cand.argmin(0)
        acc[i, j] = cost[i - 1, j - 1] + cand[best, np.arange(len(i))]
        ln[i, j] = lens[best, np.arange(len(i))] + 1
    return float(acc[r, c]), int(ln[r, c])


def dtw_jpe(preds, targets) -> float:
    """SOKE/MaDiS DTW-JPE: accumulated DTW cost, averaged over sequences (paper-compatible)."""
    return float(np.mean([dtw_cost(_pairwise_jpe(p, t))[0] for p, t in zip(preds, targets)]))


def dtw_jpe_norm(preds, targets) -> float:
    """Calibrated DTW-JPE: accumulated cost divided by warping-path length (per-step error, m)."""
    vals = []
    for p, t in zip(preds, targets):
        a, n = dtw_cost(_pairwise_jpe(p, t))
        vals.append(a / n)
    return float(np.mean(vals))


def _ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def sibleu_unigram(preds, targets) -> float:
    """MaDiS released-code SiBLEU: corpus unigram clipped precision x brevity penalty
    over sign-token IDs (order-insensitive). preds/targets: lists of token-id lists."""
    from collections import Counter
    overlap = hyp = ref = 0
    for p, t in zip(preds, targets):
        overlap += sum((Counter(p) & Counter(t)).values())
        hyp += len(p)
        ref += len(t)
    if hyp == 0:
        return 0.0
    bp = 1.0 if hyp > ref else float(np.exp(1 - ref / hyp))
    return 100.0 * bp * overlap / hyp


def sibleu4(preds, targets) -> float:
    """Order-sensitive corpus BLEU-4 over sign-token IDs (uniform weights, no smoothing)."""
    from collections import Counter
    match = np.zeros(4)
    total = np.zeros(4)
    hyp = ref = 0
    for p, t in zip(preds, targets):
        hyp += len(p)
        ref += len(t)
        for n in range(1, 5):
            pc, tc = Counter(_ngrams(p, n)), Counter(_ngrams(t, n))
            match[n - 1] += sum((pc & tc).values())
            total[n - 1] += max(len(p) - n + 1, 0)
    if hyp == 0 or np.any(match == 0):
        return 0.0
    bp = 1.0 if hyp > ref else float(np.exp(1 - ref / hyp))
    return 100.0 * bp * float(np.exp(np.mean(np.log(match / total))))


# ---------------------------------------------------------------------------
# SOKE text-to-sign protocol on SOKE 133-D feature sequences (normalised, SOKE stats).
#
# Mirrors SOKE mGPT/metrics/t2m.py (+ dtw.py, human_models.rigid_align):
#   DTW-JPE    : per frame pair, translate the generated skeleton so its root joint
#                (pelvis for body/all, wrist for hands) coincides with the reference,
#                mean L2 over the part's joints; DTW accumulated cost per sequence.
#   DTW-PA-JPE : same, but with a similarity (Procrustes, scale+rotation+translation)
#                alignment of the generated skeleton to the reference first.
#   body  : alignment on all SMPL-X joints, error on SOKE's 12 upper-body joints
#   hand  : 21 regressed joints per hand; reported "hand" = mean(left, right)
#   all   : all 144 SMPL-X output joints (body, hands, face keypoints)
# Values are averaged over sequences (paper unit: accumulated metres).
# ---------------------------------------------------------------------------


def _procrustes_pairwise(gen, ref):
    """Align every generated frame to every reference frame (SOKE rigid_align).

    gen [Tg,N,3], ref [Tr,N,3] torch tensors -> aligned generated points [Tg,Tr,N,3]."""
    import torch

    n = gen.shape[1]
    mg, mr = gen.mean(1, keepdim=True), ref.mean(1, keepdim=True)
    a, b = gen - mg, ref - mr
    H = torch.einsum("inc,jnd->ijcd", a, b) / n
    U, S, Vh = torch.linalg.svd(H)
    R = Vh.transpose(-1, -2) @ U.transpose(-1, -2)
    neg = torch.linalg.det(R) < 0
    if neg.any():
        S = S.clone()
        Vh = Vh.clone()
        S[..., -1] = torch.where(neg, -S[..., -1], S[..., -1])
        Vh[..., 2, :] = torch.where(neg[..., None], -Vh[..., 2, :], Vh[..., 2, :])
        R = Vh.transpose(-1, -2) @ U.transpose(-1, -2)
    var = a.var(1, unbiased=False).sum(-1)  # [Tg]
    c = S.sum(-1) / var[:, None]  # [Tg,Tr]
    cR = c[..., None, None] * R
    t = mr[None, :, 0, :] - torch.einsum("ijcd,id->ijc", cR, mg[:, 0, :])
    return torch.einsum("ijcd,ind->ijnc", cR, a + mg) + t[:, :, None, :]


def pairwise_part_cost(gen, ref, wanted=None, align: str = "root", chunk: int = 32):
    """[Tg,Tr] SOKE frame-pair cost. align='root' (joint 0) or 'procrustes' (all joints)."""
    import torch

    gen = torch.as_tensor(gen, dtype=torch.float64)
    ref = torch.as_tensor(ref, dtype=torch.float64, device=gen.device)
    rows = []
    for s in range(0, gen.shape[0], chunk):
        g = gen[s:s + chunk]
        if align == "root":
            x = g[:, None] - g[:, None, :1] + ref[None, :, :1]
            x = x.expand(-1, ref.shape[0], -1, -1)
        elif align == "procrustes":
            x = _procrustes_pairwise(g, ref)
        else:
            raise ValueError(align)
        y = ref[None].expand_as(x)
        if wanted is not None:
            x, y = x[:, :, wanted], y[:, :, wanted]
        rows.append(torch.sqrt(((x - y) ** 2).sum(-1)).mean(-1))
    return torch.cat(rows, 0).cpu().numpy()


_SOKE_CACHE: dict = {}


def soke_protocol_scores(preds, targets) -> dict[str, np.ndarray]:
    """Per-sequence SOKE DTW scores for normalised 133-D preds/targets (cached per call set)."""
    key = (id(preds), id(targets))
    hit = _SOKE_CACHE.get(key)
    if hit is not None and hit[0] is preds and hit[1] is targets:
        return hit[2]
    from slp.evaluation.smplx_fk import UPPER_BODY_IDX, default_fk

    fk = default_fk()
    out = {f"{m}_{p}": [] for m in ("jpe", "pa") for p in ("all", "body", "lhand", "rhand")}
    out["pa_body_ua"] = []
    for p, t in zip(preds, targets):
        p, t = np.asarray(p), np.asarray(t)
        if p.ndim != 2 or p.shape[-1] != 133 or len(p) == 0:
            raise ValueError(f"SOKE protocol expects non-empty [T,133] sequences, got {p.shape}")
        gj, gl, gr = fk(p)
        rj, rl, rr = fk(t)
        for m, align in (("jpe", "root"), ("pa", "procrustes")):
            out[f"{m}_all"].append(dtw_cost(pairwise_part_cost(gj, rj, None, align))[0])
            out[f"{m}_body"].append(dtw_cost(pairwise_part_cost(gj, rj, UPPER_BODY_IDX, align))[0])
            out[f"{m}_lhand"].append(dtw_cost(pairwise_part_cost(gl, rl, None, align))[0])
            out[f"{m}_rhand"].append(dtw_cost(pairwise_part_cost(gr, rr, None, align))[0])
        # Variant: Procrustes fitted on the 12 upper-body joints themselves (SOKE's released code
        # fits it on all joints, which gives PA > JPE for body, unlike every published table).
        ub = list(UPPER_BODY_IDX)
        out["pa_body_ua"].append(dtw_cost(pairwise_part_cost(gj[:, ub], rj[:, ub], None, "procrustes"))[0])
    res = {k: np.asarray(v) for k, v in out.items()}
    _SOKE_CACHE.clear()
    _SOKE_CACHE[key] = (preds, targets, res)
    return res


def _soke(metric: str, part: str) -> Callable:
    def fn(preds, targets) -> float:
        s = soke_protocol_scores(preds, targets)
        if part == "hand":
            return float(np.mean((s[f"{metric}_lhand"] + s[f"{metric}_rhand"]) / 2))
        return float(np.mean(s[f"{metric}_{part}"]))
    fn.__name__ = f"soke_dtw_{'pa_' if metric == 'pa' else ''}jpe_{part}"
    return fn


soke_dtw_jpe_all = _soke("jpe", "all")
soke_dtw_jpe_body = _soke("jpe", "body")
soke_dtw_jpe_hand = _soke("jpe", "hand")
soke_dtw_pa_jpe_all = _soke("pa", "all")
soke_dtw_pa_jpe_body = _soke("pa", "body")
soke_dtw_pa_jpe_hand = _soke("pa", "hand")
soke_dtw_pa_jpe_body_ua = _soke("pa", "body_ua")


# ---------------------------------------------------------------------------
# Calibration metrics for DTW-based SLP evaluation (docs/proposal7, M1-M2 + statistics)
#
# DTW-JPE rewards regression to the mean pose (shrinking a prediction towards the mean improves
# it), so every DTW comparison is reported with:
#   M2  kinematic ratios: predicted / reference low-pass speed and acceleration of the evaluated
#       points (body: 12 upper-body points relative to the pelvis; hands: 2 x 21 wrist-relative
#       points), after a Gaussian temporal low-pass that removes the fitting jitter of the GT;
#   M1  text-motion retrieval (R-precision in batches of 32, Frechet distance) on the embeddings
#       of an evaluator trained on the train split;
#   a matched-speed comparison against the "shrink-to-mean" frontier of a reference model, with
#       paired bootstrap confidence intervals over sequences.
# ---------------------------------------------------------------------------


def lowpass_kinematics(points, sigma: float = 2.0) -> tuple[float, float]:
    """Mean speed |dp| and acceleration |d^2 p| per frame of points [T,P,3] after a Gaussian
    temporal low-pass (std ``sigma`` frames, 'nearest' borders). Units of the input per frame."""
    from scipy.ndimage import gaussian_filter1d

    x = np.asarray(points, dtype=np.float64)
    if sigma > 0 and len(x) > 1:
        x = gaussian_filter1d(x, sigma, axis=0, mode="nearest")
    v = np.diff(x, axis=0)
    speed = float(np.linalg.norm(v, axis=-1).mean()) if len(v) else 0.0
    a = np.diff(v, axis=0)
    accel = float(np.linalg.norm(a, axis=-1).mean()) if len(a) else 0.0
    return speed, accel


_KIN_CACHE: dict = {}


def soke_kinematics(seqs, sigma: float = 2.0) -> dict[str, np.ndarray]:
    """Per-sequence low-pass speed / acceleration (metres per frame) of normalised 133-D sequences:
    {body,hand}_{speed,accel}. Body: UPPER_BODY_IDX relative to the pelvis; hand: both hands
    relative to their wrists (SOKE hand regressor). Cached for the last sequence list."""
    key = (id(seqs), float(sigma))
    hit = _KIN_CACHE.get(key)
    if hit is not None and hit[0] is seqs:
        return hit[1]
    from slp.evaluation.smplx_fk import UPPER_BODY_IDX, default_fk

    fk = default_fk()
    out = {k: [] for k in ("body_speed", "body_accel", "hand_speed", "hand_accel")}
    for x in seqs:
        j, l, r = (a.cpu().numpy() for a in fk(np.asarray(x)))
        body = j[:, UPPER_BODY_IDX] - j[:, :1]
        hand = np.concatenate([l - l[:, :1], r - r[:, :1]], 1)
        for part, pts in (("body", body), ("hand", hand)):
            sp, ac = lowpass_kinematics(pts, sigma)
            out[f"{part}_speed"].append(sp)
            out[f"{part}_accel"].append(ac)
    res = {k: np.asarray(v) for k, v in out.items()}
    if len(_KIN_CACHE) > 4:
        _KIN_CACHE.clear()
    _KIN_CACHE[key] = (seqs, res)
    return res


def _kin_ratio(stat: str) -> Callable:
    def fn(preds, targets) -> float:
        """Predicted / reference mean (M2); 1 = reference dynamics, < 1 = damped (mean regression)."""
        return float(soke_kinematics(preds)[stat].mean() / max(soke_kinematics(targets)[stat].mean(), 1e-12))
    fn.__name__ = f"soke_{stat.split('_')[1]}_ratio_{stat.split('_')[0]}"
    return fn


soke_speed_ratio_body = _kin_ratio("body_speed")
soke_speed_ratio_hand = _kin_ratio("hand_speed")
soke_accel_ratio_body = _kin_ratio("body_accel")
soke_accel_ratio_hand = _kin_ratio("hand_accel")


def r_precision(text_emb, motion_emb, top_k=(1, 2, 3), batch: int = 32, seed: int = 0) -> dict[str, float]:
    """Text-to-motion R-precision (T2M protocol): pairs are shuffled into batches of ``batch``;
    within a batch every text ranks the batch's motions by Euclidean distance between L2-normalised
    embeddings; R@k = fraction whose own motion is in the top k. The last incomplete batch is dropped."""
    t = np.asarray(text_emb, dtype=np.float64)
    m = np.asarray(motion_emb, dtype=np.float64)
    if t.shape != m.shape or len(t) < batch:
        raise ValueError(f"need paired embeddings with at least {batch} rows, got {t.shape} / {m.shape}")
    t = t / np.linalg.norm(t, axis=1, keepdims=True).clip(1e-12)
    m = m / np.linalg.norm(m, axis=1, keepdims=True).clip(1e-12)
    idx = np.random.default_rng(seed).permutation(len(t))
    hits = {k: 0 for k in top_k}
    n = 0
    for s in range(0, len(idx) - batch + 1, batch):
        b = idx[s:s + batch]
        d = np.linalg.norm(t[b][:, None] - m[b][None], axis=-1)
        rank = (d < np.diag(d)[:, None]).sum(1)  # motions strictly closer than the paired one
        for k in top_k:
            hits[k] += int((rank < k).sum())
        n += batch
    return {f"R@{k}": hits[k] / n for k in top_k}


def frechet_distance(feats_a, feats_b) -> float:
    """Frechet distance between Gaussians fitted to two embedding sets [N,D] (FID formula)."""
    from scipy import linalg

    a, b = np.asarray(feats_a, np.float64), np.asarray(feats_b, np.float64)
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(ca @ cb, disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(len(ca)) * 1e-6
        covmean = linalg.sqrtm((ca + off) @ (cb + off))
    covmean = covmean.real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(ca) + np.trace(cb) - 2 * np.trace(covmean))


def paired_bootstrap_ci(diff, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean of per-sequence paired differences and its percentile bootstrap (1 - alpha) interval."""
    d = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def frontier_effect(arm_scores, arm_speed: float, frontier_scores, frontier_speeds) -> tuple[np.ndarray, bool]:
    """Matched-speed effect of an arm against a reference "shrink-to-mean" frontier.

    frontier_scores [A,N]: per-sequence DTW scores of the reference shrunk by a_1 < ... < a_A;
    frontier_speeds [A]: their dataset speeds (decreasing). The frontier is interpolated linearly at
    the arm's speed (same weights for every sequence) and the per-sequence effect
    arm - frontier(arm_speed) is returned (< 0: better than mere shrinkage). An arm faster than the
    unshrunk reference is compared with the reference itself (conservative) and slower than the
    most-shrunk point with that point; the flag says whether such clamping happened."""
    F = np.asarray(frontier_scores, dtype=np.float64)
    v = np.asarray(frontier_speeds, dtype=np.float64)
    if F.ndim != 2 or F.shape[0] != len(v) or len(v) < 2:
        raise ValueError("frontier_scores must be [A,N] with A = len(frontier_speeds) >= 2")
    order = np.argsort(v)  # increasing speed
    v, F = v[order], F[order]
    clamped = not (v[0] <= arm_speed <= v[-1])
    s = float(np.clip(arm_speed, v[0], v[-1]))
    i = int(np.clip(np.searchsorted(v, s, side="right") - 1, 0, len(v) - 2))
    lam = 0.0 if v[i + 1] == v[i] else (s - v[i]) / (v[i + 1] - v[i])
    front = (1 - lam) * F[i] + lam * F[i + 1]
    return np.asarray(arm_scores, dtype=np.float64) - front, clamped


METRICS: dict[str, Callable] = {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "dtw_jpe": dtw_jpe,
    "dtw_jpe_norm": dtw_jpe_norm,
    "sibleu_unigram": sibleu_unigram,
    "sibleu4": sibleu4,
    "soke_dtw_jpe_all": soke_dtw_jpe_all,
    "soke_dtw_jpe_body": soke_dtw_jpe_body,
    "soke_dtw_jpe_hand": soke_dtw_jpe_hand,
    "soke_dtw_pa_jpe_all": soke_dtw_pa_jpe_all,
    "soke_dtw_pa_jpe_body": soke_dtw_pa_jpe_body,
    "soke_dtw_pa_jpe_hand": soke_dtw_pa_jpe_hand,
    "soke_dtw_pa_jpe_body_ua": soke_dtw_pa_jpe_body_ua,
    "soke_speed_ratio_body": soke_speed_ratio_body,
    "soke_speed_ratio_hand": soke_speed_ratio_hand,
    "soke_accel_ratio_body": soke_accel_ratio_body,
    "soke_accel_ratio_hand": soke_accel_ratio_hand,
}
