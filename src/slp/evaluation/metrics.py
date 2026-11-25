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
    value, length, _ = _dtw(cost, trace=False)
    return value, length


def dtw_path(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the minimum-cost DTW path, with the same tie breaks as dtw_cost."""
    _, length, parents = _dtw(cost, trace=True)
    gen, ref = np.empty(length, dtype=int), np.empty(length, dtype=int)
    i, j = parents.shape
    for k in range(length - 1, -1, -1):
        gen[k], ref[k] = i - 1, j - 1
        step = parents[i - 1, j - 1]
        i -= step != 2
        j -= step != 1
    return gen, ref


def _dtw(cost: np.ndarray, trace: bool):
    cost = np.asarray(cost, dtype=np.float64)
    r, c = cost.shape
    if r == 0 or c == 0:
        raise ValueError("DTW inputs must contain at least one frame")
    acc = np.full((r + 1, c + 1), np.inf)
    ln = np.zeros((r + 1, c + 1))
    acc[0, 0] = 0.0
    parents = np.empty((r, c), dtype=np.uint8) if trace else None
    for d in range(2, r + c + 1):
        i = np.arange(max(1, d - c), min(r, d - 1) + 1)
        j = d - i
        cand = np.stack([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        lens = np.stack([ln[i - 1, j - 1], ln[i - 1, j], ln[i, j - 1]])
        best = cand.argmin(0)
        acc[i, j] = cost[i - 1, j - 1] + cand[best, np.arange(len(i))]
        ln[i, j] = lens[best, np.arange(len(i))] + 1
        if parents is not None:
            parents[i - 1, j - 1] = best
    return float(acc[r, c]), int(ln[r, c]), parents


def articulatory_geometry_scores(pred: np.ndarray, target: np.ndarray,
                                 train_std: np.ndarray) -> dict[str, float]:
    """Shared-path and normalized-progress errors on the 96-D signing descriptor."""
    from scipy.spatial.distance import cdist

    pred, target, train_std = map(np.asarray, (pred, target, train_std))
    if any(x.ndim != 2 or x.shape[1] != 96 or len(x) == 0 for x in (pred, target)):
        raise ValueError("Expected non-empty 96-D frame descriptors")
    if train_std.shape != (96,) or np.any(train_std <= 0):
        raise ValueError("Expected positive train scales for the 96-D descriptor")
    groups = (slice(0, 6), slice(6, 18), slice(18, 48))
    cost = np.mean([cdist(pred[:, g] / train_std[g], target[:, g] / train_std[g])
                    / np.sqrt(len(train_std[g])) for g in groups], axis=0)
    pi, ti = dtw_path(cost)
    time = np.rint(np.linspace(0, len(pred) - 1, len(target))).astype(int)

    def errors(a, b):
        origin, delta = (a - b)[:, :3], (a - b)[:, 3:6] / 2
        wrists = np.stack((origin - delta, origin + delta), axis=1)
        na, nb = a[:, [9, 10, 11, 15, 16, 17]], b[:, [9, 10, 11, 15, 16, 17]]
        na, nb = na.reshape(-1, 2, 3), nb.reshape(-1, 2, 3)
        norm = np.linalg.norm(na, axis=-1) * np.linalg.norm(nb, axis=-1)
        cosine = (na * nb).sum(-1) / np.maximum(norm, 1e-8)
        return {
            "wrist_cm": float(100 * np.linalg.norm(wrists, axis=-1).mean()),
            "finger_cm": float(100 * np.linalg.norm((a - b)[:, 18:48].reshape(-1, 3), axis=-1).mean()),
            "palm_deg": float(np.degrees(np.arccos(cosine.clip(-1, 1))).mean()),
            "bimanual_cm": float(100 * np.abs(a[:, 48:73] - b[:, 48:73]).mean()),
            "face_mae": float((np.abs(a[:, 83:] - b[:, 83:]) / train_std[83:]).mean()),
        }

    return {**{f"dtw_{k}": v for k, v in errors(pred[pi], target[ti]).items()},
            **{f"progress_{k}": v for k, v in errors(pred[time], target).items()}}



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



# ---------------------------------------------------------------------------
# Representation diagnostics for generative latent spaces (methods/morae_slp, MoRAE Sec. 4 / C).
# Analysis helpers, not benchmark metrics: they take feature matrices, not predictions/targets.
# ---------------------------------------------------------------------------


def latent_spectrum(z, floor: float = 1e-3) -> dict:
    """PCA spectrum of vectors z [N, d] (centred, float64 covariance, no eigenvalue floor).

    Returns eigenvalues (descending), kappa = lambda_1 / lambda_d and the number of directions
    below ``floor * lambda_1`` (MoRAE Table S8 protocol)."""
    x = np.asarray(z, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2:
        raise ValueError(f"need [N, d] vectors with N >= 2, got {x.shape}")
    lam = np.linalg.eigvalsh(np.cov(x, rowvar=False).reshape(x.shape[1], x.shape[1]))[::-1]
    kappa = float(lam[0] / lam[-1]) if lam[-1] > 0 else float("inf")
    return {"eigenvalues": lam, "kappa": kappa, "n_below": int((lam < floor * lam[0]).sum())}


def directional_loadings(z, y) -> tuple[np.ndarray, np.ndarray]:
    """MoRAE Eq. (S1): with q_i the centred PCA projections of z [N, d] (variance lambda_i),
    L_Y(i) = ||Cov(q_i, Y)||^2 / (lambda_i tr Cov(Y)) is the fraction of the variance of the
    target Y [N, k] explained by a linear readout from direction i alone. Returns
    (eigenvalues descending, loadings); the loadings sum to the R^2 of the full linear readout."""
    x = np.asarray(z, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or t.ndim != 2 or len(x) != len(t) or len(x) < 2:
        raise ValueError("need paired [N, d] and [N, k] arrays")
    x = x - x.mean(0)
    t = t - t.mean(0)
    lam, vec = np.linalg.eigh(np.cov(x, rowvar=False).reshape(x.shape[1], x.shape[1]))
    order = np.argsort(lam)[::-1]
    lam, vec = lam[order], vec[:, order]
    q = x @ vec
    cov = q.T @ t / (len(x) - 1)                     # [d, k] Cov(q_i, Y)
    total = np.trace(np.cov(t, rowvar=False).reshape(t.shape[1], t.shape[1]))
    load = (cov ** 2).sum(1) / (np.clip(lam, 1e-300, None) * max(total, 1e-300))
    return lam, load


def multilabel_average_precision(scores, labels) -> float:
    """Macro average precision over the labels that have at least one positive.

    scores, labels: [N, L]; AP of a label = mean precision at the rank of each positive."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    if s.shape != y.shape or s.ndim != 2:
        raise ValueError("scores and labels must both be [N, L]")
    aps = []
    for j in range(y.shape[1]):
        pos = y[:, j]
        if not pos.any():
            continue
        order = np.argsort(-s[:, j], kind="stable")
        hits = pos[order]
        precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
        aps.append(precision[hits].mean())
    if not aps:
        raise ValueError("no label has a positive example")
    return float(np.mean(aps))

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
