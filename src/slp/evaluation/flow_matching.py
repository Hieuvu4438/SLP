"""Scoring primitives for variable-length sign motion and paired corpus metrics."""

from collections import Counter
import math

import numpy as np

from slp.evaluation.metrics import pairwise_part_cost
from slp.evaluation.smplx_fk import UPPER_BODY_IDX


def dtw_path(cost):
    n, m = cost.shape
    if not n or not m or not np.isfinite(cost).all():
        raise ValueError("DTW expects a nonempty finite cost matrix")
    accumulated = np.full((n + 1, m + 1), np.inf)
    moves = np.zeros((n, m), dtype=np.int8)
    accumulated[0, 0] = 0
    for i in range(n):
        for j in range(m):
            candidates = (accumulated[i, j], accumulated[i, j + 1], accumulated[i + 1, j])
            move = min(range(3), key=candidates.__getitem__)
            accumulated[i + 1, j + 1] = cost[i, j] + candidates[move]
            moves[i, j] = move
    path = []
    i, j = n - 1, m - 1
    while i >= 0 and j >= 0:
        path.append((i, j))
        move = moves[i, j]
        i -= move != 2
        j -= move != 1
    return float(accumulated[n, m]), np.array(path[::-1])


def geometry(pred, reference, body_pa_alignment="all"):
    result = {}
    body_path = None
    for label, mode in (("jpe", "root"), ("pa_jpe", "procrustes")):
        paths = []
        scores = []
        for region, p, r in zip(("body", "left_hand", "right_hand"), pred, reference):
            wanted = UPPER_BODY_IDX if region == "body" else None
            if label == "pa_jpe" and region == "body" and body_pa_alignment == "upper_body":
                p, r = p[:, wanted], r[:, wanted]
                wanted = None
            cost = pairwise_part_cost(p, r, wanted=wanted, align=mode)
            score, path = dtw_path(cost)
            result[f"{region}_dtw_{label}"] = score
            result[f"{region}_dtw_{label}_path_length"] = len(path)
            result[f"{region}_dtw_{label}_per_aligned_frame_mm"] = 1000 * score / len(path)
            scores.append(score)
            paths.append(path)
        result[f"hand_dtw_{label}"] = (scores[1] + scores[2]) / 2
        if mode == "root":
            body_path = paths[0]
    return result, body_path


def bleu_counts(hypothesis, reference):
    correct, total = [], []
    for n in range(1, 5):

        def counts(tokens):
            return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

        h, r = counts(hypothesis), counts(reference)
        correct.append(sum((h & r).values()))
        total.append(sum(h.values()))
    return [*correct, *total, len(hypothesis), len(reference)]


def bleu_from_counts(counts):
    if min(counts[:8]) == 0 or counts[-2] == 0:
        return 0.0
    precision = [c / t for c, t in zip(counts[:4], counts[4:8])]
    brevity = min(0.0, 1 - counts[-1] / counts[-2])
    return 100 * math.exp(brevity + sum(math.log(p) for p in precision) / 4)


def corpus_bleu4(rows):
    counts = [
        r["bt_counts"]
        if "bt_counts" in r
        else bleu_counts(r["bt_hypothesis_tokens"], r["bt_reference_tokens"])
        for r in rows
    ]
    return bleu_from_counts(np.sum(counts, axis=0))


def paired_bootstrap(baseline, challenger, repeats, seed, confidence=0.95):
    """Pair on seed/clip, then sample clusters jointly across those seeds."""

    def keyed(rows):
        keyed_rows = {(r["seed"], r["clip_id"]): r for r in rows}
        if len(keyed_rows) != len(rows):
            raise ValueError("Duplicate seed/clip")
        return keyed_rows

    base, cand = keyed(baseline), keyed(challenger)
    if base.keys() != cand.keys() or not base:
        raise ValueError("Empty or different paired cohort")
    seeds = sorted({s for s, _ in base})
    clips = sorted({c for _, c in base})
    groups = {}
    for clip in clips:
        for s in seeds:
            if (s, clip) not in base:
                raise ValueError("Each seed must cover the same clip cohort")
            a, b = base[s, clip], cand[s, clip]
            if a.get("planner_hash") != b.get("planner_hash"):
                raise ValueError(f"Mismatched length planner for seed {s}")
            for key in (
                "video_id",
                "cohort",
                "protocol",
                "corpus",
                "pose_fit_id",
                "fps_reference",
                "T_reference",
                "nfe",
            ):
                if a.get(key) != b.get(key) or a.get(key) != base[seeds[0], clip].get(key):
                    raise ValueError(f"Paired protocol mismatch in {key}: {clip}")
        row = base[seeds[0], clip]
        cluster = row.get("video_id") or row.get("text_hash") or clip
        groups.setdefault(cluster, []).append(clip)
    indices = {clip: i for i, clip in enumerate(clips)}
    clustered = [np.array([indices[clip] for clip in group]) for group in groups.values()]
    rng = np.random.default_rng(seed)
    numeric = (
        "hand_dtw_pa_jpe",
        "hand_dtw_jpe",
        "body_dtw_jpe",
        "body_dtw_pa_jpe",
        "wrist_torso_error",
        "duration_mae_seconds",
    )
    metrics = {}
    for k in numeric:
        a = np.array([[base[s, c][k] for c in clips] for s in seeds])
        b = np.array([[cand[s, c][k] for c in clips] for s in seeds])
        values = a - b
        metrics[k] = lambda draw, v=values: v[:, draw].mean(axis=1)
    if all("bt_counts" in x for x in [*baseline, *challenger]):
        a = np.array([[base[s, c]["bt_counts"] for c in clips] for s in seeds])
        b = np.array([[cand[s, c]["bt_counts"] for c in clips] for s in seeds])

        def effect(draw):
            return np.array(
                [
                    bleu_from_counts(k) - bleu_from_counts(j)
                    for k, j in zip(b[:, draw].sum(1), a[:, draw].sum(1))
                ]
            )

        metrics["corpus_bt_bleu4"] = effect
    samples = {k: [] for k in metrics}
    for _ in range(repeats):
        draw = np.concatenate(
            [clustered[i] for i in rng.integers(len(clustered), size=len(clustered))]
        )
        for k, f in metrics.items():
            samples[k].append(float(f(draw).mean()))
    alpha = 1 - confidence
    out = {}
    for k, f in metrics.items():
        points = f(np.arange(len(clips)))
        out[k] = {
            "per_seed": dict(zip(seeds, map(float, points))),
            "mean": float(points.mean()),
            "seed_std": float(points.std(ddof=1)) if len(seeds) > 1 else None,
            "ci": np.quantile(samples[k], [alpha / 2, 1 - alpha / 2]).tolist(),
            "scope": "paired clustered clips, conditional on training seeds",
        }
    return out
