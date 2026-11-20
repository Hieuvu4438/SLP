"""Manual-signing geometry, timed events and paired inference for multi-rate flows."""

from collections import defaultdict
import math

import numpy as np
import torch


def match_events(reference, prediction, tolerance_s):
    """Lexicographic one-to-one matching: most hits first, then minimal time error."""
    by_type = sorted({e["kind"] for e in (*reference, *prediction)})
    matched = []
    for kind in by_type:
        a = sorted(float(e["time_s"]) for e in reference if e["kind"] == kind)
        b = sorted(float(e["time_s"]) for e in prediction if e["kind"] == kind)
        rows = [[(0, 0., ()) for _ in range(len(b) + 1)]]
        for x in a:
            prev = rows[-1]
            row = [(0, 0., ())]
            for j, y in enumerate(b):
                options = [prev[j + 1], row[-1]]
                distance = abs(x - y)
                if distance <= tolerance_s + 1e-7:
                    n, cost, links = prev[j]
                    options.append((n + 1, cost + distance, links + (distance,)))
                row.append(min(options, key=lambda v: (-v[0], v[1])))
            rows.append(row)
        matched.extend(rows[-1][-1][2])
    tp, fp, fn = len(matched), len(prediction) - len(matched), len(reference) - len(matched)
    return dict(event_tp=tp, event_fp=fp, event_fn=fn, timing_abs_sum_s=sum(matched),
                **_rates(tp, fp, fn, "event"),
                timing_mae_ms=1000 * sum(matched) / tp if tp else None)


def _rates(tp, fp, fn, prefix):
    return {f"{prefix}_precision": tp / (tp + fp) if tp + fp else None,
            f"{prefix}_recall": tp / (tp + fn) if tp + fn else None,
            f"{prefix}_f1": (2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else None}


def hysteresis(distance, valid, enter, min_frames=2):
    """Pair-specific contact flags and confirmed boundaries at the first run frame."""
    if enter <= 0 or min_frames < 1:
        raise ValueError("Invalid contact detector settings")
    flags = torch.zeros_like(valid)
    events = []
    for pair in range(distance.shape[1]):
        state, run, have_state, known = False, 0, False, False
        for i in range(len(distance)):
            if not bool(valid[i, pair]):
                state, run, have_state, known = False, 0, False, False
                continue
            v = float(distance[i, pair])
            flags[i, pair] = state
            if not have_state:
                # Start a visible run in the non-contact state. A boundary whose
                # start coincides with first observation has no verified onset.
                first, have_state = i, True
            crossing = v > enter * 1.5 if state else v < enter
            known |= not crossing
            run = run + 1 if crossing else 0
            if run == min_frames:
                start = i - (min_frames - 1)
                state = not state
                flags[start:i + 1, pair] = state
                if known and start > first:
                    events.append((start, pair, "onset" if state else "offset"))
                known, run = True, 0
    return flags, events


def contact_counts(prediction, reference, valid):
    tp = int((prediction & reference & valid).sum())
    fp = int((prediction & ~reference & valid).sum())
    fn = int((~prediction & reference & valid).sum())
    return dict(contact_tp=tp, contact_fp=fp, contact_fn=fn, contact_frames=int(valid.sum()),
                **_rates(tp, fp, fn, "contact"))


def _mean(error, mask):
    values = error[mask]
    return float(values.mean()) if values.numel() else None


def geometry_metrics(candidate, reference, scene_width, windows, anchor_exclusion):
    """Compute spatial errors after explicit caller alignment and keep frame statistics."""
    tip_valid = candidate.tip_mask & reference.tip_mask
    wrist_valid = candidate.hand_mask & reference.hand_mask
    tip = (candidate.local_tips - reference.local_tips).norm(dim=-1) * (1000 * scene_width)
    wrist = (candidate.wrist - reference.wrist).norm(dim=-1) * (1000 * scene_width)
    palm_delta = candidate.palm.transpose(-1, -2) @ reference.palm
    cosine = (palm_delta.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    palm = torch.rad2deg(cosine.clamp(-1, 1).acos())
    body = (candidate.joints[..., :12, :] - reference.joints[..., :12, :]).norm(dim=-1) * 1000
    body_valid = candidate.joint_mask[..., :12] & reference.joint_mask[..., :12]
    sums = tip.masked_fill(~tip_valid, 0).sum(dim=(-1, -2))
    count = tip_valid.sum(dim=(-1, -2))
    window_mask = windows[:, None, None]
    return dict(
        event_tip_mm=_mean(tip, tip_valid & window_mask),
        outside_tip_mm=_mean(tip, tip_valid & ~window_mask),
        event_tip_excluding_anchors_mm=_mean(
            tip, tip_valid & window_mask & ~anchor_exclusion[:, None, None]),
        tip_mm=_mean(tip, tip_valid), wrist_mm=_mean(wrist, wrist_valid),
        palm_degrees=_mean(palm, wrist_valid), body_mm=_mean(body, body_valid),
        event_tip_count=int((tip_valid & window_mask).sum()),
        tip_count=int(tip_valid.sum()), wrist_count=int(wrist_valid.sum()),
        frame_tip_sum=sums.tolist(), frame_tip_count=count.tolist(),
    )


SCORES = (
    "event_tip_mm", "outside_tip_mm", "event_tip_excluding_anchors_mm", "tip_mm", "wrist_mm",
    "palm_degrees", "body_mm", "duration_relative", "event_precision", "event_recall",
    "event_f1", "timing_mae_ms", "contact_f1", "correction_cap_rate",
)


def _macro_by_clip(rows, field):
    grouped = defaultdict(list)
    for item in rows:
        if item.get(field) is not None:
            grouped[item["clip_id"]].append(item[field])
    return {key: float(np.mean(v)) for key, v in grouped.items()}


def _clusters(rows):
    out = {}
    for row in rows:
        clip, video = row["clip_id"], row["origin_video_id"]
        if clip in out and out[clip] != video:
            raise ValueError("Inconsistent origin video for the same clip")
        out[clip] = video
    return out


def _bootstrap_indices(keys, clusters, rng, repeats):
    groups = defaultdict(list)
    for i, key in enumerate(keys):
        groups[clusters[key]].append(i)
    values = list(groups.values())
    for _ in range(repeats):
        sampled = rng.integers(len(values), size=len(values))
        yield [i for group in sampled for i in values[group]]


def summarize(rows, cfg, seed):
    clusters = _clusters(rows)

    def aggregate(items):
        scores = {}
        for field in SCORES:
            macro = _macro_by_clip(items, field)
            keys = sorted(macro)
            values = np.asarray([macro[key] for key in keys])
            rng = np.random.default_rng(seed)
            draws = [values[index].mean() for index in _bootstrap_indices(
                keys, clusters, rng, cfg["bootstrap_resamples"]
            )] if keys else []
            scores[field] = {"n": len(keys), "excluded": len({r["clip_id"] for r in items}) - len(keys),
                             "mean": float(values.mean()) if keys else None,
                             "ci95": np.quantile(draws, [.025, .975]).tolist() if draws else None}
            if field == "duration_relative":
                scores[field]["median"] = float(np.median(values)) if keys else None
        return scores

    slices = defaultdict(list)
    for row in rows:
        for name in row.get("slices", []):
            slices[name].append(row)
    totals = {k: sum(row.get(k, 0) for row in rows) for k in (
        "event_tp", "event_fp", "event_fn", "contact_tp", "contact_fp", "contact_fn",
        "contact_frames", "event_tip_count", "tip_count", "wrist_count")}
    return {"clips": len(clusters), "samples": len(rows), "scores": aggregate(rows), "counts": totals,
            "slices": {k: aggregate(v) for k, v in sorted(slices.items())},
            "event_label_sources": sorted({r.get("event_source", "unknown") for r in rows})}


def _exclusion_metrics(a, b):
    order = ("frame_tip_sum", "frame_tip_count", "event_frames", "oracle_anchor_frames")
    if not all(key in a and key in b for key in order):
        return None, None
    if a["event_frames"] != b["event_frames"]:
        raise ValueError("Event windows differ across paired runs")
    retained = sorted(set(a["event_frames"]) - {
        f for pos in (*a["oracle_anchor_frames"], *b["oracle_anchor_frames"])
        for f in (pos - 1, pos, pos + 1)
    })
    values = []
    for row in (a, b):
        n = sum(row["frame_tip_count"][f] for f in retained)
        values.append(sum(row["frame_tip_sum"][f] for f in retained) / n if n else None)
    return tuple(values)


def align_pairs(base, candidate):
    def index(rows):
        result = {(r["clip_id"], r.get("sample_index", 0)): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate clip/sample ID in predictions")
        return result

    aa, bb = index(base), index(candidate)
    cluster = _clusters([*base, *candidate])
    ins, outs = [], []
    for key in sorted(aa.keys() & bb.keys()):
        a, b = dict(aa[key]), dict(bb[key])
        for field in ("reference_sha256", "noise_seed", "target_length", "event_frames"):
            if a.get(field) != b.get(field):
                raise ValueError(f"Incompatible pair metadata for {field}")
        if not np.allclose(a["frame_tip_count"], b["frame_tip_count"], atol=0, rtol=0):
            raise ValueError("Paired spatial metric coverage differs")
        a["event_tip_excluding_anchors_mm"], b["event_tip_excluding_anchors_mm"] = (
            _exclusion_metrics(a, b))
        # Match at sample level before averaging samples within each clip.
        for field in SCORES:
            if a.get(field) is None or b.get(field) is None:
                a[field] = b[field] = None
        ins.append(a)
        outs.append(b)
    return ins, outs, cluster


def paired_effects(base, candidate, cfg, seed):
    ins, outs, cluster = align_pairs(base, candidate)
    summary = {}
    rng = np.random.default_rng(seed)
    for field in SCORES:
        a, b = (_macro_by_clip(rows, field) for rows in (ins, outs))
        keys = sorted(a.keys() & b.keys())
        if not keys:
            summary[field] = dict(n=0, excluded=len(cluster))
            continue
        x, y = (np.asarray([v[k] for k in keys]) for v in (a, b))
        if field in ("contact_f1", "event_precision", "event_recall", "event_f1"):
            unit = "rate_fraction"
        elif field == "duration_relative":
            unit = "median_absolute_fraction"
        else:
            unit = "relative_improvement"

        def effect(a, b):
            if unit == "rate_fraction":
                return float(b.mean() - a.mean())
            if unit == "median_absolute_fraction":
                return float(np.median(a) - np.median(b))
            return float((a.mean() - b.mean()) / max(a.mean(), cfg["gate_epsilon"]))

        draws = [effect(x[idx], y[idx]) for idx in _bootstrap_indices(
            keys, cluster, rng, cfg["bootstrap_resamples"]
        )]
        ranking = sorted(zip(keys, x - y, strict=True), key=lambda pair: (pair[1], pair[0]))
        summary[field] = {"n": len(keys), "excluded": len(cluster) - len(keys),
                          "baseline": float(x.mean()), "candidate": float(y.mean()),
                          "effect": effect(x, y), "unit": unit,
                          "ci95": np.quantile(draws, [.025, .975]).tolist(),
                          "largest_regressions": [k for k, _ in ranking[:10]],
                          "largest_improvements": [k for k, _ in reversed(ranking[-10:])]}
    return summary


def gate_decision(effects, gate, min_clips, text_only=None):
    primary = effects["event_tip_mm"]
    if primary["n"] < min_clips:
        return "HOLD", ["insufficient usable event clips"]
    if gate not in ("p1", "p2"):
        raise ValueError("Only P1 and P2 have predeclared gates")
    reasons = []
    for name, max_regression, min_n in (
        ("wrist_mm", .05, min_clips), ("outside_tip_mm", .03, min_clips), ("contact_f1", .02, 30)
    ):
        result = effects[name]
        if result["n"] < min_n:
            if name != "contact_f1":
                return "HOLD", [f"insufficient guardrail coverage for {name}"]
            continue
        if result["effect"] < -max_regression:
            reasons.append(f"{name} exceeded the regression limit")
    if reasons:
        return "STOP", reasons
    if primary["ci95"][1] <= 0:
        return "STOP", ["event fingertip error did not improve"]
    if primary["effect"] < (.05 if gate == "p1" else .03) or primary["ci95"][0] <= 0:
        return "HOLD", ["primary improvement or paired interval below GO threshold"]
    if gate == "p2":
        if text_only is None:
            return "HOLD", ["text-only evidence is required"]
        duration, spatial = (text_only[k] for k in ("duration_relative", "event_tip_mm"))
        timing, recall = (text_only[k] for k in ("timing_mae_ms", "event_recall"))
        if (min(duration["n"], spatial["n"]) < min_clips or
                not (timing["n"] or recall["n"])):
            return "HOLD", ["text-only metric coverage is insufficient"]
        if (spatial["effect"] < 0 or duration["effect"] < -.02 or
                max(timing.get("effect", -math.inf), recall.get("effect", -math.inf)) < 0):
            return "STOP", ["text-only duration, spatial or timing criterion failed"]
    return "GO", []


def hierarchical_effects(seeds, field, cfg, seed=17, interaction=False):
    """Bootstrap paired seed and video clusters, using the same video draw across seeds."""
    by_seed = []
    all_clusters = {}
    for arms in seeds:
        if len(arms) != (4 if interaction else 2):
            raise ValueError("Unexpected number of arms in hierarchical comparison")
        ab, cd, cluster = align_pairs(arms[0], arms[1])
        pairs = [_macro_by_clip(rows, field) for rows in (ab, cd)]
        if interaction:
            a2, b2, other = align_pairs(arms[2], arms[3])
            for clip, video in other.items():
                if clip in cluster and cluster[clip] != video:
                    raise ValueError("Inconsistent origin video in interaction")
                cluster[clip] = video
            pairs.extend(_macro_by_clip(rows, field) for rows in (a2, b2))
        for clip, video in cluster.items():
            if clip in all_clusters and all_clusters[clip] != video:
                raise ValueError("Inconsistent origin video across seeds")
            all_clusters[clip] = video
        by_seed.append(pairs)
    if not by_seed:
        raise ValueError("No seed runs supplied")
    common = set.intersection(*(set(arm) for arms in by_seed for arm in arms))
    keys = sorted(common)
    if not keys:
        return {"n": 0, "excluded": len(all_clusters), "seeds": len(seeds)}
    values = np.asarray([[[v[k] for k in keys] for v in arms] for arms in by_seed])

    def measure(x):
        means = x.mean(axis=-1)
        if interaction:
            return (means[:, 0] - means[:, 1]) - (means[:, 2] - means[:, 3])
        if field.endswith(("_f1", "_precision", "_recall")):
            return means[:, 1] - means[:, 0]
        if field == "duration_relative":
            return np.median(x[:, 0], axis=-1) - np.median(x[:, 1], axis=-1)
        return (means[:, 0] - means[:, 1]) / np.maximum(means[:, 0], cfg["gate_epsilon"])

    rng = np.random.default_rng(seed)
    draws = []
    for video_draw in _bootstrap_indices(keys, all_clusters, rng, cfg["bootstrap_resamples"]):
        seed_draw = rng.integers(len(seeds), size=len(seeds))
        draws.append(float(measure(values[seed_draw][..., video_draw]).mean()))
    estimates = measure(values)
    return {"n": len(keys), "excluded": len(all_clusters) - len(keys), "seeds": len(seeds),
            "per_seed": estimates.tolist(), "mean": float(estimates.mean()),
            "std_across_seeds": float(estimates.std(ddof=1)) if len(estimates) > 1 else None,
            "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "unit": "difference_of_raw_improvements" if interaction else (
                "rate_fraction" if field.endswith(("_f1", "_precision", "_recall"))
                or field == "duration_relative" else "relative_improvement"),
            "bootstrap": "paired_seed_and_origin_video"}
