"""Evaluation orchestration. Metric definitions live in slp.evaluation.multirate."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from slp.evaluation.multirate import (
    contact_counts, gate_decision, geometry_metrics, hysteresis, match_events, paired_effects,
    summarize,
)

from .anchors import clip_pose, normalized_signals, pose_signals
from .data import collate_clips, file_hash, read_jsonl, write_json, write_jsonl, atomic_bytes
from .geometry import fk, interpolate_rotations

KINEMATIC_TYPES = ("shape", "turn")


def _peaks(scores, threshold, gap):
    indices = sorted((i for i in range(len(scores)) if scores[i] >= threshold and scores[i] > 0),
                     key=lambda i: (-float(scores[i]), i))
    accepted = []
    for i in indices:
        if all(abs(i-j) >= gap for j in accepted):
            accepted.append(i)
    return sorted(accepted)


def kinetic_events(pose, clock, scales, cfg, thresholds=None):
    signal, visible = pose_signals(pose, cfg["anchors"]["turn_min_speed_shoulder_widths"])
    z = signal.new_zeros(len(signal), 1)
    scores = normalized_signals(torch.cat((signal, z), -1),
                                torch.cat((visible, z.bool()), -1), scales)
    # Evaluation uses independent event typing and the entire visible sequence.
    # No input anchor positions, training NMS or fixed anchor budget enter here.
    thresholds = thresholds or {k: cfg["eval"]["event_threshold"] for k in KINEMATIC_TYPES}
    return [{"kind": kind, "time_s": float(clock[n])}
            for i, kind in enumerate(KINEMATIC_TYPES)
            for n in _peaks(scores[:, i], thresholds[kind], gap=2)]


def _contact_distances(pose, pairs, h):
    p = pose.tips.flatten(-3, -2)
    m = pose.tip_mask.flatten(-2)
    a, b = pairs.unbind(-1)
    return (p[:, a] - p[:, b]).norm(dim=-1) / h, m[:, a] & m[:, b]


def _contact_event(n, pair, edge, pairs, clock):
    a, b = pairs[pair].tolist()
    return {"kind": f"contact:{a}-{b}:{edge}", "time_s": float(clock[n])}


def _contact_evaluate(clip, p, q, time, rig, enter):
    if enter is None or not clip["row"].get("contact_source"):
        return [], [], {}
    pairs = clip["contact_pairs"].to(time.device)
    if not len(pairs):
        return [], [], {}
    ct = clip["timestamps_s"].to(time.device)
    labels = clip["contact"].to(time.device)
    known = clip["contact_valid"].to(time.device)
    d, p_valid = _contact_distances(p, pairs, rig.shoulder_width)
    _, q_valid = _contact_distances(q, pairs, rig.shoulder_width)
    known = known & q_valid
    count = min(len(time), len(ct))
    # Canonical and inferred clocks are regular at the same validated FPS.
    if not torch.allclose(time[:count], ct[:count], atol=1e-6, rtol=0):
        raise ValueError("Unaligned physical contact clocks")
    p_valid = p_valid.clone()
    p_valid[:count] &= known[:count]
    pred, events = hysteresis(d, p_valid, enter)
    p_events = [_contact_event(n, j, edge, pairs, time) for n, j, edge in events]
    change = (labels[1:] != labels[:-1]) & known[1:] & known[:-1]
    q_events = [_contact_event(n+1, j, "onset" if labels[n+1, j] else "offset", pairs, ct)
                for n, j in change.nonzero().tolist()]
    scores = contact_counts(pred[:count], labels[:count], p_valid[:count])
    return p_events, q_events, scores


def _normalize_clock(clock):
    return clock / clock[-1] if len(clock) > 1 else torch.zeros_like(clock)


def _spatial_pose(pose, pred_clock, ref_clock, rig):
    if len(pred_clock) == len(ref_clock) and torch.allclose(pred_clock, ref_clock, atol=1e-6):
        return pose
    matrices = interpolate_rotations(pose.local, _normalize_clock(pred_clock),
                                    _normalize_clock(ref_clock))
    return fk(matrices, rig)


@torch.no_grad()
def evaluate_clip(cascade, clip, protocol, seed, sample_index, calibration=None):
    cfg = cascade.cfg
    row = clip["row"]
    result = cascade.generate([row["text"]], [row["clip_id"]], protocol,
                              None if protocol == "text_only" else collate_clips([clip]),
                              seed, sample_index)
    pred = replace(result["pose"], **{k: v[0] for k, v in vars(result["pose"]).items()})
    device = result["times"].device
    truth = clip_pose(clip, cascade.rig)
    truth = replace(truth, **{k: v.to(device) for k, v in vars(truth).items()})
    time = result["times"][0]
    ref_time = clip["timestamps_s"].to(device)
    thresholds = calibration["thresholds"] if calibration else None
    enter = calibration["contact_enter"] if calibration else None
    pe = kinetic_events(pred, time, cascade.stats.event_scales, cfg, thresholds)
    annotated = row.get("events", [])
    if row.get("event_source"):
        reference = list(annotated)
        supported = set(row["event_types"])
        pe = [e for e in pe if e["kind"] in supported]
    else:
        reference = kinetic_events(truth, ref_time, cascade.stats.event_scales, cfg, thresholds)
    event_source = row.get("event_source") or "fixed_reference_kinematics_proxy"
    pc, qc, contact = _contact_evaluate(clip, pred, truth, time, cascade.rig, enter)
    # Annotated contact contributes spatial windows even without an audited timing detector.
    timing_reference = [e for e in reference if not e["kind"].startswith("contact:")] + qc
    known_windows = {(e["kind"], e["time_s"]): e for e in (*reference, *qc)}
    reference = list(known_windows.values())
    windows = torch.zeros(len(ref_time), device=device, dtype=torch.bool)
    radius = cfg["eval"]["event_window_radius_frames"] / cfg["data"]["fps"]
    for e in reference:
        windows |= (ref_time - e["time_s"]).abs() <= radius + 1e-7
    exclusion = torch.zeros_like(windows)
    oracle_frames = []
    oracle_anchors = protocol in ("oracle", "predicted_body_oracle_anchors")
    a = result["anchors"]
    if oracle_anchors:
        oracle_frames = (a.time[0, a.slot[0]] * (len(ref_time)-1)).round().long().tolist()
        for n in oracle_frames:
            exclusion[max(0, n-1):n+2] = True
    spatial = _spatial_pose(pred, time, ref_time, cascade.rig)
    geometry = geometry_metrics(spatial, truth, cascade.rig.shoulder_width, windows, exclusion)
    temporal = match_events(timing_reference, [*pe, *pc],
                            cfg["eval"]["timing_match_tolerance_frames"] / cfg["data"]["fps"])
    movement = truth.local_tips.diff(dim=0).norm(dim=-1)
    hand_shapes = ((movement > .001) & truth.tip_mask[1:] & truth.tip_mask[:-1]).any(dim=(0, 2))
    wrist_shift = truth.wrist.diff(dim=0).norm(dim=-1)
    hand_path = ((wrist_shift > .001) & truth.hand_mask[1:] & truth.hand_mask[:-1]).any(dim=0)
    both = (hand_shapes | hand_path).all()
    quality = row.get("source_quality", {}).get("fit_quality")
    slices = ["two_hands" if both else "other_hand_activity", "long" if len(ref_time) >= 128
              else "under_128_frames", *sorted({e["kind"].split(":")[0] for e in reference})]
    if quality is not None:
        slices.append(f"fit_quality_{quality}")
    if contact.get("contact_frames", 0):
        slices.append("contact_labelled")
    return {"clip_id": row["clip_id"], "origin_video_id": row["origin_video_id"],
            "sample_index": sample_index, "noise_seed": seed, "reference_sha256": row["sha256"],
            "protocol": protocol, "event_source": event_source,
            "event_frames": windows.nonzero().flatten().tolist(), "oracle_anchor_frames": oracle_frames,
            "anchor_slots": int(a.slot.sum()), "anchor_observed_features": int(a.feature.sum()),
            "anchor_observed_hands": int(a.feature.any(-1).sum()), "slices": slices,
            "predicted_length": len(time), "target_length": len(ref_time),
            "duration_relative": abs(len(time)-len(ref_time)) / len(ref_time),
            "length_clamped": bool(result["length_clamped"][0]),
            "text_truncated": bool(result["truncated"][0]), "nfe": result["nfe"],
            "latency_ms": result["latency_ms"], **geometry, **temporal, **contact}


@torch.no_grad()
def fit_evaluator(dataset, rig, scales, cfg, contact_audit=None):
    """Lock detector thresholds on dev_tune. Contact requires a reviewed audit artifact."""
    if not len(dataset) or any(row["split"] != "dev_tune" for row in dataset.rows):
        raise ValueError("Detector calibration requires dev_tune")
    contact_allowed = bool(contact_audit and contact_audit.get("approved") is True and
                           contact_audit.get("reviewed_observations", 0) >= 30)
    t_grid = (.15, .25, .35, .5, .65, .8)
    c_grid = (.01, .02, .03)
    kinetic = {x: {k: [0, 0, 0] for k in KINEMATIC_TYPES} for x in t_grid}
    contact = {x: [0, 0, 0] for x in c_grid}
    annotations = {k: 0 for k in KINEMATIC_TYPES}
    contact_frames = 0
    for c in dataset:
        pose, clock = clip_pose(c, rig), c["timestamps_s"]
        if c["row"].get("event_source"):
            events = c["row"].get("events", [])
            for k in KINEMATIC_TYPES:
                annotations[k] += sum(e["kind"] == k for e in events)
            for entry in t_grid:
                fitted = kinetic_events(pose, clock, scales, cfg, {k: entry for k in KINEMATIC_TYPES})
                for k in set(c["row"]["event_types"]) & set(KINEMATIC_TYPES):
                    result = match_events([e for e in events if e["kind"] == k], [
                        e for e in fitted if e["kind"] == k
                    ], cfg["eval"]["timing_match_tolerance_frames"] / cfg["data"]["fps"])
                    for j, key in enumerate(("event_tp", "event_fp", "event_fn")):
                        kinetic[entry][k][j] += result[key]
        pairs = c["contact_pairs"]
        if contact_allowed and len(pairs) and c["row"].get("contact_source"):
            distance, visible = _contact_distances(pose, pairs, rig.shoulder_width)
            valid = c["contact_valid"] & visible
            contact_frames += int(valid.sum())
            for x in c_grid:
                v, _ = hysteresis(distance, valid, x)
                outcome = contact_counts(v, c["contact"], valid)
                for j, key in enumerate(("contact_tp", "contact_fp", "contact_fn")):
                    contact[x][j] += outcome[key]

    def ranking(count, threshold):
        tp, fp, fn = count
        return (2 * tp / max(2 * tp + fp + fn, 1), -fp, threshold)

    thresholds = {k: (max(t_grid, key=lambda x: ranking(kinetic[x][k], x))
                      if annotations[k] else cfg["eval"]["event_threshold"]) for k in KINEMATIC_TYPES}
    return {"thresholds": thresholds, "annotations": annotations,
            "contact_enter": max(c_grid, key=lambda x: ranking(contact[x], x))
            if contact_frames and any(tp + fn for tp, _, fn in contact.values()) else None,
            "contact_valid_frames": contact_frames, "contact_audit": contact_audit,
            "manifest_sha256": file_hash(dataset.manifest), "rig_sha256": rig.sha256,
            "event_scales": scales, "fps": cfg["data"]["fps"], "split": "dev_tune",
            "event_window_radius_frames": cfg["eval"]["event_window_radius_frames"],
            "timing_match_tolerance_frames": cfg["eval"]["timing_match_tolerance_frames"],
            "turn_min_speed": cfg["anchors"]["turn_min_speed_shoulder_widths"]}


def run_evaluation(cascade, dataset, protocol, out, seed, calibration=None, cohort_hash=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not len(dataset) or len({row["split"] for row in dataset.rows}) != 1:
        raise ValueError("Evaluation requires one nonempty split")
    cfg = cascade.cfg
    metadata = {"schema": 1, "split": dataset.rows[0]["split"], "protocol": protocol,
                "config": cfg, "rig_sha256": cascade.rig.sha256,
                "manifest_sha256": file_hash(dataset.manifest), "seed": seed,
                "cohort_sha256": cohort_hash, "calibration": calibration,
                "checkpoints": {k: v.sha256 for k, v in cascade.frozen.items()},
                "confidence_sha256": cascade.confidence_hash,
                "event_scales": cascade.stats.event_scales,
                "space_alignment": "normalized_time_when_duration_differs",
                "timing_alignment": "unwarped_seconds", "scope": "manual_signing_head_neck_rotations",
                "units": {"positions": "mm", "rotation_error": "degrees", "timing": "ms",
                          "rates": "fraction"}}
    if calibration is not None:
        expected = {"rig_sha256": cascade.rig.sha256, "manifest_sha256": metadata["manifest_sha256"],
                    "event_scales": cascade.stats.event_scales, "fps": cfg["data"]["fps"],
                    "event_window_radius_frames": cfg["eval"]["event_window_radius_frames"],
                    "timing_match_tolerance_frames": cfg["eval"]["timing_match_tolerance_frames"],
                    "turn_min_speed": cfg["anchors"]["turn_min_speed_shoulder_widths"]}
        if any(calibration[k] != v for k, v in expected.items()):
            raise ValueError("Evaluation detector calibration is incompatible with this run")
    write_json(out / "protocol.json", metadata)
    rows, failures = [], []
    for clip in dataset:
        for sample_index in range(cfg["sample"]["samples_per_text"]):
            try:
                rows.append(evaluate_clip(cascade, clip, protocol, seed, sample_index, calibration))
            except Exception as error:
                failures.append({"clip_id": clip["row"]["clip_id"], "sample_index": sample_index,
                                 "type": type(error).__name__, "message": str(error)})
    write_jsonl(out / "per_clip.jsonl", rows)
    write_jsonl(out / "failures.jsonl", failures)
    if not rows:
        raise RuntimeError(f"All evaluation samples failed. See {out / 'failures.jsonl'}")
    summary = summarize(rows, cfg["eval"], seed)
    summary["failures"] = len(failures)
    summary["length_clamped"] = sum(r["length_clamped"] for r in rows)
    summary["text_truncated"] = sum(r["text_truncated"] for r in rows)
    summary["nfe"] = rows[0]["nfe"]
    summary["anchors"] = {k: sum(r[k] for r in rows) for k in (
        "anchor_slots", "anchor_observed_features", "anchor_observed_hands")}
    write_json(out / "summary.json", summary)
    latency = {}
    for name in rows[0]["latency_ms"]:
        values = [r["latency_ms"][name] for r in rows]
        latency[name] = dict(zip(("p50", "p95"), np.percentile(values, (50, 95)).tolist()))
    write_json(out / "latency.json", {"unit": "ms", "measurements": len(rows),
                                     "mode": "per_clip_sampling", "stages": latency})
    lines = [f"# {protocol} on {metadata['split']}", "", "Manual signing and head/neck rotations.",
             "", f"{summary['clips']} clips, {len(rows)} samples, {len(failures)} failed.", "",
             "| Metric | Mean | Clips |", "| --- | ---: | ---: |"]
    lines.extend(f"| {k} | {v['mean']} | {v['n']} |" for k, v in summary["scores"].items())
    atomic_bytes(out / "report.md", ("\n".join(lines) + "\n").encode())
    return summary


def _comparison_pair(baseline, candidate):
    import json

    paths = [Path(p) for p in (baseline, candidate)]
    ab = [json.loads((p / "protocol.json").read_bytes()) for p in paths]
    for field in ("rig_sha256", "manifest_sha256", "cohort_sha256", "split", "seed", "protocol",
                  "calibration", "event_scales"):
        if ab[0][field] != ab[1][field]:
            raise ValueError(f"Paired runs differ in {field}")
    for field in ("text", "data", "sample", "loss", "eval"):
        # Data paths may differ, so compare the physical/schema fields and provenance hash.
        if field == "data":
            keys = ("fps", "min_frames", "max_frames", "pilot_language", "normalize_std_floor")
            values = [{k: a["config"][field][k] for k in keys} for a in ab]
        else:
            values = [a["config"][field] for a in ab]
        if values[0] != values[1]:
            raise ValueError(f"Paired runs differ in config section {field}")
    for stage in ("length", "body"):
        if stage == "body" and ab[0]["config"]["body"]["stride"] != ab[1]["config"]["body"]["stride"]:
            continue
        if ab[0]["checkpoints"].get(stage) != ab[1]["checkpoints"].get(stage):
            raise ValueError(f"Paired runs must share the {stage} checkpoint")
    failures = [read_jsonl(p / "failures.jsonl") for p in paths]
    rows = [read_jsonl(p / "per_clip.jsonl") for p in paths]
    effects = paired_effects(*rows, ab[0]["config"]["eval"], ab[0]["seed"])
    return ab, effects, failures


def compare(baseline, candidate, gate, out, baseline_text=None, candidate_text=None):
    ab, effects, failures = _comparison_pair(baseline, candidate)
    meta = ab[0]
    p2 = None
    if gate is not None:
        expected = "oracle" if gate == "p1" else "predicted_gt_length"
        if meta["protocol"] != expected or meta["split"] != "dev_gate":
            raise ValueError(f"{gate} gate requires {expected} on dev_gate")
        if (any(row["config"]["experiment"]["pilot"] != gate for row in ab) or
                [row["config"]["experiment"]["variant"] for row in ab] != ["B", "D"]):
            raise ValueError(f"{gate} gate compares the predeclared B and D variants")
        if any(row["config"]["body"]["stride"] != 4 for row in ab):
            raise ValueError(f"{gate} gate requires stride 4 for both variants")
    if gate == "p2" and baseline_text and candidate_text:
        text_meta, p2, text_failures = _comparison_pair(baseline_text, candidate_text)
        for a, b in zip(ab, text_meta):
            matching_paths = (a["config"] == b["config"] and a["split"] == b["split"] and
                              a["manifest_sha256"] == b["manifest_sha256"] and
                              a["cohort_sha256"] == b["cohort_sha256"] and a["seed"] == b["seed"] and
                              a["calibration"] == b["calibration"])
            if (not matching_paths or b["protocol"] != "text_only" or
                    any(b["checkpoints"].get(k) != v for k, v in a["checkpoints"].items())):
                raise ValueError("P2 text-only runs must match the corresponding GT-length runs")
        failures += text_failures
    if any(failures):
        status, reasons = "HOLD", ["one or more paired samples failed"]
    elif gate:
        status, reasons = gate_decision(effects, gate, meta["config"]["eval"]["min_usable_gate_clips"],
                                        text_only=p2)
    else:
        status, reasons = "exploratory", []
    result = {"status": status, "reasons": reasons, "gate": gate, "paired": effects,
              "text_only_paired": p2, "failures": [len(x) for x in failures],
              "protocol": meta["protocol"], "seed": meta["seed"], "alignment": meta[
                  "space_alignment"]}
    out = Path(out)
    write_json(out / "comparison.json", result)
    lines = [f"# {gate or 'Paired comparison'}", "", f"Decision **{status}**."] + [
        f"- {note}" for note in reasons
    ]
    atomic_bytes(out / "report.md", ("\n".join(lines) + "\n").encode())
    return result
