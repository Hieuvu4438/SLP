"""Export a blind paired semantic test and aggregate independent judgments."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import numpy as np

from methods.articulatory_hierarchical_flow_matching.data import broadcast_id
from methods.articulatory_hierarchical_flow_matching.method import atomic_write
from methods.articulatory_hierarchical_flow_matching.semantic import terms
from methods.articulatory_hierarchical_flow_matching.subspace import checksum

KINDS = ("weekday", "number", "direction", "negation")


def write_json(path, value):
    atomic_write(path, lambda file: file.write(
        (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    ))


def reviewed_pairs(path, inputs, phase):
    body = json.loads(path.read_text())
    if not isinstance(body, dict) or body.get("phase") != phase or not isinstance(body.get("pairs"), list):
        raise ValueError(f"Expected reviewed pairs locked to the {phase} partition")
    by_id = {item["name"]: item for item in inputs}
    rows = {}
    for row in body["pairs"]:
        required = ("id", "clip", "a", "b", "kind", "reviewer", "reference")
        if (not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip()
                                            for k in required)
                or row["kind"] not in KINDS or row["reference"] not in ("a", "unknown")
                or row["clip"] not in by_id or row["id"] in rows or row.get("reviewed") is not True):
            raise ValueError("Invalid reviewed minimal pair or a clip outside the requested phase")
        a, b = terms(row["a"]), terms(row["b"])
        changed = [op for op in SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
                   if op[0] != "equal"]
        if (terms(by_id[row["clip"]]["text"]) != a or not a or not b or len(changed) != 1):
            raise ValueError(f"Expected one changed span in a source-matching pair: {row['id']}")
        if len(a) + len(b) > 2 and not any(
            tag == "equal" for tag, *_ in SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
        ):
            raise ValueError(f"A multi-word minimal pair must retain some context: {row['id']}")
        rows[row["id"]] = {**row, "broadcast": broadcast_id(by_id[row["clip"]])}
    if not rows or not any(row["reference"] == "a" for row in rows.values()):
        raise ValueError("A paired test needs a distinguishable reviewed reference")
    return rows


def export_pairs(model, manifest, phase, inputs, targets, checkpoints, source_lengths, output):
    rows = reviewed_pairs(manifest, inputs, phase)
    by_id = {item["name"]: (i, item) for i, item in enumerate(inputs)}
    indices = [by_id[row["clip"]][0] for row in rows.values()]
    original = [inputs[i] for i in indices]
    alternate = [{**item, "text": row["b"]}
                 for item, row in zip(original, rows.values(), strict=True)]
    lengths = [source_lengths[i] for i in indices]
    hidden = {}
    arrays = {}

    def add(label, checkpoint, pose_rows, expected):
        for pair, poses in zip(rows, pose_rows, strict=True):
            opaque = hashlib.sha256(
                f"{model.ctx.seed}:{manifest_hash}:{checkpoint}:{label}:{pair}".encode()
            ).hexdigest()[:24]
            row = rows[pair]
            flip = int(opaque[0], 16) % 2
            choices = [row["a"], row["b"]][::(-1 if flip else 1)]
            truth = row["reference"] if expected is None else expected
            answer = "unknown" if truth == "unknown" else (
                "right" if (truth == "b") ^ bool(flip) else "left"
            )
            hidden[opaque] = {"pair": pair, "checkpoint": checkpoint, "type": label,
                              "truth": answer, "choices": choices, "frames": len(poses)}
            arrays[opaque] = poses

    manifest_hash = checksum(manifest)
    add("gt", None, (targets[i] for i in indices), None)
    for checkpoint in checkpoints:
        model.load_checkpoint(checkpoint)
        digest = checksum(checkpoint)
        for component in ("coarse", "full"):
            for side, captions in (("a", original), ("b", alternate)):
                poses = model.generate(captions, lengths=lengths, component=component)
                add(f"{component}_{side}", digest, poses, side)
    # Send only the two blind files to a human panel. The key maps back to both model and side.
    output.mkdir(exist_ok=True, parents=True)
    sequences = output / "blind.npz"
    atomic_write(sequences, lambda file: np.savez_compressed(file, **dict(sorted(arrays.items()))))
    write_json(output / "blind.json", {
        "phase": phase, "samples": [{"id": key, "choices": hidden[key]["choices"]}
                                    for key in sorted(hidden)],
        "poses_sha256": checksum(sequences),
    })
    write_json(output / "key.json", {"phase": phase, "manifest_sha256": manifest_hash,
                                    "blind_sha256": checksum(output / "blind.json"),
                                    "seed": model.ctx.seed, "pairs": rows, "samples": hidden,
                                    "lengths": dict(zip(rows, lengths, strict=True))})


def score_pairs(key, judgments):
    if judgments.get("blind_sha256") != key["blind_sha256"]:
        raise ValueError("Judgments are for a different blind bundle")
    scorer = judgments.get("scorer")
    if (not isinstance(scorer, dict) or scorer.get("kind") not in ("human", "model")
            or not isinstance(scorer.get("id"), str) or not scorer["id"].strip()):
        raise ValueError("Identify the human panel or independent model scorer")
    if scorer["kind"] == "model" and (
        not isinstance(scorer.get("checkpoint_sha256"), str)
        or len(scorer["checkpoint_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in scorer["checkpoint_sha256"])
    ):
        raise ValueError("Model judgments require a frozen checkpoint hash")
    votes = defaultdict(dict)
    events = defaultdict(list)
    for row in judgments["votes"]:
        sample, annotator, choice = (row[k] for k in ("sample", "annotator", "choice"))
        if (sample not in key["samples"] or not isinstance(annotator, str) or not annotator.strip()
                or annotator in votes[sample] or choice not in ("left", "right", "unknown")):
            raise ValueError("Invalid, unexpected or duplicate vote")
        votes[sample][annotator] = choice
        timing = row.get("events")
        if timing is not None:
            if not isinstance(timing, list) or any(type(frame) is not int or not (
                0 <= frame < key["samples"][sample]["frames"]
            ) for frame in timing) or timing != sorted(set(timing)):
                raise ValueError("Events need unique, sorted zero-based frames within the clip")
            events[sample].append(timing)
    if set(votes) != set(key["samples"]):
        raise ValueError("Judgments need to cover all blinded samples")
    agrees = comparisons = 0
    singles = {}
    for sample, row in key["samples"].items():
        counts = Counter(votes[sample].values())
        ranked = counts.most_common()
        n = sum(counts.values())
        agrees += sum(c * (c - 1) for c in counts.values())
        comparisons += n * (n - 1)
        choice = ranked[0][0] if ranked[0][1] > n / 2 else "unknown"
        record = {"correct": choice == row["truth"], "unknown": choice == "unknown"}
        if sample in events:
            timings = events[sample]
            first = [frames[0] / max(1, row["frames"] - 1) for frames in timings if frames]
            record.update({"omitted": float(np.mean([len(frames) == 0 for frames in timings])),
                           "repeated": float(np.mean([len(frames) > 1 for frames in timings])),
                           "first_event_progress": float(np.median(first)) if first else None})
        singles.setdefault(row["checkpoint"] or "reference", {}).setdefault(row["pair"], {})[
            row["type"]] = record
    result = {"phase": key["phase"], "manifest_sha256": key["manifest_sha256"],
              "seed": key["seed"], "scorer": scorer, "lengths": key["lengths"], "scores": {},
              "raw_interrater_agreement": agrees / comparisons if comparisons else None,
              "reference": singles.pop("reference"), "pairs": key["pairs"]}
    for system, pairs in singles.items():
        summary = {}
        for component in ("full", "coarse"):
            selected = [i for i in pairs if key["pairs"][i]["reference"] == "a"]
            hit = {i: float(np.mean([pairs[i][f"{component}_{side}"]["correct"]
                                     for side in ("a", "b")])) for i in selected}
            unknown = [pairs[i][f"{component}_{side}"]["unknown"] for i in selected
                       for side in ("a", "b")]
            kinds = {kind: [value for i, value in hit.items() if key["pairs"][i]["kind"] == kind]
                     for kind in KINDS}
            summary[component] = {
                "per_pair": hit, "accuracy": float(np.mean(list(hit.values()))),
                "unknown_rate": float(np.mean(unknown)),
                "by_kind": {kind: {"n": len(hit), "accuracy": float(np.mean(hit))}
                            for kind, hit in kinds.items() if hit},
            }
            timed = [pairs[i][f"{component}_{side}"] for i in selected for side in ("a", "b")
                     if "omitted" in pairs[i][f"{component}_{side}"]]
            if timed:
                errors = []
                for i in selected:
                    gt = result["reference"][i]["gt"].get("first_event_progress")
                    pred = pairs[i][f"{component}_a"].get("first_event_progress")
                    if gt is not None and pred is not None:
                        errors.append(pred - gt)
                summary[component]["events"] = {
                    "judged_clips": len(timed), "paired_original_clips": len(errors),
                    "omission": float(np.mean([x["omitted"] for x in timed])),
                    "repetition": float(np.mean([x["repeated"] for x in timed])),
                    "original_timing_bias_progress": float(np.mean(errors)) if errors else None,
                    "original_timing_mae_progress": float(np.mean(np.abs(errors))) if errors else None,
                }
        result["scores"][system] = summary
    return result
