"""Application entry points for generating, scoring and comparing stored runs."""

import csv
from pathlib import Path

import numpy as np
import torch

from slp.evaluation.flow_matching import corpus_bleu4, paired_bootstrap

from .data import Request, digest, load_clip, sha256
from .evaluation import (
    PUBLISHED,
    cohort_hash,
    locked_protocol,
    prediction_dir,
    retrieval,
    score_clip,
    semantics,
)
from .runtime import checkpointed_records, json_read, json_write, npz_write, write_atomic


def split_rows(pipe, split):
    path = pipe.root / "splits.json"
    if sha256(path) != pipe.state["parent_hashes"]["splits.json"]:
        raise ValueError("Split manifest checksum mismatch")
    rows = [row for row in json_read(path) if row["split"] == split]
    if not rows:
        raise ValueError(f"Empty split: {split}")
    return rows


def sample(pipe, split, arms, nfe, debug, retrained):
    if split == "test" and any(a.startswith("F-") for a in arms):
        raise ValueError("Oracle inputs are restricted to dev")
    rows = {row["clip_id"]: row for row in split_rows(pipe, split) if row.get("pose_path")}
    seed = pipe.seed
    data = {**pipe.cfg["data"], "fps": pipe.stats["fps"]}
    for arm in arms:
        intervention = arm.startswith("F-") or (arm.startswith("E-") and not retrained)
        weights = "C" if intervention else arm
        sampler = pipe.sampler(arm, weights)
        out = prediction_dir(pipe.root, split, arm, nfe)
        binding = digest([sampler.version, split, arm, weights, nfe, debug, seed])

        def compute(key):
            row = rows[key]
            request = Request(
                row["text"], row["source_language"], row["target_sign_language"], key, seed
            )
            if arm.startswith("F-"):
                prediction = sampler.oracle_dev(request, load_clip(row, data), nfe, debug)
            else:
                prediction = sampler.generate(request, nfe, debug)
            path = out / f"{digest([binding, key])}.npz"
            arrays = {
                k: v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                for k, v in prediction.items()
                if isinstance(v, (torch.Tensor, str, int, float))
            }
            npz_write(path, **arrays)
            diagnostics = prediction["source_diagnostics"]
            return {
                "clip_id": key,
                "filename": path.name,
                "sha256": sha256(path),
                "weights_from": weights,
                "intervention": intervention,
                "donor_id": diagnostics.get("donor_id"),
                "fallbacks": diagnostics.get("fallbacks", 0),
                "source_diagnostics": diagnostics,
            }

        metadata = checkpointed_records(
            out / "work" / binding,
            list(rows),
            compute,
            batch_size=8,
            resume=pipe.resume,
            label=f"sample {split}/{arm}",
        )
        for item in metadata.values():
            if sha256(out / item["filename"]) != item["sha256"]:
                raise ValueError(f"Invalid cached prediction: {item['filename']}")
        json_write(
            out / "index.json",
            {
                "predictions": list(metadata.values()),
                "run": pipe.state["signature"],
                "seed": seed,
                "arm": arm,
                "nfe": nfe,
                "split": split,
            },
        )


def evaluate(pipe, split, arms, nfe, protocol_path=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = split_rows(pipe, split)
    data_cfg = {**pipe.cfg["data"], "fps": pipe.stats["fps"]}
    cfg = pipe.cfg
    protocol = locked_protocol(json_read(protocol_path) if protocol_path else cfg["protocol"])
    filesystem_hashes = {r["clip_id"]: sha256(r["pose_path"]) for r in rows if r.get("pose_path")}
    coverage = [
        {"clip_id": r["clip_id"], "reason": r.get("missing_reason", "missing reference")}
        for r in rows
        if not r.get("pose_path")
    ]
    clips = [load_clip(r, data_cfg) for r in rows if r.get("pose_path")]
    if not clips:
        raise ValueError("No scoreable references")
    fk = pipe.geometry()
    cohort = cohort_hash([dict(r, sha256=filesystem_hashes.get(r["clip_id"])) for r in rows])
    assets_path = pipe.root / "assets.lock.json"
    if sha256(assets_path) != pipe.state["parent_hashes"]["assets.lock.json"]:
        raise ValueError("Asset lock checksum mismatch")
    protocol["geometry_asset_hashes"] = {
        a["name"]: a["sha256"] for a in json_read(assets_path) if "geometry" in a["consumers"]
    }
    protocol["geometry_backend"] = "synthetic" if pipe.profile == "smoke" else "soke"
    protocol_id = digest(protocol)
    planner_hash = sha256(pipe.path("planner"))
    fields = ("body_dtw_jpe", "hand_dtw_jpe", "body_dtw_pa_jpe", "hand_dtw_pa_jpe")
    results = []
    for arm in arms:
        directory = prediction_dir(pipe.root, split, arm, nfe)
        index = json_read(directory / "index.json")
        if (
            index["run"] != pipe.state["signature"]
            or index["seed"] != pipe.seed
            or index["arm"] != arm
            or index["split"] != split
            or index["nfe"] != nfe
        ):
            raise ValueError(f"Prediction manifest mismatch: {directory}")
        indexed = {r["clip_id"]: r for r in index["predictions"]}
        if len(indexed) != len(index["predictions"]) or indexed.keys() != {c.id for c in clips}:
            raise ValueError("Prediction cohort does not match reference coverage")
        per_clip, outputs = [], {}
        for c in clips:
            entry = indexed[c.id]
            file = directory / entry["filename"]
            if sha256(file) != entry["sha256"]:
                raise ValueError(f"Prediction checksum mismatch: {file}")
            with np.load(file, allow_pickle=False) as arrays:
                pred = {
                    k: x.item() if x.ndim == 0 else torch.from_numpy(x)
                    for k in arrays
                    for x in [arrays[k]]
                }
            if pred["sample_id"] != c.id or pred["nfe"] != nfe or pred["seed"] != pipe.seed:
                raise ValueError(f"Wrong clip provenance: {file}")
            if (
                pred["pose133"].shape != (pred["T_hat"], 133)
                or not pred["pose133"].isfinite().all()
                or pred["fps"] != pipe.stats["fps"][c.row["corpus"]]
            ):
                raise ValueError(f"Invalid pose or timebase: {file}")
            result = score_clip(pred, c, fk, protocol)
            result.update(semantics(pred, c, protocol, fk=fk))
            result.update(
                clip_id=c.id,
                video_id=c.row.get("video_id"),
                text_hash=digest(c.row["text"]),
                corpus=c.row["corpus"],
                pose_fit_id=c.row["pose_fit_id"],
                seed=pipe.seed,
                arm=arm,
                split=split,
                nfe=nfe,
                cohort=cohort,
                protocol=protocol_id,
                track=cfg["track"],
                planner_hash=planner_hash,
                exact_published_protocol=(
                    protocol.get("signnmflow_v1_parity") == "verified"
                    and protocol["geometry_backend"] == "soke"
                    and result.get("bt_status") == "verified"
                ),
                weights_from=entry["weights_from"],
                donor_id=entry["donor_id"],
                fallback_count=entry["fallbacks"],
                prediction_version=pred["version"],
                latency_seconds=pred["latency_seconds"],
            )
            for key in ("plan_seconds", "source_seconds", "text_seconds", "refiner_seconds"):
                result[key] = pred[key]
            per_clip.append(result)
            outputs.setdefault(c.row["corpus"], []).append(
                {
                    "clip_id": c.id,
                    "text": c.row["text"],
                    "pose133": pred["pose133"],
                    "fps": pred["fps"],
                }
            )
        out = pipe.root / "scores" / split / arm / f"nfe_{nfe}"
        write_atomic(
            out / "per_clip.parquet", lambda p: pq.write_table(pa.Table.from_pylist(per_clip), p)
        )
        json_write(out / "protocol.json", protocol)
        json_write(
            out / "coverage.json",
            {
                "included": len(clips),
                "declared": len(rows),
                "coverage": len(clips) / len(rows),
                "missing": coverage,
            },
        )
        for corpus in outputs:
            group = [r for r in per_clip if r["corpus"] == corpus]
            has_bt = all("bt_counts" in r for r in group)
            bt_reference = None
            if has_bt:
                from .data import pack133

                gt_rows = [
                    semantics({"pose133": pack133(c.q, c.e), "fps": c.fps}, c, protocol, fk=fk)
                    for c in clips
                    if c.row["corpus"] == corpus
                ]
                bt_reference = corpus_bleu4(gt_rows)
            results.append(
                {
                    "arm": arm,
                    "corpus": corpus,
                    "result_type": "reproduced" if arm.startswith("A-") else "new",
                    "mode": "multi" if cfg["data"]["corpus"] == "multi" else "mono",
                    "cohort": cohort,
                    "scorer_id": protocol_id,
                    "scope": split,
                    "pose_fit": ",".join(sorted({r["pose_fit_id"] for r in group})),
                    "input_mode": "oracle" if arm.startswith("F-") else "text",
                    "nfe": nfe,
                    "seed": pipe.seed,
                    **{k: float(np.mean([r[k] for r in group])) for k in fields},
                    "corpus_bt_bleu4": corpus_bleu4(group) if has_bt else None,
                    "reference_bt_bleu4": bt_reference,
                    "siclip_r1": retrieval(outputs[corpus], corpus, protocol).get("recall_at_1"),
                    "bt_status": group[0]["bt_status"],
                    "jerk_coverage": float(np.mean([r["jerk_ratio"] is not None for r in group])),
                    "duration_mae_seconds": float(
                        np.mean([r["duration_mae_seconds"] for r in group])
                    ),
                    "duration_cap_rate": float(np.mean([r["from_duration_cap"] for r in group])),
                }
            )
        csv_write(out / "results.csv", [r for r in results if r["arm"] == arm])
    published = [
        {
            "corpus": corpus,
            "mode": mode,
            "arm": "SignNMFlow-v1",
            "result_type": "published",
            "scorer_id": "SignNMFlow-v1",
            "scope": "paper_test",
            **dict(zip((*fields, "corpus_bt_bleu4"), metric)),
        }
        for corpus, mode, *metric in PUBLISHED
    ]
    csv_write(pipe.root / "scores" / split / f"results_nfe_{nfe}.csv", results + published)


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))

    def dump(dest):
        with dest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_atomic(path, dump)


def compare(pipe, split, baseline, challenger, nfe, roots):
    import pyarrow.parquet as pq

    evaluation, stop = pipe.cfg["evaluation"], pipe.cfg["stopping"]

    def read(arm):
        paths = [
            Path(root) / "scores" / split / arm / f"nfe_{nfe}" / "per_clip.parquet"
            for root in roots
        ]
        return [row for p in paths for row in pq.read_table(p).to_pylist()]

    base, challenge = read(baseline), read(challenger)
    corpora = sorted({r["corpus"] for r in challenge})
    if not corpora or set(corpora) != {r["corpus"] for r in base}:
        raise ValueError("Both arms must cover the same corpora")
    combined = []
    decisions = []
    for corpus in corpora:
        a = [r for r in base if r["corpus"] == corpus]
        b = [r for r in challenge if r["corpus"] == corpus]
        binding = {(r["cohort"], r["protocol"], r["track"]) for r in a + b}
        if len(binding) != 1:
            raise ValueError("Comparison requires identical cohort, protocol and track")
        cohort, protocol, track = next(iter(binding))
        outcomes = paired_bootstrap(
            a,
            b,
            evaluation["bootstrap_repetitions"],
            evaluation["bootstrap_seed"],
            evaluation["confidence_level"],
        )
        for metric, values in outcomes.items():
            combined.append(
                {
                    "corpus": corpus,
                    "cohort": cohort,
                    "scorer_id": protocol,
                    "track": track,
                    "baseline": baseline,
                    "challenger": challenger,
                    "metric": metric,
                    "result_type": "paired",
                    "scope": values["scope"],
                    "mean": values["mean"],
                    "ci_low": values["ci"][0],
                    "ci_high": values["ci"][1],
                    "seed_std": values["seed_std"],
                    "per_seed": values["per_seed"],
                }
            )
        pa, bt = outcomes["hand_dtw_pa_jpe"], outcomes.get("corpus_bt_bleu4")
        base_pa = float(np.mean([r["hand_dtw_pa_jpe"] for r in a]))
        raw = outcomes["hand_dtw_jpe"]["mean"]
        seeds = len(pa["per_seed"])
        paired_gain = pa["ci"][0] > 0 and bt is not None and bt["ci"][0] > 0
        good_wrist = outcomes["wrist_torso_error"]["mean"] >= 0
        internal = (
            pa["mean"] >= stop["target_relative_hand_pa_gain"] * base_pa
            and bt is not None
            and bt["mean"] >= stop["target_absolute_bleu_gain"]
            and raw >= 0
        )
        status = "pending"
        exact = corpus == "csl_daily" and all(r.get("exact_published_protocol") for r in a + b)
        target_csl = None
        if exact and bt is not None:
            by_seed = {s: [r for r in b if r["seed"] == s] for s in pa["per_seed"]}
            target_csl = (
                np.mean([r["hand_dtw_pa_jpe"] for r in b])
                <= stop["target_hand_pa_csl_same_protocol"]
                and np.mean([corpus_bleu4(rows) for rows in by_seed.values()])
                >= stop["target_bt_csl_same_protocol"]
                and np.mean([r["hand_dtw_jpe"] for r in b])
                <= stop["max_hand_jpe_csl_same_protocol"]
            )
        if split == "dev":
            status = "dev_signal" if bt and pa["mean"] > 0 and bt["mean"] > 0 else "diagnose"
        elif seeds >= evaluation["min_training_seeds"] and bt is not None:
            status = (
                "paired_internal_gate_pass"
                if paired_gain and internal
                else "paired_internal_gate_unmet"
            )
            if exact and baseline == "A-matched":
                status = (
                    "published_protocol_gate_pass"
                    if target_csl and paired_gain
                    else "published_protocol_gate_unmet"
                )
        decisions.append(
            f"## {corpus}\n\nStatus **{status}**. Training seeds {seeds}. "
            f"Hand PA gain {pa['mean']:.6g} metres accumulated. "
            f"BLEU gain {bt['mean'] if bt else 'unavailable'}. "
            f"Hand raw gate {'pass' if raw >= 0 else 'unmet'}. "
            f"Wrist guard {'pass' if good_wrist else 'unmet'}. "
            f"Exact published-protocol targets {'verified' if target_csl else 'unmet' if target_csl is not None else 'pending'}. "
            "Human comprehension pending.\n"
        )
    out = pipe.root / "comparisons" / split / f"{challenger}_vs_{baseline}_nfe_{nfe}"
    csv_write(out / "results.csv", combined)
    write_atomic(out / "decision.md", lambda p: p.write_text("\n".join(decisions)))
