"""Explicit source import -> verified timing -> canonicalization; failures are ledger entries."""

import json
from pathlib import Path

import numpy as np

from cprflow.data.cache import digest, file_hash, save_numeric
from cprflow.data.lmdb_adapter import LMDBAdapter, normalized_text, source_to_canonical
from cprflow.data.manifest import check_disjoint, write_jsonl
from cprflow.data.resample import resample_motion
from cprflow.data.sources import audit_sources, require_access
from cprflow.types import DATASETS, LANGUAGES, PROFILE


def source_paths(config):
    lock = audit_sources(config)
    revision = lock["dataset"]["revision"]
    root = Path(config["paths"]["source_pose"]) / revision
    return [
        (dataset, split, root / split / f"{DATASETS[dataset][2]}{split}.lmdb")
        for dataset in config["data"]["datasets"]
        for split in ("train", "dev", "test")
    ]


def import_sources(config):
    access = require_access(config["paths"]["access_record"])
    index = {s: [] for s in ("train", "dev", "test")}
    reports = []
    missing = []
    revision = audit_sources(config)["dataset"]["revision"]
    for dataset, split, path in source_paths(config):
        if not (path / "data.mdb").is_file():
            missing.append(str((path / "data.mdb").resolve()))
            continue
        sha = file_hash(path / "data.mdb")
        adapter = LMDBAdapter(path, sha, trusted=True)
        for clip in adapter.clip_ids:
            clip = clip.decode() if isinstance(clip, bytes) else str(clip)
            index[split].append(
                {
                    "uid": f"{dataset}:{clip}",
                    "dataset_id": dataset,
                    "split": split,
                    "source_clip_id": clip,
                    "source_video_id": None,
                    "signer_id": None,
                    "source_lmdb": str(path.resolve()),
                    "source_sha256": sha,
                    "source_revision": revision,
                    "access_record_reference": access[dataset]["reference"],
                    "reliability": None,
                    "synthetic": config["run"]["synthetic"],
                }
            )
        reports.append(
            {
                "dataset": dataset,
                "split": split,
                "actual_clips": len(adapter.clip_ids),
                "source_path": str(path.resolve()),
                "source_sha256": sha,
            }
        )
        adapter.close()
    destination = Path(config["paths"]["manifests"])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "source_import_report.json").write_text(
        json.dumps({"sources": reports, "missing_paths": missing}, indent=2)
    )
    if missing:
        raise FileNotFoundError("Missing pose LMDBs (M1 blocked):\n" + "\n".join(missing))
    check_disjoint(list(index.values()))
    for split, rows in index.items():
        write_jsonl(destination / f"source_index_{split}.jsonl", rows)
    return reports


def verified_timing(config):
    path = Path(config["paths"]["source_timing"])
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing verified pose timing: {path}; raw video FPS is insufficient"
        )
    obj = json.loads(path.read_text())
    revision = audit_sources(config)["dataset"]["revision"]
    if obj.get("revision") != revision:
        raise ValueError("Timing revision mismatch")
    for dataset in config["data"]["datasets"]:
        record = obj["datasets"][dataset]
        if (
            (record.get("verified") is not True and config["data"]["require_verified_feature_fps"])
            or not record.get("verification_method")
            or not record.get("evidence")
        ):
            raise ValueError(f"Unverified effective feature FPS: {dataset}")
        if not np.isfinite(record["feature_fps"]) or record["feature_fps"] <= 0:
            raise ValueError("Invalid feature FPS")
        for item in record["evidence"]:
            if not item.get("url") and not item.get("path"):
                raise ValueError("Timing evidence needs URL or file")
            if item.get("path") and file_hash(item["path"]) != item["sha256"]:
                raise ValueError("Timing evidence hash mismatch")
    return obj


def canonicalize(config):
    timing = verified_timing(config)
    from cprflow.geometry.body_model import BodyModel

    body = BodyModel(config["paths"]["smplx_model"], config["geometry"]["fk_chunk_frames"])
    dependencies = {
        "source": audit_sources(config)["dataset"],
        "timing": timing,
        "profile": PROFILE,
        "source_profile": config["data"]["source_profile"],
        "resample_grid": config["data"]["resample_grid"],
        "anti_alias_policy": "quaternion_fir_v1_when_source_fps_gt_target",
        "target_fps": 30.0,
        "asset_profile": body.profile_hash,
    }
    preprocess_hash = digest(dependencies)
    root = Path(config["paths"]["processed"]) / preprocess_hash
    dest = Path(config["paths"]["manifests"])
    all_rows = []
    quarantine = []
    long = []
    adapters = {}
    try:
        for split in ("train", "dev", "test"):
            path = dest / f"source_index_{split}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            output = []
            previous_path = dest / f"{split}.jsonl"
            previous = (
                {
                    r["uid"]: r
                    for r in (json.loads(line) for line in previous_path.read_text().splitlines())
                }
                if previous_path.is_file()
                else {}
            )
            for row in rows:
                try:
                    old = previous.get(row["uid"])
                    if (
                        old
                        and old.get("preprocessing_hash") == preprocess_hash
                        and old.get("source_sha256") == row["source_sha256"]
                        and Path(old["processed_path"]).is_file()
                        and file_hash(old["processed_path"]) == old["processed_sha256"]
                    ):
                        output.append(old)
                        if split == "train" and old["frames"] > config["data"]["max_train_frames"]:
                            long.append({**old, "reason": "exceeds_max_train_frames"})
                        continue
                    key = row["source_lmdb"]
                    if key not in adapters:
                        adapters[key] = LMDBAdapter(key, row["source_sha256"], trusted=True)
                    source = adapters[key].read(row["source_clip_id"])
                    lang, spoken, _ = DATASETS[row["dataset_id"]]
                    expected_tag = {
                        "csl_daily": "Chinese",
                        "how2sign": "English",
                        "phoenix2014t": "German",
                    }[row["dataset_id"]]
                    allowed_tags = (
                        (lang, expected_tag, "American")
                        if row["dataset_id"] == "how2sign"
                        else (lang, expected_tag)
                    )
                    if normalized_text(source["language"]) not in allowed_tags:
                        raise ValueError("Source language tag mismatch for the known dataset")
                    text = normalized_text(source["translation"])
                    if not text:
                        raise ValueError("Empty text")
                    x = source_to_canonical(source, config["data"]["source_profile"])
                    source_frames = len(x)
                    fps = timing["datasets"][row["dataset_id"]]["feature_fps"]
                    motion, timestamps, effective = resample_motion(
                        x,
                        fps,
                        verified=timing["datasets"][row["dataset_id"]]["verified"],
                        allow_assumed=not config["data"]["require_verified_feature_fps"],
                    )
                    # Hash IDs to avoid accidental path traversal by source clip names.
                    target = root / row["dataset_id"] / split / (digest(row["uid"]) + ".npz")
                    save_numeric(target, motion=motion, timestamps=timestamps)
                    record = {
                        **row,
                        "text": text,
                        "text_sha256": digest(text),
                        "sign_language_id": LANGUAGES[lang],
                        "spoken_language": spoken,
                        "profile": PROFILE,
                        "feature_profile": config["data"]["source_profile"],
                        "mapping_version": "smplx_native_upper10_v1",
                        "source_fps": fps,
                        "feature_timing_verified": timing["datasets"][row["dataset_id"]][
                            "verified"
                        ],
                        "source_language_tag": normalized_text(source["language"]),
                        "source_hand_widths": [
                            source["left_features"].shape[1],
                            source["right_features"].shape[1],
                        ],
                        "fps": effective,
                        "frames": len(motion),
                        "source_frames": source_frames,
                        "preprocessing_hash": preprocess_hash,
                        "asset_profile_hash": body.profile_hash,
                        "processed_path": str(target.resolve()),
                        "processed_sha256": file_hash(target),
                        "quality_flags": [],
                    }
                    if split == "train" and len(motion) > config["data"]["max_train_frames"]:
                        long.append({**record, "reason": "exceeds_max_train_frames"})
                    output.append(record)
                except (ValueError, KeyError) as exc:
                    quarantine.append({**row, "reason": str(exc)})
            write_jsonl(dest / f"{split}.jsonl", output)
            all_rows.append(output)
    finally:
        for adapter in adapters.values():
            adapter.close()
    check_disjoint(all_rows)
    write_jsonl(dest / "quarantine.jsonl", quarantine)
    write_jsonl(dest / "long_train_exclusions.jsonl", long)
    counts = {}
    for dataset in config["data"]["datasets"]:
        counts[dataset] = {
            split: sum(r["dataset_id"] == dataset for r in rows)
            for split, rows in zip(("train", "dev", "test"), all_rows)
        }
    report = {
        "preprocessing_hash": preprocess_hash,
        "counts": counts,
        "quarantined": len(quarantine),
        "long_train_exclusions": len(long),
        "synthetic": config["run"]["synthetic"],
        "dependencies": dependencies,
    }
    (dest / "preprocessing_report.json").write_text(json.dumps(report, indent=2))
    return report
