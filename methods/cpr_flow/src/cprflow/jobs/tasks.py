"""Small pipeline actions with explicit receipts. Never invent dataset or human approvals."""

import argparse
import json
import runpy
import sys
from pathlib import Path

from cprflow.config import load_config
from cprflow.data.cache import file_hash
from cprflow.jobs.schema import atomic_json


def environment(config):
    import torch
    import smplx  # noqa: F401
    from cprflow.training.logging import environment as snapshot
    from cprflow.data.sources import audit_sources
    from cprflow.models.text import FrozenTextEncoder
    from cprflow.training.artifacts import text_revisions

    audit_sources(config)
    if config["run"]["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    revision, tokenizer = text_revisions(config)
    # Cached/pinned pretrained encoder only; no corpus or signing-generator download.
    text = FrozenTextEncoder(revision, tokenizer, local_files_only=True)
    del text
    return {"status": "ready", "environment": snapshot(), "synthetic": False}


def seed_smoke_indexes(config, main_config):
    base = load_config(main_config)
    from cprflow.training.artifacts import update_index

    for kind in ("motion", "relation", "activity"):
        entry = json.loads((Path(base["paths"]["stats"]) / "index.json").read_text())[kind]
        update_index(config["paths"]["stats"], kind, entry)
    for kind in ("text", "geometry"):
        original = Path(base["paths"]["cache"]) / kind
        alias = Path(config["paths"]["cache"]) / kind
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.exists() and alias.resolve() != original.resolve():
            raise ValueError(f"Refuse to replace existing smoke cache: {alias}")
        if not alias.exists():
            alias.symlink_to(original.resolve(), target_is_directory=True)
        entry = json.loads((Path(base["paths"]["cache"]) / "index.json").read_text())[kind]
        update_index(config["paths"]["cache"], kind, entry)
    return {"status": "shared_by_reference", "copied_corpus_bytes": 0}


def prompts(config):
    from cprflow.data.manifest import read_manifest

    rows = read_manifest(Path(config["paths"]["manifests"]) / "dev.jsonl", "dev")
    from cprflow.types import DATASETS

    return [
        {"request_id": r["uid"], "text": r["text"], "sign_language": DATASETS[d][0], "seed": 101}
        for d in config["data"]["datasets"]
        for r in [r for r in rows if r["dataset_id"] == d][:2]
    ]


def smoke_check(ae_log, flow_log, sample_report):
    reports = {}
    for name, path in [("ae", ae_log), ("flow", flow_log)]:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        rows = [r for r in rows if "loss" in r]
        if len(rows) < 1000:
            raise ValueError(f"Incomplete real smoke: {name}")
        import math

        if not all(math.isfinite(r["loss"]) and math.isfinite(r["gradnorm"]) for r in rows):
            raise ValueError(f"Nonfinite smoke gradient/loss: {name}")
        first = sum(r["loss"] for r in rows[:100]) / 100
        last = sum(r["loss"] for r in rows[-100:]) / 100
        if last >= first:
            raise ValueError(f"Smoke loss did not decrease: {name}: {first}->{last}")
        reports[name] = {"first100_mean": first, "last100_mean": last, "updates": len(rows)}
    sampled = json.loads(Path(sample_report).read_text())
    if sampled["num_successful"] != sampled["num_requested"] or not sampled["num_successful"]:
        raise ValueError("Smoke sampling failed or empty")
    return {
        "status": "engineering_passed",
        "linguistic_gate": False,
        "synthetic": False,
        "losses": reports,
    }


def freeze_release(inputs):
    from cprflow.evaluation.protocol import require_clean_tree

    require_clean = all(item.get("require_clean_source", True) for item in inputs.values())
    if require_clean:
        commit = require_clean_tree()
        source_state = {"status": "clean_committed"}
    else:
        import subprocess
        from cprflow.jobs.schema import source_digest

        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        source_state = {
            "status": "operator_authorized_worktree_snapshot",
            "source_digest": source_digest(Path(__file__).resolve().parents[3]),
        }
    report = {}
    for name, item in inputs.items():
        checkpoint = Path(item["checkpoint"])
        datasets = {}
        for path in item["dev_reports"]:
            row = json.loads(Path(path).read_text())
            if row["split"] != "dev" or row["oracle"] or row["synthetic"]:
                raise ValueError("Final release requires real non-oracle dev reports")
            if row["checkpoint_sha256"] != file_hash(checkpoint):
                raise ValueError("Dev reports do not match selected checkpoint")
            if row["num_successful"] == 0 or row["coverage"] <= 0:
                raise ValueError("No successful full-dev samples")
            datasets[row["dataset"]] = {"sha256": file_hash(path), "coverage": row["coverage"]}
        expected = set(item.get("expected_datasets", ["csl_daily", "how2sign", "phoenix2014t"]))
        if set(datasets) != expected:
            raise ValueError("Require all configured real dev reports before full test")
        report[name] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_hash(checkpoint),
            "dev": datasets,
            "sampler": item["sampler"],
        }
    return {
        "protocol_frozen": True,
        "source_state": source_state,
        "commit": commit,
        "selection": "training-panel EMA; confirmed on full dev",
        "authorization": "User requested complete training/evaluation jobs",
        "models": report,
    }


def external(spec_path, context):
    spec = json.loads(Path(spec_path).read_text())
    # Gate is also checked by the scheduler; repeat pin validation at execution time.
    if (
        spec.get("validated_on_real_fit_and_ae") is not True
        or file_hash(spec["adapter_file"]) != spec["adapter_sha256"]
    ):
        raise ValueError("Unvalidated external evaluator")
    argv = [arg.format(**context) for arg in spec["argv"]]
    previous = sys.argv
    try:
        sys.argv = [spec["adapter_file"], *argv]
        try:
            runpy.run_path(spec["adapter_file"], run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    finally:
        sys.argv = previous
    result = json.loads(Path(context["output"]).read_text())
    if result.get("metric_status") != "available" or not result.get("metrics"):
        raise ValueError("External evaluator did not produce real metrics")
    import math

    if not 0 < result.get("num_successful", 0) <= result.get("num_requested", 0):
        raise ValueError("External evaluator must report actual coverage/counts")
    if not all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in result["metrics"].values()
    ):
        raise ValueError("Nonfinite/non-numeric external metrics")
    for key in ("dataset", "split", "checkpoint_sha256"):
        expected = file_hash(context["checkpoint"]) if key == "checkpoint_sha256" else context[key]
        if result.get(key) != expected:
            raise ValueError(f"External evaluator provenance mismatch: {key}")
    return {"status": "evaluated", "report": context["output"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "environment",
            "seed-smoke",
            "prompts",
            "smoke-check",
            "release",
            "external",
            "resolve-timing",
        ),
    )
    parser.add_argument("--config")
    parser.add_argument("--main-config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ae-log")
    parser.add_argument("--flow-log")
    parser.add_argument("--sample-report")
    parser.add_argument("--inputs")
    parser.add_argument("--spec")
    parser.add_argument("--context")
    args = parser.parse_args()
    config = load_config(args.config) if args.config else None
    if args.action == "environment":
        result = environment(config)
    elif args.action == "resolve-timing":
        from cprflow.data.local_timing import resolve_timing

        result = resolve_timing(config)
    elif args.action == "seed-smoke":
        result = seed_smoke_indexes(config, args.main_config)
    elif args.action == "prompts":
        from cprflow.data.manifest import write_jsonl

        write_jsonl(args.output, prompts(config))
        return
    elif args.action == "smoke-check":
        result = smoke_check(args.ae_log, args.flow_log, args.sample_report)
    elif args.action == "release":
        result = freeze_release(json.loads(Path(args.inputs).read_text()))
    else:
        result = external(args.spec, json.loads(Path(args.context).read_text()))
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
