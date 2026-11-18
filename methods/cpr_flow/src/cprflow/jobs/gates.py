import json
from pathlib import Path


def evaluate_gate(gate):
    kind = gate["kind"]
    if kind == "exists":
        if not Path(gate["path"]).exists():
            raise FileNotFoundError(gate["path"])
        return {"path": gate["path"]}
    if kind == "source_snapshot":
        from cprflow.jobs.schema import source_digest

        return {
            "policy": "operator_authorized_worktree_snapshot",
            "source_digest": source_digest(gate["root"]),
        }
    if kind == "clean_git":
        from cprflow.evaluation.protocol import require_clean_tree

        return {"commit": require_clean_tree()}
    from cprflow.config import load_config

    cfg = load_config(gate["config"])
    if kind == "access":
        from cprflow.data.sources import require_access

        return require_access(cfg["paths"]["access_record"])
    if kind == "timing":
        from cprflow.data.preparation import verified_timing

        return verified_timing(cfg)
    if kind == "geometry":
        from cprflow.training.gates import require_geometry_review

        return require_geometry_review(cfg)
    if kind == "calibration":
        from cprflow.sampling.guard import load_calibration
        from cprflow.data.cache import file_hash

        preprocessing = json.loads(
            (Path(cfg["paths"]["manifests"]) / "preprocessing_report.json").read_text()
        )["preprocessing_hash"]
        return load_calibration(
            cfg["paths"]["teacher_calibration"],
            {
                "teacher_sha256": file_hash(cfg["paths"]["teacher_checkpoint"]),
                "asset_sha256": file_hash(cfg["paths"]["smplx_model"]),
                "preprocess_sha256": preprocessing,
            },
        )
    if kind == "external":
        path = Path(gate["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing audited integration: {path}")
        spec = json.loads(path.read_text())
        from cprflow.data.cache import file_hash

        if (
            spec.get("validated_on_real_fit_and_ae") is not True
            or file_hash(spec["adapter_file"]) != spec["adapter_sha256"]
        ):
            raise ValueError("External evaluator has no validated/pinned adapter")
        return {"adapter_sha256": spec["adapter_sha256"]}
    raise ValueError(f"Unknown gate: {kind}")


def check_outputs(job):
    for item in job["outputs"]:
        path = Path(item["path"])
        if not path.exists():
            raise FileNotFoundError(f"Missing job output: {path}")
        if item.get("equals") or item.get("minimum"):
            value = json.loads(path.read_text())
            for key, expected in item.get("equals", {}).items():
                if value.get(key) != expected:
                    raise ValueError(f"Invalid output {path}: {key}")
            for key, minimum in item.get("minimum", {}).items():
                if value.get(key, -1) < minimum:
                    raise ValueError(f"Incomplete output {path}: {key}")
