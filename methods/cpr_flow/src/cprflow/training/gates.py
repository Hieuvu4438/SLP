"""Evidence-backed engineering gates. No human/calibration approval is synthesized."""

import json
from pathlib import Path

from cprflow.data.cache import file_hash


def require_geometry_review(config):
    if config["run"].get("manual_gate_policy", "required") == "operator_waived":
        return {
            "status": "operator_waived",
            "approved": False,
            "scientific_geometry_gate_passed": False,
            "reference": config["run"]["policy_reference"],
        }
    path = Path(config["run"]["out_dir"]) / "geometry_audit/review.json"
    if not path.is_file():
        raise FileNotFoundError(f"M1-M2 G0 blocked: missing reviewed geometry artifact {path}")
    record = json.loads(path.read_text())
    if record.get("approved") is not True:
        raise ValueError("G0 geometry review not approved")
    for field, asset in [
        ("asset_sha256", config["paths"]["smplx_model"]),
        ("train_manifest_sha256", Path(config["paths"]["manifests"]) / "train.jsonl"),
        ("source_lock_sha256", config["paths"]["sources_lock"]),
    ]:
        if record.get(field) != file_hash(asset):
            raise ValueError(f"G0 review stale or missing {field}")
    for field in (
        "native_fitting_render_reviewed",
        "left_right_mapping_reviewed",
        "face_basis_reviewed",
    ):
        if record.get(field) is not True:
            raise ValueError(f"G0 requires completed {field}")
    if not record.get("reviewer"):
        raise ValueError("G0 requires a named reviewer")
    for dataset in config["data"]["datasets"]:
        if len(set(record.get("reviewed_uids", {}).get(dataset, []))) < 64:
            raise ValueError(f"G0 requires 64 reviewed native fitting renders: {dataset}")
    return record
