"""P1–P3 arm specifications and locked multi-seed summaries."""

import json
from pathlib import Path

from slp.evaluation.multirate import hierarchical_effects

from .data import read_jsonl, write_json
from .evaluation import _comparison_pair

ARMS = {
    "A": (1, "uniform"), "B": (4, "uniform"), "C": (1, "event"), "D": (4, "event"),
    "E": (4, "onset_mid_offset"), "F": (4, "none"),
}
CONTRASTS = (("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("B", "E"), ("F", "D"))


def plan(pilot):
    variants = ("B", "D", "F") if pilot in ("p1", "p2") else tuple(ARMS)
    seeds = [17, 23, 41] if pilot == "p3" else [17]
    return {"pilot": pilot, "seeds": seeds, "arms": {
        arm: {"cli_variant": {"E": "onset_mid_offset", "F": "no_anchor"}.get(arm, arm),
              "stride": ARMS[arm][0], "policy": ARMS[arm][1], "hand_rate": "full",
              "shared_stages": [] if pilot == "p1" else [
                  "length", f"body:stride={ARMS[arm][0]}",
                  *([f"anchors:policy={ARMS[arm][1]}"] if arm != "F" else []),
              ]} for arm in variants},
        "comparisons": [["B", "D"], ["F", "D"]] if pilot in ("p1", "p2") else CONTRASTS,
        "evaluation_protocols": ["oracle"] if pilot == "p1" else [
            "oracle", "predicted_body_oracle_anchors", "predicted_gt_length",
            "oracle_body_predicted_anchors", "text_only",
        ],
        "p3_ablations": {"confidence": ["uniform", "learned"], "correction": [True, False],
                        "loss": ["endpoint_only", "endpoint_fk", "endpoint_fk_velocity"],
                        "body_fine_nfe": [[2, 4], [4, 8], [8, 16]]} if pilot == "p3" else {},
        "compute_comparisons": ["fixed_architecture", "dev_compute_within_10_percent"]
        if pilot == "p3" else []}


def _paths(path, content, arm, seeds):
    return [path.parent / content[arm][str(seed)] for seed in seeds]


def compare_matrix(manifest, out):
    """Read completed paired P3 evals using explicit per-arm/per-seed paths."""
    manifest = Path(manifest)
    content = json.loads(manifest.read_bytes())
    field = content.get("primary_metric", "event_tip_mm")
    if field not in ("event_tip_mm", "tip_mm", "wrist_mm", "body_mm"):
        raise ValueError("Unsupported matrix primary metric")
    arms = content["arms"]
    if not set(arms) <= set(ARMS) or not {"A", "B", "C", "D"} <= set(arms):
        raise ValueError("P3 matrix needs A, B, C, D and optionally E or F")
    seeds = [17, 23, 41]
    paths = {a: _paths(manifest, arms, a, seeds) for a in arms}
    protocol = None
    rows = {}
    metadata = {}
    for arm, folders in paths.items():
        rows[arm] = []
        for folder, seed in zip(folders, seeds):
            status = json.loads((folder / "protocol.json").read_bytes())
            identity = status["config"]["experiment"]
            expected = {"E": "onset_mid_offset", "F": "no_anchor"}.get(arm, arm)
            if (status["seed"] != seed or identity["variant"] != expected or
                    identity["pilot"] != "p3" or status["config"]["body"]["stride"] != ARMS[arm][0]
                    or status["config"]["anchors"]["policy"] != ARMS[arm][1]
                    or status["split"] not in ("dev_gate", "test")):
                raise ValueError("P3 run does not match its predeclared arm, seed or split")
            if read_jsonl(folder / "failures.jsonl"):
                raise ValueError("P3 matrix input contains incomplete evaluations")
            contract = {k: status[k] for k in (
                "protocol", "split", "manifest_sha256", "rig_sha256", "cohort_sha256",
                "calibration", "event_scales")}
            if protocol is not None and contract != protocol:
                raise ValueError("P3 matrix inputs must share detector, rig, cohort and protocol")
            protocol = contract
            rows[arm].append(read_jsonl(folder / "per_clip.jsonl"))
            metadata[(arm, seed)] = status
    cfg = metadata[("A", seeds[0])]["config"]["eval"]
    comparisons = {}
    for a, b in CONTRASTS:
        if a not in paths or b not in paths:
            continue
        for aa, bb in zip(paths[a], paths[b]):
            _comparison_pair(aa, bb)
        values = hierarchical_effects(list(zip(rows[a], rows[b])), field, cfg)
        comparisons[f"{a}-{b}"] = values
    interaction = hierarchical_effects(list(zip(rows["B"], rows["D"], rows["A"], rows["C"])),
                                       field, cfg, interaction=True)
    for a, b in (("A", "B"), ("C", "D")):
        for seed in seeds:
            left, right = metadata[a, seed]["checkpoints"], metadata[b, seed]["checkpoints"]
            if left.get("anchors") != right.get("anchors"):
                raise ValueError("Same-policy stride comparisons must reuse anchor checkpoints")
    report = {"protocol": protocol, "primary_metric": field, "seeds": seeds,
              "comparisons": comparisons, "stride_by_anchor_interaction": interaction}
    write_json(out, report)
    return report
