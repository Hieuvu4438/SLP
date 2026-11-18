import numpy as np


def aggregate_reports(reports):
    if not reports:
        raise ValueError("No reports")
    keys = (
        "protocol",
        "duration_source",
        "relation_source",
        "oracle",
        "profile",
        "metric_version",
        "asset_sha256",
        "fitting_revision",
        "evaluator_sha256",
        "checkpoint_sha256",
        "run_id",
        "split",
        "seeds",
        "synthetic",
        "stats_sha256",
        "inference_correction",
        "inference_teacher_sha256",
        "inference_calibration_sha256",
    )
    for key in keys:
        if any(key not in r for r in reports):
            raise ValueError(f"Missing comparison provenance: {key}")
        if len({str(r[key]) for r in reports}) != 1:
            raise ValueError(f"Reject mixed comparison protocols: {key}")
    if reports[0]["oracle"]:
        raise ValueError("Oracle reports cannot enter a main comparison table")
    if len({r["dataset"] for r in reports}) != len(reports):
        raise ValueError("Aggregate requires one report per dataset")
    # No pooled BLEU/text scores across sign/spoken languages.
    common = set.intersection(*(set(r["metrics"]) for r in reports))
    allowed = [k for k in common if "bleu" not in k.lower() and "backtranslation" not in k.lower()]
    if any(r.get("feature_timing_verified", True) is not True for r in reports):
        allowed = [
            k
            for k in allowed
            if not any(term in k for term in ("_seconds", "angular_velocity", "high_frequency"))
        ]
    return {
        "protocol": reports[0]["protocol"],
        "feature_timing_verified": all(r.get("feature_timing_verified", True) for r in reports),
        "datasets": [r["dataset"] for r in reports],
        "metrics": {k: float(np.mean([r["metrics"][k] for r in reports])) for k in allowed},
        "coverage": {r["dataset"]: r["coverage"] for r in reports},
    }


def cluster_bootstrap(values, clusters, repeats=1000, seed=17):
    # Repeated sampling seeds stay grouped under their prompt/source video.
    groups = {c: [] for c in clusters}
    for value, cluster in zip(values, clusters):
        groups[cluster].append(value)
    rng = np.random.default_rng(seed)
    keys = list(groups)
    samples = []
    for _ in range(repeats):
        chosen = rng.choice(keys, len(keys), replace=True)
        samples.append(np.mean([np.mean(groups[k]) for k in chosen]))
    return np.quantile(samples, [0.025, 0.975]).tolist()
