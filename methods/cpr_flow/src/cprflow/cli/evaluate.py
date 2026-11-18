import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

from cprflow.cli.common import print_json, save_json
from cprflow.data.cache import file_hash
from cprflow.data.dataset import MotionDataset
from cprflow.data.manifest import write_jsonl
from cprflow.evaluation.aggregate import cluster_bootstrap
from cprflow.evaluation.backtranslation import backtranslation_status
from cprflow.evaluation.geometry import geometry_metrics
from cprflow.geometry.body_model import BodyModel
from cprflow.sampling.generator import MotionGenerator
from cprflow.types import DATASETS, GenerationRequest


def evaluate(
    generator,
    dataset,
    split="dev",
    duration_source="predicted",
    diagnostic_only=False,
    output=None,
    final_test=False,
    relation_source="generated",
):
    if relation_source not in ("generated", "oracle"):
        raise ValueError("Unknown relation source")
    if relation_source == "oracle" and not diagnostic_only:
        raise ValueError("Oracle relation requires --diagnostic-only")
    oracle = duration_source == "oracle" or relation_source == "oracle"
    if duration_source == "oracle" and not diagnostic_only:
        raise ValueError("Oracle duration requires --diagnostic-only")
    if split == "test":
        from cprflow.evaluation.protocol import authorize_final_test

        authorize_final_test(generator.config, dataset, output, final_test)
    cfg = generator.config
    manifest = Path(cfg["paths"]["manifests"]) / f"{split}.jsonl"
    ds = MotionDataset(manifest, split, [dataset])
    device = next(generator.model.parameters()).device
    from cprflow.training.artifacts import normalizer, scales, stats_path
    from cprflow.evaluation.temporal import temporal_metrics

    rn = normalizer(cfg, "relation", device)
    mn = normalizer(cfg, "motion", device)
    activity_scales = scales(cfg)
    body = BodyModel(cfg["paths"]["smplx_model"], cfg["geometry"]["fk_chunk_frames"]).to(device)
    failures = []
    records = []
    duration_prefix = "oracle_" if oracle else "controlled_v1_"
    for index in range(len(ds)):
        try:
            row = ds[index]
        except (ValueError, KeyError, OSError) as exc:
            failures.extend(
                {"uid": ds.rows[index]["uid"], "seed": seed, "reason": f"data_failure: {exc}"}
                for seed in cfg["eval"]["sample_seeds"]
            )
            continue
        if row["frames"] > cfg["data"]["max_generation_frames"]:
            failures.extend(
                {"uid": row["uid"], "seed": seed, "reason": "unsupported_reference_length_gt_2048"}
                for seed in cfg["eval"]["sample_seeds"]
            )
            continue
        for seed in cfg["eval"]["sample_seeds"]:
            request = GenerationRequest(row["text"], DATASETS[dataset][0], seed)
            try:
                for attempt in range(2):
                    try:
                        if oracle:
                            diagnostic_relation = None
                            if relation_source == "oracle":
                                from cprflow.geometry.relations import relation_operator

                                x = torch.as_tensor(row["x"], device=device)
                                diagnostic_relation = relation_operator(
                                    x[None], torch.tensor([len(x)], device=device), body, rn
                                )[0]
                            result = generator._generate(
                                request,
                                diagnostic_frames=row["frames"]
                                if duration_source == "oracle"
                                else None,
                                diagnostic_relation=diagnostic_relation,
                            )
                        else:
                            result = generator.generate(request)
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if attempt == 1:
                            raise
                result.metadata["generation_batch_size"] = 1
                result.metadata["oom_retries"] = attempt
                predicted = torch.as_tensor(result.canonical_motion, device=device)
                target = torch.as_tensor(row["x"], device=device)
                metrics = geometry_metrics(predicted, target, body, rn)
                metrics.update(
                    temporal_metrics(
                        predicted, target, body, mn, activity_scales, row["timestamps"]
                    )
                )
                if oracle:
                    metrics = {
                        key.replace("controlled_v1_", "oracle_", 1): v for key, v in metrics.items()
                    }
                metrics.update(
                    {
                        duration_prefix + "duration_relative_error": abs(
                            len(predicted) - len(target)
                        )
                        / len(target),
                        duration_prefix + "duration_mae_frames": abs(len(predicted) - len(target)),
                        duration_prefix + "duration_mae_seconds": abs(
                            (len(predicted) - 1) / 30
                            - float(row["timestamps"][-1] - row["timestamps"][0])
                        ),
                        duration_prefix + "duration_clipped": float(
                            result.metadata["duration_clipped"]
                        ),
                    }
                )
                records.append(
                    {
                        "uid": row["uid"],
                        "source_video_id": row["source_video_id"],
                        "seed": seed,
                        "metrics": metrics,
                        "cost": result.metadata,
                    }
                )
            except (ValueError, RuntimeError) as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                failures.append({"uid": row["uid"], "seed": seed, "reason": str(exc)})
    requested = len(ds) * len(cfg["eval"]["sample_seeds"])
    means = (
        {
            key: float(np.mean([r["metrics"][key] for r in records if key in r["metrics"]]))
            for key in set().union(*(r["metrics"] for r in records))
        }
        if records
        else {}
    )
    ci = {}
    if records:
        ci = {
            key: cluster_bootstrap(
                [r["metrics"][key] for r in records if key in r["metrics"]],
                [r["source_video_id"] or r["uid"] for r in records if key in r["metrics"]],
            )
            for key in means
        }
    timing_verified = all(r.get("feature_timing_verified", True) for r in ds.rows)
    report = {
        "protocol": "exploratory_operator_requested_v1"
        if cfg["run"].get("manual_gate_policy") == "operator_waived"
        else "controlled_v1",
        "feature_timing_verified": timing_verified,
        "temporal_units_status": "verified_seconds"
        if timing_verified
        else "assumed_clock_not_verified_physical_seconds",
        "geometry_review_policy": cfg["run"].get("manual_gate_policy", "required"),
        "dataset": dataset,
        "split": split,
        "run_id": cfg["run"]["id"],
        "checkpoint_sha256": generator.metadata["checkpoint_sha256"],
        "manifest_sha256": file_hash(manifest),
        "duration_source": duration_source,
        "inference_correction": generator.metadata.get("inference_correction", "off"),
        "inference_teacher_sha256": generator.metadata.get("inference_teacher_sha256"),
        "inference_calibration_sha256": generator.metadata.get("inference_calibration_sha256"),
        "oracle": oracle,
        "relation_source": relation_source,
        "oracle_relation_protocol": "normalized_time_linear_endpoint_clamped_path_v1"
        if relation_source == "oracle"
        else None,
        "profile": cfg["data"]["profile"],
        "metric_version": "exact_min_sum_dtw_path_average_v1",
        "stats_sha256": {kind: file_hash(stats_path(cfg, kind)) for kind in ("motion", "relation")},
        "asset_sha256": file_hash(cfg["paths"]["smplx_model"]),
        "fitting_revision": json.loads(Path(cfg["paths"]["sources_lock"]).read_text())["dataset"][
            "revision"
        ],
        "evaluator_sha256": file_hash(Path(__file__).parents[1] / "evaluation/geometry.py"),
        "seeds": cfg["eval"]["sample_seeds"],
        "units": "metres",
        "num_requested": requested,
        "num_successful": len(records),
        "coverage": len(records) / requested if requested else 0,
        "metrics": means,
        "metric_num_samples": {key: sum(key in r["metrics"] for r in records) for key in means},
        "bootstrap_prompt_ci95": ci,
        "unavailable_metrics": [
            "official_backtranslation",
            "contact_F1",
            "mesh_penetration",
            "independent_sign_FID",
            "human_comprehension",
        ],
        "backtranslation_status": backtranslation_status(),
        "failure_reasons": dict(collections.Counter(r["reason"] for r in failures)),
        "full_test_failure_coverage": "not_accessed"
        if split != "test"
        else len(records) / max(1, requested),
        "synthetic": cfg["run"]["synthetic"],
    }
    if output:
        save_json(output, report)
        write_jsonl(str(output) + ".samples.jsonl", records)
        write_jsonl(str(output) + ".failures.jsonl", failures)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Per-corpus evaluation, predicted-duration main protocol"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--duration-source", default="predicted", choices=("predicted", "oracle"))
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--relation-source", choices=("generated", "oracle"), default="generated")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    parser.add_argument("--guard-mode", choices=("off", "unrestricted", "feature_only", "semantic"))
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--teacher-calibration")
    args = parser.parse_args()
    generator = MotionGenerator.from_checkpoint(
        args.checkpoint,
        args.device,
        args.guard_mode,
        args.teacher_checkpoint,
        args.teacher_calibration,
    )
    output = args.output or str(
        Path(generator.config["run"]["out_dir"])
        / generator.config["run"]["id"]
        / f"metrics/{args.dataset}_{args.split}_{args.duration_source}.json"
    )
    print_json(
        evaluate(
            generator,
            args.dataset,
            args.split,
            args.duration_source,
            args.diagnostic_only,
            output,
            args.final_test,
            args.relation_source,
        )
    )


if __name__ == "__main__":
    main()
