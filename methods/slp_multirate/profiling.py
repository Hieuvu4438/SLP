"""Synchronized end-to-end sampling latency and operator FLOP estimates."""

import platform

import numpy as np
import torch

from .data import write_json
from .training import runtime_versions


def profile_text(cascade, prompts, out, warmup=10, repeats=50, estimate_flops=True):
    if warmup < 0 or repeats < 1:
        raise ValueError("Invalid warmup/measurement counts")
    texts = [v["text"] for v in prompts]
    ids = [v["clip_id"] for v in prompts]
    gpu = cascade.device.type == "cuda"
    for _ in range(warmup):
        cascade.generate(texts, ids)
    if gpu:
        torch.cuda.synchronize(cascade.device)
        torch.cuda.reset_peak_memory_stats(cascade.device)
    measurements = []
    for _ in range(repeats):
        result = cascade.generate(texts, ids)
        measurements.append(result["latency_ms"])
        lengths, nfe = result["lengths"].tolist(), result["nfe"]
        del result
    latency = {name: dict(zip(("p50", "p95"), np.percentile(
        [x[name] for x in measurements], (50, 95)
    ).tolist())) for name in measurements[-1]}
    memory = torch.cuda.max_memory_allocated(cascade.device) if gpu else None
    op_flops = None
    if estimate_flops:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if gpu:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True) as profile:
            cascade.generate(texts, ids)
        op_flops = {entry.key: entry.flops for entry in profile.key_averages() if entry.flops > 0}
    cfg = cascade.cfg
    stride = cfg["body"]["stride"]
    result = {"latency_ms": latency, "warmup": warmup, "repeats": repeats, "batch_size": len(ids),
              "predicted_lengths": lengths, "actual_nfe": nfe, "precision": "fp32",
              "peak_cuda_memory_bytes": memory, "text_hidden_cached": False,
              "fine_attention": "dense_kernel_with_local_mask", "body_stride": stride,
              "frame_token_evaluations": sum(((t + stride - 1) // stride) * nfe["body"]
                                             + t * nfe["fine"] for t in lengths),
              "flops": sum(op_flops.values()) if op_flops else None, "flops_by_operator": op_flops,
              "flops_method": "torch.profiler with_flops for supported operators",
              "runtime": runtime_versions(cascade.device), "host": platform.node(),
              "seed": cascade.seed, "prompt_ids": ids,
              "checkpoint_sha256": {name: file.sha256 for name, file in cascade.frozen.items()}}
    write_json(out, result)
    return result


def compare_compute(baseline_path, candidate_path, out, metric="total_p50_ms"):
    """Build fixed-compute comparison metadata from comparable profile reports."""
    import json
    from pathlib import Path

    baseline, candidate = [json.loads(Path(p).read_bytes()) for p in (baseline_path, candidate_path)]
    for field in ("runtime", "host", "batch_size", "predicted_lengths", "precision", "prompt_ids",
                  "warmup", "repeats", "text_hidden_cached", "seed"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"Compute profiles differ in {field}")
    if baseline["warmup"] < 10 or baseline["repeats"] < 50:
        raise ValueError("Compute comparison requires at least 10 warmups and 50 measurements")
    if metric == "total_p50_ms":
        x, y = [z["latency_ms"]["total"]["p50"] for z in (baseline, candidate)]
    elif metric == "flops":
        x, y = [z["flops"] for z in (baseline, candidate)]
    else:
        raise ValueError("Unknown profile comparison metric")
    if x is None or y is None or min(x, y) <= 0:
        raise ValueError("Both compute measurements must be present and positive")
    delta = (y - x) / x
    result = {"metric": metric, "baseline": x, "candidate": y, "relative_difference": delta,
              "within_10_percent": abs(delta) <= .10,
              "profiles": [str(baseline_path), str(candidate_path)],
              "baseline_nfe": baseline["actual_nfe"], "candidate_nfe": candidate["actual_nfe"]}
    write_json(out, result)
    return result
