"""Dev IK gate, geometry previews, and instrumented latency on fixed requests."""

import time

import numpy as np
import torch

from .data import digest, unpack133
from .fk import FINGERS, LANDMARKS, SIDES
from .pipeline import label_at
from .runtime import json_write, write_atomic
from .sampling import corpus_for
from .source import solve_ik


@torch.no_grad()
def audit_ik(pipe):
    clips = pipe.data["subsets"]["dev"]
    cfg = pipe.cfg
    requested = 100
    choice = sorted(clips, key=lambda c: digest([pipe.seed, "ik-oracle-audit", c.id]))[:requested]
    rest, fk = pipe.stats["rest"].to(pipe.device), pipe.geometry()
    records = []
    for c in choice:
        r, mask = label_at(c, pipe.grid(len(c.q)), fk, rest)
        _, errors, status = solve_ik(
            r, rest, cfg["ik"], fk, pipe.stats["scale"], cfg["planner"]["loss_weights"], mask
        )
        initial, final = errors[0][1], errors[-1][1]
        records.append(
            {
                "clip_id": c.id,
                "relative_residual_reduction": 1 - final / initial if initial > 1e-6 else None,
                "energy": errors,
                **status,
            }
        )
    fractions = [
        r["relative_residual_reduction"]
        for r in records
        if r["relative_residual_reduction"] is not None
    ]
    median = float(np.median(fractions)) if fractions else None
    status = "pending"
    if len(records) == requested and median is not None:
        status = "pass" if median >= 0.2 else "unmet"
    json_write(
        pipe.root / "ik_dev_audit.json",
        {"status": status, "median_reduction": median, "requested": requested, "clips": records},
    )


def render_preview(fk, q, path):
    """Render landmark articulation for first, middle and last frames."""
    from matplotlib import pyplot as plt

    indices = [0, len(q) // 2, len(q) - 1]
    with torch.no_grad():
        jt = fk(q[indices]).cpu().numpy()
    edges = [("pelvis", "spine3"), ("spine3", "nose")]
    for side in SIDES:
        shoulder, wrist = f"{side}_shoulder", f"{side}_wrist"
        edges.extend((("spine3", shoulder), (shoulder, wrist)))
        for finger in FINGERS:
            chain = [wrist, *(f"{side}_{finger}{k}" for k in (1, 2, 3, ""))]
            edges.extend(zip(chain[:-1], chain[1:]))
    scale = max(float(np.ptp(jt, axis=1).max()), 1e-6)
    center = jt.mean(axis=(0, 1))
    fig = plt.figure(figsize=(9, 3))
    try:
        for i, points in enumerate(jt):
            ax = fig.add_subplot(1, 3, i + 1, projection="3d")
            for a, b in edges:
                x, y, z = points[[LANDMARKS.index(a), LANDMARKS.index(b)]].T
                ax.plot(x, z, y, color="black", linewidth=1)
            ax.set_xlim(center[0] - scale, center[0] + scale)
            ax.set_ylim(center[2] - scale, center[2] + scale)
            ax.set_zlim(center[1] - scale, center[1] + scale)
            ax.set_title(f"Frame {indices[i]}")
            ax.set_axis_off()
        write_atomic(path, lambda p: fig.savefig(p, format="png", dpi=160))
    finally:
        plt.close(fig)


@torch.no_grad()
def benchmark(pipe, requests, arm, nfe, reps, warmup, batch_size):
    """Text-only latency on dynamically padded batches within length buckets."""
    if not requests or reps < 1 or warmup < 0 or batch_size < 1:
        raise ValueError("Invalid benchmark budget")
    sampler = pipe.sampler(arm)
    device = pipe.device
    fk = pipe.geometry()
    requests_by_length = {}
    for req in requests:
        length, _ = sampler.planner.predict(req, pipe.stats["cap"][corpus_for(req)])
        key = (length - 1) // 32 * 32 + 1
        requests_by_length.setdefault(key, []).append(req)
    grouped = {}
    for key, items in sorted(requests_by_length.items()):
        count = len(items)
        for i in range(warmup + reps):
            indexes = [(i * batch_size + k) % count for k in range(batch_size)]
            if i == warmup and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            begin = time.perf_counter()
            predictions = sampler.generate_batch([items[j] for j in indexes], nfe)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency = time.perf_counter() - begin
            if i < warmup:
                continue
            before = time.perf_counter()
            for pred in predictions:
                q, e = unpack133(pred["pose133"].to(device))
                fk.score_joints(q, e)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            grouped.setdefault(key, []).append(
                {
                    "latency_seconds": latency,
                    "text_seconds": predictions[0]["text_seconds"],
                    "refiner_seconds": predictions[0]["refiner_seconds"],
                    "plan_seconds": sum(p["plan_seconds"] for p in predictions),
                    "source_seconds": sum(p["source_seconds"] for p in predictions),
                    "fk_seconds": time.perf_counter() - before,
                    "peak_memory": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0,
                }
            )
    grouped = {
        key: {
            "count": len(values),
            **{k: np.quantile([x[k] for x in values], [0.5, 0.95]).tolist() for k in values[0]},
        }
        for key, values in grouped.items()
    }
    json_write(
        pipe.root / f"latency_{arm}_nfe_{nfe}_batch_{batch_size}.json",
        {
            "arm": arm,
            "batch": batch_size,
            "requests": len(requests),
            "device": str(device),
            "nfe": nfe,
            "warmup": warmup,
            "measured": reps,
            "quantiles": [0.5, 0.95],
            "by_length_bucket": grouped,
            "timebases": {k: pipe.stats["fps"][k] for k in {corpus_for(r) for r in requests}},
        },
    )
