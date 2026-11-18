import time

import numpy as np
import torch


def benchmark(callback, warmup=100, updates=200, device="cpu"):
    if updates < 1 or warmup < 0:
        raise ValueError("Invalid profiling counts")
    device = torch.device(device)
    for _ in range(warmup):
        callback()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(updates):
        start = time.perf_counter()
        callback()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)
    return {
        "warmup": warmup,
        "updates": updates,
        "median_seconds": float(np.median(durations)),
        "p95_seconds": float(np.quantile(durations, 0.95)),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else 0,
    }
