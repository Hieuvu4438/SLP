"""Exact min-sum DTW, deterministic diagonal/vertical/horizontal ties, path-average report."""

import numpy as np


def dtw_jpe(pred, target):
    p, t = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.ndim != 3 or t.ndim != 3 or p.shape[1:] != t.shape[1:] or not len(p) or not len(t):
        raise ValueError("Expected nonempty [T,J,3] trajectories with matching joints")
    if not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("Nonfinite DTW input")
    previous = np.full(len(t) + 1, np.inf)
    previous[0] = 0.0
    back = np.zeros((len(p), len(t)), dtype=np.uint8)
    for i in range(len(p)):
        current = np.full(len(t) + 1, np.inf)
        costs = np.linalg.norm(p[i, None] - t, axis=-1).mean(-1)
        for j in range(len(t)):
            options = (previous[j], previous[j + 1], current[j])
            direction = int(np.argmin(options))
            current[j + 1] = costs[j] + options[direction]
            back[i, j] = direction
        previous = current
    i, j = len(p) - 1, len(t) - 1
    path = []
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        direction = back[i, j]
        i -= int(direction in (0, 1))
        j -= int(direction in (0, 2))
    path.reverse()
    return {
        "distance": float(previous[-1] / len(path)),
        "cumulative_cost": float(previous[-1]),
        "path_length": len(path),
        "path": path,
    }
