"""Minimum-Bayes-risk style decoders over K sampled sequences (PLAN_V3 C5).

A frame-pair cost is the mean joint distance between two skeletons given as [T, J, 3] point
arrays (the caller passes forward-kinematics points in the frame used by the SOKE protocol:
upper body relative to the pelvis, hands relative to their wrists). The DTW here is the method's
own implementation used *for decoding*; reported metrics still come only from the shared
evaluator (CLAUDE.md).

* medoid: the sample with the smallest total DTW cost to the other samples (an MBR decision
  restricted to the sample set; Kumar & Byrne 2004). It is one real sample, not an average.
* dba:    DTW barycenter averaging (Petitjean et al. 2011): starting from the medoid, align every
  sample to the current centre and replace each centre frame by the mean of the frames aligned
  to it; repeat. The centre keeps the medoid's length.
Both are metric-aware decoders and must be reported as such.
"""

from __future__ import annotations

import numpy as np


def frame_costs(a: np.ndarray, b: np.ndarray, chunk: int = 32) -> np.ndarray:
    """[Ta, Tb] mean joint distance between point sequences a [Ta, J, 3] and b [Tb, J, 3].
    Computed in float32 over row chunks (a [Ta, Tb, J, 3] float64 block would need ~150 MB at
    T = 350)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 3 or b.ndim != 3 or a.shape[1:] != b.shape[1:]:
        raise ValueError(f"expected [T, J, 3] point sequences, got {a.shape} and {b.shape}")
    out = np.empty((len(a), len(b)), dtype=np.float64)
    for s in range(0, len(a), chunk):
        d = a[s:s + chunk, None] - b[None]
        out[s:s + chunk] = np.sqrt((d * d).sum(-1)).mean(-1)
    return out


def dtw(cost: np.ndarray, path: bool = False):
    """Accumulated DTW cost with steps (1,1), (1,0), (0,1); optionally the optimal path.
    Ties prefer the diagonal, then the step in the first sequence."""
    cost = np.asarray(cost, dtype=np.float64)
    r, c = cost.shape
    if r == 0 or c == 0:
        raise ValueError("DTW inputs must contain at least one frame")
    acc = np.full((r + 1, c + 1), np.inf)
    acc[0, 0] = 0.0
    move = np.zeros((r, c), dtype=np.uint8)
    for d in range(2, r + c + 1):  # anti-diagonals i + j = d (1-based)
        i = np.arange(max(1, d - c), min(r, d - 1) + 1)
        j = d - i
        cand = np.stack([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        best = cand.argmin(0)
        acc[i, j] = cost[i - 1, j - 1] + cand[best, np.arange(len(i))]
        move[i - 1, j - 1] = best
    if not path:
        return float(acc[r, c])
    pairs = []
    i, j = r, c
    while i > 0 and j > 0:
        pairs.append((i - 1, j - 1))
        step = move[i - 1, j - 1]
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return float(acc[r, c]), pairs[::-1]


def medoid(points: list[np.ndarray]) -> tuple[int, np.ndarray]:
    """Index of the DTW medoid of K point sequences and the [K, K] cost matrix."""
    k = len(points)
    if k == 0:
        raise ValueError("medoid needs at least one sequence")
    costs = np.zeros((k, k))
    for a in range(k):
        for b in range(a + 1, k):
            costs[a, b] = costs[b, a] = dtw(frame_costs(points[a], points[b]))
    return int(costs.sum(1).argmin()), costs


def dba(sequences: list[np.ndarray], points: list[np.ndarray], to_points, start: int,
        iterations: int = 5) -> np.ndarray:
    """DTW barycenter of ``sequences`` (pose features [T_k, C]) aligned on their ``points``.

    ``to_points`` maps a pose-feature sequence to its [T, J, 3] points (used to re-align the
    updated centre); ``start`` is the index of the initial centre (normally the medoid)."""
    if len(sequences) != len(points) or not sequences:
        raise ValueError("dba needs one point sequence per pose sequence")
    centre = np.asarray(sequences[start], dtype=np.float64).copy()
    centre_points = points[start]
    for _ in range(max(1, iterations)):
        total = np.zeros_like(centre)
        count = np.zeros(len(centre))
        for seq, pts in zip(sequences, points):
            _, pairs = dtw(frame_costs(centre_points, pts), path=True)
            for i, j in pairs:
                total[i] += seq[j]
                count[i] += 1
        updated = total / count[:, None]  # every centre frame is on every path (count >= 1)
        if np.allclose(updated, centre):
            centre = updated
            break
        centre = updated
        centre_points = to_points(centre)
    return centre.astype(np.float32)
