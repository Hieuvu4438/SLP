"""Matched-speed comparison against a reference's shrink-to-mean frontier (PLAN_V3 §4, P0.0).

    python -m methods.morae_slp.frontier --data csl_smplx \\
        --reference outputs/runs/signmflow_v2_main_csl_smplx/signmflow/<run>/predictions.npz \\
        --arm v3=outputs/runs/<exp>/morae_slp/<run>/predictions.npz [--arm ...] \\
        --clips confirm --limit 256 [--cluster-sentences] --out frontier.json

DEV only. For every arm and each reported column (JPE body, JPE hand, PA hand, PA body-ua):
  * raw paired difference to the reference;
  * matched-speed effect: arm - reference shrunk towards the normalised mean pose (features x
    (1 - a)) and interpolated at the arm's low-pass speed (body columns: body speed, hand
    columns: hand speed), with ``slp.evaluation.metrics.frontier_effect``;
  * bootstrap mean / 95 % interval / two-sided p-value of both, per sequence or clustered by
    sentence (``--cluster-sentences``, needed for CSL-Daily which repeats sentences). The p-value
    is the approximate bootstrap sign p-value 2 min(P(b <= 0), P(b >= 0)) (interval inversion,
    not a null-centred test), and the matched effect treats the frontier interpolation weight as
    fixed (its uncertainty is ignored): use the intervals for decisions, p only for Holm;
  * Holm-adjusted p-values over the arms of the run (per column and per effect type);
  * kinematic ratios (low-pass speed / acceleration over GT).
Clip sets match ``sweep.py``: ``tune`` = the ``--limit`` evenly spaced clips used for tuning,
``confirm`` = the remaining dev clips (acceptance decisions), ``all``; ``--drop-train-duplicates``
removes dev sentences that occur verbatim in train (retrieval arms).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

COLUMNS = ("jpe_body", "jpe_hand", "pa_hand", "pa_body_ua")
SPEED_OF = {"jpe_body": "body", "pa_body_ua": "body", "jpe_hand": "hand", "pa_hand": "hand"}


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error control)."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted.tolist()


def clip_indices(n: int, mode: str, limit: int | None) -> list[int]:
    if mode == "all" or limit is None or limit >= n:
        if mode == "confirm":
            raise ValueError("--clips confirm needs --limit smaller than the dev size")
        return list(range(n))
    tune = np.linspace(0, n - 1, limit, dtype=int).tolist()
    return tune if mode == "tune" else sorted(set(range(n)) - set(tune))


def columns(preds, targets) -> dict[str, np.ndarray]:
    from slp.evaluation.metrics import soke_protocol_scores

    s = soke_protocol_scores(preds, targets)
    return {"jpe_body": s["jpe_body"], "jpe_hand": (s["jpe_lhand"] + s["jpe_rhand"]) / 2,
            "pa_hand": (s["pa_lhand"] + s["pa_rhand"]) / 2, "pa_body_ua": s["pa_body_ua"]}


def load_arm(path: str, keys: list[str]) -> list[np.ndarray]:
    with np.load(path) as z:
        missing = [k for k in keys if k not in z.files]
        if missing:
            raise KeyError(f"{path}: {len(missing)} clips missing, e.g. {missing[:3]}")
        return [np.asarray(z[k], np.float32) for k in keys]


def summarise(arm, ref, frontier, speeds, gt_kin, clusters, n_boot) -> dict:
    from slp.evaluation.metrics import cluster_bootstrap_ci, frontier_effect

    out = {"dtw": {}, "delta": {}, "matched": {}, "kinematics": {}}
    for c in COLUMNS:
        part = SPEED_OF[c]
        out["dtw"][c] = float(arm["cols"][c].mean())
        m, lo, hi, p = cluster_bootstrap_ci(arm["cols"][c] - ref["cols"][c], clusters, n_boot)
        out["delta"][c] = {"mean": m, "ci95": [lo, hi], "p": p}
        eff, clamped = frontier_effect(arm["cols"][c], float(arm["kin"][f"{part}_speed"].mean()),
                                       np.stack([f["cols"][c] for f in frontier]),
                                       speeds[part])
        m, lo, hi, p = cluster_bootstrap_ci(eff, clusters, n_boot)
        out["matched"][c] = {"mean": m, "ci95": [lo, hi], "p": p, "clamped": bool(clamped)}
    for part in ("body", "hand"):
        for stat in ("speed", "accel"):
            key = f"{part}_{stat}"
            out["kinematics"][f"{key}_ratio"] = float(arm["kin"][key].mean()
                                                      / max(gt_kin[key].mean(), 1e-12))
    return out


def main(argv=None) -> None:
    from slp import config as config_lib
    from slp.data import load_dataset
    from slp.evaluation.metrics import soke_kinematics
    from slp.paths import PROJECT_ROOT

    from .retrieval import normalise

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--reference", required=True, help="predictions.npz of the reference run")
    ap.add_argument("--arm", action="append", default=[], help="NAME=predictions.npz")
    ap.add_argument("--clips", default="all", choices=["all", "tune", "confirm"])
    ap.add_argument("--limit", type=int, default=None, help="size of the tuning subset")
    ap.add_argument("--drop-train-duplicates", action="store_true")
    ap.add_argument("--cluster-sentences", action="store_true")
    ap.add_argument("--shrink", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--cache-dir", default=str(PROJECT_ROOT / "outputs/morae_slp/score_cache"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    splits = load_dataset(config_lib.compose("morae_slp", a.data, []))
    val = splits.val  # development split only
    idx = clip_indices(len(val), a.clips, a.limit)
    if a.drop_train_duplicates:
        seen = {normalise(item["text"]) for item in splits.train.inputs}
        idx = [i for i in idx if normalise(val.inputs[i]["text"]) not in seen]
    keys = [str(val.inputs[i]["name"]).replace("/", "__") for i in idx]
    gts = [np.asarray(val.targets[i], np.float32) for i in idx]
    sentences = [normalise(val.inputs[i]["text"]) for i in idx]
    clusters = sentences if a.cluster_sentences else None
    shrink = [float(x) for x in a.shrink.split(",")]
    if 0.0 not in shrink:
        shrink = [0.0] + shrink

    ref_preds = load_arm(a.reference, keys)
    ref_path = Path(a.reference).resolve()
    key = hashlib.sha1(f"{a.data}|{ref_path}|{ref_path.stat().st_mtime}|{shrink}|{idx}"
                       .encode()).hexdigest()[:16]
    cache = Path(a.cache_dir) / f"frontier_{a.data}_{key}.npz"
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)["data"].item()
        frontier, gt_kin = stored["frontier"], stored["gt_kin"]
        print(f"reference frontier loaded from {cache}")
    else:
        frontier = []
        for s in shrink:  # normalised features: the mean pose is 0
            p = [r * (1.0 - s) for r in ref_preds]
            frontier.append({"cols": columns(p, gts), "kin": soke_kinematics(p)})
        gt_kin = soke_kinematics(gts)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, data=np.array({"frontier": frontier, "gt_kin": gt_kin}, dtype=object))
    ref = frontier[shrink.index(0.0)]
    speeds = {part: [float(f["kin"][f"{part}_speed"].mean()) for f in frontier]
              for part in ("body", "hand")}

    report = {"data": a.data, "clips": a.clips, "limit": a.limit, "n_sequences": len(idx),
              "n_sentences": len(set(sentences)), "clustered": bool(a.cluster_sentences),
              "drop_train_duplicates": bool(a.drop_train_duplicates),
              "reference": str(a.reference), "shrink": shrink,
              "reference_dtw": {c: float(ref["cols"][c].mean()) for c in COLUMNS},
              "reference_kinematics": {f"{p}_speed_ratio": float(ref["kin"][f"{p}_speed"].mean()
                                                                 / gt_kin[f"{p}_speed"].mean())
                                       for p in ("body", "hand")},
              "arms": {}}
    for spec in a.arm:
        name, path = spec.split("=", 1)
        preds = load_arm(path, keys)
        arm = {"cols": columns(preds, gts), "kin": soke_kinematics(preds)}
        report["arms"][name] = summarise(arm, ref, frontier, speeds, gt_kin, clusters, a.n_boot)
    names = list(report["arms"])
    for kind in ("delta", "matched"):
        for c in COLUMNS:
            adj = holm([report["arms"][n][kind][c]["p"] for n in names]) if names else []
            for n, q in zip(names, adj):
                report["arms"][n][kind][c]["p_holm"] = q
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    for n in names:
        r = report["arms"][n]
        print(f"{n}: " + " | ".join(
            f"{c} {r['dtw'][c]:.3f} matched {r['matched'][c]['mean']:+.3f} "
            f"[{r['matched'][c]['ci95'][0]:+.3f},{r['matched'][c]['ci95'][1]:+.3f}]"
            for c in COLUMNS)
            + f" | speed b/h {r['kinematics']['body_speed_ratio']:.2f}/"
              f"{r['kinematics']['hand_speed_ratio']:.2f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
