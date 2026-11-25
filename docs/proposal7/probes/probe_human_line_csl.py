"""P0.S-H (docs/proposal7/08 §2.4, §6): human-to-human DTW reference line on CSL-Daily dev.

No model is run. Ground truth only, plus the saved v2 predictions of the replica (val split):
- pairs of *different signers* performing the same dev sentence (280 texts);
- SOKE protocol scores (src/slp/evaluation/metrics.py) of signer A against signer B, both directions;
- the v2 replica's saved predictions and a static mean pose, scored on the same sequences;
- paired bootstrap over texts.

    OMP_NUM_THREADS=4 python docs/proposal7/probes/probe_human_line_csl.py \
        --pred outputs/runs/signmflow_v2_main_csl_smplx/signmflow/<seed0_run>/predictions.npz
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from slp.data.sign_smplx import LazyPoses, _csl, load_stats  # noqa: E402
from slp.evaluation.metrics import soke_protocol_scores  # noqa: E402

COLS = ("jpe_body", "jpe_hand", "pa_body_ua", "pa_hand")


def columns(preds, targets) -> np.ndarray:
    s = soke_protocol_scores(preds, targets)
    return np.stack([s["jpe_body"], (s["jpe_lhand"] + s["jpe_rhand"]) / 2, s["pa_body_ua"],
                     (s["pa_lhand"] + s["pa_rhand"]) / 2], -1)  # [N, 4]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default=str(ROOT / "docs/proposal7/probes/probe_human_line_csl.json"))
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args(argv)

    root = ROOT / "data/processed/madis/CSL-Daily"
    rows = _csl(root, "val")
    mean, std = load_stats(str(ROOT / "third_party/SOKE/deps/stats"))
    poses = LazyPoses([r[1] for r in rows], mean, std, fps=None, subsample="madis")
    meta = [r[0] for r in rows]
    by_text = collections.defaultdict(list)
    for i, m in enumerate(meta):
        by_text[m["text"]].append(i)
    pairs = []
    for t, idx in sorted(by_text.items()):
        signers = {}
        for i in sorted(idx, key=lambda i: meta[i]["name"]):
            signers.setdefault(meta[i].get("name").split("_")[1], i)  # P<signer> field of the name
        if len(signers) >= 2:
            i, j = list(signers.values())[:2]
            pairs.append((i, j))
    print(f"{len(pairs)} dev texts with two different signers")

    pred = np.load(a.pred)
    A = [np.asarray(poses[i], np.float32) for i, _ in pairs]
    B = [np.asarray(poses[j], np.float32) for _, j in pairs]
    human = (columns(A, B) + columns(B, A)) / 2  # [P, 4], symmetric
    pa = [pred[meta[i]["name"]] for i, _ in pairs]
    pb = [pred[meta[j]["name"]] for _, j in pairs]
    model = (columns(pa, A) + columns(pb, B)) / 2  # per text: mean over its two sequences
    za = [np.zeros_like(x) for x in A]
    zb = [np.zeros_like(x) for x in B]
    meanpose = (columns(za, A) + columns(zb, B)) / 2  # normalised 0 = training mean pose, GT length

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(pairs), (a.boot, len(pairs)))
    diff = model - human
    ci = np.percentile(diff[idx].mean(1), [2.5, 97.5], axis=0)
    lens = np.array([[len(x), len(y)] for x, y in zip(A, B)], float)
    res = {
        "n_texts": len(pairs),
        "columns": COLS,
        "human_A_vs_B": human.mean(0).round(4).tolist(),
        "v2_replica": model.mean(0).round(4).tolist(),
        "static_mean_pose_gt_length": meanpose.mean(0).round(4).tolist(),
        "v2_minus_human": diff.mean(0).round(4).tolist(),
        "v2_minus_human_ci95": ci.round(4).tolist(),
        "frac_texts_v2_better_than_human": (diff < 0).mean(0).round(3).tolist(),
        "length_ratio_between_signers_median_p90": np.percentile(lens.max(1) / lens.min(1), [50, 90]).round(3).tolist(),
        "pred_file": a.pred,
    }
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
