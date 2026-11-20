"""Score experiment arms against a reference run (docs/proposal7 07 §1.2, task C10).

    python -m methods.signmflow_plus.analysis.score --data phoenix_smplx \
        --reference outputs/runs/<baseline_run>/predictions.npz \
        --arm E3=outputs/runs/<e3_run>/predictions.npz --arm E4c=outputs/runs/<e4c_run>/predictions.npz \
        [--evaluator outputs/signmflow_plus/evaluator/phoenix_smplx/evaluator.pt] [--text-swap] \
        --out outputs/signmflow_plus/scores/phoenix_E3_E4c.json

Per arm and per reported DTW column (JPE body, JPE hand, PA hand, PA body-ua), on the val split:
  * raw paired difference to the reference, mean and 95% paired-bootstrap CI over sequences;
  * matched-speed effect: arm - (reference shrunk towards the mean pose by a, interpolated at the
    arm's low-pass speed; body columns use the body speed, hand columns the hand speed). An arm
    "beats the frontier" when the CI of this effect lies below 0;
  * M2: low-pass speed / acceleration ratios to GT; M3: gap to the mean-pose prediction;
  * M4 (--text-swap): DTW against the GT of another sentence of the closest length, minus DTW
    against the own GT (> 0 = the motion is specific to its sentence);
  * M1 (--evaluator): R@1/R@3 and Frechet distance of the evaluator embeddings (analysis/evaluator.py).

Only the val split is read (CLAUDE.md). Metric definitions come from slp.evaluation.metrics; the
reference / frontier / mean-pose / GT scores are cached in --cache-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
COLUMNS = ("jpe_body", "jpe_hand", "pa_hand", "pa_body_ua")
SPEED_OF = {"jpe_body": "body", "pa_body_ua": "body", "jpe_hand": "hand", "pa_hand": "hand"}


def load_npz(path, names) -> list[np.ndarray]:
    with np.load(path) as z:
        missing = [n for n in names if n.replace("/", "__") not in z.files]
        if missing:
            raise KeyError(f"{path}: {len(missing)} val sequences missing, e.g. {missing[:3]}")
        return [np.asarray(z[n.replace("/", "__")], np.float32) for n in names]


class Scorer:
    """Per-sequence values of the four reported DTW columns (SOKE protocol) + kinematics."""

    def __init__(self, gts):
        from slp.evaluation.smplx_fk import default_fk

        self.fk = default_fk()
        self.gts = gts
        self._gt_fk = {}

    def _fk(self, x):
        return tuple(a.cpu().numpy() for a in self.fk(np.asarray(x)))

    def gt_fk(self, i):
        if i not in self._gt_fk:
            self._gt_fk[i] = self._fk(self.gts[i])
        return self._gt_fk[i]

    def columns(self, preds, ref_index=None) -> dict[str, np.ndarray]:
        """ref_index[i]: which GT sequence prediction i is compared with (default: its own)."""
        from slp.evaluation.metrics import dtw_cost, pairwise_part_cost
        from slp.evaluation.smplx_fk import UPPER_BODY_IDX

        ub = list(UPPER_BODY_IDX)
        out = {c: [] for c in COLUMNS}
        for i, p in enumerate(preds):
            gj, gl, gr = self._fk(p)
            rj, rl, rr = self.gt_fk(i if ref_index is None else int(ref_index[i]))
            out["jpe_body"].append(dtw_cost(pairwise_part_cost(gj, rj, UPPER_BODY_IDX, "root"))[0])
            out["jpe_hand"].append((dtw_cost(pairwise_part_cost(gl, rl, None, "root"))[0]
                                    + dtw_cost(pairwise_part_cost(gr, rr, None, "root"))[0]) / 2)
            out["pa_hand"].append((dtw_cost(pairwise_part_cost(gl, rl, None, "procrustes"))[0]
                                   + dtw_cost(pairwise_part_cost(gr, rr, None, "procrustes"))[0]) / 2)
            out["pa_body_ua"].append(dtw_cost(pairwise_part_cost(gj[:, ub], rj[:, ub], None, "procrustes"))[0])
        return {k: np.asarray(v) for k, v in out.items()}

    @staticmethod
    def kinematics(seqs) -> dict[str, np.ndarray]:
        from slp.evaluation.metrics import soke_kinematics

        return soke_kinematics(list(seqs))


def text_swap_index(texts, lengths, seed=0) -> np.ndarray:
    """For each sequence, another sequence with a different sentence and the closest length."""
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths)
    out = np.empty(len(texts), dtype=int)
    for i, t in enumerate(texts):
        cand = np.array([j for j in range(len(texts)) if texts[j] != t])
        d = np.abs(lengths[cand] - lengths[i])
        best = cand[d == d.min()]
        out[i] = int(rng.choice(best))
    return out


def summarise(arm: dict, ref: dict, frontier: dict, gt_kin: dict, mean_pose: dict, shrink, n_boot=4000) -> dict:
    from slp.evaluation.metrics import frontier_effect, paired_bootstrap_ci

    res = {"dtw": {}, "delta_vs_reference": {}, "frontier_effect": {}, "kinematics": {}, "gap_to_mean_pose": {}}
    for part in ("body", "hand"):
        for st in ("speed", "accel"):
            k = f"{part}_{st}"
            res["kinematics"][f"{k}_ratio"] = float(arm["kin"][k].mean() / gt_kin[k].mean())
    for c in COLUMNS:
        a = arm["cols"][c]
        res["dtw"][c] = float(a.mean())
        res["delta_vs_reference"][c] = paired_bootstrap_ci(a - ref["cols"][c], n_boot)
        part = SPEED_OF[c]
        speeds = [float(f["kin"][f"{part}_speed"].mean()) for f in frontier]
        eff, clamped = frontier_effect(a, float(arm["kin"][f"{part}_speed"].mean()),
                                       np.stack([f["cols"][c] for f in frontier]), speeds)
        m, lo, hi = paired_bootstrap_ci(eff, n_boot)
        res["frontier_effect"][c] = {"mean": m, "ci95": [lo, hi], "beats_frontier": hi < 0,
                                     "speed_clamped": clamped}
        res["gap_to_mean_pose"][c] = float(mean_pose["cols"][c].mean() - a.mean())
    res["frontier_points"] = {f"a={a}": {"body_speed": float(f["kin"]["body_speed"].mean()),
                                         "hand_speed": float(f["kin"]["hand_speed"].mean()),
                                         **{c: float(f["cols"][c].mean()) for c in COLUMNS}}
                              for a, f in zip(shrink, frontier)}
    return res


def main(argv=None):
    sys.path.insert(0, str(ROOT / "src"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--reference", required=True, help="predictions.npz of the reference (baseline) run")
    ap.add_argument("--arm", action="append", default=[], help="NAME=predictions.npz (repeatable)")
    ap.add_argument("--shrink", default="0,0.05,0.1,0.15,0.2", help="shrink amounts a of the frontier")
    ap.add_argument("--text-swap", action="store_true", help="also compute M4 (one more pass per arm)")
    ap.add_argument("--evaluator", default=None, help="M1 evaluator checkpoint (analysis/evaluator.py)")
    ap.add_argument("--limit", type=int, default=None, help="first N val sequences only (quick check)")
    ap.add_argument("--cache-dir", default=str(ROOT / "outputs/signmflow_plus/score_cache"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from slp import config as config_lib
    from slp.data import load_dataset

    val = load_dataset(config_lib.compose("signmflow_plus", a.data)).val  # val split only
    n = len(val.inputs) if a.limit is None else min(a.limit, len(val.inputs))
    names = [str(x["name"]) for x in val.inputs[:n]]
    texts = [x["text"] for x in val.inputs[:n]]
    gts = [np.asarray(val.targets[i], np.float32) for i in range(n)]
    shrink = [float(s) for s in a.shrink.split(",")]
    sc = Scorer(gts)

    key = hashlib.sha1(f"{a.data}|{Path(a.reference).resolve()}|{Path(a.reference).stat().st_mtime}|{shrink}|{n}"
                       .encode()).hexdigest()[:12]
    cache = Path(a.cache_dir) / f"{a.data}_{key}.npz"
    ref_preds = load_npz(a.reference, names)
    if cache.exists():
        z = dict(np.load(cache, allow_pickle=True))
        stored = z["data"].item()
        ref, frontier, mean_pose, gt_kin = stored["ref"], stored["frontier"], stored["mean_pose"], stored["gt_kin"]
        print(f"reference / frontier loaded from {cache}")
    else:
        print(f"scoring the reference and its shrink frontier {shrink} on {n} val sequences ...")
        frontier = []
        for s in shrink:
            p = [r * (1.0 - s) for r in ref_preds]  # normalised features: the mean pose is 0
            frontier.append({"cols": sc.columns(p), "kin": sc.kinematics(p)})
        ref = frontier[shrink.index(0.0)] if 0.0 in shrink else {"cols": sc.columns(ref_preds),
                                                                 "kin": sc.kinematics(ref_preds)}
        mp = [np.zeros_like(r) for r in ref_preds]
        mean_pose = {"cols": sc.columns(mp)}
        gt_kin = sc.kinematics(gts)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, data=np.array({"ref": ref, "frontier": frontier, "mean_pose": mean_pose,
                                       "gt_kin": gt_kin}, dtype=object))

    swap = text_swap_index(texts, [len(g) for g in gts]) if a.text_swap else None
    evaluator = None
    if a.evaluator:
        from .evaluator import Evaluator

        evaluator = Evaluator.load(a.evaluator)
        gt_m = evaluator.embed_motion(gts)
        txt = evaluator.embed_text(texts)

    report = {"data": a.data, "n_sequences": n, "reference": a.reference, "shrink": shrink, "arms": {}}
    arms = [("reference", a.reference)] + [tuple(s.split("=", 1)) for s in a.arm]
    for name, path in arms:
        preds = ref_preds if name == "reference" else load_npz(path, names)
        arm = ref if name == "reference" else {"cols": sc.columns(preds), "kin": sc.kinematics(preds)}
        r = summarise(arm, ref, frontier, gt_kin, mean_pose, shrink)
        if swap is not None:
            sw = sc.columns(preds, ref_index=swap)
            r["text_swap_gap"] = {c: float((sw[c] - arm["cols"][c]).mean()) for c in COLUMNS}
        if evaluator is not None:
            from slp.evaluation.metrics import frechet_distance, r_precision

            m = evaluator.embed_motion(preds)
            r["retrieval"] = {**r_precision(txt, m), "fid_to_gt": frechet_distance(m, gt_m)}
        report["arms"][name] = r
        fe = r["frontier_effect"]
        print(f"{name:>12}: " + " | ".join(f"{c} {r['dtw'][c]:.3f} (eff {fe[c]['mean']:+.3f} "
                                            f"[{fe[c]['ci95'][0]:+.3f},{fe[c]['ci95'][1]:+.3f}])" for c in COLUMNS)
              + f" | speed body {r['kinematics']['body_speed_ratio']:.2f} hand {r['kinematics']['hand_speed_ratio']:.2f}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
