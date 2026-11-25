import os
import sys
import time
import json
import yaml
import types
from pathlib import Path
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor

from slp.data import load_dataset
from slp.evaluation.metrics import dtw_cost, pairwise_part_cost
from slp.evaluation.smplx_fk import UPPER_BODY_IDX, default_fk
from methods.motionhiflow_slp.method import Method

def eval_single_pair(args):
    p, t = args
    fk = default_fk()
    gj, gl, gr = fk(p)
    rj, rl, rr = fk(t)
    row = {}
    for m, align in (("jpe", "root"), ("pa", "procrustes")):
        row[f"{m}_all"] = dtw_cost(pairwise_part_cost(gj, rj, None, align))[0]
        row[f"{m}_body"] = dtw_cost(pairwise_part_cost(gj, rj, UPPER_BODY_IDX, align))[0]
        row[f"{m}_lhand"] = dtw_cost(pairwise_part_cost(gl, rl, None, align))[0]
        row[f"{m}_rhand"] = dtw_cost(pairwise_part_cost(gr, rr, None, align))[0]
    ub = list(UPPER_BODY_IDX)
    row["pa_body_ua"] = dtw_cost(pairwise_part_cost(gj[:, ub], rj[:, ub], None, "procrustes"))[0]
    return row

def parallel_evaluate(preds, targets, num_workers=8):
    args_list = [(np.asarray(p), np.asarray(t)) for p, t in zip(preds, targets)]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(eval_single_pair, args_list))

    out = {}
    for k in results[0].keys():
        out[k] = np.array([r[k] for r in results])

    metrics = {
        "soke_dtw_jpe_all": float(np.mean(out["jpe_all"])),
        "soke_dtw_jpe_body": float(np.mean(out["jpe_body"])),
        "soke_dtw_jpe_hand": float(np.mean((out["jpe_lhand"] + out["jpe_rhand"]) / 2)),
        "soke_dtw_pa_jpe_all": float(np.mean(out["pa_all"])),
        "soke_dtw_pa_jpe_body": float(np.mean(out["pa_body"])),
        "soke_dtw_pa_jpe_hand": float(np.mean((out["pa_lhand"] + out["pa_rhand"]) / 2)),
        "soke_dtw_pa_jpe_body_ua": float(np.mean(out["pa_body_ua"])),
    }
    return metrics

def main():
    run_dir = Path("outputs/runs/motionhiflow_phoenix_improve_single/motionhiflow_slp/seed0_20261123-151247-163949")
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Make sure reference predictions can be found
    ref_run = Path("outputs/runs/motionhiflow_phoenix_opt/motionhiflow_slp/seed0_20261123-110901-538484")
    cfg["method"]["sampler"]["lengths_from_npz"] = str(ref_run / "predictions.npz")
    cfg["method"]["sampler"]["batch_size"] = 16

    ctx = types.SimpleNamespace(seed=0, run_dir=run_dir, logger=types.SimpleNamespace(info=print))
    method = Method(cfg, ctx)
    method.load(run_dir / "checkpoints")

    data = load_dataset(cfg)
    val = data.val
    print(f"[{time.strftime('%H:%M:%S')}] Generating predictions for {len(val)} val clips...", flush=True)
    t0 = time.time()
    preds = method.predict(val.inputs)
    print(f"[{time.strftime('%H:%M:%S')}] Prediction finished in {time.time()-t0:.2f}s.", flush=True)

    keys = [str(item.get("name", i)).replace("/", "__") for i, item in enumerate(val.inputs)]
    arrays = dict(zip(keys, preds, strict=True))
    with open(run_dir / "predictions.npz", "wb") as f:
        np.savez_compressed(f, **arrays)
    print("Saved predictions.npz")

    print(f"[{time.strftime('%H:%M:%S')}] Evaluating SOKE DTW metrics with 8 workers...", flush=True)
    t1 = time.time()
    metrics = parallel_evaluate(preds, val.targets, num_workers=8)
    print(f"[{time.strftime('%H:%M:%S')}] Evaluation finished in {time.time()-t1:.2f}s.", flush=True)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
