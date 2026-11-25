"""Evaluate trained MotionHiFlow model on Phoenix test split using a single thread/process."""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import json
import yaml
import types
import numpy as np
from pathlib import Path
import torch

torch.set_num_threads(1)

from slp.data import load_dataset
from slp.evaluation import evaluate
from methods.motionhiflow_slp.method import Method

def main():
    run_dir = Path("outputs/runs/motionhiflow_phoenix_opt/motionhiflow_slp/seed0_20261123-110901-538484")
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["method"]["sampler"]["batch_size"] = 16
    ctx = types.SimpleNamespace(seed=0, run_dir=run_dir)
    method = Method(cfg, ctx)
    method.load(run_dir / "checkpoints")

    data = load_dataset(cfg)
    test = data.test
    print(f"[{time.strftime('%H:%M:%S')}] Starting test prediction on {len(test)} clips...", flush=True)
    t0 = time.time()
    preds = method.predict(test.inputs)
    print(f"[{time.strftime('%H:%M:%S')}] Prediction finished in {time.time()-t0:.2f}s. Saving predictions_test.npz...", flush=True)

    keys = [str(item.get("name", i)).replace("/", "__") for i, item in enumerate(test.inputs)]
    arrays = dict(zip(keys, preds, strict=True))
    with open(run_dir / "predictions_test.npz", "wb") as f:
        np.savez_compressed(f, **arrays)

    print(f"[{time.strftime('%H:%M:%S')}] Calculating DTW-JPE metrics on {len(test)} clips (single process)...", flush=True)
    t1 = time.time()
    metrics = evaluate(preds, test.targets, cfg["eval"]["metrics"])
    print(f"[{time.strftime('%H:%M:%S')}] DTW Evaluation finished in {(time.time()-t1)/60:.2f} mins.", flush=True)

    out_path = run_dir / "metrics_test.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] Results saved to {out_path}:\n", json.dumps(metrics, indent=2), flush=True)

if __name__ == "__main__":
    main()
