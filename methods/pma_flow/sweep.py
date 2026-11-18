"""Dev-split alpha sweep of a finished PMA-Flow run (proposal 5, RQ3: the alpha curve).

    python -m methods.pma_flow.sweep --run <outputs/runs/.../seed0_...> [--metrics m1,m2]

A run writes predictions.npz (alpha = method.sampler.alpha) and predictions_alpha<a>.npz for
every method.sampler.export_alphas. This scores each file with the shared metrics of
``slp.evaluation`` against the targets of the split the run was evaluated on, and writes
<run>/alpha_sweep.json. It refuses runs evaluated on the test split: alpha is chosen on val.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    from slp import config as config_lib
    from slp.data import load_dataset
    from slp.evaluation import evaluate

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--metrics", default=None, help="comma-separated (default: the run's eval.metrics)")
    a = ap.parse_args(argv)
    run = Path(a.run)
    meta = json.load(open(run / "meta.json"))
    if meta.get("eval_split") != "val":
        sys.exit("alpha is selected on the val split only; this run was evaluated on "
                 f"{meta.get('eval_split')!r}")
    cfg = config_lib.load_yaml(run / "config.yaml")
    metrics = a.metrics.split(",") if a.metrics else cfg["eval"]["metrics"]
    data = load_dataset(cfg).val
    names = [x["name"].replace("/", "__") for x in data.inputs]
    main_alpha = float(cfg["method"]["sampler"].get("alpha", 1.0))
    out = {}
    for f in sorted(run.glob("predictions*.npz")):
        m = re.fullmatch(r"predictions(?:_alpha([0-9.]+))?\.npz", f.name)
        if not m:
            continue
        alpha = float(m.group(1)) if m.group(1) else main_alpha
        z = np.load(f)
        preds = [z[n] for n in names]
        out[f"{alpha:g}"] = evaluate(preds, data.targets, metrics)
        print(f"alpha={alpha:g} " + " ".join(f"{k}={v:.4f}" for k, v in out[f"{alpha:g}"].items()), flush=True)
    json.dump(out, open(run / "alpha_sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
