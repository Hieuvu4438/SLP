#!/usr/bin/env python
"""Single entry point to train + evaluate ANY method.

    python scripts/run.py --method <name> --data <dataset> --name <experiment> \
        --seeds 0,1,2 [--split val|test] [key.sub=value ...]

    python scripts/run.py --list

Default evaluation split is `val`. Evaluating on `test` requires a clean git
tree (so the number is reproducible from a commit) and is appended to
results/test_access_log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slp import config as config_lib  # noqa: E402
from slp.data import load_dataset  # noqa: E402
from slp.evaluation import evaluate  # noqa: E402
from slp.paths import TEST_ACCESS_LOG  # noqa: E402
from slp.registry import get_method, list_methods, method_status  # noqa: E402
from slp.tracking import create_run  # noqa: E402
from slp.utils.git import git_info  # noqa: E402
from slp.utils.seed import set_seed  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--method", help="folder name under methods/")
    p.add_argument("--data", default=None, help="dataset name (configs/data/<name>.yaml)")
    p.add_argument("--name", default="debug", help="experiment name (groups runs)")
    p.add_argument("--seeds", default="0", help="comma-separated seeds, e.g. 0,1,2")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--allow-dirty", action="store_true",
                   help="allow --split test with uncommitted changes (NOT for paper numbers)")
    p.add_argument("--list", action="store_true", help="list methods and exit")
    p.add_argument("overrides", nargs="*", help="config overrides key.sub=value")
    return p.parse_args(argv)


def run_one(cfg: dict, seed: int, experiment: str, split: str) -> dict:
    set_seed(seed, cfg.get("train", {}).get("deterministic", True))
    ctx = create_run(cfg, seed, experiment, split)
    data = load_dataset(cfg)
    method = get_method(cfg["method_name"])(cfg, ctx)

    method.fit(data.train, data.val)
    method.save()

    target_split = data.get(split)
    preds = method.predict(target_split.inputs)
    metrics = evaluate(preds, target_split.targets, cfg["eval"]["metrics"])
    ctx.save_final_metrics(metrics)
    ctx.logger.info(f"[{split}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    if split == "test":
        TEST_ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(TEST_ACCESS_LOG, "a") as f:
            f.write(json.dumps({
                "time": datetime.now().isoformat(timespec="seconds"),
                "experiment": experiment, "method": cfg["method_name"],
                "data": cfg["data"]["name"], "seed": seed,
                "commit": git_info()["commit"], "run_dir": str(ctx.run_dir),
                "metrics": metrics,
            }) + "\n")
    return metrics


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        for m in list_methods():
            print(f"{m:30s} {method_status(m)}")
        return
    if not args.method:
        sys.exit("--method is required (use --list to see methods)")

    if args.split == "test" and git_info()["dirty"] and not args.allow_dirty:
        sys.exit("Refusing to evaluate on TEST with a dirty git tree. Commit first "
                 "(or pass --allow-dirty for a non-reportable check).")

    cfg = config_lib.compose(args.method, args.data, args.overrides)
    seeds = [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        run_one(cfg, seed, args.name, args.split)


if __name__ == "__main__":
    main()
