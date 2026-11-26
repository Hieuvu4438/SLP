"""Choose sampler settings of a trained SignRAE run on the DEV split (never test).

Also a diagnostic tool: ``--lengths gt`` (oracle durations), ``--subset train`` (fit),
``--device cpu`` (when the GPU is busy).

    python -m methods.morae_slp.sweep --run outputs/runs/<exp>/morae_slp/<run_id> \
        --cfg 1.5,2.5,3.5,4.5,6.5 --steps 5,8 --solver rk4 --temperature 1.0,0.8 [--limit 256]

MoRAE selects guidance and the number of RK4 steps on validation; this script does the same with
the shared SOKE evaluator and the kinematic guardrails. Every setting reuses the per-utterance
noise of the run's seed, so rows differ only by the sampler. Results are appended to
``<run>/sweep.jsonl`` (one JSON row per setting).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from slp import config as config_lib
from slp.data import load_dataset
from slp.evaluation import evaluate
from slp.registry import get_method
from slp.tracking.run import RunContext

from .stages import resolve


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run directory with config.yaml and checkpoints/")
    p.add_argument("--checkpoint", default="model.pt", help="file inside <run>/checkpoints")
    p.add_argument("--cfg", default="1.5,2.5,3.5,4.5,6.5")
    p.add_argument("--steps", default="5")
    p.add_argument("--solver", default="rk4")
    p.add_argument("--temperature", default="1.0")
    p.add_argument("--limit", type=int, default=None, help="evenly spaced dev clips (default all)")
    p.add_argument("--export", action="store_true", help="keep predictions.npz of every setting")
    p.add_argument("--device", default=None, help="override method.device (e.g. cpu)")
    p.add_argument("--lengths", default=None,
                   help="'gt' = reference durations (oracle diagnostic) or an npz of predictions "
                        "whose durations are reused (matched comparison); default: duration head")
    p.add_argument("--subset", default="val", choices=["val", "train"],
                   help="train = fit diagnostic on training clips (never test)")
    p.add_argument("--tag", default="", help="suffix of the setting name")
    p.add_argument("--ensemble", default="1", help="pose-space mean of K samples, e.g. 1,8")
    p.add_argument("--length-scale", default="1.0",
                   help="P0.2 analysis: multiply predicted durations, e.g. 0.85,0.9,1,1.1")
    p.add_argument("--decoder", default="mean",
                   help="combination of the K samples: mean,medoid,dba (PLAN_V3 C5)")
    p.add_argument("--confirm", action="store_true",
                   help="score the dev clips NOT in the --limit tuning subset (PLAN_V3 §4 rule 1)")
    p.add_argument("--drop-train-duplicates", action="store_true",
                   help="drop dev clips whose normalised sentence occurs verbatim in train")
    p.add_argument("--data", default=None,
                   help="score on this dataset's dev split instead of the run's (e.g. a union "
                        "flow from pretrain.py --stage flow)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run = resolve(args.run)
    cfg = config_lib.load_yaml(run / "config.yaml")
    cfg.setdefault("method_name", "morae_slp")
    seed = cfg.get("seed", cfg.get("pretrain", {}).get("seed", 0))
    if args.data:
        data_cfg = config_lib.compose("morae_slp", args.data, [])
        if data_cfg["data"]["stats"] != cfg["data"]["stats"]:
            raise ValueError("--data uses a different pose normalisation than the run")
        cfg["data"] = data_cfg["data"]
    splits = load_dataset(cfg)
    data = splits.val if args.subset == "val" else splits.train  # never the test split
    idx = list(range(len(data)))
    if args.confirm:
        if args.limit is None or args.limit >= len(idx):
            raise SystemExit("--confirm needs --limit smaller than the dev size")
        # The tuning subset must be the run's own validation subset (best-checkpoint selection
        # and duration calibration used it), otherwise "confirm" clips were seen in tuning.
        run_limit = cfg["method"].get("val_limit")
        if args.subset == "val" and run_limit is not None and run_limit != args.limit:
            raise SystemExit(f"--confirm with --limit {args.limit} but the run selected on "
                             f"val_limit={run_limit} clips")
    clip_set = "all"
    if args.limit is not None and args.limit < len(idx):
        tune = np.linspace(0, len(idx) - 1, args.limit, dtype=int).tolist()
        idx = sorted(set(idx) - set(tune)) if args.confirm else tune
        clip_set = "confirm" if args.confirm else "tune"
    if args.drop_train_duplicates:
        from .retrieval import normalise

        seen = {normalise(item["text"]) for item in splits.train.inputs}
        idx = [i for i in idx if normalise(data.inputs[i]["text"]) not in seen]
        clip_set += "-nodup"
    inputs = [data.inputs[i] for i in idx]
    targets = [data.targets[i] for i in idx]
    lengths_npz = args.lengths
    if args.lengths == "gt":
        lengths_npz = run / "sweep" / f"gt_lengths_{args.subset}.npz"
        lengths_npz.parent.mkdir(parents=True, exist_ok=True)
        keys = [str(item.get("name", i)).replace("/", "__") for i, item in zip(idx, inputs)]
        np.savez(lengths_npz, **{k: np.zeros((len(t), t.shape[1]), np.float32)
                                for k, t in zip(keys, targets, strict=True)})
    metrics = cfg["eval"]["metrics"]
    logger = logging.getLogger("morae_sweep")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    decoders = [d.strip() for d in args.decoder.split(",") if d.strip()]
    grid = itertools.product([s.strip() for s in args.solver.split(",")], _ints(args.steps),
                             _floats(args.cfg), _floats(args.temperature), _ints(args.ensemble),
                             decoders, _floats(args.length_scale))
    for solver, steps, scale, temperature, members, decoder, length_scale in grid:
        if decoder != "mean" and members < 2:
            continue  # medoid / dba need K >= 2 samples
        tag = f"{solver}{steps}_w{scale:g}_tau{temperature:g}"
        tag += f"_ens{members}" if members > 1 else ""
        tag += f"_{decoder}" if decoder != "mean" else ""
        tag += f"_len{length_scale:g}" if length_scale != 1.0 else ""
        tag += f"_{clip_set}" if clip_set not in ("all", "tune") else ""
        tag += f"_{args.data}" if args.data else ""
        tag += (f"_{Path(args.checkpoint).stem}" if args.checkpoint != "model.pt" else "")
        tag += (f"_len-{'gt' if args.lengths == 'gt' else 'npz'}" if args.lengths else "")
        tag += (f"_{args.subset}" if args.subset != "val" else "") + args.tag
        out = run / "sweep" / tag
        out.mkdir(parents=True, exist_ok=True)
        setting = json.loads(json.dumps(cfg))
        setting["method"]["export_predictions"] = bool(args.export)
        setting["method"]["sampler"].update(solver=solver, steps=steps, cfg_scale=scale,
                                            temperature=temperature, ensemble=members,
                                            decoder=decoder, length_scale=length_scale,
                                            lengths_from_npz=str(lengths_npz) if lengths_npz
                                            else None)
        if args.device:
            setting["method"]["device"] = args.device
        model = get_method(cfg["method_name"])(setting, RunContext(out, seed, logger))
        model.load_checkpoint(run / "checkpoints" / args.checkpoint)
        start = time.time()
        preds = model.predict(inputs)
        row = {"setting": tag, "solver": solver, "steps": steps, "cfg_scale": scale,
               "temperature": temperature, "checkpoint": args.checkpoint, "clips": len(idx),
               "subset": args.subset, "lengths": args.lengths or "head", "ensemble": members,
               "decoder": decoder, "clip_set": clip_set, "length_scale": length_scale,
               "duration_bias": float(getattr(model, "duration_bias", 0.0)),
               "data": cfg["data"]["name"],
               "sample_seconds": round(time.time() - start, 1),
               "mean_frames": float(np.mean([len(p) for p in preds])),
               "ref_frames": float(np.mean([len(t) for t in targets])),
               **evaluate(preds, targets, metrics)}
        with open(run / "sweep.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        logger.info(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in row.items()))


if __name__ == "__main__":
    sys.exit(main())
