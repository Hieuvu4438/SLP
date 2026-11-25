"""Run proposal 10 with python -m methods.relikflow."""

import argparse
import json
from pathlib import Path
import sys

import torch

from . import commands
from .data import Request
from .fixture import fixed_endpoint_sanity, make_fixture, sample_sanity
from .pipeline import Pipeline, correction_stage_name
from .runtime import config_load, json_write, npz_write
from .source import ARMS

COMMANDS = (
    "verify",
    "prepare",
    "fit-planner",
    "fit-direct-head",
    "build-sources",
    "fit-bank",
    "train-correction",
    "train-generator",
    "sample",
    "generate",
    "evaluate",
    "compare",
    "audit-ik",
    "benchmark",
)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        p = sub.add_parser(name)
        p.add_argument("--config")
        p.add_argument(
            "--profile",
            choices=("smoke", "pilot", "full"),
            default="smoke" if name == "verify" else "full",
        )
        p.add_argument("--output")
        p.add_argument("--seed", type=int)
        p.add_argument("--resume", action="store_true")
        if name == "prepare":
            p.add_argument("--manifest", required=True)
        if name in ("fit-planner", "fit-direct-head"):
            p.add_argument("--oof", type=int)
            p.add_argument("--fit-full", action="store_true")
        if name in ("build-sources", "train-correction", "train-generator", "sample", "evaluate"):
            p.add_argument("--arms")
        if name in ("sample", "evaluate", "compare", "fit-bank"):
            p.add_argument(
                "--split",
                choices=("train",) if name == "fit-bank" else ("dev", "test"),
                default="train" if name == "fit-bank" else "dev",
            )
        if name in ("generate", "sample", "evaluate", "compare", "benchmark"):
            p.add_argument("--nfe", type=int)
        if name in ("generate", "sample"):
            p.add_argument("--debug", action="store_true")
            p.add_argument("--retrained", action="store_true")
        if name == "evaluate":
            p.add_argument("--protocol")
        if name == "compare":
            p.add_argument("--baseline", required=True, choices=ARMS)
            p.add_argument("--challenger", required=True, choices=ARMS)
            p.add_argument("--run-roots", nargs="+")
        if name == "benchmark":
            p.add_argument("--arm", choices=ARMS)
            p.add_argument("--repetitions", type=int, default=100)
            p.add_argument("--warmup", type=int, default=20)
            p.add_argument("--batch-size", type=int, default=1)
        if name == "generate":
            p.add_argument("--text", required=True)
            p.add_argument("--source-language", required=True)
            p.add_argument("--target-sign-language", required=True)
            p.add_argument("--sample-id", required=True)
            p.add_argument("--sample-seed", type=int)
            p.add_argument("--arm", choices=ARMS)
            p.add_argument("--out", required=True)
    return ap


def verify(pipe):
    from . import smoke_slp_math as smoke

    smoke.check_layout_and_grid()
    q = smoke.np.array([0.1, smoke.np.sqrt(0.99), 0, 0])
    p = q * [1, -1, 1, 1]
    raw = smoke.np.array([0.1, 0.2, -0.3, 0.4])
    smoke.check_quaternion_geometry(q, p, raw)
    smoke.check_energy(q, p, raw)
    smoke.check_damped_ik()
    smoke.check_replanning(q, p, raw)
    arms = ["A-matched", "B", "C", "D", "E-rest", "E-shuffle"]
    pipe.prepare(make_fixture(pipe.root / "fixture"))
    pipe.open()
    folds = pipe.cfg["planner"]["oof_folds"]
    pipe.fit_planners(folds, True)
    pipe.fit_head(folds, True)
    pipe.build_sources(arms)
    pipe.fit_bank()
    for arm in ("A-matched", "C", "D"):
        pipe.correction_stage(arm)
    json_write(pipe.root / "endpoint_verification.json", fixed_endpoint_sanity(pipe))
    for arm in arms[:4]:
        pipe.generator_stage(arm)
    pipe.open(need_train=False)
    sample_sanity(pipe)
    arms.append("F-oracle-dev")
    commands.sample(pipe, "dev", arms, 5, True, False)
    commands.evaluate(pipe, "dev", arms, 5)
    commands.compare(pipe, "dev", "A-matched", "C", 5, [pipe.root])
    print(f"PASS synthetic A/B/C/D and dev E/F pipeline. Outputs at {pipe.root}")


def dispatch(args):
    cfg = config_load(args.config, args.profile)
    if args.profile == "smoke":
        torch.set_num_threads(1)
    if args.command == "verify" and args.profile != "smoke":
        raise ValueError("verify requires the synthetic smoke profile")
    default = Path(cfg["runtime"]["output"])
    path = args.output or (default / "smoke" if args.profile == "smoke" else default)
    pipe = Pipeline(cfg, path, args.seed, args.resume)
    if args.command == "verify":
        return verify(pipe)
    if args.command == "prepare":
        return pipe.prepare(args.manifest)
    inference = args.command in ("generate", "sample", "evaluate", "compare", "benchmark")
    pipe.open(need_train=not inference)
    folds = getattr(args, "oof", None) or cfg["planner"]["oof_folds"]
    arms = list(dict.fromkeys((getattr(args, "arms", None) or cfg["arm"]).split(",")))
    if any(arm not in ARMS for arm in arms):
        raise ValueError(f"Invalid experiment arms: {arms}")
    nfe = getattr(args, "nfe", None)
    nfe = cfg["evaluation"]["nfe_main"] if nfe is None else nfe
    if nfe < 1:
        raise ValueError("NFE must be positive")
    if inference and pipe.process.world > 1:
        raise ValueError("Inference and scoring commands use a single process")
    command = args.command
    if command == "fit-planner":
        pipe.fit_planners(folds, args.fit_full)
    elif command == "fit-direct-head":
        pipe.fit_head(folds, args.fit_full)
    elif command == "build-sources":
        pipe.build_sources(arms)
    elif command == "fit-bank":
        pipe.fit_bank()
    elif command == "train-correction":
        for arm in {correction_stage_name(arm): arm for arm in reversed(arms)}.values():
            pipe.correction_stage(arm)
    elif command == "train-generator":
        for arm in arms:
            pipe.generator_stage(arm)
    elif command == "sample":
        commands.sample(pipe, args.split, arms, nfe, args.debug, args.retrained)
    elif command == "evaluate":
        commands.evaluate(pipe, args.split, arms, nfe, args.protocol)
    elif command == "compare":
        commands.compare(
            pipe, args.split, args.baseline, args.challenger, nfe, args.run_roots or [pipe.root]
        )
    elif command == "audit-ik":
        from .diagnostics import audit_ik

        if pipe.process.world != 1:
            raise ValueError("Use a single process for the IK audit")
        audit_ik(pipe)
    elif command == "benchmark":
        from .diagnostics import benchmark

        rows = commands.split_rows(pipe, "dev")
        requests = [
            Request(
                r["text"], r["source_language"], r["target_sign_language"], r["clip_id"], pipe.seed
            )
            for r in rows
        ]
        benchmark(
            pipe,
            requests,
            args.arm or cfg["arm"],
            nfe,
            args.repetitions,
            args.warmup,
            args.batch_size,
        )
    elif command == "generate":
        arm = args.arm or cfg["arm"]
        request = Request(
            args.text,
            args.source_language,
            args.target_sign_language,
            args.sample_id,
            pipe.seed if args.sample_seed is None else args.sample_seed,
        )
        weights = "C" if arm.startswith("E-") and not args.retrained else arm
        prediction = pipe.sampler(arm, weights).generate(request, nfe, args.debug)
        npz_write(
            args.out,
            **{
                k: v.numpy() if isinstance(v, torch.Tensor) else v
                for k, v in prediction.items()
                if not isinstance(v, dict)
            },
        )
        print(json.dumps({"out": args.out, "T_hat": prediction["T_hat"], "fps": prediction["fps"]}))


def main():
    try:
        dispatch(parser().parse_args())
    except (TypeError, ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"RelIKFlow: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
