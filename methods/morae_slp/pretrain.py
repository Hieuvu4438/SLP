"""Self-supervised stages on the UNION of train splits (no text, never the test split).

    python -m methods.morae_slp.pretrain --stage jepa \
        --datasets phoenix_smplx,csl_smplx,how2sign_smplx_soke --out outputs/morae_slp/pretrain/jepa_union
    python -m methods.morae_slp.pretrain --stage ae \
        --datasets phoenix_smplx,csl_smplx,how2sign_smplx_soke --out outputs/morae_slp/pretrain/ae_union_d64 \
        --init-jepa outputs/morae_slp/pretrain/jepa_union/jepa.pt  [method.latent.dim=32 ...]
    python -m methods.morae_slp.pretrain --stage flow \
        --datasets phoenix_smplx,csl_smplx,how2sign_smplx_soke --out outputs/morae_slp/pretrain/flow_union \
        --init-ae outputs/morae_slp/pretrain/ae_union_d64/ae.pt  [method.text.finetune=true ...]

Only ``.train`` (training) and ``.val`` (loss curves / AE checkpoint selection) of each dataset are
read. Hyper-parameters come from methods/morae_slp/config.yaml plus ``key=value`` overrides.
The outputs plug into a per-dataset run via ``method.init_jepa`` / ``method.init_ae`` /
``method.init_flow`` (stage ``flow`` trains the text-conditioned generator on the union of
train splits, in the union latent standardisation of ``--init-ae``; mBART language tokens
distinguish German, Chinese and English).
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from datetime import datetime
from pathlib import Path

import torch

from slp import config as config_lib
from slp.data import load_dataset
from slp.tracking.run import RunContext
from slp.utils.git import git_diff, git_info
from slp.utils.logging import get_logger
from slp.utils.seed import set_seed

from .autoencoder import LatentSpace
from .data import ConcatSplit
from .stages import (calibrate, load_jepa, resolve, save_space, space_spec, train_autoencoder,
                     train_jepa)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("jepa", "ae", "flow"), required=True)
    p.add_argument("--datasets", required=True, help="comma-separated data config names")
    p.add_argument("--out", required=True, help="output directory (created; must be new or empty)")
    p.add_argument("--init-jepa", default=None, help="jepa.pt for --stage ae with latent kind jepa")
    p.add_argument("--init-ae", default=None, help="calibrated union ae.pt for --stage flow")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("overrides", nargs="*", help="config overrides key.sub=value")
    return p.parse_args(argv)


def load_union(names: list[str], overrides: list[str]):
    trains, vals, stats = [], [], set()
    cfg = None
    for name in names:
        cfg = config_lib.compose("morae_slp", name, overrides)
        stats.add(cfg["data"]["stats"])
        data = load_dataset(cfg)
        trains.append(data.train)
        vals.append(data.val)
    if len(stats) != 1:
        raise ValueError(f"All datasets must share one pose normalisation, got {sorted(stats)}")
    return cfg, ConcatSplit(trains), ConcatSplit(vals), stats.pop()


def main(argv=None) -> None:
    args = parse_args(argv)
    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    out = resolve(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty; choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed, True)
    overrides = list(args.overrides)
    if args.stage == "flow":
        if not args.init_ae:
            sys.exit("--stage flow needs --init-ae (a calibrated union ae.pt)")
        # The flow must live in the union standardisation of init_ae (no per-dataset recalibration).
        overrides += [f"method.init_ae={args.init_ae}", "method.latent.recalibrate=false"]
    cfg, train, val, stats = load_union(names, overrides)
    hp = cfg["method"]
    logger = get_logger(f"morae_pretrain.{out.name}", out / "log.txt")
    ctx = RunContext(out, args.seed, logger)
    git = git_info()
    config_lib.dump({**cfg, "pretrain": {"stage": args.stage, "datasets": names,
                                         "init_jepa": args.init_jepa, "init_ae": args.init_ae,
                                         "seed": args.seed}},
                    out / "config.yaml")
    (out / "meta.json").write_text(json.dumps({
        "stage": args.stage, "datasets": names, "train_clips": len(train), "val_clips": len(val),
        "git": git, "command": " ".join(sys.argv), "host": socket.gethostname(),
        "started_at": datetime.now().isoformat(timespec="seconds")}, indent=2))
    if git["dirty"]:
        (out / "code.diff").write_text(git_diff())

    requested = hp["device"]
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if requested == "auto" else requested)
    precision = hp["precision"]
    train_limit = hp["debug_limit"]
    val_limit = hp["val_limit"] * len(names) if hp["val_limit"] is not None else None
    if train_limit is not None:
        val_limit = min(train_limit, val_limit) if val_limit is not None else train_limit
    logger.info("Union of %s: %d train / %d val clips on %s", names, len(train), len(val), device)

    if args.stage == "flow":
        from .method import Method

        cfg["method"]["val_limit"] = val_limit
        model = Method(cfg, ctx)
        model.fit(train, val)  # writes out/checkpoints/model.pt (+ snapshots) and ae.pt
        logger.info("Saved %s", ctx.ckpt_dir / "model.pt")
        return

    if args.stage == "jepa":
        train_jepa(train, val, hp, device, precision, ctx, out / "jepa.pt", train_limit, val_limit)
        logger.info("Saved %s", out / "jepa.pt")
        return

    kind = hp["latent"]["kind"]
    arch = None
    if kind == "jepa":
        if not args.init_jepa:
            sys.exit("--stage ae with latent.kind=jepa needs --init-jepa")
        jepa, arch = load_jepa(args.init_jepa, device)
    spec = space_spec(hp, arch)
    torch.manual_seed(args.seed)
    space = LatentSpace(**spec).to(device)
    if kind == "jepa":
        space.load_teacher(jepa)
        del jepa
    train_autoencoder(space, spec, train, val, hp, device, precision, ctx, out / "ae.pt",
                      train_limit, val_limit, stats)
    space.eval().requires_grad_(False)
    info = calibrate(space, train, hp, device, train_limit)
    save_space(space, spec, out / "ae.pt", extra={"calibration": info, "data": names,
                                                  "init_jepa": args.init_jepa})
    logger.info("Saved %s (calibration %s)", out / "ae.pt", info)


if __name__ == "__main__":
    main()
