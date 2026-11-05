"""Explicit train/sample/inspect commands. --help imports neither torch nor model weights."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(description="Structured temporal-plan text-to-sign method")
    commands = result.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="run stages only when explicitly requested")
    train.add_argument("--train", required=True, help="raw133 train JSONL manifest")
    train.add_argument("--val", help="raw133 validation JSONL manifest")
    train.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="cpu")
    assets = train.add_mutually_exclusive_group()
    assets.add_argument("--smplx-model-dir")
    assets.add_argument("--rotation-offsets", help="explicit [41,3] .npy, FK-disabled ablations only")
    train.add_argument("--smplx-ext", choices=("npz", "pkl"), default="npz")
    train.add_argument("--allow-download", action="store_true", help="allow pinned mT5 download")
    sample = commands.add_parser("sample", help="text-only generation from a completed checkpoint")
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--inputs", required=True, help="JSONL with name,text,lang only")
    sample.add_argument("--output", required=True)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--device", default="cpu")
    sample.add_argument("--allow-download", action="store_true")
    inspect = commands.add_parser("inspect", help="show checkpoint metadata without generation")
    inspect.add_argument("--checkpoint", required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    device = getattr(args, "device", "cpu")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.command == "inspect":
        from .checkpoint import read_checkpoint
        checkpoint = read_checkpoint(args.checkpoint)
        print(json.dumps({key: checkpoint[key] for key in (
            "format_version", "model_config", "text", "stage", "metadata",
        )}, indent=2))
        return
    if args.command == "sample":
        from .application import export_predictions, predict_records
        from .checkpoint import load_checkpoint
        records = [json.loads(line) for line in Path(args.inputs).read_text().splitlines()
                   if line.strip()]
        model, tokenizer, metadata = load_checkpoint(
            args.checkpoint, device=device, allow_download=args.allow_download,
            require_completed=True,
        )
        count = export_predictions(predict_records(model, tokenizer, records, metadata,
                                                    seed=args.seed), args.output)
        print(json.dumps({"predictions": count, "output": args.output}))
        return
    import torch
    import yaml
    from .application import create_system
    from .data import load_manifest, validate_train_val
    from .training import TrainingConfig, TrainingRunner

    document = yaml.safe_load(Path(args.config).read_text()) or {}
    config = document.get("method", document)
    training = {**config.get("training", {}), "device": device}
    train_cfg = TrainingConfig(**training)
    torch.manual_seed(train_cfg.seed)
    train = load_manifest(args.train, split="train")
    val = load_manifest(args.val, split="val") if args.val else None
    validate_train_val(train, val)
    model, tokenizer, fk = create_system(
        train, config.get("model", {}), smplx_model_dir=args.smplx_model_dir,
        rotation_offsets=args.rotation_offsets, smplx_ext=args.smplx_ext,
        allow_download=args.allow_download, text_width=config.get("text_width", 512),
    )
    checkpoint = TrainingRunner(model, tokenizer, train_cfg, args.output, fk=fk).fit(train, val)
    print(json.dumps({"checkpoint": str(checkpoint)}))


if __name__ == "__main__":
    main()
