"""python -m methods.race_sign.cli --help"""

import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from .adapters import file_hash
from .madis_adapter import load_base_adapter
from .config import RaceConfig
from .data import RecordDataset, collate_inputs
from .geometry import SMPLXGeometry, SkeletonGeometry
from .preparation import prepare_manifest, cache_candidates
from .training import train_decoder, train_selector, load_checkpoint
from .pipeline import RacePipeline


def geometry_from_path(path, device):
    p = Path(path)
    if p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as data:
            geom = (
                SkeletonGeometry(data["parents"], data["rest_joints"])
                if {"parents", "rest_joints"} <= set(data.files)
                else SMPLXGeometry(p)
            )
    else:
        geom = SMPLXGeometry(p)
    return geom.to(device).eval()


def model_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    return RaceConfig.from_dict(cfg.get("model", cfg.get("method", {}).get("model", cfg)))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="RACE-Sign staged training and native-length inference"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser(
        "prepare", help="Build explicit training caches from raw physical poses"
    )
    prep.add_argument("--raw", required=True)
    prep.add_argument("--assets", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--role", required=True)
    prep.add_argument("--mode", choices=["teacher", "generated"], default="teacher")
    prep.add_argument("--align-target", action="store_true")
    prep.add_argument("--geometry")
    prep.add_argument("--corruption", type=float, default=0)
    prep.add_argument("--repeat-probability", type=float, default=0)
    prep.add_argument("--part-dropout", type=float, default=0)
    dec = commands.add_parser("train-decoder")
    dec.add_argument("--config", required=True)
    dec.add_argument("--train", required=True)
    dec.add_argument("--val", required=True)
    dec.add_argument("--output", required=True)
    dec.add_argument("--geometry", required=True)
    dec.add_argument("--epochs", type=int, default=100)
    dec.add_argument("--batch-size", type=int, default=8)
    dec.add_argument("--lr", type=float, default=2e-4)
    dec.add_argument("--window", type=int, default=64)
    dec.add_argument("--halo", type=int, default=8)
    dec.add_argument("--resume")
    dec.add_argument("--initialize", help="Trained decoder checkpoint for stage-B fine-tuning")
    dec.add_argument("--robustness", action="store_true")
    dec.add_argument("--no-text", action="store_true")
    dec.add_argument("--no-coupling", action="store_true")
    cache = commands.add_parser("cache-candidates")
    cache.add_argument("--checkpoint", required=True)
    cache.add_argument("--manifest", required=True)
    cache.add_argument("--role", required=True)
    cache.add_argument("--output", required=True)
    cache.add_argument("--geometry", required=True)
    cache.add_argument("--steps", type=int)
    sel = commands.add_parser("train-selector")
    sel.add_argument("--checkpoint", required=True)
    sel.add_argument("--train", required=True)
    sel.add_argument("--val", required=True)
    sel.add_argument("--calibration", required=True)
    sel.add_argument("--output", required=True)
    sel.add_argument("--geometry", required=True)
    sel.add_argument("--epochs", type=int, default=30)
    sel.add_argument("--batch-size", type=int, default=8)
    sel.add_argument("--lr", type=float, default=2e-4)
    sel.add_argument("--whole-sentence", action="store_true")
    infer = commands.add_parser("infer")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--assets", required=True)
    infer.add_argument("--inputs", required=True, help="JSONL: sample_id, text, language only")
    infer.add_argument("--geometry", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument(
        "--policy", choices=["base", "flow", "coupled", "sentence", "random"], default="coupled"
    )
    infer.add_argument("--steps", type=int)
    infer.add_argument("--guidance", type=float, default=1)
    for cmd in (prep, dec, cache, sel, infer):
        cmd.add_argument("--device", default="cpu")
        cmd.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        base = load_base_adapter(args.assets, device=args.device)
        geom = geometry_from_path(args.geometry, args.device) if args.geometry else None
        result = prepare_manifest(
            args.raw,
            base,
            args.output,
            args.role,
            mode=args.mode,
            geometry=geom,
            align=args.align_target,
            corruption=args.corruption,
            repeat_probability=args.repeat_probability,
            part_dropout=args.part_dropout,
            seed=args.seed,
        )
        print(result)
    elif args.command == "train-decoder":
        cfg = model_config(args.config)
        train = RecordDataset(
            args.train, "robustness_train" if args.robustness else "decoder_train"
        )
        val = RecordDataset(args.val, "decoder_val")
        _, _, history = train_decoder(
            cfg,
            train,
            val,
            geometry_from_path(args.geometry, args.device),
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
            window=args.window,
            halo=args.halo,
            resume=args.resume,
            initialize_from=args.initialize,
            use_text=not args.no_text,
            coupled=not args.no_coupling,
        )
        (Path(args.output) / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(history[-1]))
    elif args.command == "cache-candidates":
        print(
            cache_candidates(
                args.checkpoint,
                args.manifest,
                args.role,
                geometry_from_path(args.geometry, args.device),
                args.output,
                seed=args.seed,
                steps=args.steps,
                device=args.device,
            )
        )
    elif args.command == "train-selector":
        _, history = train_selector(
            args.checkpoint,
            RecordDataset(args.train, "selector_train"),
            RecordDataset(args.val, "selector_val"),
            RecordDataset(args.calibration, "selector_calibration"),
            geometry_from_path(args.geometry, args.device),
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
            whole_sentence=args.whole_sentence,
        )
        Path(args.output).with_suffix(".history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )
        print(json.dumps(history[-1]))
    else:
        decoder, norm, selector, checkpoint = load_checkpoint(args.checkpoint, args.device)
        if args.policy in ("sentence", "coupled") and checkpoint["provenance"].get(
            "selector_whole_sentence", False
        ) != (args.policy == "sentence"):
            raise ValueError("Inference selection unit must match selector training/calibration")
        pipe = RacePipeline(
            decoder, norm, geometry_from_path(args.geometry, args.device), selector
        ).eval()
        base = load_base_adapter(args.assets, device=args.device)
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        lines = [json.loads(s) for s in Path(args.inputs).read_text().splitlines() if s.strip()]
        if any(set(row) != {"sample_id", "text", "language"} for row in lines):
            raise ValueError(
                "Text-only inference rejects targets, reference lengths and extra fields"
            )
        if len({r["sample_id"] for r in lines}) != len(lines):
            raise ValueError("Duplicate inference IDs")
        records = []
        for i, row in enumerate(lines):
            inputs = base.prepare(row["text"], row["language"], row["sample_id"])
            pred = pipe.predict(
                collate_inputs([inputs]).to(args.device),
                policy=args.policy,
                seed=args.seed,
                steps=args.steps,
                guidance=args.guidance,
            )
            n = int(pred.lengths[0])
            file = f"{i:08d}.npz"
            np.savez_compressed(
                out / file,
                pose=pred.poses[0, :n].cpu().numpy(),
                baseline=pred.base[0, :n].cpu().numpy(),
                candidate=pred.candidate[0, :n].cpu().numpy(),
                weights=pred.weights[0, :n].cpu().numpy(),
            )
            records.append(
                dict(sample_id=row["sample_id"], path=file, native_length=n, nfe=pred.nfe)
            )
        (out / "manifest.json").write_text(
            json.dumps(
                dict(
                    checkpoint_sha256=file_hash(args.checkpoint),
                    policy=args.policy,
                    seed=args.seed,
                    predictions=records,
                ),
                indent=2,
            )
            + "\n"
        )
        print(f"Wrote {len(records)} predictions to {out}")


if __name__ == "__main__":
    main()
