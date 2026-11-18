import json
from pathlib import Path

from cprflow.cli.common import config_parser, print_json, resolved
from cprflow.config import validate
from cprflow.data.dataset import MotionDataset
from cprflow.data.sampler import BalancedBatchSampler
from cprflow.training.distributed import setup
from cprflow.training.engine import Trainer, seed_everything
from cprflow.training.logging import record_run
from cprflow.training.tasks import TaskObjective, make_model, provenance, training_rows
from cprflow.training.validation import validator


def train(config, resume=None, warm_start=None, until=None):
    rank, world, device = setup(config["run"]["device"], config["distributed"]["backend"])
    validate(config, world)
    if resume and warm_start:
        raise ValueError("--resume and --warm-start are mutually exclusive")
    if (
        config["run"]["task"] == "flow"
        and config["run"]["stage"] == "continuation"
        and not (resume or warm_start)
    ):
        raise ValueError(
            "Continuation requires --warm-start parent EMA or --resume; optimizer reset is explicit"
        )
    if not config["run"]["synthetic"]:
        from cprflow.training.gates import require_geometry_review

        require_geometry_review(config)
    seed_everything(config["seed"] + rank, config["run"]["strict_reproducibility"])
    ds = MotionDataset(
        Path(config["paths"]["manifests"]) / "train.jsonl",
        "train",
        config["data"]["datasets"],
        config["data"]["max_train_frames"],
    )
    ds.rows = training_rows(config, ds.rows)
    sampler = BalancedBatchSampler(
        ds.rows,
        config["train"]["effective_batch"],
        config["seed"],
        config["data"]["datasets"],
        config["data"]["q_dataset"],
        config["data"]["bucket_boundaries"],
    )
    objective = TaskObjective(config, device)
    model = make_model(config, objective.motion_norm).to(device)
    meta = provenance(config)
    trainer = Trainer(model, config, meta, device, rank, world)
    if resume:
        trainer.resume(resume, sampler)
    else:
        if trainer.directory.exists() and (trainer.directory / "checkpoints/last.pt").exists():
            raise FileExistsError("Run already has a checkpoint; use --resume or a new run ID")
        if warm_start:
            trainer.warm_start(warm_start)
        if rank == 0:
            record_run(trainer.directory, config, meta)
            lock = json.loads(Path(config["paths"]["sources_lock"]).read_text())
            (trainer.directory / "sources.lock.json").write_text(json.dumps(lock, indent=2))
            if warm_start:
                from cprflow.data.cache import file_hash

                (trainer.directory / "parent.json").write_text(
                    json.dumps(
                        {
                            "path": str(Path(warm_start).resolve()),
                            "sha256": file_hash(warm_start),
                            "weights": "EMA",
                            "optimizer": "reset",
                            "parent_global_step": trainer.parent_global_step,
                        },
                        indent=2,
                    )
                )
    import signal

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def stop_requested(signum, frame):
        trainer.stop_requested = True

    for sig in previous:
        signal.signal(sig, stop_requested)
    try:
        trainer.run(
            ds,
            sampler,
            objective,
            validator(config, objective),
            until=until,
            logical_objective=config["run"]["task"] == "teacher",
        )
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    completed = not trainer.stop_requested and (
        trainer.step >= config["train"]["max_updates"]
        or config["run"]["task"] == "duration"
        and trainer.bad_validations >= config["train"]["early_stop_patience"]
    )
    if completed and rank == 0:
        from cprflow.cli.common import save_json

        save_json(
            trainer.directory / "training_complete.json",
            {
                "run_id": config["run"]["id"],
                "successful_updates": trainer.step,
                "early_stopped": trainer.step < config["train"]["max_updates"],
                "config_hash": meta["config"],
                "synthetic": config["run"]["synthetic"],
            },
        )
    return {
        "run_dir": str(trainer.directory),
        "successful_updates": trainer.step,
        "synthetic": config["run"]["synthetic"],
        "world_size": world,
        "interrupted": trainer.stop_requested,
    }


def main():
    parser = config_parser("Train or exact-resume a staged CPR-Flow model")
    parser.add_argument("--task", choices=("ae", "flow", "duration", "teacher"))
    parser.add_argument("--resume")
    parser.add_argument("--warm-start")
    parser.add_argument(
        "--until",
        type=int,
        help="Stop after this absolute successful update without changing the schedule",
    )
    args = parser.parse_args()
    config = resolved(args)
    if args.task:
        config["run"]["task"] = args.task
    result = train(config, args.resume, args.warm_start, args.until)
    print_json(result)
    if result["interrupted"]:
        raise SystemExit(75)


if __name__ == "__main__":
    main()
