from pathlib import Path

import torch

from cprflow.cli.common import config_parser, print_json, resolved, save_json
from cprflow.data.dataset import MotionDataset
from cprflow.evaluation.efficiency import benchmark
from cprflow.training.engine import make_optimizer
from cprflow.training.tasks import TaskObjective, make_model


def main():
    parser = config_parser("Measure real train forward/backward cost per length bucket")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--updates", type=int, default=200)
    args = parser.parse_args()
    cfg = resolved(args)
    device = cfg["run"]["device"]
    ds = MotionDataset(
        Path(cfg["paths"]["manifests"]) / "train.jsonl",
        "train",
        cfg["data"]["datasets"],
        cfg["data"]["max_train_frames"],
    )
    objective = TaskObjective(cfg, device)
    model = make_model(cfg, objective.motion_norm).to(device)
    optimizer = make_optimizer(model, cfg["train"]["lr"], cfg["train"]["weight_decay"])
    from cprflow.training.precision import Precision
    from cprflow.training.ema import EMA

    precision = Precision(device, cfg["train"]["precision"])
    ema = EMA(model, cfg["train"]["ema_decay"])
    results = {}
    lower = 0
    for upper in cfg["data"]["bucket_boundaries"]:
        indices = [i for i, r in enumerate(ds.rows) if lower < r["frames"] <= upper]
        lower = upper
        if not indices:
            results[str(upper)] = {"status": "no_train_clips"}
            continue
        rows = [ds[i] for i in indices[: cfg["train"]["microbatch_per_rank"]]]

        def update():
            optimizer.zero_grad(set_to_none=True)
            with precision.autocast():
                loss, _ = objective(model, rows, 40000, rows)
            precision.scaler.scale(loss.mean()).backward()
            precision.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            old_scale = precision.scaler.get_scale()
            precision.scaler.step(optimizer)
            precision.scaler.update()
            if precision.scaler.get_scale() >= old_scale:
                ema.update(model)

        try:
            stats = benchmark(update, args.warmup, args.updates, device)
            stats["precision"] = str(precision.dtype or torch.float32)
            stats["parameters"] = sum(p.numel() for p in model.parameters())
            stats["true_frames"] = [r["frames"] for r in rows]
            stats["microbatch"] = len(rows)
            stats["synthetic"] = cfg["run"]["synthetic"]
            results[str(upper)] = stats
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results[str(upper)] = {
                "status": "OOM",
                "action": "lower microbatch and increase accumulation; keep effective batch",
            }
    save_json(Path(cfg["run"]["out_dir"]) / "profiles" / f"{cfg['run']['id']}.json", results)
    print_json(results)


if __name__ == "__main__":
    main()
