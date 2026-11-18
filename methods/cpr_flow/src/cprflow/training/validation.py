"""Deterministic fixed-ID dev panels. No use of the sealed test split."""

from pathlib import Path

import numpy as np
import torch

from cprflow.data.cache import digest, save_numeric
from cprflow.data.dataset import MotionDataset
from cprflow.evaluation.geometry import geometry_metrics
from cprflow.models.duration import DurationModel
from cprflow.sampling.generator import MotionGenerator
from cprflow.training.artifacts import load_dependency
from cprflow.training.engine import seed_everything
from cprflow.types import DATASETS, GenerationRequest


def validator(config, objective):
    ds = MotionDataset(
        Path(config["paths"]["manifests"]) / "dev.jsonl", "dev", config["data"]["datasets"]
    )
    panels = {
        d: [i for i, r in enumerate(ds.rows) if r["dataset_id"] == d][
            : config["train"]["dev_panel_per_dataset"]
        ]
        for d in config["data"]["datasets"]
    }
    if any(not indices for indices in panels.values()):
        raise ValueError("Each corpus needs a dev panel")
    ids = [ds.rows[i]["uid"] for indices in panels.values() for i in indices]
    subset_hash = digest(ids)

    def validate(trainer):
        seed_everything(1729)
        original = {k: v.detach().clone() for k, v in trainer.raw.state_dict().items()}
        trainer.raw.load_state_dict(trainer.ema.shadow)
        trainer.raw.eval()
        objective.validation = True
        scores = {}
        detail = {}
        try:
            with torch.no_grad():
                for dataset, indices in panels.items():
                    means = []
                    for i in indices:
                        if trainer.stop_requested:
                            raise InterruptedError("Stop requested during dev validation")
                        row = ds[i]
                        vec, terms = objective(trainer.raw, [row], trainer.step, [row])
                        if config["run"]["task"] == "duration":
                            means.append(float(terms["absolute_log_length_error"][0]))
                        elif config["run"]["task"] == "teacher":
                            means.append(float(vec.mean()))
                        else:
                            means.append(float(vec.mean()))
                    detail[dataset] = {"dev_loss": float(np.mean(means)), "clips": len(means)}
                    scores[dataset] = float(np.mean(means))
                if config["run"]["task"] == "teacher":
                    from cprflow.evaluation.retrieval import retrieval_recall
                    from cprflow.data.collate import collate_motion

                    # Full per-language panel, so retrieval is not a batch-one fake score.
                    for dataset, indices in panels.items():
                        rows = [ds[i] for i in indices]
                        b = collate_motion(rows, objective.motion_norm, objective.device)
                        text = objective.text(rows)
                        m = trainer.raw(b.x, b.lengths, b.timestamps)["global"]
                        t = trainer.raw.encode_text(text.hidden, text.mask, b.language_id)
                        positive = np.array([[a["text"] == b["text"] for b in rows] for a in rows])
                        retrieval = retrieval_recall(m.cpu().numpy(), t.cpu().numpy(), positive)
                        detail[dataset].update(retrieval)
                        scores[dataset] = -retrieval["recall_at_1"]
                if config["run"]["task"] == "flow":
                    from cprflow.geometry.body_model import BodyModel

                    body = objective.body or BodyModel(
                        config["paths"]["smplx_model"], config["geometry"]["fk_chunk_frames"]
                    ).to(objective.device)
                    dc = config["duration"]
                    duration = load_dependency(
                        DurationModel(dc["hidden_dim"], dc["sigma_floor"], dc["sigma_max"]),
                        config["paths"]["duration_checkpoint"],
                        config,
                        "duration",
                        objective.device,
                    )
                    guard_kwargs = {"relation_norm": objective.relation_norm}
                    if config["guard"]["enabled"]:
                        from cprflow.geometry.relations import relation_operator

                        guard_kwargs.update(
                            {
                                "teacher": objective.teacher,
                                "calibration": getattr(objective, "calibration", None),
                                "relation_fn": lambda x, n: relation_operator(
                                    x, n, body, objective.relation_norm
                                ),
                            }
                        )
                    gen = MotionGenerator(
                        trainer.raw,
                        objective.ae,
                        duration,
                        objective.motion_norm,
                        objective.latent_norm,
                        None,
                        config,
                        **guard_kwargs,
                    )
                    for dataset, indices in panels.items():
                        selected = indices[: config["train"]["dev_prompts_per_dataset"]]
                        values = []
                        durations = []
                        for i in selected:
                            if trainer.stop_requested:
                                raise InterruptedError("Stop requested during dev sampling")
                            row = ds[i]
                            request = GenerationRequest(
                                row["text"],
                                DATASETS[dataset][0],
                                config["train"]["dev_sample_seed"],
                            )
                            text = objective.text([row])
                            result = gen._generate(request, text_features=text)
                            pred = torch.as_tensor(result.canonical_motion, device=objective.device)
                            target = torch.as_tensor(row["x"], device=objective.device)
                            metrics = geometry_metrics(pred, target, body, objective.relation_norm)
                            values.append(metrics["controlled_v1_dtw_jpe_global_hand21"])
                            durations.append(abs(len(pred) - len(target)) / len(target))
                            if (
                                trainer.step % config["train"]["dev_generation_every"] == 0
                                and trainer.rank == 0
                            ):
                                save_numeric(
                                    trainer.directory
                                    / "dev_samples"
                                    / str(trainer.step)
                                    / (digest(row["uid"]) + ".npz"),
                                    motion=result.canonical_motion,
                                    timestamps=np.arange(len(pred)) / 30,
                                )
                        scores[dataset] = float(np.mean(values))
                        detail[dataset].update(
                            global_hand21_dtw_jpe=scores[dataset],
                            duration_relative_error=float(np.mean(durations)),
                        )
        finally:
            trainer.raw.load_state_dict(original)
            objective.validation = False
        return float(np.mean(list(scores.values()))), {
            "per_dataset": detail,
            "selection_subset_hash": subset_hash,
            "panel_selected": True,
        }

    return validate
