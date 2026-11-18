from pathlib import Path

import torch

from cprflow.cli.common import config_parser, print_json, resolved
from cprflow.data.cache import NumericCache, digest, file_hash
from cprflow.data.dataset import MotionDataset
from cprflow.training.artifacts import (
    load_dependency,
    normalizer,
    stats_path,
    text_revisions,
    update_index,
)


def build_cache(config, kind, splits):
    if kind == "geometry":
        from cprflow.data.geometry_cache import build_geometry_cache

        return build_geometry_cache(config, splits)
    from cprflow.models.text import FrozenTextEncoder
    from cprflow.models.ae import MotionAE
    from cprflow.geometry.body_model import BodyModel
    from cprflow.geometry.relations import relation33
    from cprflow.geometry.temporal import pool2

    kinds = ["text"] if kind == "text" else ["latent", "relations"]
    device = config["run"]["device"]
    motion_norm = normalizer(config, "motion", device) if kind != "text" else None
    if kind == "text":
        rev, tok = text_revisions(config)
        encoder = FrozenTextEncoder(
            rev, tok, config["text"]["max_tokens"], config["text"]["local_files_only"]
        ).to(device)
        deps = {
            "text": {
                "revision": rev,
                "tokenizer_revision": tok,
                "max_tokens": config["text"]["max_tokens"],
                "prefix": "literal_square_brackets_v1",
                "eval_mode": True,
            }
        }
    else:
        ae = load_dependency(MotionAE(), config["paths"]["ae_checkpoint"], config, "ae", device)
        body = BodyModel(config["paths"]["smplx_model"], config["geometry"]["fk_chunk_frames"]).to(
            device
        )
        common = {
            "motion_stats": file_hash(stats_path(config, "motion")),
            "asset": file_hash(config["paths"]["smplx_model"]),
        }
        deps = {
            "latent": {
                **common,
                "ae": file_hash(config["paths"]["ae_checkpoint"]),
                "latent_stats": file_hash(stats_path(config, "latent")),
            },
            "relations": {
                **common,
                "relation_version": "relation33_v1",
                "definition": {
                    k: config["relation"][k] for k in ("threshold", "softness", "pooling")
                },
                "pool": "valid_pair_mean",
                "relation_stats": file_hash(stats_path(config, "relation")),
            },
        }
    for dependencies in deps.values():
        dependencies["synthetic"] = config["run"]["synthetic"]
    caches = {k: NumericCache(Path(config["paths"]["cache"]) / k, deps[k]) for k in kinds}
    counts = {}
    with torch.no_grad():
        for split in splits:
            ds = MotionDataset(
                Path(config["paths"]["manifests"]) / f"{split}.jsonl",
                split,
                config["data"]["datasets"],
            )
            if config["run"]["id"].startswith("smoke_real48"):
                if split == "train":
                    from cprflow.training.tasks import training_rows

                    ds.rows = training_rows(config, ds.rows)
                else:
                    ds.rows = [
                        r
                        for d in config["data"]["datasets"]
                        for r in [r for r in ds.rows if r["dataset_id"] == d][
                            : config["train"]["dev_panel_per_dataset"]
                        ]
                    ]
            count = 0
            for row in ds:
                if kind == "text":
                    from cprflow.types import DATASETS

                    uid = digest({"text": row["text"], "language": row["sign_language_id"]})
                    if caches["text"].path(uid).is_file():
                        cached = caches["text"].get(uid)["hidden"]
                        if cached.ndim == 2 and cached.shape[1] == 768 and 0 < len(cached) <= 512:
                            count += 1
                            continue
                    try:
                        text = encoder([row["text"]], [DATASETS[row["dataset_id"]][0]])
                    except ValueError as exc:
                        from cprflow.cli.common import save_json

                        save_json(
                            Path(config["paths"]["cache"]) / "text_cache_failure.json",
                            {
                                "uid": row["uid"],
                                "dataset": row["dataset_id"],
                                "split": split,
                                "reason": str(exc),
                                "num_requested_this_split": len(ds),
                                "num_successful_this_split": count,
                                "num_attempted_this_split": count + 1,
                                "complete": False,
                                "previous_splits": counts,
                                "synthetic": config["run"]["synthetic"],
                            },
                        )
                        raise ValueError(f"Text cache failed for {row['uid']}: {exc}") from exc
                    uid = digest({"text": row["text"], "language": row["sign_language_id"]})
                    caches["text"].put(
                        uid, hidden=text.hidden[0, text.mask[0]].cpu().half().numpy()
                    )
                else:
                    uid = digest({"uid": row["uid"], "processed": row["processed_sha256"]})
                    x = torch.as_tensor(row["x"], device=device)
                    lengths = torch.tensor([len(x)], device=device)
                    z = ae.encode(motion_norm(x)[None], lengths)[0]
                    geo = body(x)
                    r = relation33(
                        geo["joints"],
                        geo["names"],
                        config["relation"]["threshold"],
                        config["relation"]["softness"],
                    )
                    r = pool2(r[None], lengths)[0]
                    caches["latent"].put(uid, z=z.cpu().numpy())
                    caches["relations"].put(uid, r=r.cpu().numpy())
                count += 1
            counts[split] = count
    for k in kinds:
        update_index(
            config["paths"]["cache"],
            k,
            {"dependencies": deps[k], "cache_key": caches[k].key, "counts": counts},
        )
    return {"kinds": kinds, "counts": counts, "frozen": True}


def main():
    parser = config_parser("Build frozen numeric text/latent/relation caches")
    parser.add_argument("--kind", required=True, choices=("text", "latent-relation", "geometry"))
    parser.add_argument(
        "--splits", nargs="+", default=["train", "dev", "test"], choices=("train", "dev", "test")
    )
    args = parser.parse_args()
    print_json(build_cache(resolved(args), args.kind, args.splits))


if __name__ == "__main__":
    main()
