from pathlib import Path

import numpy as np
import torch

from cprflow.cli.common import config_parser, print_json, resolved
from cprflow.data.cache import digest, file_hash, save_numeric
from cprflow.data.dataset import MotionDataset
from cprflow.data.stats import balanced_moments, save_stats
from cprflow.training.artifacts import load_dependency, normalizer, update_index


def fit(config, kind, split="train"):
    if split != "train":
        raise ValueError("fit_stats is train-only")
    manifest = Path(config["paths"]["manifests"]) / "train.jsonl"
    ds = MotionDataset(
        manifest, "train", config["data"]["datasets"], config["data"]["max_train_frames"]
    )
    if config["run"]["id"].startswith("smoke_real48"):
        from cprflow.training.tasks import training_rows

        ds.rows = training_rows(config, ds.rows)
    from cprflow.geometry.body_model import BodyModel
    from cprflow.geometry.relations import relation33
    from cprflow.geometry.temporal import pool2
    from cprflow.models.ae import MotionAE
    from cprflow.objectives.geometry import GROUPS, activities

    device = config["run"]["device"]
    body = BodyModel(config["paths"]["smplx_model"], config["geometry"]["fk_chunk_frames"]).to(
        device
    )
    deps = {
        "train_manifest_sha256": file_hash(manifest),
        "asset_sha256": file_hash(config["paths"]["smplx_model"]),
        "method": config["stats"]["method"],
        "datasets": config["data"]["datasets"],
        "max_train_frames": config["data"]["max_train_frames"],
        "std_floor": config["stats"]["std_floor"],
        "relation_definition": {
            k: config["relation"][k] for k in ("threshold", "softness", "pooling")
        },
    }
    motion_norm = normalizer(config, "motion", device) if kind == "latent" else None
    ae = (
        load_dependency(MotionAE(), config["paths"]["ae_checkpoint"], config, "ae", device)
        if kind == "latent"
        else None
    )
    if ae:
        deps["ae_sha256"] = file_hash(config["paths"]["ae_checkpoint"])

    def records(name):
        # Keep only one true clip in memory, even for the full joint corpus.
        with torch.no_grad():
            for row in ds:
                x = torch.as_tensor(row["x"], device=device)
                lengths = torch.tensor([len(x)], device=device)
                if name == "motion":
                    values = row["x"]
                elif name == "latent":
                    values = (
                        ae.encode(motion_norm(x)[None], lengths)[0].permute(1, 0, 2).cpu().numpy()
                    )
                else:
                    geo = body(x)
                    r = relation33(
                        geo["joints"],
                        geo["names"],
                        config["relation"]["threshold"],
                        config["relation"]["softness"],
                    )
                    values = pool2(r[None], lengths)[0].cpu().numpy()
                yield row, values

    root = Path(config["paths"]["stats"]) / digest(deps)
    outputs = {}
    for name in ("latent",) if kind == "latent" else ("motion", "relation"):
        mean, std, counts = balanced_moments(
            records(name), config["data"]["datasets"], config["stats"]["std_floor"]
        )
        if name == "latent":
            mean, std = mean[:, None, :], std[:, None, :]
        path = root / f"{name}.npz"
        save_stats(path, mean, std, {**deps, "counts": counts, "kind": name})
        update_index(
            config["paths"]["stats"], name, {"path": str(path.resolve()), "sha256": file_hash(path)}
        )
        outputs[name] = str(path)
    if kind != "latent":
        motion_norm = normalizer(config, "motion", device)
        # Train velocity/activity RMS, equal clip and dataset moments; no per-sample inference scaling.
        scale_names = list(GROUPS) + [
            "expression",
            "activity_left",
            "activity_right",
            "activity_head",
            "activity_face",
        ]

        def activity_records():
            with torch.no_grad():
                for row in ds:
                    x = torch.as_tensor(row["x"], device=device)
                    ts = torch.as_tensor(row["timestamps"], dtype=torch.float32, device=device)
                    geo = body(x)
                    dt = ts[1:] - ts[:-1]
                    vec = []
                    for group in GROUPS:
                        velocity = (geo[group][1:] - geo[group][:-1]) / dt[:, None, None]
                        vec.append(velocity.flatten(1).square().mean(1))
                    vec.append(((x[1:, 246:] - x[:-1, 246:]) / dt[:, None]).square().mean(1))
                    activity = activities(x, geo, ts, motion_norm, {k: 1.0 for k in scale_names})
                    vec.extend(activity[:, i].square() for i in range(4))
                    yield row, torch.stack(vec, -1).cpu().numpy()

        moment, _, counts = balanced_moments(activity_records(), config["data"]["datasets"], 1e-12)
        values = {
            key: np.array(max(float(v) ** 0.5, 1e-3), dtype=np.float32)
            for key, v in zip(scale_names, moment)
        }
        path = root / "activity.npz"
        save_numeric(path, **values)
        from cprflow.cli.common import save_json

        save_json(str(path) + ".json", {**deps, "counts": counts, "array_sha256": file_hash(path)})
        update_index(
            config["paths"]["stats"],
            "activity",
            {"path": str(path.resolve()), "sha256": file_hash(path)},
        )
        outputs["activity"] = str(path)
    return outputs


def main():
    parser = config_parser("Fit train-only dataset/clip-balanced statistics")
    parser.add_argument("--kind", choices=("motion-relation", "latent"), required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    args = parser.parse_args()
    print_json(fit(resolved(args), args.kind, args.split))


if __name__ == "__main__":
    main()
