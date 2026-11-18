from pathlib import Path

from cprflow.cli.common import config_parser, print_json, resolved, save_json
from cprflow.data.preparation import canonicalize, import_sources, verified_timing


def main():
    parser = config_parser("Prepare the locked three-corpus controlled track")
    parser.add_argument(
        "--phase",
        required=True,
        choices=("import", "verify-timing", "canonicalize", "audit-geometry"),
    )
    parser.add_argument("--samples-per-dataset", type=int, default=64)
    args = parser.parse_args()
    cfg = resolved(args)
    if args.phase == "import":
        output = import_sources(cfg)
    elif args.phase == "verify-timing":
        output = verified_timing(cfg)
    elif args.phase == "canonicalize":
        output = canonicalize(cfg)
    else:
        from cprflow.data.dataset import MotionDataset
        from cprflow.geometry.body_model import BodyModel
        from cprflow.cli.render import render_motion
        import torch

        ds = MotionDataset(
            Path(cfg["paths"]["manifests"]) / "train.jsonl", "train", cfg["data"]["datasets"]
        )
        model = BodyModel(cfg["paths"]["smplx_model"], cfg["geometry"]["fk_chunk_frames"])
        output = {
            "synthetic": False,
            "native_video_review": "pending",
            "mapping_review": "pending",
            "renders": [],
        }
        for dataset in cfg["data"]["datasets"]:
            selected = [i for i, r in enumerate(ds.rows) if r["dataset_id"] == dataset][
                : args.samples_per_dataset
            ]
            if len(selected) < args.samples_per_dataset:
                raise ValueError(f"Not enough {dataset} audit samples")
            for i in selected:
                row = ds[i]
                target = (
                    Path(cfg["run"]["out_dir"]) / "geometry_audit" / dataset / (str(i) + ".mp4")
                )
                render_motion(torch.as_tensor(row["x"]), model, target, row["fps"])
                output["renders"].append({"uid": row["uid"], "path": str(target)})
        save_json(Path(cfg["run"]["out_dir"]) / "geometry_audit/report.json", output)
    print_json(output)


if __name__ == "__main__":
    main()
