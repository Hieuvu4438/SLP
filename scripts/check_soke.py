"""Audit SOKE assets/imports and optionally exercise the pretrained tokenizer.

Run with third_party/SOKE/.venv/bin/python scripts/check_soke.py --smoke.
The default device is CPU; --device cuda also checks a small GPU operation.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party/SOKE"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/soke/readiness.json")
    args = parser.parse_args()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0" if args.device == "cuda" else "")
    report = {
        "repo": str(REPO),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version,
        "imports": {},
        "assets": {},
        "smoke": {},
    }
    for name in [
        "torch", "numpy", "pytorch_lightning", "torchmetrics", "transformers",
        "diffusers", "chumpy", "smplx", "pyrender", "moviepy.editor", "cv2",
        "mGPT.archs.mgpt_mbart", "mGPT.data.H2S", "mGPT.models.mgpt", "train", "test",
    ]:
        try:
            module = importlib.import_module(name)
            report["imports"][name] = {"ok": True, "version": getattr(module, "__version__", None)}
        except Exception as exc:
            report["imports"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    assets = {
        "tokenizer": "checkpoints/tokenizer.ckpt",
        "mean": "deps/stats/mean.pt",
        "std": "deps/stats/std.pt",
        "mbart_config": "deps/mbart-h2s-csl-phoenix/config.json",
        "mbart_weights": "deps/mbart-h2s-csl-phoenix/pytorch_model.bin",
        "mbart_map_ids": "deps/mbart-h2s-csl-phoenix/map_ids.pkl",
        "mbart_sentencepiece": "deps/mbart-h2s-csl-phoenix/sentencepiece.bpe.model",
        "smplx": "deps/smpl_models/smplx/SMPLX_NEUTRAL.npz",
        "smplx_j14": "deps/smpl_models/smplx/SMPLX_to_J14.pkl",
        "smplx_hands": "deps/smpl_models/smplx/MANO_SMPLX_vertex_ids.pkl",
        "smplx_face": "deps/smpl_models/smplx/SMPL-X__FLAME_vertex_ids.npy",
        "smplx_segmentation": "deps/smpl_models/smplx_vert_segmentation.json",
        "glove_vectors": "deps/t2m/glove/our_vab_data.npy",
        "glove_words": "deps/t2m/glove/our_vab_words.pkl",
        "glove_index": "deps/t2m/glove/our_vab_idx.pkl",
        "generator_checkpoint": "experiments/mgpt/SOKE/checkpoints/last.ckpt",
    }
    for name, path in assets.items():
        p = REPO / path
        complete = p.is_file() and p.stat().st_size > 0 and not Path(str(p) + ".aria2").exists()
        report["assets"][name] = {"path": path, "present": complete}
        if complete:
            report["assets"][name]["bytes"] = p.stat().st_size

    import torch
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(ROOT / "configs/soke/soke.local.yaml")
    dataset_paths = {
        "how2sign_test_annotations": Path(cfg.DATASET.H2S.ROOT) / "test/re_aligned/how2sign_realigned_test_preprocessed_fps.csv",
        "how2sign_test_poses": Path(cfg.DATASET.H2S.ROOT) / "test/poses",
        "csl_test_annotations": Path(cfg.DATASET.H2S.CSL_ROOT) / "csl_clean.test",
        "csl_poses": Path(cfg.DATASET.H2S.CSL_ROOT) / "poses",
        "phoenix_test_annotations": Path(cfg.DATASET.H2S.PHOENIX_ROOT) / "phoenix14t.test",
        "phoenix_train_annotations": Path(cfg.DATASET.H2S.PHOENIX_ROOT) / "phoenix14t.train",
        "phoenix_dev_annotations": Path(cfg.DATASET.H2S.PHOENIX_ROOT) / "phoenix14t.dev",
    }
    report["dataset_paths"] = {k: {"path": str(p), "present": p.exists()} for k, p in dataset_paths.items()}
    # Pose coverage must be validated against annotations, not inferred from a directory.
    report["dataset_pose_coverage_validated"] = False
    report["cuda"] = {"visible": os.environ["CUDA_VISIBLE_DEVICES"], "available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}

    if args.smoke:
        try:
            from mGPT.config import instantiate_from_config
            torch.set_num_threads(2)
            state = torch.load(REPO / assets["tokenizer"], map_location="cpu")["state_dict"]
            for key, config in [("vae", "re96"), ("hand_vae", "hand192"), ("rhand_vae", "hand192")]:
                vae_cfg = OmegaConf.load(REPO / f"configs/vq/{config}.yaml")
                del vae_cfg.params["ablation"]
                vae = instantiate_from_config(vae_cfg).to(args.device).eval()
                prefix = key + "."
                weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
                if key == "vae" and not weights:
                    prefix = "motion_vae."
                    weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
                vae.load_state_dict(weights, strict=True)
                with torch.inference_mode():
                    x = torch.zeros(1, 16, vae_cfg.params.nfeats, device=args.device)
                    codes, _ = vae.encode(x)
                    decoded = vae.decode(codes[0])
                assert decoded.shape == x.shape and torch.isfinite(decoded).all()
                report["smoke"][key] = {"ok": True, "codes_shape": list(codes.shape), "decoded_shape": list(decoded.shape), "strict_weights": True}
                del vae
            for name in ["mean", "std"]:
                value = torch.load(REPO / assets[name], map_location="cpu")
                assert value.shape == (179,) and torch.isfinite(value).all()
                report["smoke"][name] = {"ok": True, "shape": list(value.shape)}
        except Exception as exc:
            report["smoke"]["error"] = f"{type(exc).__name__}: {exc}"

    imports_ok = all(x["ok"] for x in report["imports"].values())
    report["runtime_imports_ok"] = imports_ok
    report["full_text_to_sign_ready"] = False
    report["readiness_note"] = "Requires a trained generator checkpoint and verified dataset pose/annotation coverage. Tokenizer smoke uses synthetic inputs; it is not generator inference."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if imports_ok and "error" not in report["smoke"] else 1


if __name__ == "__main__":
    sys.exit(main())
