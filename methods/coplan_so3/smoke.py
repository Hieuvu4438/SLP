"""Full-architecture no-training smoke check using a training clip only."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import torch
import yaml

from slp.data.sign_smplx import feats133

from .data.dataset import collate
from .factory import ROOT, build
from .sampling import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report")
    args = parser.parse_args()
    torch.manual_seed(731)
    hp = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())["method"]
    model, codec = build(hp)
    path = sorted(
        (ROOT / "data/processed/madis/Phoenix_2014T/poses_video_level/train").glob("*.pkl")
    )[0]
    with open(path, "rb") as f:
        raw = torch.from_numpy(feats133(pickle.load(f))[:8])
    normalized = (raw - codec.mean) / codec.std
    r, e = codec.encode(normalized)
    batch = collate(
        [
            {
                "rotations": r,
                "expression": e,
                "timestamps": torch.arange(len(r)) / 25.0,
                "text": "Guten Morgen",
                "language": 0,
            }
        ]
    ).to(args.device)
    model.to(args.device)
    model.features.fit_batch(
        batch.rotations, batch.expression, batch.mask, batch.timestamps, split="train"
    )
    model.features.finalize(split="train")
    report = {
        "training_executed": False,
        "optimizer_steps": 0,
        "device": args.device,
        "train_clip": str(path.relative_to(ROOT)),
        "frames": len(r),
        "parameters": sum(p.numel() for p in model.parameters()),
        "objectives": {},
    }
    for stage in ("A", "B", "C"):
        if stage == "B":
            model.plan.eval()
            with torch.no_grad():
                z, zm = model.plan.encode_raw(
                    batch.rotations, batch.expression, batch.mask, batch.timestamps
                )
                model.plan.stats.update(z, zm, split="train")
                model.plan.stats.finalize(split="train")
            model.plan.freeze()
        model.set_stage(stage)
        model.train()
        loss, terms = model.objective(batch, torch.Generator(device=args.device).manual_seed(512))
        if not torch.isfinite(loss):
            raise AssertionError(f"Nonfinite objective {stage}: {terms}")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise AssertionError(f"Nonfinite/missing gradients at {stage}")
        report["objectives"][stage] = {
            "loss": float(loss.detach()),
            "gradient_tensors": len(gradients),
        }
        model.zero_grad(set_to_none=True)
    start = time.perf_counter()
    out = generate(model, ["Guten Morgen"], [0], lengths=[8], total_budget=5, seed=42)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    report["generation_seconds"] = time.perf_counter() - start
    report["generation_metadata"] = out.metadata
    # Shape and frame-rate conventions on an actual clip match the shared evaluator.
    codec = codec.to(args.device)
    generated = codec.decode(out.rotations[0], out.expression[0])
    if generated.shape != (8, 133) or not torch.isfinite(generated).all():
        raise AssertionError("Failed evaluation codec integration")
    if args.device.startswith("cuda"):
        report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "passed"
    output = json.dumps(report, indent=2) + "\n"
    if args.report:
        Path(args.report).write_text(output)
    print(output)


if __name__ == "__main__":
    main()
