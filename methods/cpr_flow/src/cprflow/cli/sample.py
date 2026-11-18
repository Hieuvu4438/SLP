import argparse
import json
from pathlib import Path

import numpy as np

from cprflow.cli.common import print_json, save_json
from cprflow.data.cache import digest, save_numeric
from cprflow.sampling.generator import MotionGenerator
from cprflow.types import GenerationRequest


def main():
    parser = argparse.ArgumentParser(
        description="Generate physical canonical296 from text-only requests"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nfe", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("--solver", choices=("euler", "heun"))
    parser.add_argument("--guard-mode", choices=("off", "unrestricted", "feature_only", "semantic"))
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--teacher-calibration")
    args = parser.parse_args()
    generator = MotionGenerator.from_checkpoint(
        args.checkpoint,
        args.device,
        args.guard_mode,
        args.teacher_checkpoint,
        args.teacher_calibration,
    )
    outputs = []
    failures = []
    allowed = {"request_id", "text", "sign_language", "seed", "duration_seconds"}
    rows = [
        json.loads(line) for line in Path(args.text_file).read_text().splitlines() if line.strip()
    ]
    for row in rows:
        if set(row) - allowed:
            raise ValueError(f"Forbidden generation fields: {sorted(set(row) - allowed)}")
        request_id = row.pop("request_id")
        try:
            request = GenerationRequest(**row)
            result = generator.generate(request, args.nfe, args.cfg, args.solver)
            stem = Path(args.output) / (digest({"id": request_id, "seed": request.seed}))
            save_numeric(
                stem.with_suffix(".npz"),
                motion=result.canonical_motion,
                timestamps=np.arange(result.predicted_frames, dtype=np.float64) / 30,
            )
            save_json(
                stem.with_suffix(".json"),
                {
                    "request_id": request_id,
                    "seed": request.seed,
                    "predicted_frames": result.predicted_frames,
                    "fps": result.fps,
                    **result.metadata,
                },
            )
            outputs.append(str(stem))
        except (ValueError, RuntimeError) as exc:
            failures.append({"request_id": request_id, "reason": str(exc)})
    report = {
        "num_requested": len(rows),
        "num_successful": len(outputs),
        "outputs": outputs,
        "failures": failures,
    }
    save_json(Path(args.output) / "sample_report.json", report)
    print_json(report)


if __name__ == "__main__":
    main()
