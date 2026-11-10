"""Real pretrained author base + full, UNTRAINED RACE model integration check.
No signing-quality/benchmark claim is inferred from this command.
"""

import argparse
import json
from pathlib import Path
import time
import torch
from .madis_adapter import load_base_adapter
from .geometry import SMPLXGeometry, geometry_signature
from .config import RaceConfig
from .model import FlowDecoder
from .schema import MotionNormalizer
from .selector import ImprovementSelector
from .pipeline import RacePipeline
from .data import collate_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default="csl")
    parser.add_argument("--text", default="今天的天气很好。")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    torch.manual_seed(7)
    start = time.perf_counter()
    base = load_base_adapter(args.assets, device=args.device)
    record = base.prepare(args.text, args.language, "author_checkpoint_integration", max_tokens=100)
    geometry = SMPLXGeometry(args.geometry).to(args.device)
    cfg = RaceConfig(
        code_dim=record["codes"].shape[-1], text_dim=record["text"].shape[-1], factor=base.factor
    )
    decoder = FlowDecoder(cfg).to(args.device)
    selector = ImprovementSelector(cfg).to(args.device)
    pipe = RacePipeline(decoder, MotionNormalizer().to(args.device), geometry, selector).eval()
    prediction = pipe.predict(collate_inputs([record]).to(args.device), policy="coupled", seed=7)
    result = dict(
        author_base=base.provenance["adapter"],
        native_frames=int(prediction.lengths[0]),
        shape=list(prediction.poses.shape),
        nfe=prediction.nfe,
        finite=bool(torch.isfinite(prediction.poses).all()),
        exact_nonmanual_preservation=bool(
            torch.equal(prediction.poses[..., 120:], prediction.base[..., 120:])
        ),
        text_feature_dim=record["text"].shape[-1],
        code_dim=record["codes"].shape[-1],
        race_config=cfg.to_dict(),
        race_parameter_count=sum(p.numel() for p in decoder.parameters()),
        selector_scores_finite=bool(torch.isfinite(prediction.delta).all()),
        elapsed_seconds=round(time.perf_counter() - start, 2),
        device=torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
        race_decoder_trained=False,
        selector_trained=False,
        calibration_fitted=False,
        geometry=geometry_signature(geometry),
        base_provenance=base.provenance,
        scope="Real pretrained author base + full UNTRAINED RACE decoder/selector: integration only, no quality or SOTA claim",
    )
    if (
        not result["finite"]
        or not result["exact_nonmanual_preservation"]
        or not result["selector_scores_finite"]
    ):
        raise AssertionError(result)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
