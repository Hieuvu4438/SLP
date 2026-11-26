"""Real-data smoke test of every SignRAE v3 option at tiny size (PLAN_V3; no experiment).

    python -m methods.morae_slp.smoke --data phoenix_smplx \\
        --ae outputs/morae_slp/pretrain/ae_union_jepa_d64/ae.pt [--device cpu] [--out DIR]

Fits a tiny generator on a few TRAIN clips with x1 prediction + the re-planned sampler, the
metric-frame FK loss, the gloss-consistency loss (with a recognizer trained here on the same
standardisation), the retrieval memory and the duration options; then predicts a few DEV clips
with the medoid and DBA decoders, reloads the checkpoint and checks that predictions are
finite, have the predicted lengths and are reproducible. Exit code 0 = all checks passed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def main(argv=None) -> int:
    from slp import config as config_lib
    from slp.data import Split, load_dataset
    from slp.registry import get_method
    from slp.tracking.run import RunContext

    from .gloss import GlossRecognizer, GlossVocab, save_recognizer
    from .stages import encode_split, latent_stats, load_space

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="phoenix_smplx")
    p.add_argument("--ae", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--clips", type=int, default=12)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("morae_smoke")
    root = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="morae_smoke_"))
    root.mkdir(parents=True, exist_ok=True)

    base = config_lib.compose("morae_slp", args.data, [])
    splits = load_dataset(base)
    n = args.clips
    train = Split(splits.train.inputs[:n], [splits.train.targets[i] for i in range(n)])
    dev = Split(splits.val.inputs[:4], [splits.val.targets[i] for i in range(4)])
    has_gloss = all(item.get("gloss") for item in train.inputs)

    # Recognizer in the run's standardisation (recalibrated on the same train clips).
    device = torch.device(args.device)
    space, _ = load_space(args.ae, device)
    idx = list(range(len(train)))
    raw = encode_split(space, train, idx, device, 0, standardised=False)
    mean, std = latent_stats(raw, idx, base["method"]["latent"]["min_std"])
    overrides = [
        f"method.device={args.device}", "method.precision=fp32", "method.workers=0",
        f"method.init_ae={args.ae}",
        "method.flow.width=32", "method.flow.depth=2", "method.flow.heads=2", "method.flow.ff=64",
        "method.flow.updates=2", "method.flow.eval_every=1", "method.flow.batch_size=4",
        "method.flow.microbatch=2", "method.flow.select=flow", "method.text.finetune=true",
        "method.flow.token_drop=0.1",
        "method.flow.prediction=target", "method.sampler.solver=replan_anchored",
        "method.sampler.steps=5", "method.sampler.cfg_scale=1.5", "method.sampler.temperature=0.7",
        "method.flow.fk_weight_body=0.25", "method.flow.fk_weight_hand=1.0",
        "method.flow.fk_t_min=0.0", "method.flow.fk_frames=4",
        "method.retrieval.enabled=true", "method.retrieval.k=2",
        "method.retrieval.memory.layers=1", "method.retrieval.memory.heads=2",
        "method.duration.token_count=true", "method.duration.calibrate=true",
        "method.sampler.batch_size=2", "method.export_predictions=false",
    ]
    if has_gloss:
        vocab = GlossVocab.from_items(train.inputs)
        arch = {"latent_dim": space.latent_dim, "vocab_size": len(vocab), "width": 32,
                "depth": 1, "heads": 2}
        save_recognizer(root / "recognizer.pt", GlossRecognizer(**arch), vocab, arch, mean, std,
                        {"data": args.data, "smoke": True})
        overrides += [f"method.gloss.recognizer={root / 'recognizer.pt'}",
                      "method.flow.gloss_weight=0.1", "method.flow.gloss_start=0",
                      "method.flow.gloss_t_min=0.0"]
    cfg = config_lib.compose("morae_slp", args.data, overrides)
    run = root / "run"
    run.mkdir(exist_ok=True)
    ctx = RunContext(run, 0, log)
    model = get_method("morae_slp")(cfg, ctx)
    model.fit(train, dev)
    checks = {}
    history = (run / "metrics_history.jsonl").read_text()
    checks["fk_loss_logged"] = "flow/train_fk_hand" in history
    checks["gloss_loss_logged"] = (not has_gloss) or "flow/train_gloss" in history
    checks["retrieval_saved"] = (run / "checkpoints" / "retrieval.pt").is_file()
    state = torch.load(run / "checkpoints" / "model.pt", map_location="cpu", weights_only=True)
    checks["checkpoint_v3_keys"] = (state["flow_prediction"] == "target"
                                    and state["memory_args"] is not None
                                    and state["duration_args"] == {"token_count": True})
    results = {}
    for decoder in ("mean", "medoid", "dba"):
        model.sampler.update(ensemble=1 if decoder == "mean" else 3, decoder=decoder)
        preds = model.predict(dev.inputs)
        results[decoder] = preds
        checks[f"{decoder}_finite"] = all(np.isfinite(x).all() and x.shape[1] == 133 for x in preds)
    again = get_method("morae_slp")(cfg, ctx)
    again.load(run / "checkpoints")
    again.sampler.update(ensemble=1, decoder="mean")
    reload_preds = again.predict(dev.inputs)
    checks["reload_reproducible"] = all(np.allclose(a, b, atol=1e-4)
                                        for a, b in zip(results["mean"], reload_preds))
    for name, ok in checks.items():
        log.info("%-24s %s", name, "ok" if ok else "FAILED")
    log.info("artifacts in %s", root)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
