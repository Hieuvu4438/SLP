"""Table 7: parameter count and sampling latency (s / motion) of SignNMFlow.

    python -m methods.signmflow.benchmark [--ckpt <run>/checkpoints] [--data csl_smplx] [--n 100]

Uses the method config (plus optional key=value overrides); random weights if no checkpoint.
Latency: batch size 1, NFE = sampler.steps, CUDA-synchronised, after warm-up, text encoding
included; lengths are the predicted ones.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def main():
    from slp import config as config_lib
    from slp.data import load_dataset
    from slp.tracking.run import RunContext
    from slp.utils.logging import get_logger

    from methods.signmflow.flow import sample
    from methods.signmflow.method import Method

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data", default="csl_smplx")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = config_lib.compose("signmflow", a.data, a.overrides)
    out = ROOT / "outputs" / "signmflow" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    m = Method(cfg, RunContext(out, 0, get_logger("bench", out / "log.txt")))
    if a.ckpt:
        m.load(a.ckpt)
    else:
        m._setup()
        m.gen.to(m.device).eval()
        if m.spec.path == "naturalness":
            from methods.signmflow.paths import CorrectionNet
            m.corr = CorrectionNet().to(m.device).eval()
    n_gen = sum(p.numel() for p in m.gen.parameters())
    n_corr = sum(p.numel() for p in m.corr.parameters()) if m.corr is not None else 0
    inputs = load_dataset(cfg).val.inputs[:a.n]
    steps = int(cfg["method"]["sampler"]["steps"])
    times = []
    with torch.no_grad():
        for i, x in enumerate(inputs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            text, pad = m.gen.encode_text([x["text"]], [x["src"]], m.device)
            L = int(m.gen.predict_log_length(text, pad).exp().round().clamp(40, 400))
            sample(m.gen, m.corr, m.spec, text, pad, [L], steps=steps)
            torch.cuda.synchronize()
            if i >= 5:
                times.append(time.perf_counter() - t0)
    print(f"#params generator {n_gen / 1e6:.2f}M + correction {n_corr / 1e6:.2f}M "
          f"= {(n_gen + n_corr) / 1e6:.2f}M | NFE {steps} | latency {sum(times) / len(times):.4f} s/motion "
          f"({torch.cuda.get_device_name()})")


if __name__ == "__main__":
    main()
