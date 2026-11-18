#!/usr/bin/env python
"""CPU architecture profile with explicitly synthetic inputs; never corpus throughput."""

import argparse
import copy
import json
from pathlib import Path

import torch

from cprflow.config import DEFAULTS
from cprflow.evaluation.efficiency import benchmark
from cprflow.training.logging import environment
from cprflow.training.tasks import make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = {
        "environment": environment(),
        "synthetic": True,
        "device": "cpu",
        "scope": "Main architecture, B=1 T=16, forward/backward; not dataset throughput",
        "variants": {},
    }
    for variant, relational in [("B1", False), ("C0", True), ("B3", True)]:
        config = copy.deepcopy(DEFAULTS)
        config["relation"]["enabled"] = relational
        model = make_model(config)
        z = torch.randn(1, 4, 8, 128)
        r = torch.randn(1, 8, 33) if relational else None
        t = torch.tensor([0.7])
        text = torch.randn(1, 5, 768)
        mask = torch.ones(1, 8, dtype=torch.bool)

        def update():
            model.zero_grad(set_to_none=True)
            pred = model(
                z,
                r,
                t,
                text,
                torch.ones(1, 5, dtype=torch.bool),
                torch.tensor([0]),
                torch.tensor([16]),
                mask,
                torch.arange(8)[None] / 15,
            )
            loss = pred.z.square().sum() + pred.z.sum()
            if pred.r is not None:
                loss = loss + pred.r.square().sum() + pred.r.sum()
            loss.backward()

        report = benchmark(update, warmup=1, updates=3)
        report["trainable_parameters"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        report["stream_state_elements"] = 4 * 8 * 128 + (8 * 33 if relational else 0)
        report["hidden_state_elements"] = (5 if relational else 4) * 8 * 384
        report["local_context_key_tokens_per_query"] = 9 * (5 if relational else 4)
        report["blocks"] = 8
        report["hidden"] = 384
        report["heads"] = 6
        # torch.profiler reports supported matrix ops; SDPA/elementwise work is not silently claimed.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True
        ) as prof:
            update()
        report["profiler_supported_ops_flops"] = sum(event.flops for event in prof.key_averages())
        report["flops_scope"] = "supported profiler operations; SDPA kernels may be omitted"
        results["variants"][variant] = report
    assert (
        results["variants"]["C0"]["trainable_parameters"]
        == results["variants"]["B3"]["trainable_parameters"]
    )
    assert (
        results["variants"]["C0"]["stream_state_elements"]
        == results["variants"]["B3"]["stream_state_elements"]
    )
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
