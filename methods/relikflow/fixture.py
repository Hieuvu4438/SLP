"""Synthetic clips for the integrated verification profile."""

import json
from pathlib import Path

import numpy as np

from .runtime import npz_write, write_atomic


def make_fixture(root):
    root = Path(root)
    rows = []
    rng = np.random.default_rng(1729)
    for split, count in (("train", 8), ("dev", 3), ("test", 2)):
        for i in range(count):
            clip_id = f"fixture_{split}_{i:02d}"
            length = (1, 2, 4, 5, 6, 7, 9, 10)[i]
            motion = rng.normal(0, 0.10, size=(1, 133)) + np.linspace(0, 1, length)[:, None] * (
                rng.normal(0, 0.22, size=(1, 133))
            )
            q_valid = np.ones((length, 41), dtype=bool)
            if split == "train" and length == 6:
                q_valid[:, 10:13] = False
                motion[:, 30:39] = np.nan
            file = root / f"{clip_id}.npz"
            npz_write(file, pose133=motion.astype(np.float32), q_valid=q_valid)
            rows.append(
                dict(
                    clip_id=clip_id,
                    video_id=f"{split}_{i // 2}",
                    corpus="csl_daily",
                    split=split,
                    source_language="zh_CN",
                    target_sign_language="CSL",
                    text=f"测试句子 {split} {i}",
                    pose_path=file.name,
                    fps=25.0,
                    pose_fit_id="synthetic",
                    num_frames=length,
                    timebase_source="fixture",
                    pose_layout="soke133_physical",
                )
            )
    path = root / "manifest.jsonl"
    write_atomic(path, lambda p: p.write_text("".join(json.dumps(r) + "\n" for r in rows)))
    return path


def fixed_endpoint_sanity(pipe):
    """Overfit one synthetic batch with fixed source/time and verify gradient ownership."""
    import torch

    from .bridge import bridge
    from .data import collate

    model = pipe.make_generator()
    planner = pipe.make_planner(pipe.path("planner")).requires_grad_(False)
    correction = pipe.correction("C")
    frozen = [planner, correction] if correction is not None else [planner]
    snapshots = [[p.detach().clone() for p in m.parameters()] for m in frozen]
    clips = pipe.data["subsets"]["train"][2:4]
    batch = collate(clips, pipe.device)
    requests = [c.request(pipe.seed) for c in clips]
    model.eval()
    q0 = batch["q"].roll(1, dims=-1)
    t = q0.new_full((len(q0), 1, 1, 1), 0.39)
    with torch.no_grad():
        x = bridge(q0, batch["q"], t, correction)
        et = t[..., 0] * batch["e"]
    opt = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0)
    losses = []
    encoder_received_gradient = False
    for step in range(50):
        q, e = model.predict_target(x, et, t, model.encode(requests), None, batch["frame_valid"])
        loss = 10 * (q[batch["q_valid"]] - batch["q"][batch["q_valid"]]).square().mean()
        loss = (
            loss + 0.1 * (e[batch["expr_valid"]] - batch["e"][batch["expr_valid"]]).square().mean()
        )
        losses.append(float(loss.detach()))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            encoder_received_gradient = any(
                p.grad is not None and bool(p.grad.abs().max() > 0) for p in model.text.parameters()
            )
        opt.step()
    if not (encoder_received_gradient and losses[-1] < 0.2 * losses[0]):
        raise ValueError("Fixed endpoint sanity or encoder gradient check failed")
    for snapshots_for_model, network in zip(snapshots, frozen):
        if any(
            not torch.equal(old, new) or new.grad is not None
            for old, new in zip(snapshots_for_model, network.parameters())
        ):
            raise ValueError("A frozen network changed during the endpoint update")
    return {"fixed_first": losses[0], "fixed_last": losses[-1], "encoder_gradient": True}


def sample_sanity(pipe):
    import torch

    from .commands import split_rows
    from .data import Request
    from .runtime import json_write

    sampler = pipe.sampler("C")
    rows = split_rows(pipe, "dev")
    requests = [
        Request(r["text"], r["source_language"], r["target_sign_language"], r["clip_id"], pipe.seed)
        for r in rows
    ]
    one = [sampler.generate(r, 5) for r in requests]
    batch = sampler.generate_batch(requests[::-1], 5)[::-1]
    for a, b in zip(one, batch):
        torch.testing.assert_close(a["pose133"], b["pose133"], atol=1e-5, rtol=0)
        if a["T_hat"] != b["T_hat"] or b["nfe"] != 5:
            raise ValueError("Batch sampling changed length or NFE")
    json_write(pipe.root / "sampler_verification.json", {"batch_equivalence": True, "nfe": 5})
