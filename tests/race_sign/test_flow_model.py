import torch
from methods.race_sign.flow import flow_path, endpoint, sample_euler
from methods.race_sign.model import FlowDecoder
from methods.race_sign.data import collate_training, window_batch
from methods.race_sign.schema import manual_to_6d, MotionNormalizer
from methods.race_sign.losses import decoder_objective


def conditions(batch):
    return {
        k: batch[k]
        for k in (
            "codes",
            "token_lengths",
            "text",
            "text_mask",
            "lengths",
            "language",
            "relations",
            "offsets",
        )
    }


def test_analytic_flow_path_and_actual_nfe():
    y, noise = torch.randn(2, 8, 240), torch.randn(2, 8, 240)
    times = torch.tensor([0.0, 0.8])
    state, velocity = flow_path(y, times, noise)
    torch.testing.assert_close(endpoint(state, times, velocity), y, atol=1e-6, rtol=1e-6)

    class Oracle(torch.nn.Module):
        def forward(self, state, t, **kwargs):
            return y - noise

    oracle = Oracle().eval()
    generated, nfe = sample_euler(oracle, noise, 4, {"lengths": torch.tensor([8, 8])}, guidance=2)
    torch.testing.assert_close(generated, y, atol=1e-6, rtol=1e-6)
    assert nfe == 8


def test_padding_invariance_and_training_gradients(cfg, geometry, record_factory):
    torch.manual_seed(14)
    batch = collate_training([record_factory(0, 8), record_factory(1, 12)])
    batch["relations"] = geometry(manual_to_6d(batch["baseline"])).relations
    model = FlowDecoder(cfg).eval()
    state = torch.randn(2, 12, 240)
    time = torch.tensor([0.25, 0.7])
    original = model(state, time, **conditions(batch))
    state[0, 8:] = 1e8
    batch["codes"][0, 2:] = -1e8
    batch["relations"][0, 8:] = 1e8
    modified = model(state, time, **conditions(batch))
    torch.testing.assert_close(original[0, :8], modified[0, :8])
    assert torch.equal(modified[0, 8:], torch.zeros_like(modified[0, 8:]))
    model.train()
    norm = MotionNormalizer()
    loss, _ = decoder_objective(model, norm, geometry, batch, cfg)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.code_projection[0].weight.grad.abs().sum() > 0
    assert model.blocks[0].part_attn.in_proj_weight.grad.abs().sum() > 0


def test_window_offsets_preserve_condition_correspondence(cfg, geometry, record_factory):
    batch = collate_training([record_factory(0, 20), record_factory(1, 24)])
    batch["relations"] = geometry(manual_to_6d(batch["baseline"])).relations
    win = window_batch(batch, 8, 2, cfg.factor, torch.Generator().manual_seed(12))
    for i in range(2):
        start, length = int(win["offsets"][i]), int(win["lengths"][i])
        torch.testing.assert_close(
            win["target_manual"][i, :length], batch["target_manual"][i, start : start + length]
        )
        assert win["loss_mask"][i].sum() == 8
    model = FlowDecoder(cfg).eval()
    out = model(torch.zeros_like(win["target_manual"]), torch.zeros(2), **conditions(win))
    assert torch.isfinite(out).all()


def test_cuda_forward_backward(cfg, geometry, record_factory):
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from methods.race_sign.training import to_device

    geometry = geometry.cuda()
    batch = to_device(collate_training([record_factory(0, 8)]), "cuda")
    batch["relations"] = geometry(manual_to_6d(batch["baseline"])).relations
    model = FlowDecoder(cfg).cuda().train()
    loss, _ = decoder_objective(model, MotionNormalizer().cuda(), geometry, batch, cfg)
    loss.backward()
    assert torch.isfinite(loss) and all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
