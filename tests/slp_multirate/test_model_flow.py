from dataclasses import replace

import pytest
import torch

from methods.slp_multirate.anchors import choose_indices, empty_anchors, jitter_anchors
from methods.slp_multirate.config import resolve
from methods.slp_multirate.flow import (
    Moments, Stats, anchor_loss, body_objective, masked_mean, sample_cfm, seeded_noise,
)
from methods.slp_multirate.geometry import build_body_grid, identity6, matrix_to_rot6d
from methods.slp_multirate.models import (
    AnchorAttention, AnchorPlanner, Condition, EndpointFlow, TextBatch, TextEncoder, plan_anchors,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_anchor_order_fallback_budget_and_normalized_payload():
    for n in range(1, 13):
        for policy in ("uniform", "onset_mid_offset", "event"):
            chosen = choose_indices(n, 8, policy, [], [0] * n)
            assert chosen == sorted(set(chosen)) and len(chosen) == min(n, 8)
    assert choose_indices(6, 2, "event", score=[0, 3, 3, 1, 2, 0]) == [1, 4]
    cfg = resolve("p0")
    text = TextEncoder(cfg["text"])(["test short", "test long"])
    lengths = torch.tensor([1, 12])
    learner = AnchorPlanner(text.hidden.shape[-1], cfg["anchors"])
    predicted = plan_anchors(learner, text, lengths)
    assert predicted.slot.sum(-1).tolist() == [1, 8]
    assert (predicted.time[1].diff() > 0).all()
    assert (predicted.time[1] > 0).all() and (predicted.time[1] < 1).all()
    value, _ = anchor_loss(predicted, predicted, lengths)
    assert torch.isfinite(value)
    jittered = jitter_anchors(predicted, lengths, 3, .05, torch.Generator().manual_seed(3))
    assert jittered.time[1].diff().min() >= 0
    assert jittered.payload[~jittered.feature].count_nonzero() == 0


def test_anchor_attention_zero_in_invalid_sample_and_confidence():
    cfg = resolve("p0")
    layer = AnchorAttention(32, 4, cfg["anchors"])
    query = torch.randn(2, 8, 32)
    frames = torch.ones(2, 8, dtype=torch.bool)
    missing = empty_anchors(2, 8)
    assert layer(query, missing, frames).count_nonzero() == 0
    missing.slot[0, 1] = True
    missing.feature[0, 1, 1, :5] = True
    missing.payload[0, 1, 1, :5] = 2
    assert layer(query, missing, frames).count_nonzero() == 0
    missing.confidence[0, 1, 1] = 1
    result = layer(query, missing, frames)
    assert result[0].count_nonzero() and result[1].count_nonzero() == 0


def test_padding_nan_contamination_and_attention():
    cfg = resolve("p0")
    model = EndpointFlow(8, cfg["fine"], cfg["anchors"]).eval()
    frame = torch.tensor([[True, True, True, False], [True, True, True, True]])
    coarse = torch.tensor([[True, False], [True, True]])
    text = TextBatch(torch.randn(2, 2, 8), coarse, torch.zeros(2, dtype=torch.bool))
    cond = Condition(text, torch.rand(2, 4), frame, torch.rand(2, 2, 72),
                     torch.rand(2, 2), coarse, torch.rand(2, 4, 72), empty_anchors(2, 8))
    noisy = torch.randn(2, 4, 204)
    tau = torch.tensor([.4, .1])
    dirty = replace(cond, times=cond.times.masked_fill(~frame, torch.nan),
                    coarse=cond.coarse.masked_fill(~coarse[..., None], torch.nan),
                    coarse_times=cond.coarse_times.masked_fill(~coarse, torch.nan),
                    text=replace(text, hidden=text.hidden.masked_fill(~coarse[..., None], torch.nan)),
                    full_body=cond.full_body.masked_fill(~frame[..., None], torch.nan))
    p = model(noisy, tau, cond)
    q = model(noisy.masked_fill(~frame[..., None], torch.nan), tau, dirty)
    torch.testing.assert_close(p, q)
    assert q[~frame].count_nonzero() == 0
    q.sum().backward()
    assert all(param.grad is None or param.grad.isfinite().all() for param in model.parameters())


def test_masked_statistics_and_per_clip_mean():
    error = torch.tensor([[2., 6., 4.], [10., 0., 0.]], requires_grad=True)
    valid = torch.tensor([[True, True, True], [True, False, False]])
    average, n = masked_mean(error, valid)
    assert average == 7 and n == 2
    empty, n = masked_mean(error, torch.zeros_like(valid))
    empty.backward()
    assert empty == 0 and n == 0 and error.grad.count_nonzero() == 0
    moments = Moments((2,))
    x = torch.tensor([[1., 1.], [1., float("nan")]])
    m = torch.isfinite(x)
    moments.add(x, m)
    stats = moments.finish(.001)
    assert stats.std.min() == .001
    normalized = stats.normalize(x, m)
    assert normalized.isfinite().all()
    torch.testing.assert_close(stats.denormalize(normalized)[m], x[m])


def test_perfect_endpoint_and_all_masked_backward():
    cfg = resolve("p0")
    time = torch.tensor([[0., 0.], [0., .04]])
    valid = torch.tensor([[True, False], [True, True]])
    x = identity6(2, 2, 12)
    obs = valid[..., None].expand(2, 2, 12)
    grid = build_body_grid(x, time, valid, obs, 1)
    target = matrix_to_rot6d(grid.rotations).flatten(-2).masked_fill(
        ~valid[..., None], torch.nan)
    p = target.clone().requires_grad_(True)
    stats = Stats(torch.zeros(72), torch.ones(72), torch.ones(72))
    t = torch.zeros(2)
    loss, _ = body_objective(p, target, grid, stats, t, cfg["loss"])
    assert loss < 1e-7
    loss.backward()
    assert p.grad.isfinite().all()
    grid = replace(grid, observation=torch.zeros_like(grid.observation))
    p = target.clone().requires_grad_(True)
    loss, _ = body_objective(p, target, grid, stats, t, cfg["loss"])
    loss.backward()
    assert loss == 0 and p.grad.isfinite().all()


@pytest.mark.parametrize("nfe", [1, 2, 4, 8])
def test_sampler_exact_nfe_and_stage_seeds(nfe):
    mask = torch.tensor([[True, True, False], [True, True, True]])
    start = seeded_noise(["a", "b"], 17, "body", 0, mask, 4)
    again = seeded_noise(["b", "a"], 17, "body", 0, mask.flip(0), 4)
    torch.testing.assert_close(start, again.flip(0))
    assert not torch.equal(start, seeded_noise(["a", "b"], 17, "fine", 0, mask, 4))
    calls = []

    def model(x, tau, c):
        calls.append(tau)
        assert (x[~mask] == 0).all()
        return torch.ones_like(x)

    clean = sample_cfm(model, start, None, mask, nfe)
    assert len(calls) == nfe
    torch.testing.assert_close(torch.stack(calls)[:, 0], torch.arange(nfe) / nfe)
    assert (clean[mask] == 1).all() and (clean[~mask] == 0).all()


def test_fine_endpoint_keeps_equal_wrist_and_finger_weights_under_missingness():
    from methods.slp_multirate.flow import fine_objective

    cfg = resolve("p0")
    weights = cfg["loss"] | {name: 0. for name in (
        "fingertip_fk", "wrist_fk", "palm_chordal", "velocity_fk", "contact")}
    pred = torch.zeros(2, 1, 204)
    obs = torch.zeros_like(pred, dtype=torch.bool)
    pred[:, :, :6] = 2
    pred[0, :, 6:12] = 4
    obs[:, :, :6] = True
    obs[0, :, 6:12] = True
    loss, report = fine_objective(pred, torch.zeros_like(pred), obs,
                                  {"frame_mask": torch.ones(2, 1, dtype=torch.bool)},
                                  None, None, None, None, torch.zeros(2), weights, cfg["fine"])
    assert loss == 10.
    assert report["wrist_endpoint"] == 4 and report["wrist_endpoint_n"] == 2
    assert report["finger_endpoint"] == 16 and report["finger_endpoint_n"] == 1
