import logging

import numpy as np
import pytest
import torch

from slp.config import compose
from slp.data import Split
from slp.registry import get_method
from slp.tracking.run import RunContext

from methods.articulatory_hierarchical_flow_matching.data import POSE_DIM, MotionBatch
from methods.articulatory_hierarchical_flow_matching.flow import sample, training_stages
from methods.articulatory_hierarchical_flow_matching.subspace import (
    MODES,
    Representation,
    articulatory_observations,
    calibrate,
    dct_columns,
    envelope,
    frame_intervals,
    low_pass,
)

NAME = "articulatory_hierarchical_flow_matching"


@pytest.fixture(autouse=True)
def limit_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("mode", MODES)
def test_projected_euler_and_padding(mode):
    g = torch.Generator().manual_seed(7)
    rep = Representation(mode, 16, 8).double()
    rep.scales.copy_(torch.tensor([0.8, 1.2]))
    rep.axes.copy_(torch.linalg.qr(torch.randn(rep.axes.shape, generator=g, dtype=torch.float64)).Q)
    lengths = torch.tensor([1, 16, 137, 399])
    proj = rep.projector(lengths, 399)
    start = torch.randn(4, 399, POSE_DIM, generator=g, dtype=torch.float64)
    final = torch.randn(start.shape, generator=g, dtype=start.dtype)
    target = final.masked_fill(~proj.mask[..., None], 0)
    noise = start.masked_fill(~proj.mask[..., None], 0)
    p, q = proj.coarse(final), proj.detail(final)
    torch.testing.assert_close(p + q, target)
    torch.testing.assert_close(proj.coarse(p), p, atol=2e-12, rtol=0)
    torch.testing.assert_close(proj.coarse(q), torch.zeros_like(q), atol=2e-12, rtol=0)
    assert torch.equal(proj.coarse_rank + proj.detail_rank, lengths * POSE_DIM)
    assert proj.coarse_rank[0] == POSE_DIM and proj.detail_rank[0] == 0
    torch.testing.assert_close(rep.projector(lengths[2:3], 137).coarse(final[2:3, :137]),
                               p[2:3, :137], atol=2e-12, rtol=0)

    modes = (False,) if mode == "single" else (False, True)
    for unconstrained in modes:
        stages = training_stages(target, noise, proj, rep.scales, lengths, unconstrained)

        def velocity(_, time, stage, coarse):
            if stage == 1:
                torch.testing.assert_close(coarse, stages[0].target)
            return stages[stage].target - stages[stage].start

        result = sample(velocity, noise, proj, rep, (3, 5), unconstrained)
        torch.testing.assert_close(result, target, atol=4e-12, rtol=0)
    if mode != "single":
        result = sample(lambda x, u, stage, c: -x, noise, proj, rep, (2, 2), False,
                        coarse_override=p, residual_override=-q)
        torch.testing.assert_close(proj.coarse(result), p, atol=3e-12, rtol=0)


def test_articulatory_descriptor_order_and_frame_bins():
    poses = torch.zeros(2, POSE_DIM)
    poses[:, 120:] = 3
    body = torch.zeros(2, 144, 3)
    left = torch.zeros(2, 21, 3)
    left[:, 5, 0] = 1
    left[:, 9, 1] = 1
    right = left + torch.tensor([1., 0., 0.])
    result = articulatory_observations(poses, body, left, right)
    assert result.shape == (2, 96)
    torch.testing.assert_close(result[0, :18], torch.tensor([
        .5, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1,
    ]))
    torch.testing.assert_close(result[:, 48:73], torch.ones(2, 25))
    torch.testing.assert_close(result[:, 83:], torch.full((2, 13), 3.))
    lengths = torch.tensor([1, 7])
    assert frame_intervals(lengths, 7)[1].tolist() == [0, 0, 1, 1, 2, 2, 3]


@pytest.mark.parametrize("mode", MODES)
def test_streaming_calibration_matches_dense_moments(mode):
    g = torch.Generator().manual_seed(2)
    poses = torch.randn(3, 31, POSE_DIM, dtype=torch.float64, generator=g)
    lengths = torch.tensor([1, 11, 31])
    valid = torch.arange(31)[None, :] < lengths[:, None]
    poses[~valid] = 0
    rep = Representation(mode, 8, 8).double()
    batches = [[MotionBatch(poses, lengths, ["a"] * 3, torch.zeros(3, dtype=torch.long))]]
    mapping = torch.randn(POSE_DIM, 96, dtype=poses.dtype, generator=g)

    def extract(x):
        return x @ mapping

    extract.provenance = {"fixture": True}
    cfg = {"weights": [1, 1, 1, 1, 0.5], "random_seed": 3, "unconstrained": False}
    calibrate(rep, batches, cfg, extract if mode == "articulatory" else None, ["1", "2", "3"], False)
    projector = rep.projector(lengths, 31)
    state = rep.encode(poses).masked_fill(~valid[..., None], 0)
    coarse, fine = projector.coarse(state), projector.detail(state)
    torch.testing.assert_close(rep.mean, poses[valid].mean(0))
    torch.testing.assert_close(rep.scales[0], (coarse.square().sum() / projector.coarse_rank.sum()).sqrt())
    if projector.detail_rank.sum():
        torch.testing.assert_close(rep.scales[1], (fine.square().sum() / projector.detail_rank.sum()).sqrt())
    if mode == "articulatory":
        count = (lengths + 7) // 8
        dct = dct_columns(lengths, 31, count, int(count.max()), poses.dtype)
        a = (poses - low_pass(poses, dct))[valid]
        obs = poses @ mapping
        b = (obs - low_pass(obs, dct))[valid]
        weights = torch.tensor(cfg["weights"], dtype=poses.dtype)
        sizes = torch.tensor([6, 12, 30, 35, 13])
        weights = (weights / sizes).sqrt().repeat_interleave(sizes)
        basis = torch.linalg.svd(rep.white @ (a.T @ b) * (weights / obs[valid].std(0, unbiased=False)),
                                 full_matrices=False).U[:, :8]
        torch.testing.assert_close(rep.axes @ rep.axes.T, basis @ basis.T)


def test_train_predict_reload_and_component_oracles(tmp_path, monkeypatch):
    class FixtureFK:
        provenance = {"fixture": True}

        def __init__(self, *args):
            pass

        def __call__(self, rows):
            return rows[..., :96].roll(1, dims=-1)

    monkeypatch.setattr(f"methods.{NAME}.method.DescriptorExtractor", FixtureFK)
    cfg = compose(NAME, overrides=[
        "method.device=cpu", "method.precision=fp32", "method.workers=0",
        "method.text.backend=bytes", "method.text.byte_dim=8", "method.flow.width=16",
        "method.flow.heads=2", "method.flow.depth=1", "method.flow.updates=2",
        "method.flow.batch_size=2", "method.flow.microbatch=1", "method.flow.eval_every=1",
        "method.flow.snapshot_steps=[1]", "method.flow.energy_weight=0.1",
        "method.sampler.steps=[2,2]", "method.sampler.min_frames=1",
    ])
    rng = np.random.default_rng(12)
    inputs = [{"name": f"dev/{i}", "lang": lang, "text": text}
              for i, (lang, text) in enumerate((("de_DGS", "Wind"), ("zh_CSL", "下雨"),
                                                ("en_ASL", "it snows")))]
    targets = [rng.normal(size=(t, POSE_DIM)).astype(np.float32) for t in (1, 17, 13)]
    split = Split(inputs, targets)
    ctx = RunContext(tmp_path, 23, logging.getLogger(__name__))
    method = get_method(NAME)(cfg, ctx)
    method.fit(split, split)
    method.save()
    assert (ctx.ckpt_dir / "model_step000001.pt").is_file()
    before = method.predict(inputs)
    with np.load(tmp_path / "predictions.npz") as export:
        for key, value in zip(method.prediction_keys(inputs), before, strict=True):
            np.testing.assert_array_equal(export[key], value)
    loaded = get_method(NAME)(cfg, ctx)
    loaded.load(ctx.ckpt_dir)
    after = loaded.predict(inputs)
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(a, b)
    rep = loaded.representation
    loaded.sampler["batch_size"] = 1
    lengths = [len(row) for row in targets]
    for key in ("coarse", "residual"):
        rows = loaded.generate(inputs, oracle_component=key, oracle_targets=targets)
        for length, reference, row in zip(lengths, targets, rows, strict=True):
            proj = rep.projector(torch.tensor([length]), length)
            part = proj.coarse if key == "coarse" else proj.detail
            torch.testing.assert_close(part(rep.encode(torch.from_numpy(row)[None])),
                                       part(rep.encode(torch.from_numpy(reference)[None])),
                                       atol=3e-4, rtol=3e-4)
    cfg["method"]["sampler"]["lengths_from_npz"] = str(tmp_path / "fixed.npz")
    np.savez(tmp_path / "fixed.npz", **dict(zip(method.prediction_keys(inputs), targets, strict=True)))
    assert [len(row) for row in loaded.predict(inputs)] == lengths
    y = torch.zeros(1, 7, POSE_DIM)
    energies, mask = envelope(y, torch.tensor([7]), rep)
    assert energies.shape == (1, 4, 16) and mask.all()


def test_common_path_uses_same_two_hand_alignment():
    from slp.evaluation.metrics import articulatory_geometry_scores, dtw_cost, dtw_path

    cost = np.array([[1, 2, 5], [8, 4, 2]])
    i, j = dtw_path(cost)
    assert dtw_cost(cost) == (cost[i, j].sum(), len(i))
    assert (i[0], j[0], i[-1], j[-1]) == (0, 0, 1, 2)
    target = np.zeros((3, 96))
    target[:, [11, 17]] = 1
    target[1, :3] = 0.1
    pred = np.repeat(target, [1, 2, 1], axis=0)
    scores = articulatory_geometry_scores(pred, target, np.ones(96))
    assert scores["dtw_wrist_cm"] == scores["dtw_palm_deg"] == 0
    pred[:, 48:73] = 0.02
    scores = articulatory_geometry_scores(pred, target, np.ones(96))
    assert scores["dtw_bimanual_cm"] == pytest.approx(2)
