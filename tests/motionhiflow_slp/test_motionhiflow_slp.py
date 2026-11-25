from __future__ import annotations

import copy
import logging

import numpy as np
import pytest
import torch
from torch.nn.functional import interpolate

from slp.config import compose
from slp.data import Split
from slp.registry import get_method
from slp.tracking.run import RunContext

from methods.motionhiflow_slp.codec import MotionVAE, codec_losses
from methods.motionhiflow_slp.pyramid import PyramidFlow, resample


@pytest.fixture
def small_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_resampling_excludes_padding_and_uses_each_clips_length():
    values = torch.tensor([[[1.0], [4.0], [8.0], [900.0]], [[7.0], [5.0], [2.0], [4.0]]])
    lengths, out_lengths = torch.tensor([3, 4]), torch.tensor([6, 2])
    actual = resample(values, lengths, out_lengths)
    for i, (length, out_length) in enumerate(zip(lengths.tolist(), out_lengths.tolist())):
        reference = interpolate(values[i:i + 1, :length].transpose(1, 2), size=out_length,
                                mode="linear", align_corners=False).transpose(1, 2)
        torch.testing.assert_close(actual[i:i + 1, :out_length], reference)
        assert not torch.count_nonzero(actual[i, out_length:])


@pytest.mark.parametrize("scales,gamma", [([1.0], None), ([0.3, 0.6, 1.0], None),
                                          ([0.3, 0.6, 1.0], 1.0)])
def test_pyramid_sampler_reaches_clean_target_with_oracle_velocity(scales, gamma):
    flow = PyramidFlow(scales, gamma)
    generator = torch.Generator().manual_seed(30)
    clean = torch.randn(2, 11, 4, 3, generator=generator)
    noise = torch.randn(clean.shape, generator=generator)
    lengths = torch.tensor([11, 3])
    counts = flow.step_counts(7)
    seen = []

    def velocity(state, time, scale, stage_lengths):
        index = scales.index(scale)
        stage = flow.stages[index]
        progress = (time - stage.time_start) / stage.time_width
        expected, target, _, expected_lengths = flow.training_pair(
            clean, noise, lengths, index, progress,
        )
        torch.testing.assert_close(expected_lengths, stage_lengths)
        torch.testing.assert_close(state, expected, atol=3e-6, rtol=1e-5)
        seen.append(scale)
        return target

    sampled = flow.sample(velocity, noise, lengths, 7)
    for i, length in enumerate(lengths):
        torch.testing.assert_close(sampled[i, :length], clean[i, :length], atol=3e-6, rtol=1e-5)
        assert not torch.count_nonzero(sampled[i, length:])
    assert len(seen) == 7
    assert counts == [seen.count(scale) for scale in scales]


def test_codec_reconstructs_odd_and_single_frame_lengths(small_cpu):
    model = MotionVAE(width=8, latent_dim=4, depth=1, stride=4).eval()
    poses = torch.randn(2, 11, 133)
    lengths = torch.tensor([5, 11])
    latent, _, latent_lengths = model.encode(poses, lengths)
    solo, _, _ = model.encode(poses[:1, :5], lengths[:1])
    assert latent_lengths.tolist() == [2, 3]
    torch.testing.assert_close(latent[0, :2], solo[0])
    batch_rec = model.decode(latent, lengths, 11)
    solo_rec = model.decode(solo, lengths[:1], 5)
    torch.testing.assert_close(batch_rec[0, :5], solo_rec[0])
    assert not torch.count_nonzero(batch_rec[0, 5:])

    cfg = {"part_weights": [1, 2, 2, 1], "pool_weight": 0.5, "pool_min_scale": 0.3,
           "velocity_weight": 0.5, "kl_weight": 0.02}
    model.train()
    losses = codec_losses(model, poses[:1, :1], torch.tensor([1]), cfg)
    assert losses["velocity"].item() == 0
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()


def test_fit_predict_export_and_reload(tmp_path, small_cpu):
    cfg = compose("motionhiflow_slp", "phoenix_smplx", [
        "method.device=cpu", "method.workers=0", "method.text.backend=bytes",
        "method.text.byte_dim=12", "method.codec.width=8", "method.codec.latent_dim=4",
        "method.codec.depth=1", "method.codec.updates=2", "method.codec.batch_size=3",
        "method.codec.microbatch=2", "method.flow.width=16", "method.flow.heads=2",
        "method.flow.double_blocks=1", "method.flow.single_blocks=1",
        "method.flow.time_patch=2", "method.flow.updates=2", "method.flow.batch_size=3",
        "method.flow.snapshot_steps=[1,2]",
        "method.flow.microbatch=2", "method.sampler.steps=4", "method.sampler.min_frames=3",
        "method.sampler.max_frames=14", "method.sampler.batch_size=2",
    ])
    cls = get_method("motionhiflow_slp")
    ctx = RunContext(tmp_path, 55, logging.getLogger(__name__))
    inputs = [
        {"name": "clip/one", "text": "heute Sonne", "lang": "de_DGS"},
        {"name": "clip/two", "text": "今天", "src": "csl"},
        {"name": "clip/three", "text": "hello", "lang": "en_ASL"},
    ]
    rng = np.random.default_rng(2026)
    targets = [rng.normal(size=(length, 133)).astype(np.float32) for length in (5, 9, 12)]
    split = Split(inputs, targets)
    torch.manual_seed(21)
    model = cls(cfg, ctx)
    model.fit(split, split)
    model.save()
    for step in (1, 2):
        snapshot = ctx.ckpt_dir / f"model_step{step:06d}.pt"
        state = torch.load(snapshot, map_location="cpu", weights_only=True)
        assert state["step"] == step

    prediction = model.predict(inputs)
    assert all(3 <= len(p) <= 14 and p.shape[1] == 133 and np.isfinite(p).all()
               for p in prediction)
    with np.load(tmp_path / "predictions.npz") as export:
        assert export.files == ["clip__one", "clip__two", "clip__three"]
        for key, array in zip(export.files, prediction):
            np.testing.assert_array_equal(export[key], array)
    restored = cls(cfg, ctx)
    restored.load(ctx.ckpt_dir)
    # Inputs can contain unrelated metadata. Reference duration has no effect.
    restored.sampler["batch_size"] = 1
    restored.sampler.pop("lengths_from_npz")
    repeated = restored.predict([{**item, "length": 400} for item in inputs])
    for a, b in zip(prediction, repeated):
        np.testing.assert_allclose(a, b, atol=2e-6, rtol=1e-5)

    reference = tmp_path / "predictions.npz"
    ablation_dir = tmp_path / "single_scale"
    ablation_dir.mkdir()
    ablation = copy.deepcopy(cfg)
    ablation["method"]["init_codec"] = str(ctx.ckpt_dir / "codec.pt")
    ablation["method"]["flow"]["scales"] = [1.0]
    ablation["method"]["sampler"]["lengths_from_npz"] = str(reference)
    single = cls(ablation, RunContext(ablation_dir, 55, ctx.logger))
    single.fit(split, split)
    single_prediction = single.predict(inputs)
    assert [len(x) for x in single_prediction] == [len(x) for x in prediction]
    for key, state in model.codec.state_dict().items():
        torch.testing.assert_close(state, single.codec.state_dict()[key])


def test_endpoint_fk_matches_soke_wrist_and_tip_geometry(small_cpu):
    from slp.evaluation.smplx_fk import SMPLXJoints, smplx_dir, stats_dir

    from methods.motionhiflow_slp.endpoint import SigningEndpointLoss

    if not (smplx_dir() / "smplx" / "SMPLX_NEUTRAL.npz").is_file():
        pytest.skip("SMPL-X rig is unavailable")
    ref = SMPLXJoints(device="cpu")
    fk = SigningEndpointLoss(stats_dir())
    pose = torch.randn(2, 3, 133).requires_grad_()
    world, left, right = ref(pose.detach().view(-1, 133).numpy())
    wrists = world[:, 20:22]
    tips = torch.stack((left[:, 4::4], right[:, 4::4]), dim=1)
    wr, fingers = fk.positions(pose)
    torch.testing.assert_close(wr, 100 * (wrists - world[:, :1]), atol=2e-4, rtol=1e-5)
    torch.testing.assert_close(fingers, 100 * (tips - wrists.unsqueeze(-2)), atol=2e-4, rtol=1e-5)
    a, b = fk(pose, torch.zeros_like(pose), torch.tensor([1, 3]), samples=2)
    (a + b).backward()
    assert torch.isfinite(pose.grad).all()
    assert pose.grad[0, 1:].count_nonzero().item() == 0
    assert pose.grad[:, :, 30:120].abs().sum().item() > 0
