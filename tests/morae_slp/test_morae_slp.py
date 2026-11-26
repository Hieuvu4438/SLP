from __future__ import annotations

import copy
import logging
import json
import math

import numpy as np
import pytest
import torch

from slp.config import compose
from slp.data import Split
from slp.registry import get_method
from slp.tracking.run import RunContext

from methods.morae_slp.autoencoder import LatentSpace, autoencoder_losses
from methods.morae_slp.data import (GROUPS, N_GROUPS, POSE_DIM, UNITS, ConcatSplit,
                                    group_channel_matrix, sample_group_mask, text_input,
                                    with_velocity)
from methods.morae_slp.dit import LatentDiT
from methods.morae_slp.flow import cfg_combine, integrate, training_pair
from methods.morae_slp.jepa import SignJEPA, teacher_momentum


@pytest.fixture
def small_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


TINY_ENCODER = {"width": 16, "depth": 2, "heads": 2, "mlp_ratio": 2.0, "input_velocity": True,
                "feature_layers": 2}
TINY_SPACE = {"latent_dim": 4, "encoder": TINY_ENCODER,
              "compressor": {"attn_width": 32, "heads": 4, "rope": True},
              "decoder": {"width": 16, "depth": 1, "heads": 2}}
AE_CFG = {"part_weights": [1, 2, 2, 1], "kl_weight": 1e-4, "velocity_weight": 0.5,
          "fk_weight": 0.0, "fk_frames": 4}


# ----------------------------------------------------------------------------- groups / masks
def test_groups_partition_the_133_channels():
    channels = sorted(c for g in GROUPS for c in g)
    assert channels == list(range(POSE_DIM)) and len(GROUPS) == N_GROUPS == 42
    for name, units in UNITS.items():
        assert sorted(g for u in units for g in u) == list(range(N_GROUPS)), name
    # Fingers are three consecutive joints; left index finger = SMPL-X 25..27 = channels 30:39.
    assert list(GROUPS[10]) + list(GROUPS[11]) + list(GROUPS[12]) == list(range(30, 39))
    assert (10, 11, 12) in UNITS["chain"] and tuple(range(10, 25)) in UNITS["hand"]


@pytest.mark.parametrize("probs", [{"joint": 1.0}, {"chain": 1.0}, {"hand": 1.0},
                                   {"joint": 0.4, "chain": 0.4, "hand": 0.2}])
def test_group_masks_hide_whole_units_and_reach_the_ratio(probs):
    gen = torch.Generator().manual_seed(0)
    mask = sample_group_mask(64, 0.3, probs, gen)
    assert mask.shape == (64, N_GROUPS)
    assert (mask.sum(1) >= 13).all() and (mask.sum(1) < N_GROUPS).all()
    if len(probs) == 1:  # a single granularity hides whole units only
        for unit in UNITS[next(iter(probs))]:
            part = mask[:, list(unit)]
            assert (part.all(1) | ~part.any(1)).all()
    if probs == {"joint": 1.0}:
        assert (mask.sum(1) == 13).all()
    else:  # coarser units are actually used
        assert mask.sum(1).float().mean() >= 13


def test_channel_membership_includes_velocities():
    member = group_channel_matrix(True)
    assert member.shape == (N_GROUPS, 2 * POSE_DIM) and (member.sum(0) == 1).all()
    assert member[41, 123:133].all() and member[41, 133 + 123:].all()


def test_velocity_input_is_zero_on_first_and_padded_frames():
    poses = torch.randn(2, 5, POSE_DIM)
    u = with_velocity(poses, torch.tensor([3, 5]))
    assert torch.equal(u[0, 3:], torch.zeros_like(u[0, 3:]))
    assert torch.equal(u[:, 0, POSE_DIM:], torch.zeros(2, POSE_DIM))
    torch.testing.assert_close(u[1, 4, POSE_DIM:], poses[1, 4] - poses[1, 3])


def test_concat_split_maps_indices():
    a = Split([{"name": "a0"}, {"name": "a1"}], [np.zeros((2, 1)), np.ones((2, 1))])
    b = Split([{"name": "b0"}], [np.full((3, 1), 2.0)])
    union = ConcatSplit([a, b])
    assert len(union) == 3 and union.inputs[2]["name"] == "b0"
    assert union.targets[1][0, 0] == 1 and len(union.targets[2]) == 3


# ----------------------------------------------------------------------------- Sign-JEPA
def _jepa():
    return SignJEPA(width=16, depth=2, heads=2, pred_width=8, pred_depth=1, pred_heads=2,
                    mlp_ratio=2.0, input_velocity=True, feature_layers=2)


def test_jepa_masks_inputs_and_trains_only_the_online_branch(small_cpu):
    torch.manual_seed(0)
    model = _jepa()
    poses, lengths = torch.randn(3, 7, POSE_DIM), torch.tensor([7, 4, 1])
    mask = sample_group_mask(3, 0.3, {"joint": 1.0}, torch.Generator().manual_seed(1))
    u = model.inputs(poses, lengths)
    masked = model.masked_inputs(u, mask)
    cm = model.channel_mask(mask)
    assert torch.equal(masked[:, :, ~cm[0]][0], u[0][:, ~cm[0]])
    assert torch.allclose(masked[0][:, cm[0]], model.mask_value[cm[0]].expand(7, -1))
    out = model.loss(poses, lengths, mask)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert all(p.grad is None for p in model.teacher.parameters())
    assert model.mask_value.grad is not None and model.mask_value.grad.abs().sum() > 0
    before = [p.clone() for p in model.teacher.parameters()]
    with torch.no_grad():
        for p in model.online.parameters():
            p.add_(1.0)
    model.update_teacher(0.75)
    for b, t, o in zip(before, model.teacher.parameters(), model.online.parameters()):
        torch.testing.assert_close(t, 0.75 * b + 0.25 * o)
    assert teacher_momentum(0, 10, 0.996) == 0.996 and teacher_momentum(10, 10, 0.996) == 1.0


def test_encoders_ignore_padding(small_cpu):
    torch.manual_seed(0)
    model = _jepa().eval()
    space = LatentSpace("vae", **TINY_SPACE).eval()
    poses = torch.randn(1, 6, POSE_DIM)
    padded = torch.cat([poses, torch.randn(1, 4, POSE_DIM)], 1)
    for n in (6,):
        a = model.features(poses, torch.tensor([n]))
        b = model.features(padded, torch.tensor([n]))
        torch.testing.assert_close(a, b[:, :n], atol=1e-5, rtol=1e-5)
        ma, _ = space.posterior(poses, torch.tensor([n]))
        mb, _ = space.posterior(padded, torch.tensor([n]))
        torch.testing.assert_close(ma, mb[:, :n], atol=1e-5, rtol=1e-5)
        da = space.decode(ma, torch.tensor([n]))
        db = space.decode(torch.cat([ma, torch.randn(1, 4, 4)], 1), torch.tensor([n]))
        torch.testing.assert_close(da, db[:, :n], atol=1e-5, rtol=1e-5)
        assert not db[:, n:].count_nonzero()


# ----------------------------------------------------------------------------- latent space
@pytest.mark.parametrize("kind", ["jepa", "ae", "vae", "raw"])
def test_latent_space_losses_and_parameters(kind, small_cpu):
    torch.manual_seed(0)
    space = LatentSpace(kind, **TINY_SPACE)
    if kind == "jepa":
        jepa = _jepa()
        space.load_teacher(jepa)
        torch.testing.assert_close(space.features(torch.ones(1, 3, POSE_DIM), torch.tensor([3])),
                                   jepa.features(torch.ones(1, 3, POSE_DIM), torch.tensor([3])))
    poses, lengths = torch.randn(2, 5, POSE_DIM), torch.tensor([5, 2])
    if kind == "raw":
        assert space.trainable_parameters() == []
        q = space.encode_flow(poses, lengths)
        torch.testing.assert_close(space.decode_flow(q, lengths)[0], poses[0])
        return
    encoder_trained = any(p is q for p in space.trainable_parameters()
                          for q in space.encoder.parameters())
    assert encoder_trained == (kind in ("ae", "vae"))
    space.train()
    out = autoencoder_losses(space, poses, lengths, AE_CFG)
    assert torch.isfinite(out["loss"]) and ("kl" in out) == (kind != "ae")
    out["loss"].backward()


def test_flow_standardisation_round_trip():
    space = LatentSpace("raw", **TINY_SPACE)
    space.latent_mean.fill_(2.0)
    space.latent_std.fill_(3.0)
    z = torch.randn(2, 4, POSE_DIM)
    torch.testing.assert_close(space.from_flow(space.to_flow(z)), z)


# ----------------------------------------------------------------------------- flow
def test_training_pair_is_the_linear_path():
    q, eps, t = torch.randn(3, 4, 2), torch.randn(3, 4, 2), torch.tensor([0.0, 0.5, 1.0])
    x, v = training_pair(q, eps, t)
    torch.testing.assert_close(x[0], eps[0])
    torch.testing.assert_close(x[2], q[2])
    torch.testing.assert_close(v, q - eps)


@pytest.mark.parametrize("solver,steps,tol", [("rk4", 20, 1e-4), ("heun", 200, 1e-3),
                                              ("euler", 4000, 2e-3)])
def test_solvers_follow_the_gaussian_reference_flow(solver, steps, tol):
    # Paper Eq. (S7): for z ~ N(0, lam) the optimal velocity is (t lam - (1 - t)) / q(t) * y with
    # q(t) = t^2 lam + (1 - t)^2, and the flow maps eps to sqrt(lam) * eps at t = 1.
    lam = torch.tensor([0.05, 1.0, 9.0], dtype=torch.float64)
    eps = torch.randn(5, 3, dtype=torch.float64)

    def field(y, t):
        tt = t[:, None]
        return (tt * lam - (1 - tt)) / (tt ** 2 * lam + (1 - tt) ** 2) * y

    out = integrate(field, eps.clone(), steps, solver)
    torch.testing.assert_close(out, lam.sqrt() * eps, atol=tol, rtol=tol)


def test_cfg_combine_and_masked_integration():
    c, u = torch.ones(2, 3, 1), torch.zeros(2, 3, 1)
    torch.testing.assert_close(cfg_combine(c, u, 4.5), 4.5 * c)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    out = integrate(lambda x, t: torch.ones_like(x), torch.zeros(2, 3, 1), 3, "rk4", valid)
    torch.testing.assert_close(out[..., 0], valid.float())


def test_dit_is_zero_at_init_and_dropped_text_is_ignored(small_cpu):
    torch.manual_seed(0)
    dit = LatentDiT(4, 6, width=16, depth=2, heads=2, ff=32, dropout=0.0)
    x, t = torch.randn(2, 5, 4), torch.rand(2)
    valid = torch.ones(2, 5, dtype=torch.bool)
    tokens, mask = torch.randn(2, 3, 6), torch.ones(2, 3, dtype=torch.bool)
    langs = torch.tensor([0, 1])
    drop = torch.tensor([True, True])
    assert not dit(x, t, valid, dit.condition(tokens, mask, langs, drop)).count_nonzero()
    for p in dit.parameters():  # leave the zero-initialisation behind
        p.data.add_(0.05 * torch.randn_like(p))
    a = dit(x, t, valid, dit.condition(tokens, mask, langs, drop))
    b = dit(x, t, valid, dit.condition(torch.randn(2, 4, 6), torch.ones(2, 4, dtype=torch.bool),
                                       langs, drop))
    torch.testing.assert_close(a, b)
    keep = dit(x, t, valid, dit.condition(tokens, mask, langs, ~drop))
    assert not torch.allclose(a, keep)


# ----------------------------------------------------------------------------- end to end
def _tiny_cfg(*extra):
    return compose("morae_slp", "phoenix_smplx", [
        "method.device=cpu", "method.precision=fp32", "method.workers=0",
        "method.text.backend=bytes", "method.text.byte_dim=8", "method.latent.dim=4",
        "method.jepa.width=16", "method.jepa.depth=2", "method.jepa.heads=2",
        "method.jepa.pred_width=8", "method.jepa.pred_depth=1", "method.jepa.pred_heads=2",
        "method.jepa.feature_layers=2", "method.jepa.updates=2", "method.jepa.batch_size=3",
        "method.jepa.microbatch=2", "method.jepa.crop_min=2", "method.jepa.crop_max=6",
        "method.ae.compressor.attn_width=32", "method.ae.compressor.heads=4",
        "method.ae.decoder.width=16", "method.ae.decoder.depth=1", "method.ae.decoder.heads=2",
        "method.ae.updates=2", "method.ae.batch_size=3", "method.ae.microbatch=2",
        "method.ae.crop_min=2", "method.ae.crop_max=12", "method.ae.velocity_weight=0.5",
        "method.flow.width=16", "method.flow.depth=2", "method.flow.heads=2", "method.flow.ff=32",
        "method.flow.updates=2", "method.flow.batch_size=3", "method.flow.microbatch=2",
        "method.flow.snapshot_steps=[1]", "method.flow.max_frames=64",
        "method.sampler.steps=2", "method.sampler.min_frames=3", "method.sampler.max_frames=14",
        "method.sampler.batch_size=2", "method.sampler.cfg_scale=2.0", *extra,
    ])


def _toy_split():
    inputs = [{"name": "clip/one", "text": "heute Sonne", "lang": "de_DGS"},
              {"name": "clip/two", "text": "regen", "src": "phoenix"},
              {"name": "clip/three", "text": "morgen wind", "lang": "de_DGS"}]
    rng = np.random.default_rng(2026)
    targets = [rng.normal(size=(n, POSE_DIM)).astype(np.float32) for n in (5, 9, 12)]
    return inputs, Split(inputs, targets)


def test_fit_predict_export_reload_and_reuse(tmp_path, small_cpu):
    cfg = _tiny_cfg()
    cls = get_method("morae_slp")
    ctx = RunContext(tmp_path, 7, logging.getLogger(__name__))
    inputs, split = _toy_split()
    torch.manual_seed(3)
    model = cls(cfg, ctx)
    model.fit(split, split)
    model.save()
    for name in ("jepa.pt", "ae.pt", "model.pt", "model_step000001.pt"):
        assert (ctx.ckpt_dir / name).is_file(), name
    pred = model.predict(inputs)
    assert all(3 <= len(p) <= 14 and p.shape[1] == POSE_DIM and np.isfinite(p).all()
               for p in pred)
    with np.load(tmp_path / "predictions.npz") as export:
        assert export.files == ["clip__one", "clip__two", "clip__three"]
        for key, arr in zip(export.files, pred):
            np.testing.assert_array_equal(export[key], arr)

    restored = cls(cfg, ctx)
    restored.load(ctx.ckpt_dir)
    restored.sampler["batch_size"] = 1  # batch-size independent noise and lengths
    again = restored.predict(inputs)
    for a, b in zip(pred, again):
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)

    # Reuse the compact latent and lock the durations: only the flow is retrained.
    reuse = copy.deepcopy(cfg)
    reuse["method"]["init_ae"] = str(ctx.ckpt_dir / "ae.pt")
    reuse["method"]["sampler"]["lengths_from_npz"] = str(tmp_path / "predictions.npz")
    other = tmp_path / "reuse"
    other.mkdir()
    second = cls(reuse, RunContext(other, 7, ctx.logger))
    second.fit(split, split)
    assert not (other / "checkpoints" / "jepa.pt").exists()
    assert [len(p) for p in second.predict(inputs)] == [len(p) for p in pred]
    for key, value in model.space.state_dict().items():
        if key.startswith("decoder."):
            torch.testing.assert_close(value, second.space.state_dict()[key])


@pytest.mark.parametrize("kind", ["raw", "vae"])
def test_controls_fit_and_predict(kind, tmp_path, small_cpu):
    cfg = _tiny_cfg(f"method.latent.kind={kind}", "method.sampler.solver=euler",
                    "method.sampler.cfg_scale=1.0", "method.flow.text_mode=adaln")
    cls = get_method("morae_slp")
    inputs, split = _toy_split()
    model = cls(cfg, RunContext(tmp_path, 1, logging.getLogger(__name__)))
    model.fit(split, split)
    assert not (tmp_path / "checkpoints" / "jepa.pt").exists()
    assert model.space.latent_dim == (POSE_DIM if kind == "raw" else 4)
    pred = model.predict(inputs[:2])
    assert len(pred) == 2 and all(np.isfinite(p).all() for p in pred)


def test_config_validation_rejects_bad_sampler(tmp_path):
    cfg = _tiny_cfg("method.sampler.solver=dopri5")
    with pytest.raises(ValueError):
        get_method("morae_slp")(cfg, RunContext(tmp_path, 0, logging.getLogger(__name__)))


def test_duration_prior_matches_training_lengths(tmp_path, small_cpu):
    cfg = _tiny_cfg("method.flow.updates=1", "method.flow.ema_decay=0.0")
    cls = get_method("morae_slp")
    inputs, split = _toy_split()
    model = cls(cfg, RunContext(tmp_path, 0, logging.getLogger(__name__)))
    model.fit(split, split)
    bias = model.gen.duration.mlp[-1].bias.item()
    assert abs(bias - np.mean(np.log([5, 9, 12]))) < 0.05
    assert math.isfinite(bias)


# ----------------------------------------------------------------------------- diagnostics
def test_cross_joint_readout_recovers_linear_coordination():
    from methods.morae_slp.diagnostics import CrossJoint, group_r2, raw_cp

    rng = np.random.default_rng(0)
    base = rng.normal(size=(4000, 8))
    mix = rng.normal(size=(8, POSE_DIM))
    frames = (base @ mix + 0.01 * rng.normal(size=(4000, POSE_DIM))).astype(np.float32)
    model = CrossJoint(frames[:2000])
    r2 = group_r2(model.standardise(frames[2000:]), model.predict(frames[2000:]))
    assert r2.shape == (N_GROUPS,) and r2.min() > 0.95
    noise = rng.normal(size=(4000, POSE_DIM)).astype(np.float32)
    null = CrossJoint(noise[:2000])
    r2_noise = group_r2(null.standardise(noise[2000:]), null.predict(noise[2000:]))
    assert r2_noise.max() < 0.02  # only the overfitting penalty (about -p/n) remains
    split = Split([{"name": str(i)} for i in range(20)],
                  [frames[i * 200:(i + 1) * 200] for i in range(20)])
    out = raw_cp(split, np.random.default_rng(1), clips=8, frames=50)
    assert out["all_other_groups"]["mean_lhand"] > 0.9
    assert out["hand_shortcut_r2_drop"] > -0.05


def test_spectrum_and_reconstruction_on_a_tiny_space(small_cpu):
    from methods.morae_slp.diagnostics import reconstruction, spectrum

    torch.manual_seed(0)
    space = LatentSpace("vae", **TINY_SPACE).eval()
    rng = np.random.default_rng(0)
    split = Split([{"name": str(i)} for i in range(12)],
                  [rng.normal(size=(20, POSE_DIM)).astype(np.float32) for _ in range(12)])
    out = spectrum(space, split, torch.device("cpu"), np.random.default_rng(1), clips=6,
                   frames=8, workers=0)
    assert out["dim"] == 4 and out["kappa"] >= 1 and len(out["eigenvalues"]) == 4
    assert -1 <= out["rho_lambda_LC_over_LP"] <= 1
    pytest.importorskip("smplx")
    from slp.evaluation.smplx_fk import smplx_dir

    if not (smplx_dir() / "smplx" / "SMPLX_NEUTRAL.npz").is_file():
        pytest.skip("SMPL-X rig is unavailable")
    rec = reconstruction(LatentSpace("raw", **TINY_SPACE), split, torch.device("cpu"), 2,
                         ["soke_dtw_jpe_body", "soke_dtw_jpe_hand"])
    assert rec["clips"] == 2 and rec["soke_dtw_jpe_body"] < 1e-6  # raw space is lossless


def test_sampler_settings_change_a_trained_generator(tmp_path, small_cpu):
    cfg = _tiny_cfg("method.flow.updates=1", "method.export_predictions=false")
    cls = get_method("morae_slp")
    inputs, split = _toy_split()
    model = cls(cfg, RunContext(tmp_path, 0, logging.getLogger(__name__)))
    model.fit(split, split)
    torch.manual_seed(5)
    with torch.no_grad():  # move away from the adaLN-zero start, where every velocity is 0
        for p in model.gen.dit.parameters():
            p.add_(0.1 * torch.randn_like(p))
    lengths = {"lengths_from_npz": None}

    def run(**sampler):
        model.sampler.update(lengths, **sampler)
        return np.concatenate([p.ravel() for p in model.predict(inputs)])

    base = run(solver="rk4", steps=2, cfg_scale=1.0, temperature=1.0)
    np.testing.assert_allclose(base, run(solver="rk4", steps=2, cfg_scale=1.0, temperature=1.0))
    assert not np.allclose(base, run(solver="rk4", steps=2, cfg_scale=3.0, temperature=1.0))
    assert not np.allclose(base, run(solver="euler", steps=2, cfg_scale=1.0, temperature=1.0))
    assert not np.allclose(base, run(solver="rk4", steps=2, cfg_scale=1.0, temperature=0.5))
    # Guided sampling batches both branches; it must equal the two separate evaluations.
    cond = run(solver="euler", steps=1, cfg_scale=1.0, temperature=0.0)
    model.gen.dit.text.pool.weight.data.zero_()  # no text effect at all -> guidance is a no-op
    for block in model.gen.dit.blocks:
        block.xattn.proj.weight.data.zero_()
        block.xattn.proj.bias.data.zero_()
    a = run(solver="euler", steps=1, cfg_scale=1.0, temperature=0.0)
    b = run(solver="euler", steps=1, cfg_scale=5.0, temperature=0.0)
    np.testing.assert_allclose(a, b, atol=1e-5)
    assert cond.shape == a.shape


def test_equal_part_weights_give_the_plain_mse():
    from methods.morae_slp.autoencoder import part_mse

    pred, target = torch.randn(2, 6, POSE_DIM), torch.zeros(2, 6, POSE_DIM)
    pred[..., :30] *= 5  # unequal errors per part
    valid = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
    total, _ = part_mse(pred, target, valid, {"body": 1, "lhand": 1, "rhand": 1, "face": 1})
    plain = (pred[valid] ** 2).mean()
    torch.testing.assert_close(total, plain)


def test_duration_head_is_fp32_under_autocast():
    from methods.morae_slp.dit import DurationHead

    head = DurationHead(8, hidden=16)
    head.init_prior(5.3)
    tokens = torch.randn(64, 3, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        head.mlp[-1].weight.data.normal_(0, 0.01)
        out = head(tokens, torch.ones(64, 3, dtype=torch.bool), torch.zeros(64, dtype=torch.long))
    assert out.dtype == torch.float32
    assert len(torch.unique(out.exp().round())) > 8  # not quantised to a few bf16 levels


def test_uncalibrated_latent_checkpoint_is_refused_without_recalibration(tmp_path):
    from methods.morae_slp.stages import load_space, save_space

    spec = {"kind": "vae", **TINY_SPACE}
    save_space(LatentSpace(**spec), spec, tmp_path / "ae.pt")
    load_space(tmp_path / "ae.pt", torch.device("cpu"))
    with pytest.raises(ValueError):
        load_space(tmp_path / "ae.pt", torch.device("cpu"), need_calibration=True)
    save_space(LatentSpace(**spec), spec, tmp_path / "ae.pt", extra={"calibration": {"clips": 1}})
    load_space(tmp_path / "ae.pt", torch.device("cpu"), need_calibration=True)


def _fitted_tiny(tmp_path, *extra):
    cfg = _tiny_cfg("method.flow.updates=1", "method.export_predictions=false", *extra)
    model = get_method("morae_slp")(cfg, RunContext(tmp_path, 0, logging.getLogger(__name__)))
    inputs, split = _toy_split()
    model.fit(split, split)
    return model, inputs, split, cfg


def test_ensemble_is_the_mean_of_member_samples(tmp_path, small_cpu):
    from methods.morae_slp.data import text_input

    model, inputs, _, _ = _fitted_tiny(tmp_path)
    torch.manual_seed(5)
    with torch.no_grad():
        for p in model.gen.dit.parameters():
            p.add_(0.1 * torch.randn_like(p))
    texts, langs = zip(*(text_input(x) for x in inputs))
    langs = torch.tensor(langs)
    keys = [f"{lang}/{x['name']}" for lang, x in zip(langs.tolist(), inputs)]
    lengths = [5, 9, 12]
    m0 = model._sample(list(texts), langs, keys, lengths)
    m1 = model._sample(list(texts), langs, [k + "#1" for k in keys], lengths)
    model.sampler["ensemble"] = 2
    both = model._sample(list(texts), langs, keys, lengths)
    for a, b, e in zip(m0, m1, both):
        assert not np.allclose(a, b)
        np.testing.assert_allclose(e, (a + b) / 2, atol=1e-5, rtol=1e-5)
    # Member 0 is the single-sample prediction (ensemble=1 runs stay comparable).
    model.sampler["ensemble"] = 1
    for a, b in zip(m0, model._sample(list(texts), langs, keys, lengths)):
        np.testing.assert_array_equal(a, b)


def test_token_drop_keeps_a_subset_with_at_least_one_token(tmp_path):
    model = get_method("morae_slp")(_tiny_cfg("method.flow.token_drop=0.9"),
                                    RunContext(tmp_path, 0, logging.getLogger(__name__)))
    mask = torch.zeros(64, 7, dtype=torch.bool)
    for i in range(64):
        mask[i, : 1 + i % 7] = True
    torch.manual_seed(0)
    kept = model._drop_tokens(mask)
    assert (kept <= mask).all() and kept.any(1).all() and kept.sum() < mask.sum()
    model.flow_cfg["token_drop"] = 0.0
    assert torch.equal(model._drop_tokens(mask), mask)


def test_init_flow_warm_starts_the_generator(tmp_path, small_cpu):
    (tmp_path / "a").mkdir()
    first, inputs, split, cfg = _fitted_tiny(tmp_path / "a")
    ckpt = first.ctx.ckpt_dir
    bad = copy.deepcopy(cfg)
    bad["method"].update(init_ae=str(ckpt / "ae.pt"), init_flow=str(ckpt / "model.pt"))
    with pytest.raises(ValueError, match="recalibrate"):
        get_method("morae_slp")(bad, first.ctx)
    warm = copy.deepcopy(bad)
    warm["method"]["latent"]["recalibrate"] = False
    warm["method"]["flow"]["lr"] = 0.0  # no update: the warm start must survive unchanged
    (tmp_path / "b").mkdir()
    second = get_method("morae_slp")(warm, RunContext(tmp_path / "b", 0, first.ctx.logger))
    second.fit(split, split)
    for key, value in first.gen.state_dict().items():
        torch.testing.assert_close(value, second.gen.state_dict()[key])
    # A different latent standardisation is refused.
    other = copy.deepcopy(warm)
    other["method"]["init_flow"] = str(ckpt / "model.pt")
    state = torch.load(ckpt / "model.pt", weights_only=True)
    state["space"]["latent_std"] = state["space"]["latent_std"] * 2
    torch.save(state, tmp_path / "shifted.pt")
    other["method"]["init_flow"] = str(tmp_path / "shifted.pt")
    (tmp_path / "c").mkdir()
    third = get_method("morae_slp")(other, RunContext(tmp_path / "c", 0, first.ctx.logger))
    with pytest.raises(ValueError, match="latent_std"):
        third.fit(split, split)


def test_optimizer_trains_an_extra_group_at_a_scaled_rate():
    from methods.morae_slp.training import make_optimizer

    fast, slow = torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)
    opt = make_optimizer(fast.parameters(), {"lr": 1e-3, "weight_decay": 0.1,
                                             "betas": [0.9, 0.99]},
                         extra=(list(slow.parameters()) + list(fast.parameters()), 0.1))
    lrs = sorted({g["lr"] for g in opt.param_groups})
    assert lrs == pytest.approx([1e-4, 1e-3])
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == 4


# ============================================================================================
# v3 (PLAN_V3.md): C1 target prediction / re-planned samplers, C2 FK loss, C3 gloss loss,
# C4 retrieval memory, C5 decoders, C6 duration options, tools.

def _smplx_available() -> bool:
    try:
        from methods.morae_slp.autoencoder import PoseFKLoss

        PoseFKLoss(None)
        return True
    except Exception:
        return False


needs_smplx = pytest.mark.skipif(not _smplx_available(), reason="SMPL-X model files missing")


def test_target_and_velocity_parameterisations_are_consistent():
    from methods.morae_slp.flow import target_weight, to_target, to_velocity

    torch.manual_seed(0)
    x0, x1, v_hat = torch.randn(4, 6, 3), torch.randn(4, 6, 3), torch.randn(4, 6, 3)
    t = torch.tensor([0.0, 0.3, 0.6, 0.85])
    x_t, v = training_pair(x1, x0, t)
    x1_hat = to_target(v_hat, x_t, t, "velocity")
    torch.testing.assert_close(to_velocity(x1_hat, x_t, t, "target", min_gap=1e-9), v_hat)
    # ||x1 - x1_hat||^2 / (1 - t)^2 is exactly the velocity loss (weight not capped here).
    w = target_weight(t, "velocity")[:, None, None]
    torch.testing.assert_close(w * (x1_hat - x1).square(), (v_hat - v).square())
    assert torch.equal(target_weight(t, "uniform"), torch.ones(4))
    with pytest.raises(ValueError):
        to_target(v_hat, x_t, t, "noise")


@pytest.mark.parametrize("mode", ["replan_implied", "replan_anchored"])
def test_replan_samplers_follow_the_signmflow_updates(mode):
    from methods.morae_slp.flow import replan

    torch.manual_seed(1)
    x0, target = torch.randn(2, 5, 3), torch.randn(2, 5, 3)
    for steps in (1, 3, 5):  # a perfect x1 predictor lands exactly on the target
        torch.testing.assert_close(replan(lambda x, t: target, x0, steps, mode), target)
    # One step of a 4-step grid from t = 0: both rules give (1 - 1/4) x0 + 1/4 x1_hat.
    seen = []

    def predictor(x, t):
        seen.append(float(t[0]))
        return target if len(seen) == 1 else x
    replan(predictor, x0, 4, mode)
    assert seen[:2] == [0.0, 0.25]
    # The first step of both rules is identical and the last step returns the last x1
    # estimate, so with 2 steps they coincide; a state-dependent predictor separates them
    # from 3 steps on.
    f = (lambda x, t: 0.5 * x + target)
    torch.testing.assert_close(replan(f, x0, 2, "replan_implied"),
                               replan(f, x0, 2, "replan_anchored"))
    assert not torch.allclose(replan(f, x0, 3, "replan_implied"),
                              replan(f, x0, 3, "replan_anchored"))
    valid = torch.tensor([[True] * 5, [True] * 3 + [False] * 2])
    out = replan(lambda x, t: target, x0, 3, mode, valid)
    assert torch.all(out[1, 3:] == 0)


def test_latent_dataset_extras_and_collate():
    from methods.morae_slp.data import Batch, LatentDataset, collate_microbatches

    items = [{"text": "a b", "lang": "de_DGS"}, {"text": "c", "lang": "de_DGS"}]
    latents = [torch.randn(4, 2), torch.randn(6, 2)]
    plain = LatentDataset(latents, items)
    assert len(plain[0]) == 3
    batch = collate_microbatches([plain[0], plain[1]], 2)[0]
    assert batch.poses is None and batch.glosses is None and batch.memory is None
    rich = LatentDataset(latents, items, poses=[torch.randn(4, POSE_DIM), torch.randn(6, POSE_DIM)],
                         glosses=[torch.tensor([2, 3]), torch.tensor([4])],
                         memory=torch.tensor([[1, -1], [0, 1]]))
    batch = collate_microbatches([rich[1], rich[0]], 2)[0]  # sorted by length inside
    assert batch.poses.shape == (2, 6, POSE_DIM) and batch.lengths.tolist() == [4, 6]
    assert [g.tolist() for g in batch.glosses] == [[2, 3], [4]]
    assert batch.memory.tolist() == [[1, -1], [0, 1]]
    assert isinstance(batch.to(torch.device("cpu")), Batch)
    with pytest.raises(ValueError):
        LatentDataset(latents, items, poses=[torch.randn(5, POSE_DIM), torch.randn(6, POSE_DIM)])


def test_text_index_excludes_own_sentence_and_round_trips():
    from methods.morae_slp.retrieval import TextIndex, gather_memory, normalise

    texts = ["Morgen Regen im Norden.", "morgen regen im norden", "heute sonne im süden",
             "今天天气很好", "明天下雨", "wind aus westen"]
    assert normalise(texts[0]) == normalise(texts[1])
    index = TextIndex(texts)
    train = index.query(texts, 2, exclude_groups=True, query_ids=list(range(len(texts))))
    for q, row in enumerate(train):
        assert all(index.group[j] != index.group[q] for j in row if j >= 0)
    assert index.query(["morgen regen im osten"], 1)[0, 0] in (0, 1)
    assert index.query(["明天天气很好"], 1)[0, 0] == 3
    again = TextIndex.from_state(index.state())
    np.testing.assert_allclose(again.scores(["wind im westen"]), index.scores(["wind im westen"]))
    assert index.duplicate_of_train(["morgen, regen im norden", "neu"]).tolist() == [True, False]
    with pytest.raises(ValueError):
        index.query(texts, 2, exclude_groups=True)
    bank = [torch.randn(n, 3) for n in (5, 7, 2)]
    mem, valid = gather_memory(bank, torch.tensor([[0, -1], [2, 1]]), 6, torch.device("cpu"))
    assert mem.shape == (2, 2, 6, 3) and valid.sum(-1).tolist() == [[5, 0], [2, 6]]
    torch.testing.assert_close(mem[1, 1, :6], bank[1][:6])


def test_memory_tokens_join_the_text_context_and_follow_text_dropout(small_cpu):
    dit = LatentDiT(4, 8, width=16, depth=2, heads=2, ff=32,
                    memory={"slots": 2, "layers": 1, "heads": 2, "stride": 2})
    tokens, mask = torch.randn(3, 5, 8), torch.ones(3, 5, dtype=torch.bool)
    lang, drop = torch.zeros(3, dtype=torch.long), torch.tensor([False, True, False])
    mem, valid = torch.randn(3, 2, 7, 4), torch.ones(3, 2, 7, dtype=torch.bool)
    valid[2] = False  # sample 2 lost its memory (retrieval dropout)
    c = dit.condition(tokens, mask, lang, drop, mem, valid)
    assert c["context"].shape == (3, 1 + 5 + 2 * 4, 16)
    assert c["context_valid"].sum(1).tolist() == [5 + 8, 1, 5]
    out = dit(torch.randn(3, 6, 4), torch.rand(3), torch.ones(3, 6, dtype=torch.bool), c)
    assert torch.all(out == 0)  # adaLN-zero / zero cross-attention output at initialisation
    plain = LatentDiT(4, 8, width=16, depth=2, heads=2, ff=32)
    with pytest.raises(ValueError):
        plain.condition(tokens, mask, lang, drop, mem, valid)
    with pytest.raises(ValueError):
        LatentDiT(4, 8, width=16, depth=2, heads=2, ff=32, text_mode="adaln", memory={"slots": 1})


def test_decoding_dtw_matches_the_shared_dtw_and_medoid_dba():
    from methods.morae_slp.decode import dba, dtw, frame_costs, medoid
    from slp.evaluation.metrics import dtw_cost

    rng = np.random.default_rng(0)
    for _ in range(10):
        cost = rng.random((int(rng.integers(1, 20)), int(rng.integers(1, 20))))
        value, path = dtw(cost, path=True)
        assert value == pytest.approx(dtw_cost(cost)[0])
        assert value == pytest.approx(sum(cost[i, j] for i, j in path))
        assert path[0] == (0, 0) and path[-1] == (cost.shape[0] - 1, cost.shape[1] - 1)
    a = rng.normal(size=(9, 4, 3))
    assert medoid([a + 5, a, a + 0.01])[0] in (1, 2)
    assert frame_costs(a, a).diagonal().max() == 0
    seq = rng.normal(size=(9, 6))
    np.testing.assert_allclose(dba([seq, seq], [a, a], lambda c: a, 0), seq, rtol=1e-6)
    # DBA of a time-shifted copy keeps the medoid's length.
    out = dba([seq, np.concatenate([seq[:1], seq])], [a, np.concatenate([a[:1], a])],
              lambda c: a, 0)
    assert out.shape == seq.shape


def test_gloss_recognizer_learns_ctc_and_corruption_keeps_padding(small_cpu):
    from methods.morae_slp.gloss import (GlossRecognizer, GlossVocab, corrupt, ctc, greedy,
                                         word_error_rate)

    vocab = GlossVocab.from_items([{"gloss": "A B C"}, {"gloss": "B D"}, {}])
    assert vocab.tokens[:2] == ["<blank>", "<unk>"] and vocab.encode("A X").tolist() == [2, 1]
    torch.manual_seed(0)
    model = GlossRecognizer(4, len(vocab), width=32, depth=1, heads=2)
    z, lengths = torch.randn(3, 12, 4), torch.tensor([12, 9, 7])
    valid = torch.arange(12)[None] < lengths[:, None]
    targets = [vocab.encode("A B"), vocab.encode(""), vocab.encode("C D B")]
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    first = None
    for _ in range(150):
        loss = ctc(model(z, valid), lengths, targets)
        first = float(loss.detach()) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.detach()) < 0.1 * first
    hyps = greedy(model(z, valid), lengths)
    assert hyps[0] == [2, 3] and hyps[2] == [4, 5, 3]
    assert word_error_rate([[1, 2]], [[1, 3]]) == 0.5
    noisy = corrupt(z, valid, torch.Generator().manual_seed(0))
    assert noisy.shape == z.shape and noisy[1, 9:].abs().sum() == 0
    x = z.clone().requires_grad_(True)
    ctc(model(x, valid), lengths, targets).backward()
    assert x.grad.abs().sum() > 0  # the frozen recognizer passes gradients to its input


def _v3_model(tmp_path, *extra, split=None):
    cfg = _tiny_cfg("method.flow.updates=1", "method.export_predictions=false", *extra)
    model = get_method("morae_slp")(cfg, RunContext(tmp_path, 0, logging.getLogger(__name__)))
    inputs, toy = _toy_split()
    model.fit(split or toy, split or toy)
    torch.manual_seed(5)
    with torch.no_grad():  # move away from the zero-output initialisation
        for p in model.gen.dit.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return model, inputs, cfg


def test_target_prediction_fit_predict_reload(tmp_path, small_cpu):
    model, inputs, cfg = _v3_model(tmp_path, "method.flow.prediction=target",
                                   "method.sampler.solver=replan_anchored")
    state = torch.load(model.ctx.ckpt_dir / "model.pt", weights_only=True)
    assert state["flow_prediction"] == "target"
    preds = model.predict(inputs)
    assert all(np.isfinite(p).all() and p.shape[1] == POSE_DIM for p in preds)
    for solver in ("replan_implied", "rk4", "euler"):
        model.sampler["solver"] = solver
        assert all(np.isfinite(p).all() for p in model.predict(inputs))
    model.sampler["solver"] = "replan_anchored"
    restored = get_method("morae_slp")(_tiny_cfg("method.export_predictions=false",
                                                 "method.sampler.solver=replan_anchored"),
                                       model.ctx)
    restored.load(model.ctx.ckpt_dir)  # the config says velocity: the checkpoint wins
    assert restored.prediction == "target"
    restored.gen.load_state_dict(model.gen.state_dict())
    for a, b in zip(preds, restored.predict(inputs)):
        np.testing.assert_allclose(a, b, atol=1e-5)


def test_checkpoints_without_v3_keys_load_as_v2(tmp_path, small_cpu):
    model, inputs, cfg = _v3_model(tmp_path)
    path = model.ctx.ckpt_dir / "model.pt"
    state = torch.load(path, weights_only=True)
    for key in ("flow_prediction", "memory_args", "duration_args", "duration_bias"):
        state.pop(key)
    state["generator"] = model.gen.state_dict()
    torch.save(state, tmp_path / "old.pt")
    before = model.predict(inputs)
    old = get_method("morae_slp")(cfg, model.ctx)
    old.load_checkpoint(tmp_path / "old.pt")
    assert old.prediction == "velocity" and old.memory_args is None and old.duration_bias == 0
    for a, b in zip(before, old.predict(inputs)):
        np.testing.assert_allclose(a, b, atol=1e-5)


@needs_smplx
def test_fk_loss_reaches_the_generator_but_not_the_frozen_decoder(tmp_path, small_cpu):
    from methods.morae_slp.data import collate_microbatches

    model, inputs, cfg = _v3_model(tmp_path, "method.flow.fk_weight_hand=1.0",
                                   "method.flow.fk_weight_body=0.5", "method.flow.fk_t_min=0.0",
                                   "method.flow.fk_frames=4")
    _, split = _toy_split()
    from methods.morae_slp.data import LatentDataset

    q = [model.space.encode_flow(torch.from_numpy(t)[None], torch.tensor([len(t)]))[0]
         for t in split.targets]
    ds = LatentDataset(q, inputs, poses=[torch.from_numpy(t) for t in split.targets])
    batch = collate_microbatches([ds[i] for i in range(3)], 3)[0]
    model.gen.train().requires_grad_(True)
    losses = model._flow_losses(batch, model.gen)
    assert {"fk_body", "fk_hand"} <= set(losses) and torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gen.dit.parameters())
    assert all(p.grad is None for p in model.space.parameters())


def test_gloss_loss_with_a_matching_recognizer(tmp_path, small_cpu):
    from methods.morae_slp.gloss import GlossRecognizer, GlossVocab, save_recognizer

    inputs, toy = _toy_split()
    glossed = [{**item, "gloss": g} for item, g in zip(inputs, ["SONNE", "REGEN", "WIND MORGEN"])]
    split = Split(glossed, toy.targets)
    (tmp_path / "a").mkdir()
    first, _, cfg = _v3_model(tmp_path / "a", split=split)
    ckpt = first.ctx.ckpt_dir
    vocab = GlossVocab.from_items(glossed)
    arch = {"latent_dim": first.space.latent_dim, "vocab_size": len(vocab), "width": 16,
            "depth": 1, "heads": 2}
    save_recognizer(tmp_path / "r.pt", GlossRecognizer(**arch), vocab, arch,
                    first.space.latent_mean, first.space.latent_std, {})
    gl = (f"method.gloss.recognizer={tmp_path / 'r.pt'}", "method.flow.gloss_weight=0.5",
          "method.flow.gloss_start=0", "method.flow.gloss_t_min=0.0",
          f"method.init_ae={ckpt / 'ae.pt'}", "method.latent.recalibrate=false")
    (tmp_path / "b").mkdir()
    second, _, _ = _v3_model(tmp_path / "b", *gl, split=split)
    history = (tmp_path / "b" / "metrics_history.jsonl").read_text()
    assert "flow/train_gloss" in history and "flow/val_gloss" in history
    # A recognizer from another standardisation is refused.
    save_recognizer(tmp_path / "bad.pt", GlossRecognizer(**arch), vocab, arch,
                    first.space.latent_mean + 1, first.space.latent_std, {})
    (tmp_path / "c").mkdir()
    with pytest.raises(ValueError, match="standardisation"):
        _v3_model(tmp_path / "c", *gl[1:], f"method.gloss.recognizer={tmp_path / 'bad.pt'}",
                  split=split)


def test_retrieval_memory_fit_predict_reload(tmp_path, small_cpu):
    model, inputs, cfg = _v3_model(tmp_path, "method.retrieval.enabled=true",
                                   "method.retrieval.k=2", "method.retrieval.memory.stride=1",
                                   "method.retrieval.memory.layers=1",
                                   "method.retrieval.memory.heads=2")
    assert (model.ctx.ckpt_dir / "retrieval.pt").is_file()
    assert model.memory_args == {"slots": 2, "layers": 1, "heads": 2, "stride": 1}
    assert model.gen.dit.memory is not None
    preds = model.predict(inputs)
    assert all(np.isfinite(p).all() for p in preds)
    restored = get_method("morae_slp")(_tiny_cfg("method.export_predictions=false"), model.ctx)
    restored.load(model.ctx.ckpt_dir)
    assert restored.memory_args == model.memory_args and len(restored.bank) == 3
    restored.gen.load_state_dict(model.gen.state_dict())
    for a, b in zip(preds, restored.predict(inputs)):
        np.testing.assert_allclose(a, b, atol=1e-5)
    # Training queries never retrieve their own sentence.
    own = model._retrieve([text_input(i)[0] for i in inputs], [0, 1, 2])
    assert all(int(j) != q for q, row in enumerate(own) for j in row)


@needs_smplx
def test_medoid_and_dba_decoders(tmp_path, small_cpu):
    model, inputs, cfg = _v3_model(tmp_path)
    members = [torch.randn(2, 6, POSE_DIM) * 0.1 for _ in range(3)]
    model.sampler["decoder"] = "medoid"
    picked = model._combine(members, [6, 4])
    for i, (p, n) in enumerate(zip(picked, [6, 4])):
        assert p.shape == (n, POSE_DIM)
        assert any(np.allclose(p, m[i, :n].numpy()) for m in members)
    model.sampler["decoder"] = "dba"
    same = [members[0]] * 3
    np.testing.assert_allclose(model._combine(same, [6, 4])[0], members[0][0].numpy(), atol=1e-6)
    model.sampler.update(ensemble=2, decoder="medoid")
    assert [len(p) for p in model.predict(inputs)] == [len(p) for p in model.predict(inputs)]


def test_duration_options_and_calibration(tmp_path, small_cpu):
    model, inputs, cfg = _v3_model(tmp_path, "method.duration.token_count=true",
                                   "method.duration.loss=l2", "method.duration.calibrate=true")
    state = torch.load(model.ctx.ckpt_dir / "model.pt", weights_only=True)
    assert state["duration_args"] == {"token_count": True}
    assert state["duration_bias"] == pytest.approx(model.duration_bias)
    assert model.gen.duration.mlp[0].in_features == model.gen.duration.language.embedding_dim + 1
    restored = get_method("morae_slp")(_tiny_cfg("method.export_predictions=false"), model.ctx)
    restored.load(model.ctx.ckpt_dir)
    assert restored.duration_bias == pytest.approx(model.duration_bias)
    base = [len(p) for p in model.predict(inputs)]
    model.sampler["length_scale"] = 2.0
    longer = [len(p) for p in model.predict(inputs)]
    assert all(b2 >= b1 for b1, b2 in zip(base, longer)) and longer != base


@pytest.mark.parametrize("override", [
    "method.flow.prediction=noise", "method.flow.target_weighting=x",
    "method.sampler.decoder=medoid", "method.sampler.solver=replan", "method.flow.gloss_weight=1",
    "method.duration.loss=l3", "method.sampler.length_scale=0", "method.flow.fk_t_min=1.0"])
def test_v3_config_validation(override, tmp_path):
    with pytest.raises(ValueError):
        get_method("morae_slp")(_tiny_cfg(override), RunContext(tmp_path, 0,
                                                                logging.getLogger(__name__)))
    with pytest.raises(ValueError):
        get_method("morae_slp")(_tiny_cfg("method.retrieval.enabled=true",
                                          "method.flow.text_mode=adaln"),
                                RunContext(tmp_path, 0, logging.getLogger(__name__)))


def test_frontier_tools_holm_and_clip_sets():
    from methods.morae_slp.frontier import clip_indices, holm

    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5]
    tune = clip_indices(10, "tune", 4)
    confirm = clip_indices(10, "confirm", 4)
    assert sorted(tune + confirm) == list(range(10)) and not set(tune) & set(confirm)
    assert tune == np.linspace(0, 9, 4, dtype=int).tolist()
    with pytest.raises(ValueError):
        clip_indices(10, "confirm", None)


@needs_smplx
def test_all_v3_options_together(tmp_path, small_cpu):
    """The smoke.py configuration on toy data: every v3 option at once, fit -> decoders ->
    reload; checks that the options compose (shapes, logged losses, checkpoint keys)."""
    from methods.morae_slp.gloss import GlossRecognizer, GlossVocab, save_recognizer

    inputs, toy = _toy_split()
    glossed = [{**item, "gloss": g} for item, g in zip(inputs, ["SONNE", "REGEN", "WIND MORGEN"])]
    split = Split(glossed, toy.targets)
    (tmp_path / "a").mkdir()
    first, _, _ = _v3_model(tmp_path / "a", split=split)
    vocab = GlossVocab.from_items(glossed)
    arch = {"latent_dim": first.space.latent_dim, "vocab_size": len(vocab), "width": 16,
            "depth": 1, "heads": 2}
    save_recognizer(tmp_path / "r.pt", GlossRecognizer(**arch), vocab, arch,
                    first.space.latent_mean, first.space.latent_std, {})
    (tmp_path / "b").mkdir()
    model, _, cfg = _v3_model(
        tmp_path / "b", f"method.init_ae={first.ctx.ckpt_dir / 'ae.pt'}",
        "method.latent.recalibrate=false", "method.flow.prediction=target",
        "method.sampler.solver=replan_anchored", "method.flow.fk_weight_hand=1.0",
        "method.flow.fk_weight_body=0.25", "method.flow.fk_t_min=0.0", "method.flow.fk_frames=4",
        f"method.gloss.recognizer={tmp_path / 'r.pt'}", "method.flow.gloss_weight=0.1",
        "method.flow.gloss_start=0", "method.flow.gloss_t_min=0.0",
        "method.retrieval.enabled=true", "method.retrieval.k=2",
        "method.retrieval.memory.layers=1", "method.retrieval.memory.heads=2",
        "method.duration.token_count=true", "method.duration.calibrate=true", split=split)
    history = (tmp_path / "b" / "metrics_history.jsonl").read_text()
    for key in ("flow/train_fk_hand", "flow/train_gloss", "flow/val_fk_body", "flow/val_gloss"):
        assert key in history
    for decoder, members in (("mean", 1), ("medoid", 3), ("dba", 3)):
        model.sampler.update(ensemble=members, decoder=decoder)
        preds = model.predict(glossed)
        assert all(np.isfinite(p).all() and p.shape[1] == POSE_DIM for p in preds)
    model.sampler.update(ensemble=1, decoder="mean")
    before = model.predict(glossed)
    again = get_method("morae_slp")(cfg, model.ctx)
    again.load(model.ctx.ckpt_dir)
    again.gen.load_state_dict(model.gen.state_dict())
    again.sampler.update(ensemble=1, decoder="mean")
    for a, b in zip(before, again.predict(glossed)):
        np.testing.assert_allclose(a, b, atol=1e-5)


def test_review_fixes_retrieval_never_returns_the_query_itself():
    from methods.morae_slp.retrieval import TextIndex

    index = TextIndex(["guten morgen", "heute regen", "guten morgen", "morgen sonne"])
    rows = index.query(["guten morgen", "heute regen"], 2, query_ids=[0, 1])
    assert 0 not in rows[0].tolist() and 1 not in rows[1].tolist()
    assert rows[0, 0] == 2  # the verbatim repeat is allowed without exclude_groups
    rows = index.query(["guten morgen"], 2, exclude_groups=True, query_ids=[0])
    assert 2 not in rows[0].tolist()


def test_review_fixes_duration_token_count_sees_the_undropped_mask(tmp_path, small_cpu):
    from methods.morae_slp.data import LatentDataset, collate_microbatches

    model, inputs, _ = _v3_model(tmp_path, "method.duration.token_count=true",
                                 "method.flow.token_drop=0.9")
    _, split = _toy_split()
    q = [model.space.encode_flow(torch.from_numpy(t)[None], torch.tensor([len(t)]))[0]
         for t in split.targets]
    batch = collate_microbatches([LatentDataset(q, inputs)[i] for i in range(3)], 3)[0]
    seen = []
    forward = model.gen.duration.forward
    model.gen.duration.forward = lambda tokens, mask, lang: (seen.append(mask.clone()),
                                                             forward(tokens, mask, lang))[1]
    full, _ = model.gen.text(batch.texts, batch.languages)
    torch.manual_seed(0)
    model._flow_losses(batch, model.gen)
    assert torch.equal(seen[0], model.gen.text(batch.texts, batch.languages)[1])


def test_review_fixes_target_prediction_refuses_solvers_that_evaluate_t1(tmp_path, small_cpu):
    with pytest.raises(ValueError, match="evaluates t = 1"):
        get_method("morae_slp")(_tiny_cfg("method.flow.prediction=target",
                                          "method.sampler.solver=rk4"),
                                RunContext(tmp_path, 0, logging.getLogger(__name__)))
    model, _, _ = _v3_model(tmp_path, "method.flow.prediction=target",
                            "method.sampler.solver=euler")
    sweep_like = get_method("morae_slp")(_tiny_cfg("method.sampler.solver=heun"), model.ctx)
    with pytest.raises(ValueError, match="evaluates t = 1"):
        sweep_like.load(model.ctx.ckpt_dir)


def test_review_fixes_frame_costs_chunking_is_exact():
    from methods.morae_slp.decode import frame_costs

    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(70, 5, 3)), rng.normal(size=(40, 5, 3))
    np.testing.assert_allclose(frame_costs(a, b, chunk=7), frame_costs(a, b, chunk=1000),
                               rtol=1e-6)
    ref = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).mean(-1)
    np.testing.assert_allclose(frame_costs(a, b), ref, rtol=1e-5)


# ---------------------------------------------------------------- select_best (TEST command)
def _fake_run(root, experiment, metrics, command, sweep=None, seed="seed0_x"):
    run = root / experiment / "morae_slp" / seed
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps(metrics))
    (run / "meta.json").write_text(json.dumps({"command": command}))
    if sweep:
        (run / "sweep.jsonl").write_text("\n".join(json.dumps(r) for r in sweep))
    return run


def _m(hand, body=5.0, speed=0.8):
    return {"soke_dtw_jpe_hand": hand, "soke_dtw_jpe_body": body, "soke_dtw_pa_jpe_body_ua": 4.0,
            "soke_dtw_pa_jpe_hand": 1.3, "soke_speed_ratio_body": speed,
            "soke_speed_ratio_hand": speed}


def test_select_best_picks_the_lowest_dev_jpe_hand_and_rejects_static_outputs(tmp_path, monkeypatch):
    from methods.morae_slp import select_best as sb

    monkeypatch.setattr(sb, "RUNS", tmp_path)
    cmd = ("scripts/run.py --method morae_slp --data phoenix_smplx --name X --seeds 0 --split val "
           "method.workers=6 method.sampler.cfg_scale=1.5 method.sampler.temperature=0.7")
    _fake_run(tmp_path, "morae_v2ctrl_phoenix_only", _m(7.0), cmd)
    _fake_run(tmp_path, "morae_v3_phoenix_target", _m(6.8), cmd)
    _fake_run(tmp_path, "morae_v3_phoenix_fk_target", _m(6.0, speed=0.2), cmd)  # near static
    candidates = (
        sb.Candidate("phoenix_smplx", "base", "morae_v2ctrl_phoenix_only", "metrics"),
        sb.Candidate("phoenix_smplx", "target", "morae_v3_phoenix_target", "metrics"),
        sb.Candidate("phoenix_smplx", "fk", "morae_v3_phoenix_fk_target", "metrics"),
        sb.Candidate("phoenix_smplx", "missing", "morae_v3_phoenix_lr03", "metrics"))
    rows = sb.rank("phoenix_smplx", candidates)
    assert [(c.label, v) for c, _, v in rows] == [
        ("target", "valid"), ("base", "valid"), ("fk", "rejected: speed ratio < 0.5"),
        ("missing", "not available")]
    assert sb.choose("phoenix_smplx", candidates).label == "target"
    # Tie on JPE-hand: the lower JPE-body wins.
    _fake_run(tmp_path, "morae_v3_phoenix_lr03", _m(6.8, body=4.9), cmd)
    assert sb.choose("phoenix_smplx", candidates).label == "missing"
    with pytest.raises(RuntimeError):
        sb.choose("csl_smplx", candidates)


def test_select_best_uses_only_whole_dev_sweep_rows(tmp_path, monkeypatch):
    from methods.morae_slp import select_best as sb

    monkeypatch.setattr(sb, "RUNS", tmp_path)
    row = {**_m(7.6), "clips": 1077, "ensemble": 1, "decoder": "mean", "lengths": "head",
           "length_scale": 1.0, "subset": "val", "solver": "euler", "cfg_scale": 1.0,
           "temperature": 0.7}
    cmd = "scripts/run.py --method morae_slp --data csl_smplx --name X --seeds 0 --split val"
    _fake_run(tmp_path, "morae_v2ctrl_csl_60k_cont", _m(7.9), cmd,
              sweep=[{**row, "clips": 256, **_m(5.0)},            # tuning subset: ignored
                     {**row, "ensemble": 8, **_m(5.0)},           # K > 1: ignored
                     {**row, "lengths": "gt", **_m(5.0)},         # oracle durations: ignored
                     {**row, "temperature": 1.0, **_m(5.0)},      # other setting: ignored
                     row])
    cand = sb.Candidate("csl_smplx", "euler", "morae_v2ctrl_csl_60k_cont", "sweep", (),
                        {"solver": "euler", "cfg_scale": 1.0, "temperature": 0.7})
    assert sb.dev_metrics(cand)["soke_dtw_jpe_hand"] == 7.6


def test_select_best_test_command_switches_split_name_and_sampler():
    from methods.morae_slp import select_best as sb

    cmd = ("scripts/run.py --method morae_slp --data csl_smplx --name run_a --seeds 1 --split val "
           "method.workers=6 method.sampler.cfg_scale=1.5 method.flow.snapshot_steps=[20000] "
           "method.sampler.temperature=0.7 --allow-dirty")
    sampler = ("method.sampler.solver=euler", "method.sampler.cfg_scale=1")
    argv = sb.test_argv(cmd, "run_a", sampler, dirty=False, python="py")
    assert argv[:3] == ["py", "-u", "scripts/run.py"]
    assert argv.count("--split") == 1 and argv[argv.index("--split") + 1] == "test"
    assert argv[argv.index("--name") + 1] == "run_a_test" and "val" not in argv
    assert argv[argv.index("--seeds") + 1] == "1"
    assert "method.sampler.cfg_scale=1.5" not in argv and "method.sampler.cfg_scale=1" in argv
    assert "method.sampler.temperature=0.7" in argv and "method.flow.snapshot_steps=[20000]" in argv
    assert "--allow-dirty" not in argv
    assert sb.test_argv(cmd, "run_a", sampler, dirty=True, python="py")[-1] == "--allow-dirty"


def test_select_best_registry_covers_every_dataset_and_is_gloss_free():
    from methods.morae_slp import select_best as sb

    assert {c.dataset for c in sb.CANDIDATES} == set(sb.DEV_SIZE)
    assert all("gloss" not in c.label and "retrieval" not in c.label for c in sb.CANDIDATES)


def test_settle_test_log_commits_only_the_test_log(tmp_path):
    import subprocess

    from methods.morae_slp import select_best as sb

    def git(*a):
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a],
                              cwd=tmp_path, capture_output=True, text=True, check=True).stdout

    git("init", "-q")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "test_access_log.jsonl").write_text("{}\n")
    (tmp_path / "a.txt").write_text("a")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    assert sb.settle_test_log(tmp_path) is False                     # clean tree: nothing to do
    (tmp_path / "results" / "test_access_log.jsonl").write_text("{}\n{}\n")
    (tmp_path / "a.txt").write_text("changed")                       # another change pending
    assert sb.settle_test_log(tmp_path) is False and git("status", "--porcelain").count("\n") == 2
    git("checkout", "a.txt")
    import os
    env_backup = {k: os.environ.get(k) for k in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL",
                                                 "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL")}
    os.environ.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                      GIT_COMMITTER_EMAIL="t@t")
    try:
        assert sb.settle_test_log(tmp_path) is True
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert git("status", "--porcelain") == "" and "Record test access log" in git("log", "-1")
