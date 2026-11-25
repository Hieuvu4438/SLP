from __future__ import annotations

import copy
import logging
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
                                    group_channel_matrix, sample_group_mask, with_velocity)
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
