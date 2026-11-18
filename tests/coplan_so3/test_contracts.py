"""Checks independent of the network's own implementation."""

import copy
from dataclasses import replace

import pytest
import torch

from methods.coplan_so3.flow import SourceConfig, ar1_noise
from methods.coplan_so3.geometry import so3
from methods.coplan_so3.geometry.pullback import feature_jvp
from methods.coplan_so3.losses import contrastive
from methods.coplan_so3.sampling import generate
from methods.coplan_so3.training import restore_checkpoint, save_checkpoint
from .conftest import freeze_plan


def test_explicit_jacobian_gradient(features, batch):
    features.double()
    r = batch.rotations[:1, :3].double()
    e = batch.expression[:1, :3].double()
    m = batch.mask[:1, :3]
    t = batch.timestamps[:1, :3].double()
    features.fit_batch(r, e, m, t, split="train")
    features.finalize(split="train")
    n_w = r.numel() // 3

    def function(x):
        w = x[:n_w].reshape(1, 3, 5, 3)
        v = x[n_w:].reshape_as(e)
        return torch.cat([z.flatten() for z in features(r @ so3.exp(w), e + v, m, t).values()])

    x = torch.zeros(n_w + e.numel(), dtype=r.dtype)
    jac = torch.autograd.functional.jacobian(function, x)
    delta = (torch.randn_like(x) * 0.01).requires_grad_()
    jv = feature_jvp(
        features, r, e, delta[:n_w].reshape(1, 3, 5, 3), delta[n_w:].reshape_as(e), m, t
    )
    grad = torch.autograd.grad(jv.square().sum(), delta)[0]
    torch.testing.assert_close(grad, 2 * jac.T @ jac @ delta, atol=1e-8, rtol=1e-8)


def test_padding_and_hidden_values_do_not_change_network(model, batch):
    model.eval()
    r, e, m, t = batch.rotations, batch.expression, batch.mask, batch.timestamps
    z, zm = model.plan.encode_raw(r, e, m, t)
    rpad = torch.cat((r, so3.exp(torch.randn(2, 4, 5, 3))), 1)
    epad = torch.cat((e, torch.randn(2, 4, 2)), 1)
    mpad = torch.nn.functional.pad(m, (0, 4))
    tpad = torch.cat((t, torch.randn(2, 4)), 1)
    zp, zmp = model.plan.encode_raw(rpad, epad, mpad, tpad)
    torch.testing.assert_close(z, zp[:, : z.shape[1]], atol=2e-6, rtol=2e-5)
    text, tm = model.encode_text(batch.texts, batch.languages)
    s = torch.tensor([0.2, 0.8])
    out = model.dense(r, e, s, text, tm, z, zm, batch.languages, m, t)
    padded = model.dense(rpad, epad, s, text, tm, zp, zmp, batch.languages, mpad, tpad)
    for a, b in zip(out[:2], padded[:2]):
        torch.testing.assert_close(a, b[:, : r.shape[1]], atol=2e-6, rtol=2e-5)


def test_checkpoint_roundtrip_and_wrong_law_rejected(model, batch, tmp_path):
    freeze_plan(model, batch)
    model.set_stage("C")
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, {}, stage="C")
    other = copy.deepcopy(model)
    with torch.no_grad():
        next(other.dense.parameters()).add_(1)
    restore_checkpoint(path, other)
    a = generate(model, ["hello"], [0], lengths=[4], seed=2)
    b = generate(other, ["hello"], [0], lengths=[4], seed=2)
    torch.testing.assert_close(a.rotations, b.rotations, atol=0, rtol=0)
    other.posterior_sigma = 0.9
    with pytest.raises(ValueError, match="noise law"):
        restore_checkpoint(path, other)
    other.posterior_sigma = model.posterior_sigma
    other.manifest = replace(other.manifest, fps=30.0)
    with pytest.raises(ValueError, match="manifest mismatch"):
        restore_checkpoint(path, other)


def test_ar1_stationary_full_rank():
    x = ar1_noise((40000, 4, 1), 0.7, torch.zeros(()), torch.Generator().manual_seed(42)).squeeze(
        -1
    )
    assert abs(float(x.var(0).mean()) - 1) < 0.02
    cov = torch.cov(x.T)
    assert torch.linalg.eigvalsh(cov).min() > 0.1
    assert abs(float(cov[0, 1]) - 0.7) < 0.02
    with pytest.raises(ValueError):
        SourceConfig(ar1=1)


def test_contrastive_positives_and_false_negative_filtering():
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    lang = torch.zeros(3, dtype=torch.long)
    texts = ["same", "same", "different"]
    matched = contrastive(z, z, lang, texts)
    mismatched = contrastive(z, z.flip(0), lang, texts)
    assert matched < mismatched
    exclude = torch.ones(3, 3, dtype=torch.bool)
    assert contrastive(z, z, lang, texts, exclude=exclude) == 0


def test_length_overflow_is_not_silently_cropped(model, batch):
    freeze_plan(model, batch)
    with pytest.raises(ValueError, match="no silent truncation"):
        generate(model, ["hello"], [0], lengths=[1000], max_frames=400)
    with torch.no_grad():
        model.length_model.head[-1].weight.zero_()
        model.length_model.head[-1].bias.copy_(torch.tensor([100.0, 0.0]))
    with pytest.raises(ValueError, match="exceeds max_frames"):
        generate(model, ["hello"], [0])


def test_language_and_plan_conditions_reach_dense_velocity(model, batch):
    model.eval()
    text, tm = model.encode_text(batch.texts, batch.languages)
    z, zm = model.plan.encode_raw(batch.rotations, batch.expression, batch.mask, batch.timestamps)

    def run(text, z, language):
        return model.dense(
            batch.rotations,
            batch.expression,
            torch.ones(2) * 0.5,
            text,
            tm,
            z,
            zm,
            language,
            batch.mask,
            batch.timestamps,
        )[0]

    a = run(text, z, batch.languages)
    assert not torch.allclose(a, run(text.flip(0), z, batch.languages))
    assert not torch.allclose(a, run(text, z + 3.0, batch.languages))
    assert not torch.allclose(a, run(text, z, batch.languages + 1))


def test_two_condition_multimodal_oracle_prior():
    """Known population CFM field, no trained network: both modes and conditions survive."""
    from methods.coplan_so3.sampling import sample_plan

    n = 4000
    language = torch.arange(n) % 2
    text = torch.zeros(n, 1, 2, dtype=torch.float64)
    mask = torch.ones(n, 1, dtype=torch.bool)
    lengths = torch.ones(n, dtype=torch.long)

    def oracle(z, s, text, text_mask, language, lengths, zmask):
        # Independent N(0,1) source; data is an equal mixture of N(c-1,.01),
        # N(c+1,.01), with c=-3/+3 determined by the condition.
        center = (language.to(z) * 6 - 3)[:, None, None]
        means = torch.stack((center - 1, center + 1), -1)
        ss = s[:, None, None, None]
        variance = (1 - ss).square() + ss.square() * 0.01
        residual = z[..., None] - ss * means
        responsibility = torch.softmax(-0.5 * residual.square() / variance, -1)
        conditional_velocity = means + (0.01 * ss - (1 - ss)) / variance * residual
        return (responsibility * conditional_velocity).sum(-1)

    z = sample_plan(
        oracle, text, mask, language, lengths, mask, 1, 250, torch.Generator().manual_seed(81)
    )[:, 0, 0]
    for lang in (0, 1):
        centered = z[language == lang] - (6 * lang - 3)
        assert abs(float(centered.mean())) < 0.08
        assert 0.85 < float(centered.std()) < 1.1
        # An averaged single mode at the center would fail this test.
        assert float((centered.abs() > 0.65).double().mean()) > 0.95
        assert 0.44 < float((centered > 0).double().mean()) < 0.56


def test_multilingual_sampling_probabilities():
    from types import SimpleNamespace
    from methods.coplan_so3.training import language_sampler

    ds = SimpleNamespace(records=[{"sign_language": "DGS"}] * 4 + [{"sign_language": "ASL"}] * 16)
    s = language_sampler(ds, 0.5)
    mass = s.weights[:4].sum() / s.weights.sum()
    torch.testing.assert_close(mass, torch.tensor(1 / 3, dtype=mass.dtype))


def test_mixed_output_frame_rates(model, batch):
    freeze_plan(model, batch)
    model.fps_by_language = (25.0, 30.0, 24.0)
    out = generate(model, ["hello", "world"], [0, 1], lengths=[3, 3], seed=3, total_budget=5)
    assert out.metadata["fps"] == [25.0, 30.0]


def test_cache_provenance_and_tamper_rejection(manifest, batch, tmp_path):
    import json
    import numpy as np
    from methods.coplan_so3.data.dataset import CanonicalMotionDataset, collate
    from methods.coplan_so3.data.manifest import file_sha256

    path = tmp_path / "motion.npz"
    np.savez(
        path,
        rotations=batch.rotations[0].numpy(),
        expression=batch.expression[0].numpy(),
        timestamps=batch.timestamps[0].numpy(),
    )
    record = {
        "clip_id": "clip",
        "parent_video_id": "parent",
        "dataset_id": "fixture",
        "split": "train",
        "text": "hello",
        "sign_language": "DGS",
        "extraction_version": "v1",
        "annotation_version": "v1",
        "manifest_sha256": manifest.digest,
        "path": path.name,
        "sha256": file_sha256(path),
        "expression_coordinates": "normalized",
    }
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps(record) + "\n")
    ds = CanonicalMotionDataset(index, manifest, split="train")
    output = collate([ds[0]])
    torch.testing.assert_close(output.rotations[0], batch.rotations[0])
    with pytest.raises(ValueError, match="split/schema"):
        CanonicalMotionDataset(index, manifest, split="val")
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        ds[0]


def test_wrong_source_checkpoint_rejected(model, batch, tmp_path):
    from methods.coplan_so3.flow import SourceConfig

    freeze_plan(model, batch)
    model.set_stage("C")
    path = tmp_path / "source.pt"
    save_checkpoint(path, model, {}, stage="C")
    model.source_config = SourceConfig(ar1=0.9)
    with pytest.raises(ValueError, match="law mismatch"):
        restore_checkpoint(path, model)


def test_optional_endpoint_branch_backward(model, batch):
    from methods.coplan_so3.models.part_temporal_transformer import MotionHead

    freeze_plan(model, batch)
    model.dense.endpoint = MotionHead(model.manifest, model.dense.dim, 6)
    model.auxiliary_weight = 0.05
    model.set_stage("C")
    loss, terms = model.objective(batch, torch.Generator().manual_seed(5))
    assert "auxiliary" in terms and torch.isfinite(loss)
    loss.backward()
    grad = model.dense.endpoint.body[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
