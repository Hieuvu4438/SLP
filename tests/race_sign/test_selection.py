import pytest
import torch
from methods.race_sign.data import collate_inputs
from methods.race_sign.model import FlowDecoder
from methods.race_sign.schema import MotionNormalizer, manual_to_6d
from methods.race_sign.selector import ImprovementSelector, improvement_targets, blend_candidates
from methods.race_sign.pipeline import RacePipeline


def test_improvement_sign_and_calibration(cfg, geometry):
    target = manual_to_6d(torch.zeros(1, 8, 133))
    damaged_pose = torch.zeros(1, 8, 133)
    damaged_pose[..., 3] = 0.5
    damaged = manual_to_6d(damaged_pose)
    gain, mask = improvement_targets(damaged, target, target, torch.tensor([8]), geometry, cfg)
    assert (gain[mask] > 0).all()
    selector = ImprovementSelector(cfg)
    selector.calibrate(
        torch.tensor([[0.0, 1.0, 2.0]]),
        torch.tensor([[1.0, 3.0, 5.0]]),
        torch.ones(1, 3, dtype=torch.bool),
    )
    torch.testing.assert_close(selector.calibration_scale, torch.tensor(2.0))
    torch.testing.assert_close(selector.calibration_bias, torch.tensor(1.0))


def test_slerp_selection_exact_endpoints(cfg):
    a = manual_to_6d(torch.randn(1, 12, 133) * 0.1)
    b = manual_to_6d(torch.randn(1, 12, 133) * 0.1)
    for value, expected in ((-1.0, a), (1.0, b)):
        out, _ = blend_candidates(a, b, torch.full((1, 3), value), torch.tensor([12]), 4, 1)
        assert torch.equal(out, expected)
    out, weights = blend_candidates(
        a, b, torch.tensor([[-1.0, 1.0, -1.0]]), torch.tensor([12]), 4, 1
    )
    assert (weights >= 0).all() and (weights <= 1).all()
    assert torch.isfinite(out).all()


def test_inference_no_gt_and_checkpoint_roundtrip(cfg, geometry, record_factory, tmp_path):
    from methods.race_sign.training import save_checkpoint, load_checkpoint

    batch = collate_inputs([record_factory(0, 8), record_factory(1, 12)])
    decoder = FlowDecoder(cfg)
    selector = ImprovementSelector(cfg)
    pipe = RacePipeline(decoder, MotionNormalizer(), geometry, selector).eval()
    result = pipe.predict(batch, seed=42)
    again = pipe.predict(batch, seed=42)
    torch.testing.assert_close(result.poses, again.poses)
    assert torch.equal(result.poses[0, :8, 120:], batch.baseline[0, :8, 120:])
    assert result.nfe == cfg.flow_steps
    assert torch.equal(result.lengths, batch.lengths)
    with pytest.raises(TypeError):
        pipe.predict({"target": torch.zeros(1)})
    path = tmp_path / "model.pt"
    save_checkpoint(
        path, decoder, pipe.normalizer, selector=selector, stage="selector", provenance={}
    )
    d, n, s, _ = load_checkpoint(path)
    loaded = RacePipeline(d, n, geometry, s).eval().predict(batch, seed=42)
    torch.testing.assert_close(result.poses, loaded.poses)


def test_noise_and_predictions_independent_of_batch_padding(cfg, geometry, record_factory):
    decoder = FlowDecoder(cfg)
    pipe = RacePipeline(decoder, MotionNormalizer(), geometry).eval()
    one = record_factory(0, 8)
    single = pipe.predict(collate_inputs([one]), policy="flow", seed=17)
    pair = pipe.predict(collate_inputs([one, record_factory(1, 20)]), policy="flow", seed=17)
    torch.testing.assert_close(single.poses[0], pair.poses[0, :8], atol=2e-5, rtol=2e-5)
    with pytest.raises(ValueError):
        pipe.predict(collate_inputs([one]), policy="flow", steps=0)
    baseline = pipe.predict(collate_inputs([one]), policy="base")
    assert torch.equal(baseline.poses[0], one["baseline"])


def test_switch_blending_uses_native_length_not_batch_padding():
    a = manual_to_6d(torch.randn(1, 16, 133) * 0.1)
    b = manual_to_6d(torch.randn(1, 16, 133) * 0.1)
    short, weights = blend_candidates(a, b, torch.tensor([[-1.0, 1.0]]), torch.tensor([16]), 12, 6)
    padded_a = torch.nn.functional.pad(a, (0, 0, 0, 8))
    padded_b = torch.nn.functional.pad(b, (0, 0, 0, 8))
    padded, padded_weights = blend_candidates(
        padded_a, padded_b, torch.tensor([[-1.0, 1.0]]), torch.tensor([16]), 12, 6
    )
    torch.testing.assert_close(short, padded[:, :16])
    torch.testing.assert_close(weights, padded_weights[:, :16])


def test_calibration_surrogate_diagnostics(cfg):
    from methods.race_sign.selector import calibration_diagnostics

    selector = ImprovementSelector(cfg)
    predicted = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    observed = predicted * 2 + 0.2
    selector.calibrate(predicted, observed, torch.ones(4, dtype=torch.bool))
    report = calibration_diagnostics(
        selector, predicted, observed, torch.where(observed > 0, 20.0, -20.0)
    )
    assert report["pair_count"] == 4 and report["surrogate_mae"] < 1e-6
    assert report["auxiliary_win_brier"] < 1e-6
    assert sum(b["count"] for b in report["reliability_bins"]) == 4
    assert report["surrogate_coverage_curve"][0]["mean_observed_improvement"] > 0


def test_inference_rejects_asset_geometry_and_sampling_mismatch(
    cfg, geometry, record_factory, tmp_path
):
    from dataclasses import replace
    from methods.race_sign.geometry import geometry_signature
    from methods.race_sign.training import save_checkpoint, load_checkpoint

    path = tmp_path / "contract.pt"
    policy = dict(candidate_steps=cfg.flow_steps, candidate_use_text=True, candidate_coupled=True)
    provenance = dict(
        base_signature={"tokenizer_sha256": "author_fixture"},
        geometry_signature=geometry_signature(geometry),
        selector_sampling_policy=policy,
    )
    save_checkpoint(
        path,
        FlowDecoder(cfg),
        MotionNormalizer(),
        selector=ImprovementSelector(cfg),
        stage="selector",
        provenance=provenance,
    )
    decoder, norm, selector, _ = load_checkpoint(path)
    pipe = RacePipeline(decoder, norm, geometry, selector).eval()
    inputs = collate_inputs([record_factory(0, 8)])
    with pytest.raises(ValueError, match="tokenizer"):
        pipe.predict(inputs)
    inputs = replace(inputs, base_signature=provenance["base_signature"])
    with pytest.raises(ValueError, match="Sampling policy"):
        pipe.predict(inputs, steps=cfg.flow_steps + 1)
    geometry.rest[0, 0] += 0.1
    with pytest.raises(ValueError, match="geometry"):
        pipe.predict(inputs, policy="flow")
