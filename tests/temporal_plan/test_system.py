import copy

import pytest
import torch

from methods.temporal_plan.losses import LossWeights
from methods.temporal_plan.segments import TeacherConfig
from methods.temporal_plan.system import ModelConfig, TemporalPlanSystem
from methods.temporal_plan.teacher import teacher_targets
from methods.temporal_plan.text import TextConditioner


def make_tiny_system():
    transformers = pytest.importorskip("transformers")
    text = TextConditioner(transformers.MT5EncoderModel(transformers.MT5Config(
        vocab_size=16, d_model=8, d_kv=4, d_ff=16, num_layers=1, num_heads=2, dropout_rate=0,
    )), output_width=7)
    config = ModelConfig(
        max_latents=4, latent_width=2, codec_hidden=8, content_width=3, segment_hidden=12,
        hidden=16, heads=2, planner_layers=1, denoiser_layers=1, mixtures=2, dropout=0,
        diffusion_steps=8, sampling_steps=3, exposure_probability=1,
        teacher=TeacherConfig(max_duration=3), losses=LossWeights(fk=0),
    )
    return TemporalPlanSystem(config, text, torch.zeros(41, 3))


def take_step(model, loss):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    optimizer.step()


def test_complete_staged_losses_freezing_exposure_generation_and_state_roundtrip(monkeypatch):
    """Only tiny CPU optimization steps; no dataset, GPU, pretrained weights or efficacy claim."""
    torch.manual_seed(12)
    model = make_tiny_system()
    raw = torch.randn(2, 8, 133)*.05
    lengths = torch.tensor([5, 8])
    raw[0, 5:] = float("nan")
    ids = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])
    text_mask, languages = ids != 0, torch.tensor([0, 1])
    generator = torch.Generator().manual_seed(2)
    model.eval()
    with pytest.raises(RuntimeError, match="stages"):
        model.generate(ids, text_mask, languages)
    with pytest.raises(RuntimeError, match="preceding"):
        model.set_stage("planner")
    model.fit_motion_statistics([(raw, lengths)])
    model.set_stage("codec")
    take_step(model, model.codec_loss(raw, lengths, generator=generator)["total"])
    assert all(p.grad is None for p in model.planner.parameters())
    model.mark_stage_completed()
    model.fit_latent_statistics([(raw, lengths)])

    model.set_stage("segment")
    take_step(model, model.segment_loss(raw, lengths, generator=generator)["total"])
    assert all(p.grad is None for p in model.codec.parameters())
    model.mark_stage_completed()

    model.set_stage("planner")
    model.fit_content_statistics([(raw, lengths)], generator=generator)
    features, latent_lengths = model.reference_latents(raw, lengths)
    targets = teacher_targets(features, latent_lengths, model.segment_codec, model.content_norm,
                               model.config.teacher, generator=generator)
    assert targets.source_indices.tolist() == [0, 0, 1, 1]
    assert targets.plan.latent_lengths.tolist() == [3, 3, 4, 4]
    take_step(model, model.planner_loss(raw, lengths, ids, text_mask, languages,
                                       generator=generator)["total"])
    assert all(p.grad is None for p in model.text.encoder.parameters())
    assert model.text.projection.weight.grad is not None
    model.mark_stage_completed()
    # Force many short predicted units: exercises K > reference latent length in exposure.
    with torch.no_grad():
        model.planner.duration[-1].weight.zero_()
        model.planner.duration[-1].bias.zero_()
        model.planner.duration[-1].bias[0] = 100
        model.planner.stop[-1].weight.zero_()
        model.planner.stop[-1].bias.fill_(-100)

    model.set_stage("generator")
    generator_losses = model.generator_loss(raw, lengths, ids, text_mask, languages,
                                             generator=generator)
    auxiliary_grads = torch.autograd.grad(
        generator_losses["decoded"],
        [p for p in model.denoiser.parameters() if p.requires_grad],
        retain_graph=True, allow_unused=True,
    )
    assert any(g is not None and g.abs().sum() > 0 for g in auxiliary_grads)
    assert all(g is None or torch.isfinite(g).all() for g in auxiliary_grads)
    take_step(model, generator_losses["total"])
    assert all(p.grad is None for p in model.codec.parameters())
    assert all(p.grad is None for p in model.planner.parameters())
    assert all(p.grad is None for p in model.text.parameters())
    assert not model.codec.training and not model.planner.training and not model.text.training
    model.mark_stage_completed()

    model.set_stage("exposure")
    exposure = model.generator_loss(raw, lengths, ids, text_mask, languages, generator=generator)
    assert float(exposure["exposure_fraction"]) == 1
    assert float(exposure["subcell_units"]) > 0
    take_step(model, exposure["total"])
    model.mark_stage_completed()
    model.eval()
    # Any attempt to derive test-time teacher targets/reference latents fails this test.
    monkeypatch.setattr(model, "reference_latents", lambda *args, **kwargs: pytest.fail(
        "reference motion used during primary inference"
    ))
    result = model.generate(ids, text_mask, languages, generator=torch.Generator().manual_seed(3))
    assert result.frame_lengths.tolist() == [8, 8]
    assert result.raw133.shape == (2, 8, 133) and torch.isfinite(result.raw133).all()
    assert result.stop_reasons == ("capacity", "capacity")
    with pytest.raises(TypeError):
        model.generate(ids, text_mask, languages, reference_lengths=lengths)
    restored = make_tiny_system()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    restored.eval()
    again = restored.generate(ids, text_mask, languages, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(again.raw133, result.raw133)
