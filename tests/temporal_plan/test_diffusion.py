import pytest
import torch
from torch import nn

from methods.temporal_plan.denoiser import JointDenoiser
from methods.temporal_plan.diffusion import CosineDiffusion


def test_cosine_schedule_identity_and_reverse_step_match_closed_form():
    diffusion = CosineDiffusion(steps=20)
    assert diffusion.alpha_bar[0] == 1
    assert (diffusion.alpha_bar[1:] < diffusion.alpha_bar[:-1]).all()
    assert diffusion.alpha_bar[-1] > 0
    clean, noise = torch.randn(2, 3, 4, 2), torch.randn(2, 3, 4, 2)
    torch.testing.assert_close(diffusion.q_sample(clean, torch.zeros(2, dtype=torch.long), noise),
                               clean)
    ts, previous = torch.tensor([19, 12]), torch.tensor([8, 0])
    noisy = diffusion.q_sample(clean, ts, noise)
    restored = diffusion.ddim_step(noisy, ts, previous, clean)
    expected = diffusion.q_sample(clean, previous, noise)
    torch.testing.assert_close(restored, expected)
    with pytest.raises(ValueError, match="decreasing"):
        diffusion.ddim_step(noisy, previous, ts, clean)


class ConstantClean(nn.Module):
    def __init__(self, clean):
        super().__init__()
        self.register_buffer("clean", clean)

    def forward(self, noisy, *_):
        return self.clean.expand_as(noisy)


def test_sampler_ends_at_x0_without_clipping_and_handles_single_step():
    diffusion = CosineDiffusion(12)
    clean = torch.full((1, 4, 4, 2), 3.7)
    model = ConstantClean(clean).eval()
    kwargs = dict(lengths=torch.tensor([4]), memory=torch.zeros(1, 2, 5),
                  text_valid=torch.ones(1, 2, dtype=torch.bool),
                  condition=torch.zeros(1, 4, 5), latent_width=2)
    for steps in (1, 4, 12):
        result = diffusion.sample(model, sampling_steps=steps, **kwargs)
        torch.testing.assert_close(result, clean)
    with pytest.raises(ValueError, match="sampling_steps"):
        diffusion.sample(model, sampling_steps=13, **kwargs)


def test_joint_denoiser_padding_invariance_conditions_and_finite_backward():
    torch.manual_seed(1)
    model = JointDenoiser(latent_width=3, content_width=4, text_width=7,
                          hidden=16, heads=2, layers=2, dropout=0).eval()
    lengths = torch.tensor([3, 6])
    noisy = torch.randn(2, 8, 4, 3)
    noisy[0, 3:] = float("nan")
    condition = torch.randn(2, 8, 6)
    condition[0, 3:] = float("nan")
    memory = torch.randn(2, 5, 7)
    valid = torch.tensor([[True, True, True, False, False], [True]*5])
    memory[0, 3:] = float("nan")
    times = torch.tensor([.3, .8])
    prediction = model(noisy, times, memory, valid, condition, lengths)
    alone = model(noisy[:1, :3], times[:1], memory[:1, :3], valid[:1, :3],
                   condition[:1, :3], lengths[:1])
    torch.testing.assert_close(prediction[0, :3], alone[0], atol=1e-6, rtol=1e-6)
    assert prediction[0, 3:].count_nonzero() == 0
    changed = model(noisy[:1, :3], times[:1], memory[:1, :3], valid[:1, :3],
                     condition[:1, :3]+3, lengths[:1])
    assert not torch.allclose(alone, changed)
    prediction.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_diffusion_training_masks_nan_padding_and_backpropagates():
    model = JointDenoiser(latent_width=2, content_width=3, text_width=5,
                          hidden=8, heads=2, layers=1, dropout=0)
    diffusion = CosineDiffusion(12)
    clean = torch.randn(2, 5, 4, 2)
    clean[0, 2:] = float("nan")
    lengths = torch.tensor([2, 5])
    result = diffusion.training_loss(model, clean, lengths, torch.randn(2, 3, 5),
                                     torch.ones(2, 3, dtype=torch.bool),
                                     torch.randn(2, 5, 5), generator=torch.Generator().manual_seed(8))
    assert (result.timesteps >= 1).all() and (result.timesteps <= 12).all()
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
