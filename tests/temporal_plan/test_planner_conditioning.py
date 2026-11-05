import pytest
import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from methods.temporal_plan.conditioning import interval_weights, temporal_condition
from methods.temporal_plan.planner import TextToPlan, mixture_nll
from methods.temporal_plan.plans import PlanBatch


def tiny_planner(**kwargs):
    return TextToPlan(content_width=3, text_width=7, hidden=16, heads=2, layers=2,
                      mixtures=2, max_duration=4, max_latents=9, dropout=0, **kwargs)


def fixture_plan():
    return PlanBatch(torch.randn(2, 4, 3),
                     torch.tensor([[1, 3, 2, 1], [2, 1, 0, 0]]), torch.tensor([4, 2]))


def test_mixture_nll_matches_independent_torch_distribution_and_gradients():
    torch.manual_seed(4)
    logits = torch.randn(2, 4, requires_grad=True)
    means = torch.randn(2, 4, 3, requires_grad=True)
    logvars = torch.randn(2, 4, 3, requires_grad=True)
    target = torch.randn(2, 3)
    distribution = MixtureSameFamily(Categorical(logits=logits),
                                     Independent(Normal(means, (.5*logvars).exp()), 1))
    actual = mixture_nll(logits, means, logvars, target)
    torch.testing.assert_close(actual, -distribution.log_prob(target)/3)
    actual.mean().backward()
    assert all(torch.isfinite(x.grad).all() for x in (logits, means, logvars))


def test_planner_does_not_read_current_or_future_target_in_its_content_prior():
    torch.manual_seed(1)
    model = tiny_planner().eval()
    memory = torch.randn(2, 5, 7)
    valid = torch.ones(2, 5, dtype=torch.bool)
    plan = fixture_plan()
    baseline = model(memory, valid, plan)
    changed = PlanBatch(plan.contents.clone(), plan.durations.clone(), plan.lengths)
    changed.contents[0, 2:] = 1e3
    changed.durations[0, 2:] = 4
    other = model(memory, valid, changed)
    # Step 2 is allowed to see only units 0,1, not target unit 2 or 3.
    torch.testing.assert_close(baseline.means[0, :3], other.means[0, :3])
    torch.testing.assert_close(baseline.duration_logits[0, :2], other.duration_logits[0, :2])
    torch.testing.assert_close(baseline.stop_logits[0, :2], other.stop_logits[0, :2])
    assert not torch.allclose(baseline.means[0, 3], other.means[0, 3])


def test_planner_padding_invariance_and_finite_backward():
    torch.manual_seed(2)
    model = tiny_planner().eval()
    memory = torch.randn(2, 5, 7)
    valid = torch.tensor([[True]*5, [True]*3+[False]*2])
    memory[1, 3:] = float("nan")
    plan = fixture_plan()
    plan.contents[1, 2:] = float("nan")
    output = model(memory, valid, plan)
    single = PlanBatch(plan.contents[1:2, :2], plan.durations[1:2, :2], torch.tensor([2]))
    alone = model(memory[1:2, :3], valid[1:2, :3], single)
    torch.testing.assert_close(output.means[1, :2], alone.means[0], atol=1e-6, rtol=1e-6)
    loss = model.loss(memory, valid, plan)
    loss["total"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert output.logvars.min() >= -6 and output.logvars.max() <= 2


def test_generation_enforces_duration_support_cap_and_distinguishes_eos():
    model = tiny_planner().eval()
    # Prefer duration 4 but never stop voluntarily.
    with torch.no_grad():
        model.duration[-1].weight.zero_()
        model.duration[-1].bias.zero_()
        model.duration[-1].bias[3] = 100
        model.stop[-1].weight.zero_()
        model.stop[-1].bias.fill_(-100)
    memory, mask = torch.randn(1, 3, 7), torch.ones(1, 3, dtype=torch.bool)
    result = model.generate(memory, mask, sample_content=False)
    assert result.plan.latent_lengths.tolist() == [9]
    assert result.plan.durations.tolist() == [[4, 4, 1]]
    assert result.stop_reasons == ("capacity",)
    with torch.no_grad():
        model.stop[-1].bias.fill_(100)
    eos = model.generate(memory, mask, sample_content=False)
    assert eos.plan.lengths.tolist() == [1]
    assert eos.stop_reasons == ("eos",)
    model.train()
    with pytest.raises(RuntimeError, match="eval"):
        model.generate(memory, mask)


def test_generation_randomness_is_reproducible_without_any_motion_input():
    model = tiny_planner().eval()
    memory, mask = torch.randn(1, 3, 7), torch.ones(1, 3, dtype=torch.bool)
    a = model.generate(memory, mask, generator=torch.Generator().manual_seed(7))
    b = model.generate(memory, mask, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(a.plan.contents, b.plan.contents)
    torch.testing.assert_close(a.plan.durations, b.plan.durations)


def test_interval_membership_matches_manual_fractional_intersections():
    weights, phases, effective = interval_weights(torch.ones(5, dtype=torch.long), 2,
                                                   rescale=True, overlap=0)
    expected = torch.tensor([[.4, .4, .2, 0, 0], [0, 0, .2, .4, .4]], dtype=torch.float64)
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, dtype=torch.float64))
    assert (phases >= 0).all() and (phases <= 1).all()
    torch.testing.assert_close(effective, torch.full((5,), .4, dtype=torch.float64))
    with pytest.raises(ValueError, match="forbidden"):
        interval_weights(torch.ones(5, dtype=torch.long), 2)


def test_conditioning_preserves_constants_and_gradients_even_with_subcell_units():
    content = torch.ones(5, 3, requires_grad=True)
    plan = PlanBatch.from_sequences([content], [[1]*5])
    condition = temporal_condition(plan, canvas_lengths=torch.tensor([2]),
                                    rescale=True, overlap=2)
    torch.testing.assert_close(condition.features[0, :, :3], torch.ones(2, 3))
    assert condition.subcell_units.tolist() == [5]
    condition.features[..., :3].sum().backward()
    assert (content.grad > 0).all()
    w, _, _ = interval_weights(torch.tensor([1, 3, 1]), 5, overlap=2)
    assert (w >= 0).all()
    torch.testing.assert_close(w.sum(-1), torch.ones(5, dtype=torch.float64))


def test_condition_padding_does_not_use_padded_content_or_duration():
    plan = fixture_plan()
    plan.contents[1, 2:] = float("nan")
    condition = temporal_condition(plan)
    assert torch.isfinite(condition.features).all()
    assert condition.features[1, 3:].count_nonzero() == 0
