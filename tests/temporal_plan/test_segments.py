from collections import Counter
import math

import pytest
import torch

from methods.temporal_plan.segments import (
    PartitionDistribution, SegmentCodec, TeacherConfig, boundary_evidence, phase_samples,
    segment_scores,
)


def enumerate_covers(length, maximum):
    if length == 0:
        return [()]
    return [(d, *tail) for d in range(1, min(length, maximum)+1)
            for tail in enumerate_covers(length-d, maximum)]


def reference_score(scores, ds):
    end, total = 0, scores.new_zeros(())
    for d in ds:
        end += d
        total = total + scores[end-1, d-1]
    return total


def test_dynamic_program_matches_exhaustive_normalizer_map_and_marginals():
    torch.manual_seed(9)
    scores = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    temp = .7
    dist = PartitionDistribution(scores, temp)
    paths = enumerate_covers(6, 3)
    log_weights = torch.stack([reference_score(scores, ds)/temp for ds in paths])
    reference_z = torch.logsumexp(log_weights, 0)
    torch.testing.assert_close(dist.log_normalizer, reference_z)
    probabilities = log_weights.softmax(0)
    expected_marginals = torch.zeros_like(scores)
    for ds, probability in zip(paths, probabilities):
        end = 0
        for d in ds:
            end += d
            expected_marginals[end-1, d-1] += probability
    torch.testing.assert_close(dist.segment_marginals(), expected_marginals)
    gradient, = torch.autograd.grad(dist.log_normalizer, scores)
    torch.testing.assert_close(gradient, expected_marginals/temp)
    assert tuple(dist.mode()) == paths[int(log_weights.argmax())]
    torch.testing.assert_close(torch.stack([dist.log_prob(ds).exp() for ds in paths]).sum(),
                               torch.tensor(1., dtype=torch.float64))


def test_backward_sampler_distribution_and_reproducibility():
    dist = PartitionDistribution(torch.tensor([[.3, -math.inf], [.4, .8]]), .5)
    g = torch.Generator().manual_seed(33)
    samples = [tuple(dist.sample(g)) for _ in range(1800)]
    frequency = Counter(samples)
    p_two = float(dist.log_prob([2]).exp())
    assert abs(frequency[(2,)]/len(samples)-p_two) < .04
    assert set(frequency) == {(1, 1), (2,)}
    g.manual_seed(33)
    assert samples[:20] == [tuple(dist.sample(g)) for _ in range(20)]
    with pytest.raises(ValueError):
        dist.log_prob([1])
    with pytest.raises(ValueError):
        PartitionDistribution(torch.full((2, 2), -torch.inf))


def test_phase_interpolation_endpoint_constant_and_linear_reference():
    sequence = torch.stack((torch.arange(5.), torch.arange(5.)*2), -1)
    phase = phase_samples(sequence, 8)
    torch.testing.assert_close(phase[0], sequence[0])
    torch.testing.assert_close(phase[-1], sequence[-1])
    torch.testing.assert_close(phase_samples(phase, 5), sequence)
    constant = torch.tensor([[3., 7.]])
    torch.testing.assert_close(phase_samples(constant), constant.expand(8, -1))


def test_teacher_distortion_and_endpoint_reward_match_formula():
    features = torch.zeros(5, 4, 2)
    cfg = TeacherConfig(max_duration=4, penalty=.3, beta=.5)
    scores = segment_scores(features, cfg)
    # R=0, constant interior evidence=.5; final endpoint reward is zero.
    torch.testing.assert_close(scores[3, :4], torch.full((4,), -.05))
    torch.testing.assert_close(scores[4, :4], torch.full((4,), -.3))
    assert torch.isneginf(scores[0, 1:]).all()
    for ds in (PartitionDistribution(scores).sample(torch.Generator().manual_seed(4)),
               PartitionDistribution(scores).mode()):
        assert sum(ds) == 5 and min(ds) > 0 and max(ds) <= 4
    one = segment_scores(torch.randn(1, 4, 2), cfg)
    assert PartitionDistribution(one).sample() == [1]


def test_segment_codec_backpropagates_and_validates_phase_shape():
    codec = SegmentCodec(feature_width=8, content_width=3, hidden=12)
    phases = torch.randn(4, 8, 8)
    reconstructed, codes = codec(phases)
    assert codes.shape == (4, 3)
    (reconstructed-phases).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in codec.parameters())
    with pytest.raises(ValueError):
        codec(torch.randn(4, 7, 8))


def test_vectorized_segment_scores_equal_interval_by_interval_reference():
    torch.manual_seed(71)
    features = torch.randn(9, 4, 3, dtype=torch.float64)
    config = TeacherConfig(max_duration=5)
    actual = segment_scores(features, config)
    reference = torch.full_like(actual, -torch.inf)
    boundary = boundary_evidence(features)
    flat = features.flatten(1)
    for end in range(1, len(features)+1):
        for d in range(1, min(end, config.max_duration)+1):
            interval = flat[end-d:end]
            recon = phase_samples(phase_samples(interval, 8), d)
            reference[end-1, d-1] = (-(recon-interval).square().mean() +
                                      config.beta*boundary[end]-config.penalty)
    torch.testing.assert_close(actual, reference)
