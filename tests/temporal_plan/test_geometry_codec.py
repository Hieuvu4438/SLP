import math

import pytest
import torch

from methods.temporal_plan.codec import MotionCodec, MotionPosterior
from methods.temporal_plan.geometry import (
    axis_angle_to_matrix, forward_kinematics, learning256_to_soke133,
    matrix_to_axis_angle, matrix_to_rotation6d, rotation6d_to_matrix,
    soke133_to_learning256, unpack_smplx,
)
from methods.temporal_plan.losses import LossWeights, motion_reconstruction, posterior_kl
from methods.temporal_plan.normalization import FeatureNormalizer


def test_rotations_known_axes_zero_pi_and_raw_channel_round_trip():
    v = torch.tensor([[0., 0., 0.], [0., 0., math.pi/2], [math.pi, 0., 0.]],
                     dtype=torch.float64)
    matrices = axis_angle_to_matrix(v)
    torch.testing.assert_close(matrices[1] @ v.new_tensor([1., 0., 0.]),
                               v.new_tensor([0., 1., 0.]), atol=1e-12, rtol=0)
    torch.testing.assert_close(axis_angle_to_matrix(matrix_to_axis_angle(matrices)), matrices)
    torch.testing.assert_close(rotation6d_to_matrix(matrix_to_rotation6d(matrices)), matrices)
    raw = torch.randn(2, 7, 133, dtype=torch.float64) * .2
    encoded = soke133_to_learning256(raw)
    restored = learning256_to_soke133(encoded)
    torch.testing.assert_close(raw, restored)
    torch.testing.assert_close(restored[..., 123:], raw[..., 123:], rtol=0, atol=0)
    packed = unpack_smplx(raw[0])
    assert torch.count_nonzero(packed["body_pose"][:, :33]) == 0
    torch.testing.assert_close(packed["body_pose"][:, 33:], raw[0, :, :30])


def test_degenerate_six_dimensional_outputs_are_valid_rotations_with_finite_gradients():
    d = torch.tensor([[0., 0., 0., 0., 0., 0.], [1., 0., 0., 2., 0., 0.]],
                     requires_grad=True)
    m, fallback = rotation6d_to_matrix(d, return_fallback=True)
    assert fallback.all()
    torch.testing.assert_close(m @ m.transpose(-1, -2), torch.eye(3).expand(2, 3, 3))
    torch.testing.assert_close(torch.linalg.det(m), torch.ones(2))
    m.square().sum().backward()
    assert torch.isfinite(d.grad).all()
    v = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    axis_angle_to_matrix(v).sum().backward()
    assert torch.isfinite(v.grad).all()


def test_rotation_gradient_matches_finite_differences():
    v = torch.tensor([[.2, -.3, .1]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: matrix_to_axis_angle(axis_angle_to_matrix(x)), (v,), atol=1e-5
    )


def test_fk_rotates_child_offset_by_parent_not_child_and_backpropagates():
    angles = torch.tensor([[0., 0., math.pi/2], [0., 0., .3]], requires_grad=True)
    positions, _ = forward_kinematics(axis_angle_to_matrix(angles),
                                     torch.tensor([[0., 0., 0.], [1., 0., 0.]]), (-1, 0))
    torch.testing.assert_close(positions[1], torch.tensor([0., 1., 0.]), atol=1e-6, rtol=0)
    positions[1, 0].backward()
    assert angles.grad[0, 2].abs() > .9
    torch.testing.assert_close(angles.grad[1], torch.zeros(3))


def test_train_statistics_are_streaming_and_ignore_nonfinite_padding():
    first = torch.tensor([[[1., 2.], [3., 4.], [float("nan"), float("nan")]]])
    second = torch.tensor([[[9., 8.]]])
    norm = FeatureNormalizer(2)
    with pytest.raises(RuntimeError):
        norm(first)
    norm.fit([(first, torch.tensor([2])), (second, torch.tensor([1]))])
    all_values = torch.tensor([[1., 2.], [3., 4.], [9., 8.]])
    torch.testing.assert_close(norm.mean, all_values.mean(0))
    torch.testing.assert_close(norm.std, all_values.std(0, correction=0))
    torch.testing.assert_close(norm.inverse(norm(all_values)), all_values)
    assert int(norm.count) == 3
    restored = FeatureNormalizer(2)
    restored.load_state_dict(norm.state_dict())
    torch.testing.assert_close(restored(all_values), norm(all_values))


def test_codec_is_padding_invariant_including_odd_frames_and_backward():
    torch.manual_seed(17)
    codec = MotionCodec(latent_width=3, hidden=8)
    lengths = torch.tensor([5, 8])
    x = torch.randn(2, 12, 256)
    x[0, 5:] = float("nan")
    x[1, 8:] = 12345
    reconstructed, posterior = codec(x, lengths, sample=False)
    assert posterior.lengths.tolist() == [3, 4]
    for i, n in enumerate(lengths):
        single, post_single = codec(x[i:i+1, :n], torch.tensor([n]), sample=False)
        torch.testing.assert_close(posterior.mean[i, :post_single.lengths[0]],
                                   post_single.mean[0], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(reconstructed[i, :n], single[0, :n],
                                   atol=2e-6, rtol=2e-6)
    (reconstructed.square().mean() + posterior_kl(posterior)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in codec.parameters())


def test_losses_ignore_padding_and_positive_fk_cannot_silently_disappear():
    lengths = torch.tensor([1, 3])
    raw = torch.zeros(2, 5, 133)
    target = soke133_to_learning256(raw)
    predicted = target.clone().requires_grad_(True)
    target[0, 1:] = float("nan")
    target[1, 3:] = float("nan")
    losses = motion_reconstruction(predicted, target, lengths, weights=LossWeights(fk=0))
    torch.testing.assert_close(losses["total"], torch.tensor(0.))
    losses["total"].backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[0, 1:].count_nonzero() == 0
    with pytest.raises(ValueError, match="FK"):
        motion_reconstruction(predicted, target, lengths)
    mean = torch.zeros(1, 1, 4, 2)
    post = MotionPosterior(mean, mean.clone(), torch.tensor([1]))
    torch.testing.assert_close(posterior_kl(post), torch.tensor(0.))
