import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from methods.race_sign.rotations import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
    slerp_matrix,
    geodesic_distance,
)
from methods.race_sign.schema import manual_to_6d, restore_pose


def test_against_scipy_and_near_pi():
    rng = np.random.default_rng(9)
    a = np.concatenate(
        (rng.normal(size=(100, 3)), np.zeros((1, 3)), [[np.pi, 0, 0], [0, -np.pi, 0], [1e-9, 0, 0]])
    )
    t = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    r = axis_angle_to_matrix(t)
    np.testing.assert_allclose(r.detach(), Rotation.from_rotvec(a).as_matrix(), atol=1e-10)
    back = matrix_to_axis_angle(r)
    torch.testing.assert_close(axis_angle_to_matrix(back), r, atol=1e-9, rtol=1e-9)
    loss = back.square().sum() + r.sum()
    loss.backward()
    assert torch.isfinite(t.grad).all()


@pytest.mark.parametrize("d", [[0, 0, 0, 0, 0, 0], [1, 0, 0, 2, 0, 0]])
def test_degenerate_6d_valid_gradients(d):
    x = torch.tensor(d, dtype=torch.float64, requires_grad=True)
    r = rotation_6d_to_matrix(x)
    torch.testing.assert_close(r @ r.T, torch.eye(3, dtype=r.dtype))
    torch.testing.assert_close(torch.linalg.det(r), torch.tensor(1.0, dtype=r.dtype))
    r.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_geodesic_slerp_and_schema_preserve_nonmanual():
    pose = torch.randn(2, 8, 133) * 0.15
    manual = manual_to_6d(pose)
    restored = restore_pose(manual, pose)
    torch.testing.assert_close(restored, pose, atol=1e-6, rtol=1e-6)
    assert torch.equal(restored[..., 120:], pose[..., 120:])
    a = torch.eye(3)
    b = axis_angle_to_matrix(torch.tensor([0.0, 0.0, torch.pi / 2]))
    midpoint = slerp_matrix(a, b, torch.tensor(0.5))
    torch.testing.assert_close(geodesic_distance(a, midpoint), torch.tensor(torch.pi / 4))
    torch.testing.assert_close(matrix_to_rotation_6d(a), torch.tensor([1.0, 0, 0, 0, 1, 0]))


def test_physical_hand_codec_preserves_nonprincipal_branches():
    pose = torch.zeros(1, 4, 133)
    pose[..., 31] = 4.2
    pose[..., 79] = -3.8
    mean = torch.zeros(120)
    mean[30:120] = torch.linspace(-0.4, 0.4, 90)
    physical = manual_to_6d(pose, mean)
    restored = restore_pose(physical, pose, mean)
    torch.testing.assert_close(
        axis_angle_to_matrix((restored[..., :120] + mean).reshape(1, 4, 40, 3)),
        axis_angle_to_matrix((pose[..., :120] + mean).reshape(1, 4, 40, 3)),
        atol=2e-6,
        rtol=2e-6,
    )
