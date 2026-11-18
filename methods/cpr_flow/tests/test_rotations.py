import math

import pytest
import torch

from cprflow.geometry.body_model import assemble_parameters
from cprflow.geometry.rotations import (
    geodesic_distance,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


def test_row_convention_identity_degenerate():
    r = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    d = matrix_to_rotation_6d(r)
    assert torch.equal(d, torch.tensor([0.0, -1, 0, 1, 0, 0]))
    assert torch.allclose(rotation_6d_to_matrix(d), r)
    out, deg = rotation_6d_to_matrix(torch.zeros(6), True)
    assert deg and torch.equal(out, torch.eye(3))
    random = torch.randn(300, 6)
    mats = rotation_6d_to_matrix(random)
    torch.testing.assert_close(torch.linalg.det(mats), torch.ones(300), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        mats @ mats.transpose(-1, -2), torch.eye(3).expand(300, 3, 3), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("angle", [0.0, 1e-7, math.pi - 1e-7, math.pi])
def test_geodesic_edge_gradient(angle):
    d = torch.tensor(
        [math.cos(angle), -math.sin(angle), 0, math.sin(angle), math.cos(angle), 0],
        requires_grad=True,
    )
    distance = geodesic_distance(rotation_6d_to_matrix(d), torch.eye(3))
    distance.backward()
    assert torch.isfinite(d.grad).all()
    assert abs(float(distance.detach()) - angle) < 2e-5


def test_body_mapping(identity_motion):
    x = identity_motion(2)
    x[:, 240:246] = torch.tensor([0.0, -1, 0, 1, 0, 0])
    x[:, 246:] = torch.arange(50)
    params, deg = assemble_parameters(x)
    assert torch.equal(params["body_pose"][:, :11], torch.eye(3).expand(2, 11, 3, 3))
    assert torch.equal(params["body_pose"][:, 11:], torch.eye(3).expand(2, 10, 3, 3))
    assert not torch.equal(params["jaw_pose"][:, 0], params["body_pose"][:, 14])
    assert torch.equal(params["expression"], x[:, 246:])
    assert not deg.any()
