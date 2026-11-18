import math

import numpy as np
import pytest
import torch

from cprflow.data.resample import resample_motion
from cprflow.geometry.rotations import rotation_6d_to_matrix


def test_slerp_and_endpoint_grid(identity_motion):
    x = identity_motion(2)
    x[1, :6] = torch.tensor([0.0, -1, 0, 1, 0, 0])
    x[1, 246:] = 2
    out, times, fps = resample_motion(x.numpy(), 15, verified=True)
    assert len(out) == 3 and fps == 30
    assert times[-1] == pytest.approx(1 / 15)
    np.testing.assert_allclose(out[1, 246:], 1)
    expected = torch.tensor(
        [[math.sqrt(0.5), -math.sqrt(0.5), 0], [math.sqrt(0.5), math.sqrt(0.5), 0], [0, 0, 1]]
    )
    torch.testing.assert_close(
        rotation_6d_to_matrix(torch.from_numpy(out[1, :6])), expected, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(out[[0, -1]], x.numpy(), atol=1e-5)
    with pytest.raises(ValueError, match="Verified"):
        resample_motion(x, 15)


def test_downsample_antialias_reduces_high_frequency_rotations_and_expression(identity_motion):
    times = torch.arange(121) / 120
    x = identity_motion(121)
    angle = 0.4 * torch.sin(2 * torch.pi * 25 * times)
    x[:, 0] = angle.cos()
    x[:, 1] = -angle.sin()
    x[:, 3] = angle.sin()
    x[:, 4] = angle.cos()
    x[:, 246] = angle
    out, grid, fps = resample_motion(x, 120, verified=True)
    naive = x[::4].numpy()
    assert len(out) == 31 and fps == 30
    assert np.abs(out[2:-2, 246]).max() < np.abs(naive[2:-2, 246]).max() * 0.2
    matrices = rotation_6d_to_matrix(torch.from_numpy(out[:, :6]))
    yaw = torch.atan2(matrices[:, 1, 0], matrices[:, 0, 0]).numpy()
    assert np.abs(yaw[2:-2]).max() < 0.1
    np.testing.assert_allclose(out[[0, -1]], x.numpy()[[0, -1]], atol=1e-5)


def test_unverified_clock_requires_explicit_exploratory_permission(identity_motion):
    with pytest.raises(ValueError, match="Verified"):
        resample_motion(identity_motion(3), 30, verified=False)
    result, _, _ = resample_motion(identity_motion(3), 30, verified=False, allow_assumed=True)
    assert result.shape == (3, 296)
