import os
from pathlib import Path

import numpy as np
import pytest
import torch

from cprflow.data.stats import Normalizer


def pytest_sessionstart(session):
    torch.set_num_threads(2)


@pytest.fixture
def identity_motion():
    def make(frames=5):
        x = torch.zeros(frames, 296)
        x[:, :246] = torch.tensor([1.0, 0, 0, 0, 1, 0]).repeat(41)
        return x

    return make


@pytest.fixture
def normalizer():
    return Normalizer(np.zeros(296), np.ones(296))


@pytest.fixture
def body_model():
    from cprflow.geometry.body_model import BodyModel

    path = Path(os.environ.get("CPRFLOW_SMPLX", "methods/cpr_flow/assets/smplx/SMPLX_NEUTRAL.npz"))
    if not path.exists():
        path = Path(__file__).parents[1] / "assets/smplx/SMPLX_NEUTRAL.npz"
    if not path.is_file():
        pytest.skip(f"Missing licensed expression50 SMPL-X asset: {path.resolve()}")
    return BodyModel(path, chunk_frames=2)
