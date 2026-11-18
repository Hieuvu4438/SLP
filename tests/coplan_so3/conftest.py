import pytest
import torch

from methods.coplan_so3.data.dataset import collate
from methods.coplan_so3.data.manifest import MotionManifest
from methods.coplan_so3.geometry import so3
from methods.coplan_so3.geometry.fk import TreeFK
from methods.coplan_so3.geometry.physical_time_features import SequenceFeatures
from methods.coplan_so3.model import CoPlanSO3
from methods.coplan_so3.models.text_encoder import ByteTextEncoder


@pytest.fixture
def manifest():
    return MotionManifest(
        ("root", "lw", "lf", "rw", "rf"),
        (0, 1, 2, 3, 4),
        (0,),
        (1, 2),
        (3, 4),
        ((1,), (2,)),
        ((3,), (4,)),
        2,
        25.0,
        "synthetic",
        "fixture",
        "fixture",
        "fixed",
        "physical",
        "identity_fixed",
        {},
    )


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(42)
    return collate(
        [
            {
                "rotations": so3.exp(torch.randn(t, 5, 3, generator=g) * 0.2),
                "expression": torch.randn(t, 2, generator=g) * 0.1,
                "timestamps": torch.arange(t) * 0.04,
                "text": text,
                "language": lang,
            }
            for t, text, lang in [(5, "hello", 0), (3, "world", 0)]
        ]
    )


@pytest.fixture
def features():
    fk = TreeFK(
        [-1, 0, 1, 0, 3, 2, 4],
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
        [0, 1, 2, 3, 4],
        [0, 1, 3],
        [1, 5],
        [3, 6],
        0,
    )
    return SequenceFeatures(
        fk, {"joints": 9, "wrists": 6, "handshape": 6, "palms": 12, "relative": 6}
    )


@pytest.fixture
def model(manifest, features, batch):
    torch.manual_seed(10)
    features.fit_batch(
        batch.rotations, batch.expression, batch.mask, batch.timestamps, split="train"
    )
    features.finalize(split="train")
    return CoPlanSO3(
        manifest,
        ByteTextEncoder(16),
        features,
        plan_dim=16,
        plan_hidden=16,
        plan_layers=1,
        dense_hidden=16,
        dense_layers=1,
        heads=2,
        dropout=0.0,
    )


def freeze_plan(model, batch):
    model.plan.eval()
    with torch.no_grad():
        z, mask = model.plan.encode_raw(
            batch.rotations, batch.expression, batch.mask, batch.timestamps
        )
        model.plan.stats.update(z, mask, split="train")
        model.plan.stats.finalize(split="train")
    model.plan.freeze()
