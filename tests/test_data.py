import numpy as np

from slp.config import compose
from slp.data import load_dataset


def test_splits_disjoint_and_fixed():
    cfg = compose("baseline_majority", "toy")
    a, b = load_dataset(cfg), load_dataset(cfg)
    # same split every time (independent of run seed)
    np.testing.assert_array_equal(a.test.inputs, b.test.inputs)
    # no overlap between splits (no leakage)
    rows = {s: {r.tobytes() for r in getattr(a, s).inputs} for s in ("train", "val", "test")}
    assert not (rows["train"] & rows["val"])
    assert not (rows["train"] & rows["test"])
    assert not (rows["val"] & rows["test"])
