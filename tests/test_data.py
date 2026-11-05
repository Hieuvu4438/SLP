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


# ------------------------------------------------------------ sign SMPL-X data
import pytest  # noqa: E402

from slp.data.sign_smplx import PARTS, feats133, madis_crop  # noqa: E402


def test_feats133_layout():
    T = 5
    keys = {"smplx_root_pose": 3, "smplx_body_pose": 63, "smplx_lhand_pose": 45, "smplx_rhand_pose": 45,
            "smplx_jaw_pose": 3, "smplx_shape": 10, "smplx_expr": 10}
    off, poses = 0, {}
    for k, n in keys.items():  # encode the original column index as the value
        poses[k] = np.tile(np.arange(off, off + n, dtype=np.float32), (T, 1)); off += n
    f = feats133(poses)
    assert f.shape == (T, 133)
    assert f[0, 0] == 36 and f[0, 29] == 65          # body joints 12..21 (after root + 11 joints)
    assert f[0, 30] == 66 and f[0, 119] == 155       # both hands
    assert f[0, 120] == 156 and f[0, 123] == 169     # jaw, then expression (shape dropped)
    assert sorted(sum(PARTS.values(), [])) == list(range(133))


def test_madis_crop_rule():
    assert len(madis_crop(np.zeros((10, 1)))) == 40
    assert len(madis_crop(np.zeros((1000, 1)))) == 400
    x = np.arange(103)[:, None]
    c = madis_crop(x)
    assert len(c) == 100 and c[0, 0] == 1            # centre crop to a multiple of 4


@pytest.mark.skipif(not (__import__("pathlib").Path("data/processed/madis/Phoenix_2014T").exists()),
                    reason="packed Phoenix data not present")
def test_phoenix_smplx_official_splits():
    from slp.data.sign_smplx import _phoenix
    from pathlib import Path
    root = Path("data/processed/madis/Phoenix_2014T")
    names = {s: {r[0]["name"] for r in _phoenix(root, s)} for s in ("train", "val", "test")}
    assert len(names["val"]) == 519 and len(names["test"]) == 642
    assert not (names["train"] & names["val"]) and not (names["train"] & names["test"])
