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


def test_how2sign_fps24_indices_soke_vs_madis():
    from slp.data.sign_smplx import fps24_indices

    n, fps = 100, 29.97
    count = int(24 * n / fps)
    soke = fps24_indices(n, fps, "soke")
    assert soke.tolist() == [int(i * n / count) for i in range(count)]  # SOKE load_data.sample
    madis = fps24_indices(n, fps, "madis")
    assert len(madis) == len(soke) == count and madis[-1] == n - 1 and soke[0] == 0
    assert (np.diff(soke) > 0).all() and soke.max() < n


def test_soke_bad_how2sign_ids_match_soke_source():
    from pathlib import Path

    from slp.data.sign_smplx import SOKE_BAD_HOW2SIGN_IDS

    src = Path(__file__).resolve().parents[1] / "third_party/SOKE/mGPT/data/humanml/dataset_t2m_token.py"
    if not src.exists():
        pytest.skip("SOKE source not available")
    lst = src.read_text().split("bad_how2sign_ids = [")[1].split("]")[0]
    ids = {x.strip().strip("'\"") for x in lst.split(",") if x.strip()}
    assert ids == SOKE_BAD_HOW2SIGN_IDS and len(ids) == 31
