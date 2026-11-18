import io
import pickle

import lmdb
import numpy as np
import pytest

from cprflow.data.cache import file_hash
from cprflow.data.lmdb_adapter import (
    LMDBAdapter,
    canonical_to_source,
    normalized_text,
    source_to_canonical,
)
from cprflow.data.manifest import check_disjoint
from cprflow.data.stats import balanced_moments


def test_native_source_roundtrip(identity_motion):
    x = identity_motion().numpy()
    x[:, 60:66] = [0, -1, 0, 1, 0, 0]
    x[:, 150:156] = [1, 0, 0, 0, 0, -1]
    source = canonical_to_source(x)
    assert source["body_features"].shape[-1] == 126
    assert np.array_equal(source_to_canonical(source), x)
    assert not np.array_equal(x[:, 60:150], x[:, 150:240])
    source["face_features"] = np.zeros((5, 10))
    with pytest.raises(ValueError, match="face_features"):
        source_to_canonical(source)


def test_schema_unknown_misaligned(identity_motion):
    source = canonical_to_source(identity_motion().numpy())
    source["body_features"] = np.zeros((5, 132))
    with pytest.raises(ValueError):
        source_to_canonical(source)
    source = canonical_to_source(identity_motion().numpy())
    source["right_features"] = source["right_features"][:-1]
    with pytest.raises(ValueError, match="Misaligned"):
        source_to_canonical(source)
    assert normalized_text(np.array([" e\u0301  Foo! "])) == "é Foo!"


def test_lmdb_trusted_boundary(tmp_path, identity_motion):
    root = tmp_path / "fixture.lmdb"
    env = lmdb.open(str(root), map_size=1 << 20)
    data = canonical_to_source(identity_motion().numpy())
    buffer = io.BytesIO()
    np.savez(
        buffer,
        **data,
        translation=np.array(["Hello"], dtype=object),
        language=np.array(["ASL"], dtype=object),
    )
    with env.begin(write=True) as txn:
        txn.put(b"__meta__", pickle.dumps({"clip_ids": ["fixture"], "num_clips": 1}))
        txn.put(b"fixture", buffer.getvalue())
    env.close()
    sha = file_hash(root / "data.mdb")
    with pytest.raises(ValueError, match="trusted"):
        LMDBAdapter(root, sha)
    adapter = LMDBAdapter(root, sha, trusted=True)
    assert source_to_canonical(adapter.read("fixture")).shape == (5, 296)
    adapter.close()
    with pytest.raises(ValueError, match="hash mismatch"):
        LMDBAdapter(root, "0" * 64, trusted=True)


def test_balanced_moments_and_leakage():
    records = [
        ({"split": "train", "dataset_id": "a", "uid": "a1"}, np.ones((1, 1)) * 0),
        ({"split": "train", "dataset_id": "a", "uid": "a2"}, np.ones((100, 1)) * 2),
        ({"split": "train", "dataset_id": "b", "uid": "b1"}, np.ones((10, 1)) * 5),
    ]
    mean, std, _ = balanced_moments(records, ["a", "b"])
    np.testing.assert_allclose(mean, [3])
    np.testing.assert_allclose(std, [np.sqrt(4.5)])
    with pytest.raises(ValueError, match="train"):
        balanced_moments(records, ["a", "b"], split="dev")
    with pytest.raises(ValueError, match="leakage"):
        check_disjoint([[{"uid": "same"}], [{"uid": "same"}]])


def test_actual_release_hand96_duplicate_wrists_retains_canonical296(identity_motion):
    x = identity_motion(5).numpy()
    source = canonical_to_source(x, "signspark_body126_hands96_wristdup_face56")
    np.testing.assert_array_equal(
        source_to_canonical(source, "signspark_body126_hands90_or96_wristdup_face56"), x
    )
    source["left_features"][0, 90] = 123
    with pytest.raises(ValueError, match="trailing fields"):
        source_to_canonical(source, "signspark_body126_hands90_or96_wristdup_face56")
    source = canonical_to_source(x)
    np.testing.assert_array_equal(
        source_to_canonical(source, "signspark_body126_hands90_or96_wristdup_face56"), x
    )
