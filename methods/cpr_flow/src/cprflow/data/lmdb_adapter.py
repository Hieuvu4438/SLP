"""Read-only, process-local LMDB adapter; explicit trusted-source pickle boundary."""

import io
import os
import pickle
import unicodedata
from pathlib import Path

import numpy as np

from cprflow.data.cache import file_hash
from cprflow.geometry.rotations import rotation_6d_to_matrix


def normalized_text(value):
    a = np.asarray(value)
    if a.size != 1:
        raise ValueError("Expected a scalar or single-element string")
    scalar = a.reshape(-1)[0]
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf8")
    if not isinstance(scalar, str):
        raise ValueError("Expected string")
    return " ".join(unicodedata.normalize("NFC", scalar).split())


def source_to_canonical(source, profile="signspark_body126_hands90_face56"):
    dims = {
        "signspark_body126_hands90_face56": (126, 90),
        "signspark_upper10_hands90_face56": (60, 90),
        "signspark_body126_hands96_wristdup_face56": (126, 96),
        "signspark_body126_hands90_or96_wristdup_face56": (126, 0),
    }
    if profile not in dims:
        raise ValueError(f"Unknown source layout {profile}")
    body_dim, hand_dim = dims[profile]
    arrays = []
    for name, width in [
        ("body_features", body_dim),
        ("left_features", hand_dim),
        ("right_features", hand_dim),
        ("face_features", 56),
    ]:
        a = np.asarray(source[name], dtype=np.float32)
        if hand_dim == 0 and name in ("left_features", "right_features"):
            if a.ndim != 2 or a.shape[1] not in (90, 96):
                raise ValueError("Known release hand layout must be 90D or 96D wrist-duplicate")
            width = a.shape[1]
        if a.ndim != 2 or a.shape[1] != width or not np.isfinite(a).all():
            raise ValueError(f"{name}: expected finite [T,{width}], got {a.shape}")
        arrays.append(a)
    if len({len(a) for a in arrays}) != 1:
        raise ValueError("Misaligned modalities")
    if len(arrays[0]) <= 1:
        raise ValueError("T<=1 production clip")
    body = arrays[0][:, 66:126] if body_dim == 126 else arrays[0]
    if hand_dim in (0, 96):
        # The real pinned release appends the same native wrist rot6D already in body.
        for hand, index in ((arrays[1], 19), (arrays[2], 20)):
            if hand.shape[1] == 90:
                continue
            if not np.allclose(
                hand[:, 90:96], arrays[0][:, index * 6 : (index + 1) * 6], atol=1e-5, rtol=1e-5
            ):
                raise ValueError(
                    "Unknown hand96 trailing fields: expected duplicated body wrist rotation"
                )
    x = np.concatenate((body, arrays[1][:, :90], arrays[2][:, :90], arrays[3]), -1)
    import torch

    _, degenerate = rotation_6d_to_matrix(torch.from_numpy(x[:, :246].reshape(-1, 6)), True)
    if degenerate.any():
        raise ValueError("Degenerate ground-truth rot6D")
    return x


def canonical_to_source(x, profile="signspark_body126_hands90_face56"):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 296:
        raise ValueError("Expected [T,296]")
    if profile in ("signspark_body126_hands90_face56", "signspark_body126_hands96_wristdup_face56"):
        identity = np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), (len(x), 11))
        body = np.concatenate((identity, x[:, :60]), -1)
    elif profile == "signspark_upper10_hands90_face56":
        body = x[:, :60]
    else:
        raise ValueError("Unknown source layout")
    left, right = x[:, 60:150], x[:, 150:240]
    if profile == "signspark_body126_hands96_wristdup_face56":
        left = np.concatenate((left, body[:, 114:120]), -1)
        right = np.concatenate((right, body[:, 120:126]), -1)
    return dict(
        body_features=body, left_features=left, right_features=right, face_features=x[:, 240:]
    )


class LMDBAdapter:
    def __init__(self, path, expected_sha256, trusted=False, max_readers=1024, readahead=False):
        self.path = str(Path(path).resolve())
        self.max_readers, self.readahead = max_readers, readahead
        self._env, self._pid = None, None
        if not trusted or not expected_sha256:
            raise ValueError("Source metadata pickle requires trusted release and recorded SHA256")
        if file_hash(Path(path) / "data.mdb") != expected_sha256:
            raise ValueError("Source LMDB hash mismatch")
        # Only the verified upstream metadata uses pickle. Numeric clip fields stay allow_pickle=False.
        raw = self._read(b"__meta__")
        meta = pickle.loads(raw)
        self.clip_ids = meta["clip_ids"]
        if meta.get("num_clips", len(self.clip_ids)) != len(self.clip_ids):
            raise ValueError("Metadata count mismatch")
        self.close()

    def _read(self, key):
        import lmdb

        if self._env is None or self._pid != os.getpid():
            self._env = lmdb.open(
                self.path,
                readonly=True,
                lock=False,
                readahead=self.readahead,
                meminit=False,
                max_readers=self.max_readers,
            )
            self._pid = os.getpid()
        with self._env.begin() as tx:
            value = tx.get(key)
        if value is None:
            raise KeyError(f"Missing LMDB key {key!r} in {self.path}")
        return value

    def read(self, clip_id):
        key = clip_id.encode() if isinstance(clip_id, str) else clip_id
        raw = self._read(key)
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            result = {
                k: data[k]
                for k in ("body_features", "left_features", "right_features", "face_features")
            }
            for field in ("translation", "language"):
                try:
                    result[field] = data[field]
                except ValueError as exc:
                    if "Object arrays" not in str(exc):
                        raise
                    # Only original metadata of the already trusted/hash-verified release.
                    # Pose arrays above and every derived cache remain allow_pickle=False.
                    with np.load(io.BytesIO(raw), allow_pickle=True) as metadata:
                        value = metadata[field]
                    normalized_text(value)  # Reject non-string metadata layouts.
                    result[field] = value
        return result

    def close(self):
        if self._env is not None:
            self._env.close()
        self._env, self._pid = None, None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"], state["_pid"] = None, None
        return state
