"""Read author's Lightning checkpoints without executing their legacy dataset classes.
Only fixed, known metadata classes are permitted; no weights_only=False fallback.
"""

from collections import defaultdict
import typing
import importlib
import numpy as np
import torch


class DiscardedLegacyMetadata:
    def __setstate__(self, state):
        # Historical word-vectorizer objects are not part of the inference model.
        return None


def load_author_state(path):
    try:
        reconstruct = importlib.import_module("numpy._core.multiarray")._reconstruct
    except ImportError:
        reconstruct = importlib.import_module("numpy.core.multiarray")._reconstruct
    allow = [
        dict,
        list,
        tuple,
        int,
        float,
        str,
        defaultdict,
        typing.Any,
        np.ndarray,
        np.dtype,
        (reconstruct, "numpy.core.multiarray._reconstruct"),
        reconstruct,
        (DiscardedLegacyMetadata, "mGPT.data.humanml.utils.word_vectorizer.WordVectorizer"),
        (DiscardedLegacyMetadata, "mGPT.utils.load_checkpoint._LegacyWordVectorizer"),
    ]
    if hasattr(np, "dtypes"):
        allow += [
            value
            for value in vars(np.dtypes).values()
            if isinstance(value, type) and issubclass(value, np.dtype)
        ]
    else:
        allow += [
            type(np.dtype(dtype))
            for dtype in (
                np.float16,
                np.float32,
                np.float64,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.bool_,
                object,
            )
        ]
    try:
        from omegaconf import DictConfig, ListConfig
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.nodes import AnyNode

        allow += [DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode]
    except ImportError:
        pass
    with torch.serialization.safe_globals(allow):
        try:
            record = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError as error:
            if "mmap can only be used" not in str(error):
                raise
            record = torch.load(path, map_location="cpu", weights_only=True)
    state = record.get("state_dict", record)
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a tensor-only author state_dict")
    return state
