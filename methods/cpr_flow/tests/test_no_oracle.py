import copy
import inspect

import numpy as np
import pytest
import torch

from cprflow.config import DEFAULTS
from cprflow.data.stats import Normalizer
from cprflow.models.ae import MotionAE
from cprflow.models.duration import DurationModel
from cprflow.sampling.generator import MotionGenerator
from cprflow.types import GenerationRequest, TextBatch
from test_attention_masks import nonzero_model


def test_request_rejects_targets_and_duration_prediction_controls_shape(normalizer):
    assert set(inspect.signature(GenerationRequest).parameters) == {
        "text",
        "sign_language",
        "seed",
        "duration_seconds",
    }
    with pytest.raises(TypeError):
        GenerationRequest(text="a", sign_language="ASL", seed=7, target_length=20)
    config = copy.deepcopy(DEFAULTS)
    config["run"]["synthetic"] = True
    model = nonzero_model()
    duration = DurationModel()
    with torch.no_grad():
        duration.net[-1].weight.zero_()
        duration.net[-1].bias[0] = np.log(6)
    ae = MotionAE(normalizer).freeze()

    def encode(texts, languages):
        return TextBatch(torch.zeros(1, 2, 768), torch.ones(1, 2, dtype=torch.bool))

    latent_norm = Normalizer(np.zeros((4, 1, 128)), np.ones((4, 1, 128)))
    generator = MotionGenerator(
        model, ae, duration, normalizer, latent_norm, encode, config, {"synthetic": True}
    )
    request = GenerationRequest("fixture", "ASL", 17)
    a = generator.generate(request, nfe=2)
    generator.generate(GenerationRequest("other", "CSL", 42), nfe=2)
    b = generator.generate(request, nfe=2)
    assert a.predicted_frames == 6 and a.duration_source == "predicted"
    np.testing.assert_array_equal(a.canonical_motion, b.canonical_motion)
    assert a.metadata["batch_equivalent_forwards"] == 2
