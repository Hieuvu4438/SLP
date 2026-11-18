import torch

from cprflow.models.dit import Prediction


def classifier_free_guidance(model, scale, **inputs):
    b = len(inputs["z"])
    device = inputs["z"].device
    conditional = model(**inputs, drop_text=torch.zeros(b, device=device, dtype=torch.bool))
    if scale == 1:
        return conditional, 1
    if scale < 1:
        raise ValueError("CFG scale must be >=1")
    null = model(**inputs, drop_text=torch.ones(b, device=device, dtype=torch.bool))
    r = None if conditional.r is None else null.r + scale * (conditional.r - null.r)
    return Prediction(null.z + scale * (conditional.z - null.z), r, conditional.hidden), 2
