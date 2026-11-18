"""Explicit SOKE codec. No hand-mean changes in matrix coordinates."""

import torch
from torch import nn

from ..geometry import so3

SOKE_MAPPING = tuple(range(12, 22)) + tuple(range(25, 40)) + tuple(range(40, 55)) + (22,)


class SOKECodec(nn.Module):
    def __init__(self, manifest, mean, std, hand_mean):
        super().__init__()
        if (
            manifest.smplx_joint_indices != SOKE_MAPPING
            or manifest.expression_dim != 10
            or manifest.hand_convention != "physical=axis_angle+smplx_hand_mean"
        ):
            raise ValueError("SOKE codec requires its verified mapping and hand convention")
        for name, x, shape in (
            ("mean", mean, (133,)),
            ("std", std, (133,)),
            ("hand_mean", hand_mean, (41, 3)),
        ):
            x = torch.as_tensor(x).float()
            if x.shape != shape or not torch.isfinite(x).all():
                raise ValueError(f"Invalid {name}")
            self.register_buffer(name, x)
        if (self.std <= 0).any():
            raise ValueError("Positive standard deviations required")

    def encode(self, features):
        raw = features[..., :123] * self.std[:123] + self.mean[:123]
        return so3.exp(raw.unflatten(-1, (41, 3)) + self.hand_mean), features[..., 123:]

    def decode(self, r, e):
        aa = so3.log(r) - self.hand_mean
        return torch.cat(((aa.flatten(-2) - self.mean[:123]) / self.std[:123], e), -1)

    def expression_raw(self, e):
        return e * self.std[123:] + self.mean[123:]
