"""Deterministic four-part stride-2 AE, exact true-length convolution boundaries."""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from cprflow.types import PART_DIMS


def conv(cin, cout, k=3, stride=1, dilation=1):
    return nn.Conv1d(
        cin,
        cout,
        k,
        stride=stride,
        padding=dilation * (k // 2),
        dilation=dilation,
        padding_mode="replicate",
    )


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class Residual(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            ChannelNorm(channels),
            nn.SiLU(),
            conv(channels, channels, dilation=dilation),
            ChannelNorm(channels),
            nn.SiLU(),
            conv(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class PartEncoder(nn.Sequential):
    def __init__(self, dim):
        super().__init__(
            conv(dim, 128),
            nn.SiLU(),
            Residual(128, 1),
            Residual(128, 2),
            conv(128, 256, stride=2),
            Residual(256, 1),
            Residual(256, 2),
            conv(256, 128, 1),
        )


class PartDecoder(nn.Sequential):
    def __init__(self, dim):
        super().__init__(
            conv(128, 256),
            Residual(256, 2),
            Residual(256, 1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv(256, 128),
            Residual(128, 2),
            Residual(128, 1),
            conv(128, dim, 1),
        )


class MotionAE(nn.Module):
    def __init__(self, motion_normalizer=None, execution="per_clip"):
        super().__init__()
        if execution not in ("per_clip", "masked_batch"):
            raise ValueError("Unknown AE execution mode")
        self.execution = execution
        self.encoders = nn.ModuleList([PartEncoder(d) for d in PART_DIMS])
        self.decoders = nn.ModuleList([PartDecoder(d) for d in PART_DIMS])
        self.fusion = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 512)
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        if motion_normalizer is not None:
            identity = torch.zeros(296)
            identity[:246] = torch.tensor([1.0, 0, 0, 0, 1, 0]).repeat(41)
            identity[246:] = motion_normalizer.mean.detach().cpu()[246:]
            bias = (
                identity - motion_normalizer.mean.detach().cpu()
            ) / motion_normalizer.std.detach().cpu()
            for part, dec in zip(bias.split(PART_DIMS), self.decoders):
                nn.init.normal_(dec[-1].weight, std=1e-3)
                dec[-1].bias.data.copy_(part)

    def encode(self, x, lengths):
        if self.execution == "masked_batch" and len(x) > 1:
            from cprflow.models.temporal_batch import encode

            return encode(self, x, lengths)
        clips = []
        for sample, n in zip(x, lengths.detach().cpu().tolist()):
            parts = sample[: int(n)].split(PART_DIMS, -1)
            clips.append(torch.stack([enc(p.T[None])[0].T for enc, p in zip(self.encoders, parts)]))
        # Pad time only; pad_sequence uses time as first dimension.
        return pad_sequence([z.permute(1, 0, 2) for z in clips], batch_first=True).permute(
            0, 2, 1, 3
        )

    def decode(self, z, lengths):
        if self.execution == "masked_batch" and len(z) > 1:
            from cprflow.models.temporal_batch import decode

            return decode(self, z, lengths)
        outputs = []
        for latent, n in zip(z, lengths.detach().cpu().tolist()):
            count = (int(n) + 1) // 2
            latent = latent[:, :count]
            concat = latent.permute(1, 0, 2).reshape(count, 512)
            mixed = (concat + self.fusion(concat)).reshape(count, 4, 128).permute(1, 0, 2)
            outputs.append(
                torch.cat(
                    [dec(mixed[p].T[None])[0].T[: int(n)] for p, dec in enumerate(self.decoders)],
                    -1,
                )
            )
        return pad_sequence(outputs, batch_first=True)

    def forward(self, x, lengths):
        z = self.encode(x, lengths)
        return self.decode(z, lengths), z

    def freeze(self):
        self.requires_grad_(False)
        return self.eval()
