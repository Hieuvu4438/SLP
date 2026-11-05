"""Four-part, stride-two variational motion codec with length-aware convolutions."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .geometry import PART_SLICES
from .tensors import check_sequence, zero_padding


class MaskedConv(nn.Module):
    def __init__(self, incoming: int, outgoing: int, *, stride: int = 1,
                 dilation: int = 1):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(incoming, outgoing, 3, stride=stride,
                              padding=dilation, dilation=dilation)

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        x = zero_padding(x, lengths)
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out_lengths = (lengths + self.stride - 1) // self.stride
        return zero_padding(y, out_lengths), out_lengths


class TemporalResidual(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        # LayerNorm acts on channels per frame; GroupNorm would mix padded time.
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.conv1 = MaskedConv(width, width, dilation=dilation)
        self.conv2 = MaskedConv(width, width, dilation=dilation)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        h, _ = self.conv1(F.gelu(self.norm1(x)), lengths)
        h, _ = self.conv2(F.gelu(self.norm2(h)), lengths)
        return zero_padding(x + h, lengths)


class PartEncoder(nn.Module):
    def __init__(self, channels: int, latent_width: int, hidden: int):
        super().__init__()
        self.input = MaskedConv(channels, hidden)
        self.blocks = nn.ModuleList(TemporalResidual(hidden, d) for d in (1, 2, 4))
        self.down = MaskedConv(hidden, hidden, stride=2)
        self.output = nn.Linear(hidden, 2 * latent_width)

    def forward(self, x: Tensor, lengths: Tensor):
        h, _ = self.input(x, lengths)
        for block in self.blocks:
            h = block(h, lengths)
        h, latent_lengths = self.down(h, lengths)
        mean, logvar = self.output(h).chunk(2, -1)
        return (zero_padding(mean, latent_lengths),
                zero_padding(logvar.clamp(-10, 6), latent_lengths), latent_lengths)


class PartDecoder(nn.Module):
    def __init__(self, latent_width: int, hidden: int):
        super().__init__()
        self.input = MaskedConv(latent_width, hidden)
        self.blocks = nn.ModuleList(TemporalResidual(hidden, d) for d in (4, 2, 1))
        self.up = MaskedConv(hidden, hidden)

    def forward(self, z: Tensor, latent_lengths: Tensor, frame_lengths: Tensor) -> Tensor:
        h, _ = self.input(z, latent_lengths)
        h = zero_padding(h.repeat_interleave(2, dim=1), frame_lengths)
        h, _ = self.up(h, frame_lengths)
        for block in self.blocks:
            h = block(h, frame_lengths)
        return h


@dataclass
class MotionPosterior:
    mean: Tensor                  # [B,L,4,D]
    logvar: Tensor
    lengths: Tensor

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        noise = torch.randn(self.mean.shape, dtype=self.mean.dtype,
                            device=self.mean.device, generator=generator)
        return zero_padding(self.mean + torch.exp(0.5 * self.logvar) * noise, self.lengths)


class MotionCodec(nn.Module):
    def __init__(self, latent_width: int = 64, hidden: int = 128):
        super().__init__()
        if latent_width < 1 or hidden < 1:
            raise ValueError("codec widths must be positive")
        self.latent_width = latent_width
        self.encoders = nn.ModuleList(
            PartEncoder(part.stop - part.start, latent_width, hidden) for part in PART_SLICES
        )
        self.decoders = nn.ModuleList(PartDecoder(latent_width, hidden) for _ in PART_SLICES)
        self.joint = MaskedConv(4 * hidden, hidden)
        self.output = nn.Linear(hidden, 256)

    def encode(self, x: Tensor, lengths: Tensor) -> MotionPosterior:
        check_sequence(x, lengths, 256)
        encoded = [enc(x[..., part], lengths) for enc, part in zip(self.encoders, PART_SLICES)]
        return MotionPosterior(torch.stack([p[0] for p in encoded], -2),
                               torch.stack([p[1] for p in encoded], -2), encoded[0][2])

    def decode(self, z: Tensor, latent_lengths: Tensor,
               frame_lengths: Tensor | None = None) -> Tensor:
        check_sequence(z, latent_lengths, self.latent_width)
        if z.ndim != 4 or z.shape[2] != 4:
            raise ValueError("expected four latent streams [B,L,4,D]")
        frame_lengths = 2 * latent_lengths if frame_lengths is None else frame_lengths
        if frame_lengths.device != z.device or frame_lengths.shape != latent_lengths.shape:
            raise ValueError("frame and latent lengths must match batch and device")
        if bool((((frame_lengths + 1) // 2) != latent_lengths).any()):
            raise ValueError("frame lengths must invert the stride-two encoder")
        h = torch.cat([dec(z[:, :, p], latent_lengths, frame_lengths)
                       for p, dec in enumerate(self.decoders)], -1)
        h, _ = self.joint(h, frame_lengths)
        x = zero_padding(self.output(h), frame_lengths)
        return x[:, :int(frame_lengths.max())]

    def forward(self, x: Tensor, lengths: Tensor, *, sample: bool = True,
                generator: torch.Generator | None = None):
        posterior = self.encode(x, lengths)
        z = posterior.sample(generator) if sample else posterior.mean
        return self.decode(z, posterior.lengths, lengths), posterior
