"""Train-only calibration and batched orthogonal projections."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from slp.evaluation.smplx_fk import SMPLXJoints, smplx_dir, stats_dir

from .chart import RotationChart
from .data import POSE_DIM

GROUP_DIMS = (6, 12, 30, 35, 13)
TIP_INDICES = (4, 8, 12, 16, 20)
MODES = ("articulatory", "pca", "random", "temporal", "parts", "single")
HAND_DIM = 90


def coefficient_counts(lengths: Tensor, mode: str, rank: int, stride: int):
    slow = (lengths + stride - 1) // stride
    if mode == "single":
        return lengths, lengths, POSE_DIM * lengths
    extra = (lengths - slow) * rank
    if mode == "temporal":
        slow = slow + (extra + POSE_DIM // 2) // POSE_DIM
        return slow, slow, slow * POSE_DIM
    if mode == "parts":
        hand_count = torch.minimum(lengths, slow + (extra + HAND_DIM // 2) // HAND_DIM)
        return slow, hand_count, slow * POSE_DIM + (hand_count - slow) * HAND_DIM
    return slow, slow, slow * POSE_DIM + extra


def dct_columns(lengths: Tensor, size: int, counts: Tensor, width: int, dtype) -> Tensor:
    """Orthonormal DCT-II columns on each clip, with zero rows/columns for padding."""
    frames = torch.arange(size, device=lengths.device)
    modes = torch.arange(width, device=lengths.device)
    angles = (frames.to(dtype) + 0.5)[None, :, None] * modes[None, None, :]
    angles = angles * (math.pi / lengths.to(dtype))[:, None, None]
    scales = ((2 - (modes == 0).to(dtype)) / lengths[:, None].to(dtype)).sqrt()
    valid = (frames[None, :, None] < lengths[:, None, None]) & (
        modes[None, None, :] < counts[:, None, None]
    )
    return (angles.cos() * scales[:, None, :]).masked_fill(~valid, 0)


def low_pass(sequence: Tensor, columns: Tensor) -> Tensor:
    return columns @ (columns.mT @ sequence)


def articulatory_observations(poses: Tensor, body: Tensor, left: Tensor, right: Tensor) -> Tensor:
    shoulder = (body[:, 16] + body[:, 17]) * 0.5
    location = torch.cat(((left[:, 0] + right[:, 0]) * 0.5 - shoulder,
                          right[:, 0] - left[:, 0]), dim=-1)
    directions, shapes = [], []
    for hand in (left, right):
        across = F.normalize(hand[:, 5] - hand[:, 17], eps=1e-8)
        normal = F.normalize(torch.linalg.cross(across, hand[:, 9] - hand[:, 0]), eps=1e-8)
        along = torch.linalg.cross(normal, across)
        frame = torch.stack((across, along, normal), dim=-1)
        directions.extend((along, normal))
        shapes.append(((hand[:, TIP_INDICES] - hand[:, :1]) @ frame).flatten(1))
    left_tips, right_tips = left[:, TIP_INDICES], right[:, TIP_INDICES]
    bimanual = (left_tips[:, :, None] - right_tips[:, None, :]).norm(dim=-1).flatten(1)
    nose = (torch.cat((left_tips, right_tips), dim=1) - body[:, 55:56]).norm(dim=-1)
    return torch.cat((location, *directions, *shapes, bimanual, nose, poses[:, 120:]), dim=-1)


class Projector:
    """Reusable operators for the valid frame grids of one batch."""

    def __init__(self, representation: Representation, lengths: Tensor, size: int):
        self.mode = representation.mode
        self.mask = torch.arange(size, device=lengths.device)[None, :] < lengths[:, None]
        slow, fast, self.coarse_rank = coefficient_counts(
            lengths, self.mode, representation.rank, representation.stride,
        )
        self.detail_rank = lengths * POSE_DIM - self.coarse_rank
        self.axes = representation.axes
        if self.mode != "single":
            with torch.autocast(device_type=lengths.device.type, enabled=False):
                same_counts = self.mode != "parts"
                width = int((slow if same_counts else fast).max())
                self.low = dct_columns(lengths, size, slow, width, self.axes.dtype)
                if not same_counts:
                    self.hand_band = dct_columns(lengths, size, fast, width, self.axes.dtype)
                    self.hand_band = self.hand_band - self.low

    def coarse(self, value: Tensor) -> Tensor:
        with torch.autocast(device_type=value.device.type, enabled=False):
            value = value.to(self.axes.dtype).masked_fill(~self.mask[..., None], 0)
            if self.mode == "single":
                return value
            low = low_pass(value, self.low)
            if self.mode == "temporal":
                return low
            band = low_pass(value, self.hand_band) if self.mode == "parts" else value - low
            return low + (band @ self.axes) @ self.axes.T

    def detail(self, value: Tensor) -> Tensor:
        return (value.to(self.axes.dtype) - self.coarse(value)).masked_fill(
            ~self.mask[..., None], 0,
        )


class Representation(nn.Module):
    def __init__(self, mode: str, rank: int, stride: int, tangent: bool = False):
        super().__init__()
        self.mode, self.rank, self.stride = mode, rank, stride
        dims = HAND_DIM if mode == "parts" else (0 if mode in ("single", "temporal") else rank)
        self.register_buffer("axes", torch.zeros(POSE_DIM, dims))
        self.register_buffer("white", torch.eye(POSE_DIM))
        self.register_buffer("unwhite", torch.eye(POSE_DIM))
        self.register_buffer("mean", torch.zeros(POSE_DIM))
        self.register_buffer("scales", torch.ones(2))
        self.register_buffer("observation_std", torch.ones(sum(GROUP_DIMS)))
        self.register_buffer("energy_axes", torch.zeros(POSE_DIM, rank))
        self.register_buffer("energy_std", torch.ones(4, rank))
        self.chart = RotationChart() if tangent else None

    def encode(self, poses: Tensor) -> Tensor:
        with torch.autocast(device_type=poses.device.type, enabled=False):
            poses = poses.to(self.white.dtype)
            if self.chart is not None:
                poses = self.chart.to_tangent(poses)
            return (poses - self.mean) @ self.white

    def decode(self, state: Tensor) -> Tensor:
        coordinates = state @ self.unwhite + self.mean
        return self.chart.from_tangent(coordinates) if self.chart is not None else coordinates

    def projector(self, lengths: Tensor, size: int) -> Projector:
        return Projector(self, lengths, size)


def frame_intervals(lengths: Tensor, size: int) -> Tensor:
    # Same convention as array_split: remainder frames go to the first bins.
    cuts = torch.arange(1, 4, device=lengths.device)
    boundaries = lengths[:, None] // 4 * cuts + torch.minimum(lengths[:, None] % 4, cuts)
    frames = torch.arange(size, device=lengths.device)
    return (frames[None, :, None] >= boundaries[:, None, :]).sum(-1)


def envelope(sequence: Tensor, lengths: Tensor, representation: Representation) -> tuple[Tensor, Tensor]:
    with torch.autocast(device_type=sequence.device.type, enabled=False):
        base = (lengths + representation.stride - 1) // representation.stride
        size = sequence.shape[1]
        columns = dct_columns(lengths, size, base, int(base.max()), sequence.dtype)
        fast = sequence - low_pass(sequence, columns)
        coefficients = (fast @ representation.energy_axes).square()
        bins = F.one_hot(frame_intervals(lengths, sequence.shape[1]), 4).to(sequence.dtype)
        valid = torch.arange(size, device=lengths.device)[None, :] < lengths[:, None]
        bins = bins * valid[..., None]
        counts = bins.sum(1)
        return ((bins.mT @ coefficients) / counts.clamp_min(1)[..., None]).log1p(), counts > 0


class Moments:
    """Sufficient statistics for whitening, regression and stage RMS."""

    def __init__(self, device, observations: bool):
        self.count = self.rank = 0
        self.total = torch.zeros(POSE_DIM, device=device, dtype=torch.float64)
        self.second = self.total.new_zeros(POSE_DIM, POSE_DIM)
        self.fast = torch.zeros_like(self.second)
        self.alternate_fast = torch.zeros_like(self.second)
        if observations:
            self.obs_sum = self.total.new_zeros(sum(GROUP_DIMS))
            self.obs_square = torch.zeros_like(self.obs_sum)
            self.cross = self.total.new_zeros(POSE_DIM, sum(GROUP_DIMS))

    def add(self, poses, columns, alternate, rank, observations):
        d = poses.shape[-1]
        high = (poses - low_pass(poses, columns)).reshape(-1, d)
        flat = poses.reshape(-1, d)
        self.total += flat.sum(0)
        self.second.addmm_(flat.T, flat)
        self.fast.addmm_(high.T, high)
        if alternate is not None:
            other = (poses - low_pass(poses, alternate)).reshape(-1, d)
            self.alternate_fast.addmm_(other.T, other)
        if observations is not None:
            obs = observations.reshape(-1, observations.shape[-1])
            self.obs_sum += obs.sum(0)
            self.obs_square += obs.square().sum(0)
            high_obs = (observations - low_pass(observations, columns)).flatten(0, 1)
            self.cross.addmm_(high.T, high_obs)
        self.count += int(rank[0])
        self.rank += int(rank[1])


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DescriptorExtractor:
    def __init__(self, stats: str | None, device: torch.device, chunk: int):
        model, directory = rig_assets(stats)
        self.provenance = rig_checksums(model, directory)
        self.fk = SMPLXJoints(device=str(device), model_dir=model, stats=directory)
        self.chunk = chunk

    def __call__(self, poses: Tensor) -> Tensor:
        flat = poses.flatten(0, 1)
        body, left, right = self.fk(flat.cpu().numpy(), chunk=self.chunk)
        return articulatory_observations(flat.to(body.device), body, left, right).view(
            *poses.shape[:2], -1,
        )


def rig_assets(stats=None):
    model = smplx_dir()
    if model.is_dir():
        npz = model / "smplx/SMPLX_NEUTRAL.npz"
        model = npz if npz.is_file() else model / "smplx/SMPLX_NEUTRAL.pkl"
    return model, Path(stats) if stats else stats_dir()


def rig_checksums(model, directory):
    return {"smplx": checksum(model),
            **{name: checksum(directory / name) for name in ("mean.pt", "std.pt")}}


@torch.no_grad()
def calibrate(representation: Representation, batches: Iterable, settings: dict,
              descriptor: DescriptorExtractor | None, manifest: list[str], use_energy: bool):
    # Small float64 matrices accumulate on CPU, including when FK runs on GPU.
    mode, rank, stride = representation.mode, representation.rank, representation.stride
    moments = Moments("cpu", descriptor is not None)
    for microbatches in batches:
        for batch in microbatches:
            lengths = batch.lengths
            size = batch.poses.shape[1]
            slow, fast, ranks = coefficient_counts(lengths, mode, rank, stride)
            # FK once on the concatenated valid frames, then scatter for per-clip DCT.
            mask = torch.arange(size)[None, :] < lengths[:, None]
            obs = None
            if descriptor is not None:
                values = descriptor(batch.poses[mask][None]).squeeze(0).to("cpu", torch.float64)
                obs = values.new_zeros(len(lengths), size, values.shape[-1])
                obs[mask] = values
            poses = batch.poses
            if representation.chart is not None:
                chart = representation.chart
                poses = chart.to_tangent(poses.to(chart.reference)).cpu()
                poses = poses.masked_fill(~mask[..., None], 0)
            poses = poses.double()
            # Every ablation shares whitening learned using the same base cutoff.
            base = (lengths + stride - 1) // stride
            columns = dct_columns(lengths, size, base, int(base.max()), poses.dtype)
            extra = None
            if mode in ("temporal", "parts"):
                counts = slow if mode == "temporal" else fast
                extra = dct_columns(lengths, size, counts, int(counts.max()), poses.dtype)
            moments.add(poses, columns, extra, (lengths.sum(), ranks.sum()), obs)

    n = moments.count
    mean = moments.total / n
    center = moments.second - n * torch.outer(mean, mean)
    eig, pca = torch.linalg.eigh(moments.fast / n)
    ridge = max(0.01 * eig.clamp_min(0).sum().item() / POSE_DIM, 1e-10)
    radius = (eig.clamp_min(0) + ridge).sqrt()
    white, unwhite = (pca / radius) @ pca.T, (pca * radius) @ pca.T

    axes = representation.axes.cpu().double()
    if descriptor is not None:
        std = (moments.obs_square / n - (moments.obs_sum / n).square()).clamp_min(0).sqrt()
        std = std.clamp_min(1e-4)
        weights = std.new_tensor(settings["weights"]).div(std.new_tensor(GROUP_DIMS)).sqrt()
        weights = weights.repeat_interleave(torch.tensor(GROUP_DIMS))
        u, _, _ = torch.linalg.svd((white @ moments.cross / n) * (weights / std), full_matrices=False)
        relations = u[:, :rank]
        representation.observation_std.copy_(std)
        if use_energy:
            representation.energy_axes.copy_(relations)
        if mode == "articulatory":
            axes = relations
    if mode == "pca":
        axes = pca[:, -rank:].flip(1)
    elif mode == "random":
        generator = torch.Generator().manual_seed(settings["random_seed"])
        axes = torch.linalg.qr(torch.randn(POSE_DIM, rank, dtype=torch.float64,
                                          generator=generator)).Q
    elif mode == "parts":
        axes = torch.linalg.qr(white[30:120].T).Q

    total = torch.trace(white @ center @ white)
    leftover = white @ moments.fast @ white
    if mode in ("temporal", "parts"):
        shorter_band = white @ moments.alternate_fast @ white
        if mode == "parts":
            fine = torch.trace(leftover) - torch.trace(axes.T @ (leftover - shorter_band) @ axes)
        else:
            fine = torch.trace(shorter_band)
    else:
        fine = torch.trace(leftover) - torch.trace(axes.T @ leftover @ axes)

    if mode == "single":
        fine = total.new_zeros(())
    total = total.clamp_min(0)
    fine = torch.minimum(fine.clamp_min(0), total)
    coarse = total - fine
    if settings["unconstrained"]:
        fine = total
    detail_dof = n * POSE_DIM - moments.rank
    if settings["unconstrained"]:
        detail_dof = n * POSE_DIM
    scales = torch.stack((coarse / moments.rank, fine / max(1, detail_dof)))
    scales = scales.clamp_min(1e-12).sqrt()
    for name, tensor in (("mean", mean), ("white", white), ("unwhite", unwhite),
                         ("axes", axes), ("scales", scales)):
        getattr(representation, name).copy_(tensor)

    if use_energy:
        energy_sum = mean.new_zeros(4, rank)
        energy_sq = mean.new_zeros(4, rank)
        counts = mean.new_zeros(4, 1)
        device = representation.white.device
        for microbatches in batches:
            for batch in microbatches:
                lengths, poses = batch.lengths.to(device), batch.poses.to(device)
                energy, valid = envelope(representation.encode(poses), lengths, representation)
                energy = energy.cpu().double() * valid.cpu()[..., None]
                energy_sum += energy.sum(0)
                energy_sq += energy.square().sum(0)
                counts += valid.cpu().sum(0)[:, None]
        std = (energy_sq / counts.clamp_min(1) - (energy_sum / counts.clamp_min(1)).square())
        representation.energy_std.copy_(std.clamp_min(1e-8).sqrt())

    return {
        "manifest_sha256": hashlib.sha256(json.dumps(manifest).encode()).hexdigest(),
        "clips": len(manifest),
        "frames": n,
        "coarse_fraction": moments.rank / (n * POSE_DIM),
        "rig": descriptor.provenance if descriptor is not None else None,
    }
