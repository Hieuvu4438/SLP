"""The two representation stages (Sign-JEPA, compact latent), calibration and checkpoints.

Used by ``method.Method.fit`` (single dataset) and by ``pretrain.py`` (union of train splits).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from slp.paths import PROJECT_ROOT
from slp.tracking.run import RunContext

from .autoencoder import LatentSpace, PoseFKLoss, autoencoder_losses
from .data import Batch, PoseDataset, sample_group_mask, subset_indices
from .jepa import ARCH_KEYS, SignJEPA, teacher_momentum
from .training import evaluate_loader, make_loader, make_optimizer, train_stage

JEPA_VERSION = 1
AE_VERSION = 1


def atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
        temp = Path(file.name)
    try:
        torch.save(obj, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def resolve(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


# --------------------------------------------------------------------------- Sign-JEPA
def jepa_arch(cfg: dict) -> dict:
    return {k: cfg[k] for k in ARCH_KEYS}


def save_jepa(model: SignJEPA, arch: dict, path: Path, extra: dict | None = None) -> None:
    atomic_save({"version": JEPA_VERSION, "arch": arch, "model": model.state_dict(),
                 **(extra or {})}, path)


def load_jepa(path: str | Path, device: torch.device) -> tuple[SignJEPA, dict]:
    ckpt = torch.load(resolve(path), map_location="cpu", weights_only=True)
    if ckpt.get("version") != JEPA_VERSION:
        raise ValueError(f"{path}: unsupported Sign-JEPA checkpoint version")
    model = SignJEPA(**ckpt["arch"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["arch"]


def train_jepa(train, val, hp: dict, device: torch.device, precision: str, ctx: RunContext,
               out_path: Path, train_limit: int | None = None, val_limit: int | None = None
               ) -> tuple[SignJEPA, dict]:
    cfg = hp["jepa"]
    arch = jepa_arch(cfg)
    model = SignJEPA(**arch).to(device)
    params = [*model.online.parameters(), *model.predictor.parameters(), model.mask_value]
    optimizer = make_optimizer(params, cfg)
    crop = (cfg["crop_min"], cfg["crop_max"])
    loader = make_loader(PoseDataset(train, train_limit, crop), cfg, hp["workers"], device,
                         ctx.seed, shuffle=True)
    val_loader = make_loader(PoseDataset(val, val_limit), cfg, hp["workers"], device, ctx.seed,
                             shuffle=False)
    mask_cfg = cfg["mask"]
    rng = torch.Generator().manual_seed(ctx.seed)

    def criterion(generator: torch.Generator) -> Callable[[Batch], dict]:
        def fn(batch: Batch) -> dict:
            mask = sample_group_mask(len(batch.lengths), mask_cfg["ratio"],
                                     mask_cfg["unit_probs"], generator)
            return model.loss(batch.values, batch.lengths, mask.to(device))
        return fn

    def validate() -> dict:
        return evaluate_loader(val_loader, criterion(torch.Generator().manual_seed(ctx.seed + 1)),
                               device, precision)

    total = cfg["updates"]
    ctx.logger.info("Sign-JEPA: %d clips, %d updates, units %s", len(loader.dataset), total,
                    mask_cfg["unit_probs"])
    train_stage("jepa", loader, cfg, optimizer, (model,), criterion(rng), device, precision, ctx,
                validate=validate, select=None,
                save=lambda: save_jepa(model, arch, out_path),
                after_step=lambda step: model.update_teacher(
                    teacher_momentum(step, total, cfg["momentum"])))
    return model.eval(), arch


# --------------------------------------------------------------------------- compact latent
def space_spec(hp: dict, encoder_arch: dict | None = None) -> dict:
    latent = hp["latent"]
    enc = encoder_arch or jepa_arch(hp["jepa"])
    return {"kind": latent["kind"], "latent_dim": latent["dim"],
            "encoder": {k: enc[k] for k in ("width", "depth", "heads", "mlp_ratio",
                                            "input_velocity", "feature_layers")},
            "compressor": dict(hp["ae"]["compressor"]), "decoder": dict(hp["ae"]["decoder"])}


def save_space(space: LatentSpace, spec: dict, path: Path, extra: dict | None = None) -> None:
    atomic_save({"version": AE_VERSION, "spec": spec, "model": space.state_dict(),
                 **(extra or {})}, path)


def load_space(path: str | Path, device: torch.device, need_calibration: bool = False
               ) -> tuple[LatentSpace, dict]:
    """Load ae.pt. ``need_calibration``: refuse a checkpoint whose flow statistics were never
    estimated (e.g. written by train_autoencoder of an interrupted run)."""
    ckpt = torch.load(resolve(path), map_location="cpu", weights_only=True)
    if ckpt.get("version") != AE_VERSION:
        raise ValueError(f"{path}: unsupported latent checkpoint version")
    if need_calibration and "calibration" not in ckpt:
        raise ValueError(f"{path} has no latent calibration; set method.latent.recalibrate=true")
    space = LatentSpace(**ckpt["spec"])
    space.load_state_dict(ckpt["model"])
    return space.to(device).eval().requires_grad_(False), ckpt["spec"]


def train_autoencoder(space: LatentSpace, spec: dict, train, val, hp: dict,
                      device: torch.device, precision: str, ctx: RunContext, out_path: Path,
                      train_limit: int | None = None, val_limit: int | None = None,
                      stats: str | None = None) -> None:
    """Fit compressor + decoder (+ encoder for ae/vae); keep the lowest validation MSE (paper)."""
    cfg = hp["ae"]
    if space.kind == "raw":
        return
    fk = None
    if cfg.get("fk_weight", 0.0):
        fk = PoseFKLoss(resolve(stats) if stats else None).to(device)
    optimizer = make_optimizer(space.trainable_parameters(), cfg)
    crop = (cfg["crop_min"], cfg["crop_max"])
    loader = make_loader(PoseDataset(train, train_limit, crop), cfg, hp["workers"], device,
                         ctx.seed, shuffle=True)
    val_loader = make_loader(PoseDataset(val, val_limit), cfg, hp["workers"], device, ctx.seed,
                             shuffle=False)

    def criterion(batch: Batch) -> dict:
        return autoencoder_losses(space, batch.values, batch.lengths, cfg, fk)

    def validate() -> dict:
        state = torch.random.get_rng_state()
        try:
            return evaluate_loader(val_loader, criterion, device, precision)
        finally:
            torch.random.set_rng_state(state)

    ctx.logger.info("Compact latent (%s, d=%d): %d clips, %d updates", space.kind,
                    space.latent_dim, len(loader.dataset), cfg["updates"])
    train_stage("ae", loader, cfg, optimizer, (space,), criterion, device, precision, ctx,
                validate=validate, select="mse", save=lambda: save_space(space, spec, out_path))
    best, _ = load_space(out_path, device)
    space.load_state_dict(best.state_dict())


# --------------------------------------------------------------------------- calibration
def _ordered_collate(items):
    poses, indices = zip(*items)
    return (pad_sequence(poses, batch_first=True), torch.tensor([len(p) for p in poses]),
            list(indices))


class _Indexed(torch.utils.data.Dataset):
    def __init__(self, split, indices: list[int]):
        self.data = PoseDataset(split)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.data[self.indices[i]][0], self.indices[i]


@torch.no_grad()
def encode_split(space: LatentSpace, split, indices: list[int], device: torch.device,
                 workers: int, batch_size: int = 32, standardised: bool = True,
                 ) -> dict[int, Tensor]:
    """Posterior means of whole clips in FP32 (paper: latent extraction uses FP32) -> CPU."""
    space.eval()
    loader = DataLoader(_Indexed(split, indices), batch_size=batch_size, shuffle=False,
                        num_workers=workers, collate_fn=_ordered_collate)
    out = {}
    for poses, lengths, idx in loader:
        poses, lengths = poses.to(device), lengths.to(device)
        mean, _ = space.posterior(poses, lengths)
        if standardised:
            mean = space.to_flow(mean)
        mean = mean.float().cpu()
        for row, (i, n) in enumerate(zip(idx, lengths.tolist())):
            out[i] = mean[row, :n].clone()
    return out


@torch.no_grad()
def calibrate(space: LatentSpace, split, hp: dict, device: torch.device,
              limit: int | None = None) -> dict:
    """Per-channel latent mean/std on train clips (flow coordinates) and the duration prior."""
    indices = subset_indices(len(split), limit)
    latents = encode_split(space, split, indices, device, hp["workers"], standardised=False)
    values = torch.cat([latents[i] for i in indices]).double()
    mean = values.mean(0)
    std = values.std(0).clamp_min(hp["latent"]["min_std"])
    space.latent_mean.copy_(mean.float())
    space.latent_std.copy_(std.float())
    log_lengths = np.log([len(latents[i]) for i in indices])
    return {"mean_log_length": float(log_lengths.mean()), "clips": len(indices),
            "frames": int(len(values)), "std_min": float(std.min()), "std_max": float(std.max())}
