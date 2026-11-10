"""Staged trainers with explicit roles, no benchmark scoring or test-split access."""

from copy import deepcopy
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader
from .config import RaceConfig
from .adapters import file_hash
from .model import FlowDecoder
from .schema import MotionNormalizer, manual_to_6d
from .selector import (
    ImprovementSelector,
    improvement_targets,
    selector_objective,
    calibration_diagnostics,
)
from .data import (
    collate_training,
    collate_inputs,
    require_disjoint,
    window_batch,
    dataset_signature,
)
from .losses import decoder_objective
from .geometry import geometry_signature


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def save_checkpoint(
    path,
    decoder,
    normalizer,
    *,
    selector=None,
    stage,
    provenance,
    optimizer=None,
    epoch=0,
    step=0,
    best_validation=None,
):
    record = dict(
        version=2,
        config=decoder.cfg.to_dict(),
        decoder=decoder.state_dict(),
        normalizer=normalizer.state_dict(),
        selector=None if selector is None else selector.state_dict(),
        stage=stage,
        provenance=provenance,
        epoch=epoch,
        step=step,
        rng_cpu=torch.get_rng_state(),
        rng_cuda=torch.cuda.get_rng_state_all(),
        best_validation=best_validation,
    )
    if optimizer is not None:
        record["optimizer"] = optimizer.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(record, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get("version") != 2 or data.get("stage") not in ("decoder", "selector"):
        raise ValueError("Not a supported RACE checkpoint")
    cfg = RaceConfig.from_dict(data["config"])
    decoder = FlowDecoder(cfg).to(device)
    decoder.load_state_dict(data["decoder"], strict=True)
    decoder.default_use_text = data["provenance"].get("use_text", True)
    decoder.default_coupled = data["provenance"].get("coupled", True)
    decoder.expected_base_signature = data["provenance"].get("base_signature")
    decoder.expected_geometry_signature = data["provenance"].get("geometry_signature")
    normalizer = MotionNormalizer().to(device)
    normalizer.load_state_dict(data["normalizer"], strict=True)
    selector = None
    if data["selector"] is not None:
        selector = ImprovementSelector(cfg).to(device)
        selector.load_state_dict(data["selector"], strict=True)
        selector.whole_sentence = data["provenance"].get("selector_whole_sentence", False)
        selector.sampling_policy = data["provenance"].get("selector_sampling_policy")
    return decoder, normalizer, selector, data


def _check_step(loss, model, optimizer, clip):
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite objective; refusing optimizer update")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def train_decoder(
    cfg,
    train,
    val,
    geometry,
    output,
    *,
    epochs=1,
    batch_size=8,
    lr=2e-4,
    device="cpu",
    seed=0,
    window=64,
    halo=8,
    resume=None,
    initialize_from=None,
    use_text=True,
    coupled=True,
    log=None,
):
    if train.role not in ("decoder_train", "robustness_train") or val.role != "decoder_val":
        raise ValueError("Wrong decoder split roles")
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("Positive epochs/batch size/lr required")
    require_disjoint(train, val)
    signature = dataset_signature(train)
    if signature != dataset_signature(val):
        raise ValueError("Train and validation caches use different base assets")
    seed_all(seed)
    decoder, norm = FlowDecoder(cfg).to(device), MotionNormalizer().to(device)
    geometry = geometry.to(device).eval().requires_grad_(False)
    provenance = {
        "decoder_train_groups": sorted(train.groups),
        "decoder_val_groups": sorted(val.groups),
        "current_train_groups": sorted(train.groups),
        "current_val_groups": sorted(val.groups),
        "seed": seed,
        "normalization_role": train.role,
        "use_text": use_text,
        "coupled": coupled,
        "window": window,
        "halo": halo,
        "base_signature": signature,
        "geometry_signature": geometry_signature(geometry),
    }
    if initialize_from and resume:
        raise ValueError("Use initialization or resume, not both")
    if train.role == "robustness_train" and not (initialize_from or resume):
        raise ValueError("Robustness tuning requires decoder initialization")
    if initialize_from:
        decoder, norm, _, initial = load_checkpoint(initialize_from, device)
        if initial["config"] != cfg.to_dict():
            raise ValueError("Initialization architecture differs")
        prior = initial["provenance"]
        if prior.get("base_signature") != signature:
            raise ValueError("Robustness cache uses different base assets")
        if prior.get("geometry_signature") != geometry_signature(geometry):
            raise ValueError("Robustness geometry/pose codec differs")
        if set(prior.get("decoder_val_groups", [])) & train.groups:
            raise ValueError("Fine-tuning would train on previous validation data")
        if set(prior.get("decoder_train_groups", [])) & val.groups:
            raise ValueError("Fine-tuning validation overlaps previous training data")
        provenance["decoder_train_groups"] = sorted(
            train.groups | set(prior.get("decoder_train_groups", []))
        )
        provenance["decoder_val_groups"] = sorted(
            val.groups | set(prior.get("decoder_val_groups", []))
        )
        provenance["initialized_from"] = file_hash(initialize_from)
        provenance["normalization_role"] = prior["normalization_role"]
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr)
    start, step = 0, 0
    if resume:
        decoder, norm, selector, ckpt = load_checkpoint(resume, device)
        if "initialized_from" in ckpt["provenance"]:
            for key in (
                "initialized_from",
                "normalization_role",
                "decoder_train_groups",
                "decoder_val_groups",
            ):
                provenance[key] = ckpt["provenance"][key]
        if (
            selector is not None
            or ckpt["config"] != cfg.to_dict()
            or ckpt["provenance"] != provenance
        ):
            raise ValueError("Resume checkpoint config/split provenance mismatch")
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr)
        optimizer.load_state_dict(ckpt["optimizer"])
        start = ckpt["epoch"]
        step = ckpt["step"]
        if epochs <= start:
            raise ValueError("Resume epochs must exceed the completed checkpoint epoch")
        torch.set_rng_state(ckpt["rng_cpu"].cpu())
        if device != "cpu" and ckpt["rng_cuda"]:
            torch.cuda.set_rng_state_all([r.cpu() for r in ckpt["rng_cuda"]])
    elif not initialize_from:
        norm.fit((train[i]["target"] for i in range(len(train))), geometry.manual_mean)
    best = ckpt.get("best_validation", float("inf")) if resume else float("inf")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(start, epochs):
        # Epoch-derived shuffle/window streams make continuation reproducible.
        shuffle = torch.Generator().manual_seed(seed + epoch)
        windows = torch.Generator().manual_seed(seed + epoch + 100000)
        loader = DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            generator=shuffle,
            collate_fn=collate_training,
        )
        decoder.train()
        train_losses = []
        for batch in loader:
            batch = to_device(batch, device)
            with torch.no_grad():
                batch["relations"] = geometry(
                    manual_to_6d(batch["baseline"], geometry.manual_mean)
                ).relations
            if window:
                batch = window_batch(batch, window, halo, cfg.factor, windows)
            loss, terms = decoder_objective(
                decoder, norm, geometry, batch, cfg, use_text=use_text, coupled=coupled
            )
            _check_step(loss, decoder, optimizer, 1.0)
            step += 1
            train_losses.append(float(loss.detach()))
            if log:
                log({k: float(v.detach()) for k, v in terms.items()}, step)
        decoder.eval()
        val_sum = 0.0
        val_count = 0
        rng = torch.Generator(device=device).manual_seed(seed + 200000)
        with torch.no_grad():
            for batch in DataLoader(val, batch_size=batch_size, collate_fn=collate_training):
                batch = to_device(batch, device)
                batch["relations"] = geometry(
                    manual_to_6d(batch["baseline"], geometry.manual_mean)
                ).relations
                loss, _ = decoder_objective(
                    decoder,
                    norm,
                    geometry,
                    batch,
                    cfg,
                    generator=rng,
                    use_text=use_text,
                    coupled=coupled,
                )
                val_sum += float(loss) * len(batch["lengths"])
                val_count += len(batch["lengths"])
        row = {
            "epoch": epoch + 1,
            "train_loss": sum(train_losses) / len(train_losses),
            "validation_loss": val_sum / val_count,
        }
        history.append(row)
        improved = row["validation_loss"] < best
        best = min(best, row["validation_loss"])
        save_checkpoint(
            output / "last.pt",
            decoder,
            norm,
            stage="decoder",
            provenance=provenance,
            optimizer=optimizer,
            epoch=epoch + 1,
            step=step,
            best_validation=best,
        )
        if improved:
            save_checkpoint(
                output / "best.pt",
                decoder,
                norm,
                stage="decoder",
                provenance=provenance,
                epoch=epoch + 1,
                step=step,
                best_validation=best,
            )
    return decoder.eval(), norm, history


def train_selector(
    decoder_checkpoint,
    train,
    val,
    calibration,
    geometry,
    output,
    *,
    epochs=1,
    batch_size=8,
    lr=2e-4,
    device="cpu",
    seed=0,
    whole_sentence=False,
):
    if (train.role, val.role, calibration.role) != (
        "selector_train",
        "selector_val",
        "selector_calibration",
    ):
        raise ValueError("Wrong selector roles")
    require_disjoint(train, val, calibration)
    seed_all(seed)
    decoder, norm, _, saved = load_checkpoint(decoder_checkpoint, device)
    decoder.eval().requires_grad_(False)
    geometry = geometry.to(device).eval().requires_grad_(False)
    cfg = decoder.cfg
    if saved["provenance"].get("geometry_signature") != geometry_signature(geometry):
        raise ValueError("Selector geometry/pose codec differs from decoder training")
    prior_groups = set(saved["provenance"].get("decoder_train_groups", [])) | set(
        saved["provenance"].get("decoder_val_groups", [])
    )
    if prior_groups & (train.groups | val.groups | calibration.groups):
        raise ValueError("Selector data overlaps decoder development groups")
    if any(
        dataset_signature(d) != saved["provenance"].get("base_signature")
        for d in (train, val, calibration)
    ):
        raise ValueError("Selector data uses different frozen base assets")
    selector = ImprovementSelector(cfg).to(device)
    selector.whole_sentence = whole_sentence
    expected_hash = file_hash(decoder_checkpoint)
    sampling_policy = None
    for dataset in (train, val, calibration):
        for i in range(len(dataset)):
            metadata = dataset[i]["provenance"]
            if metadata.get("candidate_decoder_sha256") != expected_hash:
                raise ValueError(
                    "Candidate cache was not produced by this frozen decoder checkpoint"
                )
            policy = {
                key: metadata[key]
                for key in ("candidate_steps", "candidate_use_text", "candidate_coupled")
            }
            if sampling_policy is not None and sampling_policy != policy:
                raise ValueError("Inconsistent candidate sampling policies in selector data")
            sampling_policy = policy
    selector.sampling_policy = sampling_policy
    optimizer = torch.optim.AdamW(selector.parameters(), lr=lr)
    best, best_weights = float("inf"), None

    def forward(records):
        batch = collate_inputs(records).to(device).validate(cfg)
        target = torch.nn.utils.rnn.pad_sequence(
            [r["target"] for r in records], batch_first=True
        ).to(device)
        candidate = torch.nn.utils.rnn.pad_sequence(
            [r["candidate"] for r in records], batch_first=True
        ).to(device)
        if candidate.shape != batch.baseline.shape or candidate.shape != target.shape:
            raise ValueError("Candidate cache must share the native timeline")
        with torch.no_grad():
            b, c, y = (
                manual_to_6d(x, geometry.manual_mean) for x in (batch.baseline, candidate, target)
            )
            br, cr = geometry(b).relations, geometry(c).relations
            truth, _ = improvement_targets(
                b, c, y, batch.lengths, geometry, cfg, whole_sentence=whole_sentence
            )
        result = selector(
            norm(b),
            norm(c),
            br,
            cr,
            batch.text,
            batch.text_mask,
            batch.lengths,
            batch.language,
            codes=batch.codes,
            whole_sentence=whole_sentence,
        )
        return result, truth

    history = []
    for epoch in range(epochs):
        selector.train()
        train_losses = []
        loader = DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=list,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        for records in loader:
            out, truth = forward(records)
            loss = selector_objective(out, truth)
            _check_step(loss, selector, optimizer, 1.0)
            train_losses.append(float(loss.detach()))
        selector.eval()
        losses, counts = [], []
        with torch.no_grad():
            for records in DataLoader(val, batch_size=batch_size, collate_fn=list):
                out, truth = forward(records)
                losses.append(float(selector_objective(out, truth)))
                counts.append(int(out["mask"].sum()))
        score = sum(loss * count for loss, count in zip(losses, counts)) / sum(counts)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": sum(train_losses) / len(train_losses),
                "validation_loss": score,
            }
        )
        if score < best:
            best, best_weights = score, deepcopy(selector.state_dict())
    if best_weights is None:
        raise ValueError("Positive selector training epochs required")
    selector.load_state_dict(best_weights)
    selector.eval()
    predicted, observed, win_logits = [], [], []
    with torch.no_grad():
        for records in DataLoader(calibration, batch_size=batch_size, collate_fn=list):
            out, truth = forward(records)
            predicted.append(out["raw_delta"][out["mask"]])
            observed.append(truth[out["mask"]])
            win_logits.append(out["win_logit"][out["mask"]])
    predicted, observed = torch.cat(predicted), torch.cat(observed)
    selector.calibrate(predicted, observed, torch.ones_like(predicted, dtype=torch.bool))
    provenance = dict(
        saved["provenance"],
        selector_train_groups=sorted(train.groups),
        selector_val_groups=sorted(val.groups),
        calibration_groups=sorted(calibration.groups),
        selector_whole_sentence=whole_sentence,
        selector_sampling_policy=sampling_policy,
        calibration_diagnostics=calibration_diagnostics(
            selector, predicted, observed, torch.cat(win_logits)
        ),
    )
    save_checkpoint(
        output,
        decoder,
        norm,
        selector=selector,
        stage="selector",
        provenance=provenance,
        epoch=epochs,
    )
    return selector, history
