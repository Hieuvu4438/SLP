"""Portable method checkpoint; never pickle a Python model/tokenizer or include FK assets."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from .losses import LossWeights
from .segments import TeacherConfig
from .system import ModelConfig, TemporalPlanSystem, STAGES
from .text import TextConditioner

FORMAT_VERSION = 1


def tokenizer_digest(directory: Path):
    digest = hashlib.sha256()
    for item in sorted(directory.iterdir()):
        if item.is_file():
            digest.update(item.name.encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()


def model_config(values: dict) -> ModelConfig:
    values = dict(values)
    values["teacher"] = TeacherConfig(**values.get("teacher", {}))
    values["losses"] = LossWeights(**values.get("losses", {}))
    return ModelConfig(**values)


def _cpu_state(model, exclude_backbone):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if not (exclude_backbone and name.startswith("text.encoder."))}


def save_checkpoint(model: TemporalPlanSystem, path: str | Path, *, tokenizer=None,
                    metadata: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pretrained = model.text.pretrained_spec is not None
    if pretrained:
        text = {"kind": "pretrained", **model.text.pretrained_spec}
    else:
        # Explicitly injected MT5 architecture: tests/from-scratch experiments, no fallback.
        text = {"kind": "mt5_config", "config": model.text.encoder.config.to_dict()}
    text["output_width"] = model.text.output_width
    text["num_languages"] = model.text.language.num_embeddings
    if text["num_languages"] != 3:
        raise ValueError("checkpoint language contract requires ASL/CSL/DGS")
    token_dir = None
    token_hash = None
    if tokenizer is not None:
        # Immutable/content-addressed tokenizer: a failed save cannot invalidate an
        # older checkpoint by overwriting its tokenizer in place.
        with tempfile.TemporaryDirectory(prefix=".tokenizer.", dir=path.parent) as temporary:
            temporary = Path(temporary)
            tokenizer.save_pretrained(temporary)
            token_hash = tokenizer_digest(temporary)
            token_dir = "tokenizer_" + token_hash
            destination = path.parent / token_dir
            if not destination.exists():
                os.replace(temporary, destination)
            elif tokenizer_digest(destination) != token_hash:
                raise ValueError("existing tokenizer store is corrupted")
    metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
    state = _cpu_state(model, pretrained)
    if any(value.is_floating_point() and not bool(torch.isfinite(value).all())
           for value in state.values()):
        raise FloatingPointError("refusing to save non-finite model state")
    payload = {
        "format_version": FORMAT_VERSION, "model_config": asdict(model.config),
        "state_dict": state, "text": text,
        "tokenizer_dir": token_dir, "tokenizer_sha256": token_hash,
        "stage": model.stage, "metadata": metadata,
        "torch_version": str(torch.__version__),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def read_checkpoint(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION or payload.get("stage") not in STAGES:
        raise ValueError("unsupported or malformed temporal-plan checkpoint")
    return payload


def restore_into(model: TemporalPlanSystem, payload: dict):
    if asdict(model_config(payload["model_config"])) != asdict(model.config):
        raise ValueError("checkpoint model config does not match current architecture")
    if payload["text"]["kind"] == "pretrained":
        expected_spec = {k: payload["text"][k] for k in ("model_id", "revision")}
        if model.text.pretrained_spec != expected_spec:
            raise ValueError("checkpoint pretrained backbone revision does not match")
    actual = model.state_dict()
    expected = ({name for name in model.state_dict() if name.startswith("text.encoder.")}
                if payload["text"]["kind"] == "pretrained" else set())
    missing = set(actual) - set(payload["state_dict"])
    unexpected = set(payload["state_dict"]) - set(actual)
    if missing != expected or unexpected:
        raise ValueError(f"incomplete checkpoint: missing={sorted(missing)}, "
                         f"unexpected={sorted(unexpected)}")
    for name, tensor in payload["state_dict"].items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != actual[name].shape:
            raise ValueError(f"checkpoint tensor shape mismatch: {name}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"checkpoint contains non-finite state: {name}")
    completed = payload["state_dict"]["completed_stages"]
    if completed.dtype != torch.bool or any(
        bool(completed[i]) and not bool(completed[:i].all()) for i in range(len(completed))
    ):
        raise ValueError("checkpoint stage-completion flags must be a contiguous prefix")
    for prefix in ("motion_norm", "latent_norm", "content_norm"):
        if bool((payload["state_dict"][f"{prefix}.std"] <= 0).any()):
            raise ValueError("checkpoint has nonpositive normalization scale")
        count = int(payload["state_dict"][f"{prefix}.count"])
        if count < 0 or (bool(completed.all()) and count == 0):
            raise ValueError("completed checkpoint requires fitted normalization statistics")
    pose_hash = payload.get("metadata", {}).get("pose_mean_sha256")
    if pose_hash is not None and pose_hash != hashlib.sha256(
        payload["state_dict"]["representation.rotation_offsets"].cpu().numpy().tobytes()
    ).hexdigest():
        raise ValueError("checkpoint pose-mean provenance does not match its state")
    # Check BEFORE mutating the live model.
    model.load_state_dict(payload["state_dict"], strict=False)
    model.stage = payload["stage"]
    model.eval()


def load_checkpoint(path: str | Path, *, device="cpu", allow_download: bool = False,
                    require_tokenizer: bool = True, require_completed: bool = False):
    path = Path(path)
    payload = read_checkpoint(path)
    if require_completed and not bool(payload["state_dict"]["completed_stages"].all()):
        raise ValueError("inference requires a completed five-stage checkpoint")
    description = payload["text"]
    if description["kind"] == "pretrained":
        text, _ = TextConditioner.from_pretrained(output_width=description["output_width"],
                                                  local_files_only=not allow_download,
                                                  spec=description)
    elif description["kind"] == "mt5_config":
        from transformers import MT5Config, MT5EncoderModel
        encoder = MT5EncoderModel(MT5Config(**description["config"]))
        text = TextConditioner(encoder, description["output_width"])
    else:
        raise ValueError("unknown text-backbone checkpoint kind")
    model = TemporalPlanSystem(model_config(payload["model_config"]), text,
                                payload["state_dict"]["representation.rotation_offsets"])
    restore_into(model, payload)
    tokenizer = None
    if payload.get("tokenizer_dir"):
        token_path = (path.parent / payload["tokenizer_dir"]).resolve()
        if token_path.parent != path.parent.resolve():
            raise ValueError("tokenizer must be stored beside its checkpoint")
        if tokenizer_digest(token_path) != payload.get("tokenizer_sha256"):
            raise ValueError("checkpoint tokenizer integrity check failed")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(token_path, local_files_only=True)
    if require_tokenizer and tokenizer is None:
        raise ValueError("checkpoint does not contain its tokenizer")
    return model.to(device).eval(), tokenizer, payload["metadata"]
