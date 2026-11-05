"""Construction and plain-text prediction; reference motions are not accepted by inference."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from .checkpoint import model_config
from .data import tokenize_texts
from .kinematics import SMPLXForward
from .representation import MotionRepresentation
from .system import TemporalPlanSystem
from .text import LANGUAGES, TextConditioner


def create_system(train, options: dict, *, smplx_model_dir=None, rotation_offsets=None,
                  smplx_ext="npz", allow_download=False, text_width=512):
    options = dict(options)
    if options.get("max_latents") is not None:
        raise ValueError("max_latents is derived from TRAIN data, not an input length/config override")
    options["max_latents"] = train.max_latents
    config = model_config(options)
    fk = None
    if smplx_model_dir is not None:
        if rotation_offsets is not None:
            raise ValueError("provide SMPL-X assets or explicit offsets, not both")
        fk = SMPLXForward.from_model_dir(smplx_model_dir, ext=smplx_ext)
        offsets = fk.rotation_offsets.detach().clone()
    elif config.losses.fk > 0:
        raise ValueError("positive FK objective needs licensed SMPL-X model assets")
    elif rotation_offsets is not None:
        offsets = torch.from_numpy(np.load(rotation_offsets, allow_pickle=False)).float()
    else:
        raise ValueError("explicit pose-mean offsets are required even for an FK-disabled ablation")
    MotionRepresentation(offsets)  # fail before loading a large text backbone
    text, tokenizer = TextConditioner.from_pretrained(output_width=text_width,
                                                      local_files_only=not allow_download)
    return TemporalPlanSystem(config, text, offsets), tokenizer, fk


def predict_records(model, tokenizer, records, metadata: dict, *, seed: int = 0):
    """Whitelist text/name/language ONLY. Other input fields cannot affect outputs."""
    prepared, names = [], set()
    fps = metadata.get("fps_by_language", {})
    for record in records:
        row = {key: record[key] for key in ("name", "text", "lang")}
        if any(not isinstance(row[k], str) or not row[k].strip() for k in row):
            raise ValueError("name/text/language must be nonempty strings")
        if row["lang"] not in LANGUAGES or row["lang"] not in fps:
            raise ValueError("inference target language was not declared in the training checkpoint")
        key = (row["lang"], row["name"])
        if key in names:
            raise ValueError("duplicate prediction identifier")
        names.add(key)
        prepared.append(row)
    model.eval()
    device = next(model.parameters()).device
    max_text = metadata.get("training", {}).get("max_text_length", 128)
    for row in prepared:
        identity = f"{seed}\0{row['lang']}\0{row['name']}"
        sample_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")
        sample_seed %= 2**63
        generator = torch.Generator(device=device).manual_seed(sample_seed)
        ids, mask = tokenize_texts(tokenizer, [row["text"]], max_length=max_text, device=device)
        language = torch.tensor([LANGUAGES[row["lang"]]], device=device)
        result = model.generate(ids, mask, language, generator=generator)
        length = int(result.frame_lengths[0])
        k = int(result.plans.lengths[0])
        yield {
            **row, "fps": fps[row["lang"]], "frames": length,
            "raw133": result.raw133[0, :length].cpu().numpy(),
            "durations": result.plans.durations[0, :k].cpu().tolist(),
            "stop_reason": result.stop_reasons[0],
            "rotation_fallbacks": int(result.rotation_fallbacks[0]),
            "sample_seed": sample_seed,
        }


def export_predictions(predictions, output_dir: str | Path):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("prediction directory must be fresh to avoid mixing runs")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    identifiers = set()
    with tempfile.TemporaryDirectory(prefix=".predictions.", dir=output_dir.parent) as stage:
        stage = Path(stage)
        with (stage / "predictions.jsonl").open("w") as handle:
            for prediction in predictions:
                record = dict(prediction)
                array = np.asarray(record.pop("raw133"), dtype=np.float32)
                if (array.ndim != 2 or array.shape[1] != 133 or len(array) == 0 or
                        not np.isfinite(array).all()):
                    raise ValueError("prediction must be a finite nonempty [T,133] array")
                if record.get("frames", len(array)) != len(array):
                    raise ValueError("prediction frame metadata disagrees with array")
                record["frames"] = len(array)
                identity = (record["lang"], record["name"])
                if identity in identifiers:
                    raise ValueError("duplicate exported prediction identifier")
                identifiers.add(identity)
                filename = hashlib.sha256(
                    f"{record['lang']}\0{record['name']}".encode()
                ).hexdigest() + ".npy"
                np.save(stage / filename, array, allow_pickle=False)
                handle.write(json.dumps({**record, "motion": filename, "format": "raw133"}) + "\n")
                count += 1
        if output_dir.exists():
            output_dir.rmdir()  # refuses a concurrently populated directory
        os.replace(stage, output_dir)
    return count
