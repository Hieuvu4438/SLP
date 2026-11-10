"""Explicit cached contracts. Inference inputs have no ground-truth fields."""

from dataclasses import dataclass, fields
from pathlib import Path
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from .schema import manual_to_6d

TRAIN_ROLES = {
    "decoder_train",
    "decoder_val",
    "selector_train",
    "selector_val",
    "selector_calibration",
    "robustness_train",
}


def validate_official_split(role, split):
    allowed = {"train", "dev"} if role in {"decoder_val", "selector_val"} else {"train"}
    if split not in allowed:
        raise ValueError(f"{role} requires official_split in {sorted(allowed)}; test is forbidden")


def base_signature(record):
    metadata = record.get("provenance", {})
    return {
        key: metadata[key]
        for key in (
            "tokenizer_sha256",
            "generator_sha256",
            "mean_sha256",
            "std_sha256",
            "mapping_sha256",
            "upstream_code_sha256",
            "effective_tokenizer_source",
            "text_encoder_mode",
        )
        if key in metadata
    }


def dataset_signature(dataset):
    expected = None
    for i in range(len(dataset)):
        actual = base_signature(dataset[i])
        if expected is not None and expected != actual:
            raise ValueError("Mixed base assets in one dataset")
        expected = actual
    return expected


@dataclass
class InferenceBatch:
    codes: torch.Tensor  # B,K,3,C; frozen codebook embeddings
    token_lengths: torch.Tensor  # common synchronized token length
    text: torch.Tensor  # B,L,D; frozen pretrained encoder states
    text_mask: torch.Tensor
    language: torch.Tensor
    baseline: torch.Tensor  # B,T,133, physical radians; jaw/expression preserved
    lengths: torch.Tensor  # base-generated native length, NOT reference length
    sample_ids: tuple[str, ...]
    base_signature: dict | None = None

    def to(self, device):
        return InferenceBatch(
            **{
                f.name: (
                    getattr(self, f.name).to(device)
                    if isinstance(getattr(self, f.name), torch.Tensor)
                    else getattr(self, f.name)
                )
                for f in fields(self)
            }
        )

    def validate(self, cfg):
        b, t, d = self.baseline.shape
        if (
            d != 133
            or b < 1
            or self.codes.shape[0] != b
            or self.codes.shape[2:] != (3, cfg.code_dim)
        ):
            raise ValueError("Invalid cached input schema")
        if self.text.shape[:2] != self.text_mask.shape or self.text.shape[-1] != cfg.text_dim:
            raise ValueError("Invalid text features")
        if self.text.shape[0] != b or len(self.sample_ids) != b or len(set(self.sample_ids)) != b:
            raise ValueError("Sample ID/batch mismatch")
        if (
            self.lengths.shape != (b,)
            or self.token_lengths.shape != (b,)
            or self.language.shape != (b,)
        ):
            raise ValueError("Length/language vectors must have shape [B]")
        if (self.lengths <= 0).any() or (self.lengths > t).any():
            raise ValueError("Invalid native lengths")
        if (self.token_lengths <= 0).any() or (self.token_lengths > self.codes.shape[1]).any():
            raise ValueError("Invalid token lengths")
        if not torch.equal(self.lengths, self.token_lengths * cfg.factor):
            raise ValueError("Native timeline must be exactly token length * tokenizer factor")
        if (self.language < 0).any() or (self.language >= cfg.languages).any():
            raise ValueError("Invalid sign-language ID")
        if self.text_mask.dtype != torch.bool:
            raise ValueError("text_mask must be boolean")
        if any(x.dtype != torch.long for x in (self.lengths, self.token_lengths, self.language)):
            raise ValueError("Lengths and language IDs must be int64 tensors")
        for x in (self.codes, self.text, self.baseline):
            if not x.is_floating_point() or not torch.isfinite(x).all():
                raise ValueError("Nonfinite input")
        return self


def collate_inputs(records):
    signature = base_signature(records[0])
    if any(base_signature(r) != signature for r in records):
        raise ValueError("Mixed base assets in an input batch")
    return InferenceBatch(
        pad_sequence([r["codes"] for r in records], batch_first=True),
        torch.tensor([len(r["codes"]) for r in records], dtype=torch.long),
        pad_sequence([r["text"] for r in records], batch_first=True),
        pad_sequence(
            [torch.ones(len(r["text"]), dtype=torch.bool) for r in records], batch_first=True
        ),
        torch.tensor([r["language"] for r in records], dtype=torch.long),
        pad_sequence([r["baseline"] for r in records], batch_first=True),
        torch.tensor([len(r["baseline"]) for r in records], dtype=torch.long),
        tuple(r["sample_id"] for r in records),
        signature,
    )


def collate_training(records):
    batch = collate_inputs(records)
    if any(r["target"].shape != r["baseline"].shape for r in records):
        raise ValueError("Training targets need explicit correspondence to the native timeline")
    out = {f.name: getattr(batch, f.name) for f in fields(batch)}
    target = pad_sequence([r["target"] for r in records], batch_first=True)
    out["target_manual"] = manual_to_6d(target)
    out["target_pose"] = target
    out["offsets"] = torch.zeros_like(batch.lengths)
    out["loss_mask"] = torch.arange(target.shape[1])[None] < batch.lengths[:, None]
    return out


class RecordDataset(Dataset):
    """JSONL rows: path, sample_id, source_group, role; record is tensor-only safe .pt."""

    def __init__(self, manifest, role):
        if role not in TRAIN_ROLES:
            raise ValueError("Training dataset role must be explicit; test/inference is prohibited")
        self.manifest = Path(manifest).resolve()
        self.role = role
        self.rows = [
            json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()
        ]
        if not self.rows or any(r.get("role") != role for r in self.rows):
            raise ValueError(f"Manifest must contain only {role} records")
        for row in self.rows:
            validate_official_split(role, row.get("official_split"))
        ids = [r["sample_id"] for r in self.rows]
        if len(set(ids)) != len(ids) or any(not r.get("source_group") for r in self.rows):
            raise ValueError("Unique sample IDs and original-video source_group required")

    @property
    def groups(self):
        return {r["source_group"] for r in self.rows}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        p = (self.manifest.parent / row["path"]).resolve()
        record = torch.load(p, map_location="cpu", weights_only=True)
        if (
            record.get("sample_id") != row["sample_id"]
            or record.get("source_group") != row["source_group"]
        ):
            raise ValueError("Record/manifest identity mismatch")
        if record.get("role") != self.role or not record.get("provenance"):
            raise ValueError("Missing role or upstream provenance")
        if record.get("official_split") != row["official_split"]:
            raise ValueError("Record/manifest official split mismatch")
        if (
            self.role.startswith("selector_")
            and record["provenance"].get("preparation") != "generated"
        ):
            raise ValueError("Selector cache must declare generated, not teacher/oracle, tokens")
        for key in ("codes", "text", "baseline", "target"):
            x = record[key]
            if not isinstance(x, torch.Tensor) or not torch.isfinite(x).all():
                raise ValueError(f"Invalid record tensor {key}")
        if record["target"].shape != record["baseline"].shape:
            raise ValueError(
                "Reference must be aligned explicitly at preparation, never at inference"
            )
        return record


def require_disjoint(*datasets):
    for i, a in enumerate(datasets):
        for b in datasets[i + 1 :]:
            overlap = a.groups & b.groups
            if overlap:
                raise ValueError(
                    f"Source-video leakage between {a.role} and {b.role}: {sorted(overlap)[:3]}"
                )


def window_batch(batch, window, halo, factor, generator=None):
    """Random supervised windows with temporal halo and global token/frame correspondence."""
    if window < 1 or halo < 0:
        raise ValueError("Invalid window/halo")
    result = dict(batch)
    positions, crops, masks, lengths = [], [], [], []
    for i, length in enumerate(batch["lengths"].tolist()):
        center = int(torch.randint(max(1, length - window + 1), (1,), generator=generator))
        start, end = max(0, center - halo), min(length, center + window + halo)
        positions.append(start)
        crops.append((start, end))
        mask = torch.arange(start, end, device=batch["lengths"].device)
        masks.append((mask >= center) & (mask < min(center + window, length)))
        lengths.append(end - start)
    for key in ("baseline", "target_manual", "target_pose", "relations"):
        if key in batch:
            result[key] = pad_sequence(
                [batch[key][i, a:b] for i, (a, b) in enumerate(crops)], batch_first=True
            )
    result["lengths"] = batch["lengths"].new_tensor(lengths)
    result["offsets"] = batch["lengths"].new_tensor(positions)
    result["loss_mask"] = pad_sequence(masks, batch_first=True)
    return result
