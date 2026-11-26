"""Fixed train-only coarse prototypes for the optional retrieval track."""

from __future__ import annotations

import hashlib

import torch
from torch import Tensor
from torch.nn import functional as F

from .data import broadcast_id, text_input


@torch.no_grad()
def text_vector(encoder, texts, languages):
    states, mask = encoder(texts, languages)
    values = (states.float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
    return F.normalize(values, dim=-1)


@torch.no_grad()
def fit_index(method, data):
    inputs = [data.split.inputs[index] for index in data.indices]
    keys = [text_input(item) for item in inputs]
    rows = list(dict.fromkeys(keys))
    inverse = {key: i for i, key in enumerate(rows)}
    group_ids = {}
    groups = []
    for i, item in enumerate(inputs):
        group = broadcast_id(item, i)
        groups.append(group_ids.setdefault(group, len(group_ids)))
    if len(group_ids) < 2:
        raise ValueError("Retrieval needs more than one train broadcast")
    vectors = []
    encoder = method.retrieval_text
    encoder.eval()
    bsz = method.semantic_cfg["text_batch"]
    for start in range(0, len(rows), bsz):
        batch = rows[start:start + bsz]
        lang = torch.tensor([key[1] for key in batch], device=method.device)
        vectors.append(text_vector(encoder, [key[0] for key in batch], lang))
    prototypes, durations = [], []
    rep = method.representation
    anchors = method.retrieval_cfg["anchors"]
    for packed in method._loader(data, False, calibrating=True):
        for batch in packed:
            batch = batch.to(method.device)
            operator = rep.projector(batch.lengths, batch.poses.shape[1])
            coarse = operator.coarse(rep.encode(batch.poses)) / rep.scales[0]
            for i, length, pose in zip(batch.indices.tolist(), batch.lengths.tolist(), coarse,
                                       strict=True):
                proto = F.interpolate(pose[:length].T[None], size=anchors, mode="linear",
                                      align_corners=True)[0].T.cpu()
                prototypes.append((i, proto))
                durations.append((i, length))
    prototypes.sort(key=lambda row: row[0])
    durations.sort()
    return {
        "vectors": torch.cat(vectors),
        "classes": torch.tensor([inverse[key] for key in keys], device=method.device),
        "groups": torch.tensor(groups, device=method.device),
        "languages": torch.tensor([key[1] for key in keys], device=method.device),
        "durations": torch.tensor([length for _, length in durations], device=method.device),
        "prototypes": torch.stack([pose for _, pose in prototypes]).to(method.device),
    }


@torch.no_grad()
def neighbors(index, queries, languages, captions, mode, seed, train_rows=None):
    score = (queries @ index["vectors"].T)[:, index["classes"]]
    available = languages[:, None] == index["languages"]
    if train_rows is not None:
        train_rows = train_rows.to(queries.device)
        available = available & (index["groups"][train_rows, None] != index["groups"])
    if not bool(available.any(-1).all()):
        raise ValueError("A retrieval query has no eligible training broadcast in its language")
    best = score.masked_fill(~available, -torch.inf).argmax(-1)
    if mode == "nearest":
        return best
    # Pick a different train clip matched as closely as possible to the retrieved duration.
    # No ground truth duration of the query enters this control.
    distance = (index["durations"][best, None] - index["durations"]).abs()
    other = available.scatter(1, best[:, None], False)
    distance = distance.masked_fill(~other, torch.iinfo(distance.dtype).max)
    nearest = other & (distance == distance.amin(-1, keepdim=True))
    choices = []
    for pick, possible, language, text in zip(best, nearest, languages.tolist(), captions,
                                               strict=True):
        ids = possible.nonzero().flatten()
        digest = hashlib.sha256(f"{seed}/{language}/{text}".encode()).digest()
        choices.append(ids[int.from_bytes(digest[:8], "little") % len(ids)] if len(ids) else pick)
    return torch.stack(choices)


@torch.no_grad()
def condition_retrieval(method, texts, langs, lengths, size, indices=None) -> Tensor:
    index = method.retrieval_index
    with torch.autocast(method.device.type, enabled=False):
        if indices is not None:
            query = index["vectors"][index["classes"][indices.to(method.device)]]
        else:
            query = text_vector(method.retrieval_text, texts, langs)
        retrieved = neighbors(index, query, langs, texts, method.retrieval_cfg["mode"],
                              method.ctx.seed, indices)
        rows = []
        for prototype, length in zip(index["prototypes"][retrieved], lengths.tolist(), strict=True):
            aligned = F.interpolate(prototype.T[None], size=length, mode="linear",
                                    align_corners=True)[0].T
            rows.append(F.pad(aligned, (0, 0, 0, size - length)))
        return torch.stack(rows)
