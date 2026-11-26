"""Train-only pose-to-text teacher and a frozen, differentiable semantic loss."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from slp.paths import PROJECT_ROOT
from slp.utils.torch_training import train_stage, validation

from .data import POSE_DIM, text_input

WORD = re.compile(r"\d+(?:[.,:]\d+)*|[^\W\d_]+(?:[-’'][^\W\d_]+)*", re.UNICODE)
PRIORITY = frozenset("""
kein keine keinen keiner keines keinem nicht weder nie niemals ohne
montag dienstag mittwoch donnerstag freitag samstag sonntag sonnabend
norden süden osten westen nord süd ost west nordwest nordost südwest südost
links rechts oben unten
""".split())


def terms(text: str) -> list[str]:
    return WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def priority(word: str) -> bool:
    return word[0].isdigit() or word in PRIORITY


def canonical(text: str) -> str:
    return " ".join(terms(text))


@torch.no_grad()
def frozen_text_vectors(encoder, sentences: list[str], words: list[str], device, batch_size):
    """Read the frozen contextual sentence vectors and lexical mBART embedding table."""
    def chunks(rows):
        return (rows[i:i + batch_size] for i in range(0, len(rows), batch_size))

    contextual = []
    for rows in chunks(sentences):
        token, mask = encoder(rows, torch.zeros(len(rows), device=device, dtype=torch.long))
        contextual.append(((token.float() * mask[..., None]).sum(1) /
                           mask.sum(1, keepdim=True)).cpu())

    lexical = [torch.zeros(1, encoder.output_dim)]
    for rows in chunks(words):
        if encoder.backend == "bytes":
            groups = [list(word.encode("utf-8")) for word in rows]
            embedding = encoder.encoder
        else:
            tokenizer = encoder.tokenizer
            raw = tokenizer(list(rows), add_special_tokens=False).input_ids
            groups = [[encoder.id_map.get(i, encoder.unk_id) if encoder.id_map is not None else i
                       for i in row] or [encoder.unk_id] for row in raw]
            embedding = encoder.encoder.embed_tokens
        sizes = torch.tensor([len(row) for row in groups], device=device)
        ids = torch.tensor([i for row in groups for i in row], device=device)
        values = embedding(ids).float()
        packed = values.new_zeros(len(groups), values.shape[-1])
        packed.index_add_(0, torch.arange(len(groups), device=device).repeat_interleave(
            sizes, output_size=len(ids)), values)
        lexical.append((packed / sizes[:, None]).cpu())
    return torch.cat(contextual), torch.cat(lexical)


def reviewed_negatives(path: str | None, by_text: dict[str, int], train_ids: set[int]):
    reviewed = {}
    if path is not None:
        rows = json.loads((PROJECT_ROOT / Path(path).expanduser()).read_text())
        if not isinstance(rows, list):
            raise ValueError("Reviewed train negatives must be a JSON list")
        for row in rows:
            if (not isinstance(row, dict) or row.get("reviewed") is not True
                    or row.get("kind") not in ("weekday", "number", "direction", "negation")
                    or not isinstance(row.get("a"), str) or not isinstance(row.get("b"), str)):
                raise ValueError("Reviewed train negatives need a, b, kind, and reviewed=true")
            a, b = (by_text.get(canonical(row[k]), -1) for k in ("a", "b"))
            if a == b or not {a, b} <= train_ids:
                raise ValueError("Both sides of a reviewed negative must be distinct teacher-train texts")
            reviewed.setdefault(a, set()).add(b)
            reviewed.setdefault(b, set()).add(a)
    return reviewed


@torch.no_grad()
def mine_negatives(sentence: Tensor, median: Tensor, classes: list[int], cap: int,
                   reviewed: dict[int, set[int]]) -> Tensor:
    """Use semantic proximity and paired frame lengths, with a fixed deduplicated bank."""
    ids = torch.tensor(classes)
    if len(ids) < 2:
        raise ValueError("Teacher fitting and monitoring each require two distinct captions")
    size = min(cap, len(ids) - 1)
    result = torch.full((len(sentence), cap), -1, dtype=torch.long)
    features = F.normalize(sentence[ids], dim=-1)
    duration = median[ids].log()
    for start in range(0, len(ids), 192):
        end = min(start + 192, len(ids))
        difference = (duration[start:end, None] - duration[None, :]).abs()
        distance = features[start:end] @ features.T - 1.5 * difference
        distance[torch.arange(end - start), torch.arange(start, end)] = -torch.inf
        for row, anchor in enumerate(ids[start:end].tolist()):
            preferred = sorted(reviewed.get(anchor, ()))
            if len(preferred) > size:
                raise ValueError("Increase semantic.negatives to fit reviewed negatives")
            # Stable ties keep the same bank for the sentence and word controls.
            ranked = ids[distance[row].argsort(descending=True, stable=True)[:size + len(preferred)]]
            excluded = set(preferred) | {anchor}
            rest = [i for i in ranked.tolist() if i not in excluded]
            result[anchor, :size] = torch.tensor((preferred + rest)[:size])
    return result


@dataclass
class SemanticBank:
    sentence: Tensor
    lexical: Tensor
    word_ids: Tensor
    weights: Tensor
    clip_class: Tensor
    duration_proxy: Tensor
    fit_negatives: Tensor
    monitor_negatives: Tensor

    def to(self, device):
        return replace(self, sentence=self.sentence.to(device), lexical=self.lexical.to(device),
                       word_ids=self.word_ids.to(device), weights=self.weights.to(device),
                       duration_proxy=self.duration_proxy.to(device))

    def candidates(self, indices: Tensor, monitor: bool = False):
        anchor = self.clip_class[indices]
        negatives = (self.monitor_negatives if monitor else self.fit_negatives)[anchor]
        result = torch.cat((anchor[:, None], negatives), dim=1)
        return result.to(self.sentence.device)


def build_bank(data, kept, held, encoder, device, config):
    texts = []
    lengths = []
    for index in data.indices:
        text, language = text_input(data.split.inputs[index])
        if language != 0:
            raise ValueError("The semantic teacher uses a German lexical vocabulary")
        texts.append(text)
        lengths.append(len(data.split.targets[index]))
    keys = [canonical(text) for text in texts]
    if not all(keys):
        raise ValueError("Semantic training requires at least one lexical token per caption")
    by_text = {key: i for i, key in enumerate(sorted(set(keys)))}
    clip_class = torch.tensor([by_text[key] for key in keys])
    # The canonical caption is the same for all recordings in its positive class.
    captions = sorted(by_text)
    tokens = [list(dict.fromkeys(terms(text))) for text in captions]
    train_ids = {int(clip_class[i]) for i in kept}
    frequencies = Counter(word for i in train_ids for word in tokens[i])
    vocabulary = sorted(frequencies)
    index = {word: i for i, word in enumerate(vocabulary, 1)}
    weights = {word: 1. + math.log((1 + len(train_ids)) / (1 + df))
               for word, df in frequencies.items()}
    selected = []
    for row in tokens:
        available = [word for word in row if word in index]
        informative = [word for word in available
                       if priority(word) or frequencies[word] / len(train_ids) <= config["max_df"]]
        selected.append(sorted(informative or available, key=lambda word: (
            not priority(word), -weights[word], word
        ))[:config["max_words"]])
    word_ids = torch.zeros((len(captions), config["max_words"]), dtype=torch.long)
    reliability = torch.zeros_like(word_ids, dtype=torch.float32)
    for i, row in enumerate(selected):
        word_ids[i, :len(row)] = torch.tensor([index[word] for word in row], dtype=torch.long)
        reliability[i, :len(row)] = torch.tensor([min(4., weights[word]) for word in row])
    if not bool(word_ids[list(train_ids)].any(dim=1).all()):
        raise ValueError("A caption in teacher training has no usable lexical target")

    sentence, lexical = frozen_text_vectors(encoder, captions, vocabulary, device,
                                            config["text_batch"])
    # Train and monitor mining are independent even when they share a repeated caption.
    def lengths_by_class(rows):
        lengths_by_class = {}
        for i in rows:
            lengths_by_class.setdefault(int(clip_class[i]), []).append(lengths[i])
        med = torch.ones(len(captions))
        for i, values in lengths_by_class.items():
            med[i] = torch.tensor(values, dtype=torch.float32).median()
        return med, sorted(lengths_by_class)

    fit_median, fit_classes = lengths_by_class(kept)
    monitor_median, monitor_classes = lengths_by_class(held)
    reviewed_path = config["reviewed_negatives"]
    reviewed = reviewed_negatives(reviewed_path, by_text, train_ids)
    hard_fit = mine_negatives(sentence, fit_median, fit_classes, config["negatives"], reviewed)
    hard_monitor = mine_negatives(sentence, monitor_median, monitor_classes, config["negatives"], {})
    # The duration-only diagnostic needs estimates from text. A monitor class's own
    # GT length would give the length baseline the exact answer for singletons.
    features = F.normalize(sentence, dim=-1).double()
    features = features - features[fit_classes].mean(0)
    design = features[fit_classes]
    target = fit_median[fit_classes].double().log()
    gram = design.T @ design
    gram.diagonal().add_(1.0)
    duration_proxy = (target.mean() + features @ torch.linalg.solve(
        gram, design.T @ (target - target.mean()),
    )).float()
    provenance = {
        "lexicon_sha256": hashlib.sha256(json.dumps(
            [(word, frequencies[word]) for word in vocabulary], ensure_ascii=False
        ).encode()).hexdigest(),
        "hard_fit_sha256": hashlib.sha256(hard_fit.numpy().tobytes()).hexdigest(),
        "hard_monitor_sha256": hashlib.sha256(hard_monitor.numpy().tobytes()).hexdigest(),
        "reviewed_pairs": sum(map(len, reviewed.values())) // 2,
        "fit_texts": len(fit_classes),
        "monitor_texts": len(monitor_classes),
        "monitor_unseen_texts": len(set(monitor_classes) - train_ids),
    }
    return SemanticBank(sentence, lexical, word_ids, reliability, clip_class,
                        duration_proxy, hard_fit, hard_monitor), provenance


def masked_soft_best(values: Tensor, valid: Tensor, dim: int, temperature: float) -> Tensor:
    """Smooth max with count correction so padding and candidate length do not inflate logits."""
    values = values.masked_fill(~valid, -1e4)
    return temperature * (torch.logsumexp(values / temperature, dim)
                          - valid.sum(dim).clamp_min(1).log())


class PoseTextTeacher(nn.Module):
    """The pose tower has no access to text, time from CFM, or frame-count embeddings."""

    def __init__(self, input_dim: int, config: dict):
        super().__init__()
        self.mode = config["mode"]
        self.window = config["window"]
        self.temperature = config["temperature"]
        self.alignment_temperature = config["alignment_temperature"]
        self.null_similarity = config["null_similarity"]
        width = config["width"]
        self.pose_in = nn.Linear(POSE_DIM, width)
        self.refine = nn.Conv1d(width, width, 3, padding=1)
        self.pose_out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.text_out = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))

    def pose(self, poses: Tensor, lengths: Tensor):
        frames = torch.arange(poses.shape[1], device=poses.device)
        valid = frames[None, :] < lengths[:, None]
        h = F.gelu(self.pose_in(poses)).masked_fill(~valid[..., None], 0)
        h = self.pose_out(h + F.gelu(self.refine(h.transpose(1, 2)).transpose(1, 2)))
        h = h.masked_fill(~valid[..., None], 0)
        extra = (-len(frames)) % self.window
        windows = F.pad(h, (0, 0, 0, extra)).unflatten(1, (-1, self.window)).sum(2)
        counts = F.pad(valid, (0, extra)).unflatten(1, (-1, self.window)).sum(2)
        return F.normalize(windows / counts.clamp_min(1)[..., None], dim=-1), counts

    def prepare_text(self, values):
        return F.normalize(self.text_out(values), dim=-1)

    def scores(self, windows, counts, keys, bank, *, frozen=False):
        safe = keys.clamp_min(0)
        if self.mode == "sentence":
            pooled = F.normalize((windows * counts[..., None]).sum(1), dim=-1)
            targets = bank.sentence[safe]
            targets = targets if frozen else self.prepare_text(targets)
            similarities = torch.einsum("bd,bkd->bk", pooled, targets)
            null = similarities.new_zeros(len(windows))
        else:
            ids = bank.word_ids[safe]
            valid = ids != 0
            weights = bank.weights[safe]
            targets = bank.lexical[ids]
            targets = targets if frozen else self.prepare_text(targets)
            cosines = torch.einsum("bwd,bkld->bkwl", windows, targets)
            tau = self.alignment_temperature
            word_scores = masked_soft_best(cosines, counts[:, None, :, None] > 0, 2, tau)
            word_to_motion = (word_scores * weights).sum(-1) / weights.sum(-1).clamp_min(1)
            window_scores = masked_soft_best(cosines, valid[:, :, None, :], 3, tau)
            # A transition can align to null. The fixed null floor cannot absorb the word recall
            # term, and the soft maximum remains differentiable near the transition threshold.
            excess = (window_scores - self.null_similarity) / tau
            motion_to_word = (F.softplus(excess) * tau * counts[:, None]).sum(-1)
            motion_to_word = motion_to_word / counts.sum(-1, keepdim=True)
            similarities = 0.5 * (word_to_motion + motion_to_word)
            null = ((-excess[:, 0]).sigmoid() * counts).sum(1) / counts.sum(1)
        return similarities, null

    def objective(self, poses, lengths, keys, bank, *, frozen=False, active=None):
        windows, counts = self.pose(poses, lengths)
        similarities, null = self.scores(windows, counts, keys, bank, frozen=frozen)
        valid = self.valid_candidates(keys, bank)
        row_valid = valid[:, 0] & valid[:, 1:].any(-1)
        if active is not None:
            row_valid = row_valid & active
        logits = (similarities / self.temperature).masked_fill(~valid, -1e4)
        row = logits.logsumexp(-1) - logits[:, 0]
        # A clip's caption may have multiple positive recordings within the same batch.
        if not frozen:
            same = keys[:, :1] == keys[:, 0]
            reverse, _ = self.scores(windows, counts, keys[:, 0].expand(len(keys), -1), bank)
            reverse = reverse.T / self.temperature
            reverse_candidates = valid[:, 0].expand_as(same)
            reverse = reverse.masked_fill(~reverse_candidates, -1e4)
            reverse_loss = reverse.logsumexp(-1) - reverse.masked_fill(
                ~(reverse_candidates & same), -1e4
            ).logsumexp(-1)
            # The per-row hard bank provides negatives even for one eligible clip.
            has_reverse_negatives = (reverse_candidates & ~same).any(-1)
            row = torch.where(has_reverse_negatives, (row + reverse_loss) * 0.5, row)
        accuracies = logits.argmax(-1) == 0
        n = row_valid.sum().clamp_min(1)
        loss_sum = (row * row_valid).sum()
        return (loss_sum / (len(poses) if frozen else n),
                (accuracies & row_valid).sum() / n, (null * row_valid).sum() / n, row_valid)

    @staticmethod
    def valid_candidates(keys, bank):
        return (keys >= 0) & bank.word_ids[keys.clamp_min(0)].ne(0).any(-1)

    @torch.no_grad()
    def freeze_bank(self, bank):
        sentence = torch.cat([self.prepare_text(chunk) for chunk in bank.sentence.split(256)])
        lexical = torch.cat([self.prepare_text(chunk) for chunk in bank.lexical.split(256)])
        return replace(bank, sentence=sentence, lexical=lexical)


def fit_teacher(method, data, kept, held, train_loader, monitor_loader):
    cfg = method.semantic_cfg
    text = method.text if not method.text_cfg["finetune"] else None
    if text is None:
        raise ValueError("Use the separate encoder fine-tuning control without the semantic loss")
    text.eval()
    bank, metadata = build_bank(data, kept, held, text, method.device, cfg)
    bank = bank.to(method.device)
    teacher = PoseTextTeacher(text.output_dim, cfg).to(method.device)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=cfg["lr"], weight_decay=0.01)
    best = {}

    def criterion(batch, monitor=False):
        lengths = batch.lengths
        poses = batch.poses
        rep = method.representation
        with torch.autocast(method.device.type, enabled=False):
            coarse = rep.decode(rep.projector(lengths, poses.shape[1]).coarse(rep.encode(poses)))
            keys = bank.candidates(batch.indices, monitor)
            loss, recall, null, eligible = teacher.objective(coarse, lengths, keys, bank)
            distance = (lengths[:, None].float().log() -
                        bank.duration_proxy[keys.clamp_min(0)]).abs()
            distance = distance.masked_fill(~teacher.valid_candidates(keys, bank), torch.inf)
            # Tied candidates count fractionally, including common same-length captions.
            ties = distance == distance.amin(-1, keepdim=True)
            length_baseline = ties[:, 0] / ties.sum(-1)
            eligible_rate = eligible.float().mean()
            return {"loss": loss, "retrieval_top1": recall * eligible_rate,
                    "null_rate": null * eligible_rate, "eligible_rate": eligible_rate,
                    "length_only_top1": (length_baseline * eligible).mean()}

    def save_best():
        best["state_dict"] = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}

    train_stage("semantic_teacher", train_loader, monitor_loader, cfg, optimizer, (teacher,),
                criterion, lambda: lambda batch: criterion(batch, monitor=True), save_best,
                method.device, "fp32", method.ctx)
    teacher.load_state_dict(best["state_dict"])
    teacher.eval().requires_grad_(False)
    metadata["monitor"] = validation(monitor_loader, lambda batch: criterion(batch, monitor=True),
                                     method.device, "fp32")
    metrics = metadata["monitor"]
    if metrics["eligible_rate"] <= 0:
        raise ValueError("No eligible train-monitor captions for the semantic teacher")
    for key in ("retrieval_top1", "null_rate", "length_only_top1"):
        metrics[key] /= metrics["eligible_rate"]
    gap = metrics["retrieval_top1"] - metrics["length_only_top1"]
    if gap < cfg["min_retrieval_gain"]:
        raise ValueError(f"Teacher retrieval gain over length only is {gap:.3f}; "
                         f"required {cfg['min_retrieval_gain']:.3f}")
    return teacher, teacher.freeze_bank(bank), metadata
