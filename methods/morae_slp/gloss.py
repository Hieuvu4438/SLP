"""Frozen gloss recognizer for the gloss-consistency loss (PLAN_V3 C3).

R maps a standardised latent sequence (the flow coordinates of a SignRAE run) to per-frame gloss
log-probabilities and is trained with CTC on TRAIN clips only. Because the flow applies it to
blurry x1 estimates, R is trained on the kind of input it will see: clean train latents and
x1-like corruptions of them (temporal smoothing, shrinkage towards the mean, noise; optionally
the x1 predictions of a trained flow). It is selected on a held-out part of TRAIN, never on dev.

    # train R (latent space and standardisation identical to the flow run that will use it)
    python -m methods.morae_slp.gloss train --data phoenix_smplx \\
        --ae outputs/morae_slp/pretrain/ae_union_jepa_d64/ae.pt --out outputs/morae_slp/gloss/phoenix
    # gate P0.3 (dev, analysis only): own sentence vs a length-matched swapped sentence
    python -m methods.morae_slp.gloss check --recognizer outputs/morae_slp/gloss/phoenix/recognizer.pt \\
        --data phoenix_smplx --ae <same ae.pt> --predictions <run>/predictions.npz --out gate.json

Gloss annotations exist for Phoenix-2014T and CSL-Daily (not How2Sign). Using them in training
puts a model in the "gloss-supervised" comparison class (PLAN_V3 §6).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers import Transformer, init_weights, sinusoidal_table

BLANK, UNK = 0, 1


class GlossVocab:
    """id 0 = CTC blank, 1 = <unk>, then the train glosses in sorted order."""

    def __init__(self, glosses: list[str]):
        self.tokens = ["<blank>", "<unk>"] + sorted(set(glosses) - {"<blank>", "<unk>"})
        self.index = {g: i for i, g in enumerate(self.tokens)}

    @classmethod
    def from_items(cls, items: list[dict]) -> GlossVocab:
        return cls([g for item in items for g in str(item.get("gloss") or "").split()])

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, gloss: str | None) -> Tensor:
        return torch.tensor([self.index.get(g, UNK) for g in str(gloss or "").split()],
                            dtype=torch.long)

    def decode(self, ids) -> list[str]:
        return [self.tokens[int(i)] for i in ids]


class GlossRecognizer(nn.Module):
    def __init__(self, latent_dim: int, vocab_size: int, width: int = 256, depth: int = 2,
                 heads: int = 4, max_frames: int = 1024):
        super().__init__()
        self.inp = nn.Linear(latent_dim, width)
        self.register_buffer("positions", sinusoidal_table(max_frames, width), persistent=False)
        self.transformer = Transformer(width, depth, heads, 4.0, rope=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)
        init_weights(self)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        """[B, T, d] standardised latents -> [B, T, V] log-probabilities (fp32)."""
        h = self.inp(x.float()) + self.positions[: x.shape[1]]
        h = self.norm(self.transformer(h, valid)[-1])
        return F.log_softmax(self.head(h).float(), dim=-1)


def ctc(log_probs: Tensor, lengths: Tensor, targets: list[Tensor]) -> Tensor:
    """Mean CTC loss over the samples with a non-empty target (0 if there is none)."""
    keep = [i for i, t in enumerate(targets) if len(t) > 0]
    if not keep:
        return log_probs.sum() * 0.0
    idx = torch.tensor(keep, device=log_probs.device)
    lp = log_probs.index_select(0, idx).transpose(0, 1)  # [T, B, V]
    tgt = [targets[i] for i in keep]
    with torch.autocast(log_probs.device.type, enabled=False):
        return F.ctc_loss(lp.float(), torch.cat(tgt).to(log_probs.device),
                          lengths.index_select(0, idx).cpu(), torch.tensor([len(t) for t in tgt]),
                          blank=BLANK, reduction="mean", zero_infinity=True)


def greedy(log_probs: Tensor, lengths: Tensor) -> list[list[int]]:
    best = log_probs.argmax(-1).cpu()
    out = []
    for row, n in zip(best, lengths.tolist()):
        ids, prev = [], BLANK
        for i in row[:n].tolist():
            if i != prev and i != BLANK:
                ids.append(i)
            prev = i
        out.append(ids)
    return out


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def word_error_rate(hyps: list[list[int]], refs: list[list[int]]) -> float:
    errors = sum(edit_distance(h, r) for h, r in zip(hyps, refs))
    return errors / max(1, sum(len(r) for r in refs))


def corrupt(z: Tensor, valid: Tensor, generator: torch.Generator) -> Tensor:
    """x1-like corruption of clean standardised latents [B, T, d]: temporal Gaussian smoothing
    (sigma ~ U[0, 3] frames), shrinkage towards the latent mean 0 (factor ~ U[0, 0.5]) and noise
    (std ~ U[0, 0.2]); one draw per clip."""
    b, t, d = z.shape
    sigma = torch.rand(b, generator=generator) * 3.0
    shrink = torch.rand(b, generator=generator) * 0.5
    noise = torch.rand(b, generator=generator) * 0.2
    out = []
    span = torch.arange(-9, 10, dtype=torch.float32)
    for i in range(b):
        x = z[i]
        if sigma[i] > 0.05:
            k = torch.exp(-0.5 * (span / sigma[i]) ** 2)
            k = (k / k.sum()).to(z.device)
            mask = valid[i].float()[:, None]                                   # [T, 1]
            kern = k.view(1, 1, -1)
            num = F.conv1d((x * mask).T.unsqueeze(1), kern, padding=9).squeeze(1).T  # [T, d]
            den = F.conv1d(mask.T.unsqueeze(0), kern, padding=9)[0, 0].unsqueeze(1)  # [T, 1]
            x = num / den.clamp_min(1e-6)
        x = (1 - shrink[i]) * x
        x = x + noise[i] * torch.randn(x.shape, generator=generator).to(z.device)
        out.append(x * valid[i].float()[:, None])
    return torch.stack(out)


# ---------------------------------------------------------------------------- persistence
def save_recognizer(path: Path, model: GlossRecognizer, vocab: GlossVocab, arch: dict,
                    mean: Tensor, std: Tensor, extra: dict) -> None:
    from .stages import atomic_save

    atomic_save({"version": 1, "arch": arch, "vocab": list(vocab.tokens),
                 "state": model.state_dict(), "latent_mean": mean.cpu(), "latent_std": std.cpu(),
                 **extra}, path)


def load_recognizer(path, device) -> tuple[GlossRecognizer, GlossVocab, Tensor, Tensor]:
    from .stages import resolve

    state = torch.load(resolve(path), map_location="cpu", weights_only=True)
    if state.get("version") != 1:
        raise ValueError("Unsupported gloss recognizer checkpoint")
    vocab = GlossVocab([])
    vocab.tokens = list(state["vocab"])
    vocab.index = {g: i for i, g in enumerate(vocab.tokens)}
    model = GlossRecognizer(**state["arch"])
    model.load_state_dict(state["state"])
    model.to(device).eval().requires_grad_(False)
    return model, vocab, state["latent_mean"], state["latent_std"]


# ---------------------------------------------------------------------------- CLI helpers
def _space_and_latents(args, split_name: str):
    """Latent space of ``--ae`` with the run's standardisation, and the split's standardised
    latents. Standardisation = train statistics (recalibrated like Method._cache_latents unless
    --no-recalibrate, in which case the calibration stored in ae.pt is used)."""
    from slp import config as config_lib
    from slp.data import load_dataset

    from .data import subset_indices
    from .stages import encode_split, latent_stats, load_space

    cfg = config_lib.compose("morae_slp", args.data, list(args.overrides))
    hp = cfg["method"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    space, _ = load_space(args.ae, device, need_calibration=not args.recalibrate)
    splits = load_dataset(cfg)
    train = splits.train
    if args.recalibrate:
        idx = subset_indices(len(train), hp["debug_limit"])
        raw = encode_split(space, train, idx, device, hp["workers"], standardised=False)
        mean, std = latent_stats(raw, idx, hp["latent"]["min_std"])
        space.latent_mean.copy_(mean)
        space.latent_std.copy_(std)
    split = train if split_name == "train" else splits.val  # never the test split
    idx = subset_indices(len(split), hp["debug_limit"])
    lat = encode_split(space, split, idx, device, hp["workers"], standardised=True)
    return cfg, space, split, [lat[i] for i in idx], [split.inputs[i] for i in idx], device


def _batches(latents: list[Tensor], order: list[int], size: int):
    from torch.nn.utils.rnn import pad_sequence

    for s in range(0, len(order), size):
        chunk = order[s:s + size]
        z = pad_sequence([latents[i] for i in chunk], batch_first=True)
        yield chunk, z, torch.tensor([len(latents[i]) for i in chunk])


def _evaluate(model, latents, targets, order, device, size=64) -> float:
    hyps, refs = [], []
    with torch.no_grad():
        for chunk, z, lengths in _batches(latents, order, size):
            valid = torch.arange(z.shape[1])[None] < lengths[:, None]
            lp = model(z.to(device), valid.to(device))
            hyps += greedy(lp, lengths)
            refs += [targets[i].tolist() for i in chunk]
    return word_error_rate(hyps, refs)


def train_main(args) -> None:
    from .retrieval import normalise

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, space, split, latents, items, device = _space_and_latents(args, "train")
    if not any(item.get("gloss") for item in items):
        sys.exit(f"{args.data} has no gloss annotations")
    vocab = GlossVocab.from_items(items)
    targets = [vocab.encode(item.get("gloss")) for item in items]
    # Hold out whole sentences (CSL repeats sentences), ~args.holdout of the unique sentences.
    rng = np.random.default_rng(args.seed)
    sentences = sorted({normalise(item["text"]) for item in items})
    held = set(rng.choice(sentences, max(1, int(len(sentences) * args.holdout)),
                          replace=False).tolist())
    # WER is only defined on clips with a reference gloss sequence.
    hold = [i for i, item in enumerate(items)
            if normalise(item["text"]) in held and len(targets[i]) > 0]
    held_set = set(hold)
    fit = [i for i in range(len(items)) if i not in held_set and len(targets[i]) > 0]
    arch = {"latent_dim": space.latent_dim, "vocab_size": len(vocab), "width": args.width,
            "depth": args.depth, "heads": args.heads}
    torch.manual_seed(args.seed)
    model = GlossRecognizer(**arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(args.seed)
    best, history = math.inf, []
    step = 0
    while step < args.steps:
        order = [fit[i] for i in torch.randperm(len(fit), generator=gen).tolist()]
        for chunk, z, lengths in _batches(latents, order, args.batch_size):
            model.train()
            valid = torch.arange(z.shape[1])[None] < lengths[:, None]
            if torch.rand((), generator=gen) < args.corrupt:
                z = corrupt(z, valid, gen)
            lp = model(z.to(device), valid.to(device))
            loss = ctc(lp, lengths.to(device), [targets[i] for i in chunk])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.eval_every == 0 or step == args.steps:
                model.eval()
                wer = _evaluate(model, latents, targets, hold, device)
                history.append({"step": step, "loss": float(loss), "holdout_wer": wer})
                print(json.dumps(history[-1]), flush=True)
                if wer < best:
                    best = wer
                    save_recognizer(out / "recognizer.pt", model, vocab, arch, space.latent_mean,
                                    space.latent_std, {"data": args.data, "ae": str(args.ae),
                                                       "holdout_wer": wer, "step": step})
            if step >= args.steps:
                break
    (out / "train_log.json").write_text(json.dumps({"best_holdout_wer": best,
                                                    "holdout_clips": len(hold),
                                                    "history": history}, indent=2))


def check_main(args) -> None:
    """Gate P0.3: R must read the gloss sequence from generated motion of the *own* sentence
    clearly better than from a length-matched prediction of another sentence."""
    from .data import POSE_DIM

    cfg, space, split, _, items, device = _space_and_latents(args, "val")
    model, vocab, mean, std = load_recognizer(args.recognizer, device)
    if not (torch.allclose(mean, space.latent_mean.cpu()) and torch.allclose(std, space.latent_std.cpu())):
        sys.exit("recognizer standardisation differs from this latent space")
    keys = [str(item["name"]).replace("/", "__") for item in items]
    with np.load(args.predictions) as z:
        preds = [np.asarray(z[k], np.float32) for k in keys]
    if any(p.ndim != 2 or p.shape[1] != POSE_DIM for p in preds):
        sys.exit("predictions must be [T, 133] arrays")
    lat = []
    with torch.no_grad():
        for p in preds:
            x = torch.from_numpy(p)[None].to(device)
            lat.append(space.encode_flow(x, torch.tensor([len(p)], device=device))[0].cpu())
    targets = [vocab.encode(item.get("gloss")) for item in items]
    order = [i for i, t in enumerate(targets) if len(t) > 0]
    lengths = np.array([len(p) for p in preds])
    swap = []
    for i in order:  # closest length, different sentence
        cand = [j for j in order if items[j]["text"] != items[i]["text"]]
        swap.append(min(cand, key=lambda j: (abs(lengths[j] - lengths[i]), j)))
    own = _evaluate(model, lat, targets, order, device)
    swapped_lat = [lat[j] for j in swap]
    swapped = _evaluate(model, swapped_lat, [targets[i] for i in order], list(range(len(order))),
                        device)
    result = {"clips": len(order), "wer_own": own, "wer_swapped": swapped,
              "gap": swapped - own, "predictions": str(args.predictions)}
    print(json.dumps(result))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("train", "check"):
        q = sub.add_parser(name)
        q.add_argument("--data", required=True)
        q.add_argument("--ae", required=True)
        q.add_argument("--device", default=None)
        q.add_argument("--no-recalibrate", dest="recalibrate", action="store_false",
                       help="use the calibration stored in ae.pt (runs with recalibrate=false)")
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("overrides", nargs="*", help="config overrides key.sub=value")
    t = sub.choices["train"]
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=8000)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--width", type=int, default=256)
    t.add_argument("--depth", type=int, default=2)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--corrupt", type=float, default=0.67, help="fraction of corrupted batches")
    t.add_argument("--holdout", type=float, default=0.05, help="held-out share of sentences")
    t.add_argument("--eval-every", type=int, default=1000)
    c = sub.choices["check"]
    c.add_argument("--recognizer", required=True)
    c.add_argument("--predictions", required=True, help="predictions.npz of dev clips")
    c.add_argument("--out", default=None)
    args = p.parse_args(argv)
    (train_main if args.cmd == "train" else check_main)(args)


if __name__ == "__main__":
    main()
