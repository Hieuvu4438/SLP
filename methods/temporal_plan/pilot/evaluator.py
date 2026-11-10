"""Pilot-only independent pose-to-German evaluator. Not a published PHX BT score."""

from __future__ import annotations
from collections import Counter
import math
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from ..tensors import valid_mask, zero_padding


class PoseTranslator(nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.lm = language_model.requires_grad_(False).eval()
        width = language_model.config.d_model
        self.conv1 = nn.Conv1d(256, 256, 5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(256, width, 5, stride=2, padding=2)
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 8, width * 2, dropout=0.1, batch_first=True),
            2,
            enable_nested_tensor=False,
        )
        self.cross_names = []
        for name, parameter in self.lm.named_parameters():
            if name.startswith("decoder.") and ("EncDecAttention" in name or "layer_norm" in name):
                parameter.requires_grad_(True)
                self.cross_names.append(name)

    def train(self, mode=True):
        super().train(mode)
        self.lm.eval()  # frozen LM dropout must stay off
        return self

    def encode(self, features, lengths):
        features = zero_padding(features, lengths)
        x = self.conv1(features.transpose(1, 2)).transpose(1, 2)
        x = torch.nn.functional.gelu(x)
        lengths = (lengths + 1) // 2
        x = zero_padding(x, lengths)
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = torch.nn.functional.gelu(x)
        lengths = (lengths + 1) // 2
        valid = valid_mask(lengths, x.shape[1])
        # Sinusoidal position is explicit; the decoder receives motion, no source text.
        pos = torch.arange(x.shape[1], device=x.device)[:, None]
        freq = torch.exp(
            torch.arange(0, x.shape[-1], 2, device=x.device) * (-math.log(10000.0) / x.shape[-1])
        )
        pe = torch.zeros(x.shape[1], x.shape[-1], device=x.device)
        pe[:, 0::2], pe[:, 1::2] = (pos * freq).sin(), (pos * freq).cos()
        x = self.temporal(x + pe[None], src_key_padding_mask=~valid)
        return BaseModelOutput(last_hidden_state=x), valid.long()

    def forward(self, features, lengths, labels):
        memory, mask = self.encode(features, lengths)
        return self.lm(
            encoder_outputs=memory, attention_mask=mask, labels=labels, use_cache=False
        ).loss

    @torch.no_grad()
    def translate(self, features, lengths, tokenizer):
        memory, mask = self.encode(features, lengths)
        tokens = self.lm.generate(
            encoder_outputs=memory,
            attention_mask=mask,
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
        )
        return tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def compact_state(self):
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {n: t.detach().cpu() for n, t in self.state_dict().items() if n in names}

    def restore_compact(self, state):
        expected = set(self.compact_state())
        if set(state) != expected:
            raise ValueError("incomplete evaluator checkpoint")
        self.load_state_dict(state, strict=False)


def corpus_bleu(hypotheses, references):
    """Case-sensitive whitespace corpus BLEU-4, no smoothing, [0,100]."""
    if len(hypotheses) != len(references) or not references:
        raise ValueError("paired nonempty corpus required")
    matches, totals, hlen, rlen = [0] * 4, [0] * 4, 0, 0
    for hyp, ref in zip(hypotheses, references):
        h, r = hyp.split(), ref.split()
        hlen += len(h)
        rlen += len(r)
        for n in range(1, 5):
            hc = Counter(tuple(h[i : i + n]) for i in range(len(h) - n + 1))
            rc = Counter(tuple(r[i : i + n]) for i in range(len(r) - n + 1))
            matches[n - 1] += sum((hc & rc).values())
            totals[n - 1] += sum(hc.values())
    if not hlen or any(m == 0 or t == 0 for m, t in zip(matches, totals)):
        return 0.0
    return 100 * math.exp(
        min(0.0, 1 - rlen / hlen) + sum(math.log(m / t) for m, t in zip(matches, totals)) / 4
    )


def calibration(gt, reconstructed, references):
    real = corpus_bleu(gt, references)
    shuffled = corpus_bleu(gt[1:] + gt[:1], references)
    codec = corpus_bleu(reconstructed, references)
    return {
        "metric": "pilot_BT_whitespace_BLEU4",
        "real": real,
        "shuffled": shuffled,
        "codec": codec,
        "usable": real >= 5 and real - shuffled >= 2 and codec >= 0.75 * real,
        "thresholds": {"real_min": 5, "real_minus_shuffle_min": 2, "codec_fraction_min": 0.75},
        "published_benchmark_comparable": False,
    }


def checked_optimizer_step(parameters, optimizer, max_norm=1.0):
    """Float64 norm avoids float32 square-sum overflow; reject truly NaN/Inf grads."""
    active = [p for p in parameters if p.grad is not None]
    norms = [torch.linalg.vector_norm(p.grad.detach(), ord=2, dtype=torch.float64) for p in active]
    if not norms:
        raise RuntimeError("no evaluator gradients")
    total = torch.linalg.vector_norm(torch.stack(norms), ord=2)
    if not bool(torch.isfinite(total)):
        optimizer.zero_grad(set_to_none=True)
        return None
    value = float(total)
    scale = min(1.0, max_norm / (value + 1e-6))
    for p in active:
        p.grad.mul_(scale)
    optimizer.step()
    return value
