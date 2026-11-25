"""Token-level text features: the SOKE mBART encoder, frozen by default or fine-tuned with the
generator (``text.finetune``), or a byte embedding for CPU tests."""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from torch import Tensor, nn

from slp.paths import PROJECT_ROOT

# Same order as data.LANGUAGES.
MBART_LANGUAGES = ("de_DE", "zh_CN", "en_XX")


class TextEncoder(nn.Module):
    def __init__(self, backend: str, model: str | None, max_tokens: int, byte_dim: int = 64,
                 finetune: bool = False):
        super().__init__()
        self.finetune = bool(finetune)
        if max_tokens < 2:
            raise ValueError("text.max_tokens must be at least 2")
        self.backend, self.max_tokens, self.id_map = backend, max_tokens, None
        if backend == "bytes":
            self.encoder = nn.Embedding(258, byte_dim)
            self.position = nn.Embedding(max_tokens, byte_dim)
            self.output_dim, self.pad_id = byte_dim, 257
        elif backend == "mbart":
            from transformers import MBartForConditionalGeneration, MBartTokenizer

            if not model:
                raise ValueError("text.model must point to a local mBART directory")
            path = PROJECT_ROOT / Path(model).expanduser()
            self.tokenizer = MBartTokenizer.from_pretrained(path, local_files_only=True,
                                                           fix_mistral_regex=False)
            self.encoder = MBartForConditionalGeneration.from_pretrained(
                path, local_files_only=True).get_encoder()
            self.output_dim = self.encoder.config.d_model
            self.pad_id = self.tokenizer.pad_token_id
            self.unk_id = self.tokenizer.unk_token_id
            mapping = path / "map_ids.pkl"
            if mapping.is_file():  # SOKE ships a compact embedding table with remapped ids
                with mapping.open("rb") as f:
                    self.id_map = pickle.load(f)
                self.pad_id = self.id_map[self.pad_id]
                self.unk_id = self.id_map[self.unk_id]
            self.encoder.requires_grad_(self.finetune)
        else:
            raise ValueError("text.backend must be mbart or bytes")

    @property
    def trained(self) -> bool:
        """Learned with the generator: the byte table always, mBART only with ``finetune``."""
        return self.backend == "bytes" or self.finetune

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.trained:
            self.encoder.eval()
        return self

    def tokenize(self, texts: list[str], languages: list[int]) -> list[list[int]]:
        if self.backend == "bytes":
            return [list(t.encode("utf-8")[:self.max_tokens - 1]) + [256] for t in texts]
        rows = []
        for text, lang in zip(texts, languages, strict=True):
            self.tokenizer.src_lang = MBART_LANGUAGES[lang]
            ids = self.tokenizer(text, truncation=True, max_length=self.max_tokens).input_ids
            if self.id_map is not None:
                ids = [self.id_map.get(i, self.unk_id) for i in ids]
            rows.append(ids)
        return rows

    def forward(self, texts: list[str], languages: Tensor) -> tuple[Tensor, Tensor]:
        """-> tokens [B, L, D] (float32), valid mask [B, L]."""
        rows = self.tokenize(texts, languages.tolist())
        size = max(map(len, rows))
        ids = torch.tensor([r + [self.pad_id] * (size - len(r)) for r in rows],
                           device=languages.device)
        mask = ids != self.pad_id
        if self.backend == "bytes":
            return self.encoder(ids) + self.position(torch.arange(size, device=ids.device)), mask
        with torch.set_grad_enabled(self.finetune and torch.is_grad_enabled()):
            tokens = self.encoder(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        return tokens.float(), mask


def masked_pool(tokens: Tensor, mask: Tensor) -> Tensor:
    weight = mask.to(tokens.dtype).unsqueeze(-1)
    return (tokens * weight).sum(1) / weight.sum(1).clamp_min(1)
