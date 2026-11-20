"""Token-level multilingual conditioning for the motion transformer."""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from torch import Tensor, nn

from slp.paths import PROJECT_ROOT

# Same order as data.LANGUAGES.
MBART_LANGUAGES = ("de_DE", "zh_CN", "en_XX")


class TextEncoder(nn.Module):
    def __init__(self, backend: str, model: str | None, max_tokens: int, finetune: bool,
                 byte_dim: int):
        super().__init__()
        if max_tokens < 2:
            raise ValueError("text.max_tokens must be at least 2")
        self.backend = backend
        self.max_tokens = max_tokens
        self.finetune = finetune
        self.id_map = None
        if backend == "bytes":
            self.encoder = nn.Embedding(258, byte_dim)
            self.position = nn.Embedding(max_tokens, byte_dim)
            self.output_dim = byte_dim
            self.pad_id = 257
        elif backend == "mbart":
            from transformers import MBartForConditionalGeneration, MBartTokenizer

            if not model:
                raise ValueError("text.model must point to a local mBART directory")
            path = PROJECT_ROOT / Path(model).expanduser()
            self.tokenizer = MBartTokenizer.from_pretrained(
                path, local_files_only=True, fix_mistral_regex=False,
            )
            self.encoder = MBartForConditionalGeneration.from_pretrained(
                path, local_files_only=True
            ).get_encoder()
            self.output_dim = self.encoder.config.d_model
            self.pad_id = self.tokenizer.pad_token_id
            self.unk_id = self.tokenizer.unk_token_id
            mapping = path / "map_ids.pkl"
            if mapping.is_file():
                # SOKE distributes a compact embedding table with remapped token IDs.
                with mapping.open("rb") as f:
                    self.id_map = pickle.load(f)
                self.pad_id = self.id_map[self.pad_id]
                self.unk_id = self.id_map[self.unk_id]
            self.encoder.requires_grad_(finetune)
        else:
            raise ValueError("text.backend must be mbart or bytes")

    @property
    def trained(self) -> bool:
        return self.backend == "bytes" or self.finetune

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.trained:
            self.encoder.eval()
        return self

    def tokenize(self, texts: list[str], language_ids: list[int]) -> list[list[int]]:
        if self.backend == "bytes":
            return [list(text.encode("utf-8")[:self.max_tokens - 1]) + [256] for text in texts]
        tokens = []
        for text, language in zip(texts, language_ids, strict=True):
            self.tokenizer.src_lang = MBART_LANGUAGES[language]
            sequence = self.tokenizer(text, truncation=True, max_length=self.max_tokens).input_ids
            if self.id_map is not None:
                sequence = [self.id_map.get(token, self.unk_id) for token in sequence]
            tokens.append(sequence)
        return tokens

    def forward(self, texts: list[str], languages: Tensor) -> tuple[Tensor, Tensor]:
        token_rows = self.tokenize(texts, languages.tolist())
        size = max(map(len, token_rows))
        ids = torch.tensor(
            [row + [self.pad_id] * (size - len(row)) for row in token_rows],
            device=languages.device,
        )
        mask = ids != self.pad_id
        if self.backend == "bytes":
            positions = torch.arange(size, device=languages.device)
            return self.encoder(ids) + self.position(positions), mask
        with torch.set_grad_enabled(self.training and self.trained and torch.is_grad_enabled()):
            tokens = self.encoder(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        return tokens, mask
