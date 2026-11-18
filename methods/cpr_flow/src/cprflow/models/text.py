import re

import torch
from torch import nn

from cprflow.data.lmdb_adapter import normalized_text
from cprflow.types import LANGUAGES, TextBatch


class FrozenTextEncoder(nn.Module):
    def __init__(self, revision, tokenizer_revision=None, max_tokens=512, local_files_only=False):
        super().__init__()
        if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("mT5 requires a resolved 40-character model commit; run audit_sources")
        tokenizer_revision = tokenizer_revision or revision
        if not re.fullmatch(r"[0-9a-f]{40}", tokenizer_revision):
            raise ValueError("Tokenizer revision must be a resolved commit")
        from transformers import AutoTokenizer, MT5EncoderModel

        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/mt5-base", revision=tokenizer_revision, local_files_only=local_files_only
        )
        self.model = MT5EncoderModel.from_pretrained(
            "google/mt5-base", revision=revision, local_files_only=local_files_only
        )
        self.model.requires_grad_(False)
        self.max_tokens = max_tokens
        self.revision, self.tokenizer_revision = revision, tokenizer_revision
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, texts, languages):
        sentences = []
        for text, lang in zip(texts, languages):
            if lang not in LANGUAGES:
                raise ValueError("Unknown sign language")
            sentences.append(f"[{lang}] {normalized_text(text)}")
        data = self.tokenizer(sentences, padding=True, truncation=False, return_tensors="pt")
        if (data["attention_mask"].sum(1) > self.max_tokens).any():
            raise ValueError("unsupported_text_length: tokenization exceeds 512; no truncation")
        data = {k: v.to(next(self.model.parameters()).device) for k, v in data.items()}
        hidden = self.model(**data).last_hidden_state
        if hidden.shape[-1] != 768:
            raise ValueError("mT5-base encoder must output 768D")
        mask = data["attention_mask"].bool()
        return TextBatch(hidden * mask[..., None], mask)
