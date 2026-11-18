"""Production mBART wrapper, plus an explicit byte encoder for offline tests."""

import pickle
from pathlib import Path

import torch
from torch import nn

from .components import embedding

LANGUAGES = {"DGS": 0, "CSL": 1, "ASL": 2}


class ByteTextEncoder(nn.Module):
    """Untrained test fixture; never presented as a pretrained text baseline."""

    def __init__(self, dim=64, max_tokens=128):
        super().__init__()
        self.dim, self.max_tokens = dim, max_tokens
        self.tokens = nn.Embedding(257, dim, padding_idx=0)

    def forward(self, texts, languages=None):
        rows = [[b + 1 for b in x.encode("utf-8")[: self.max_tokens]] or [1] for x in texts]
        ids = torch.zeros(
            len(rows), max(map(len, rows)), dtype=torch.long, device=self.tokens.weight.device
        )
        for i, row in enumerate(rows):
            ids[i, : len(row)] = torch.tensor(row, device=ids.device)
        mask = ids != 0
        pos = embedding(torch.arange(ids.shape[1], device=ids.device).float(), self.dim)
        return (self.tokens(ids) + pos) * mask[..., None], mask


class MBartTextEncoder(nn.Module):
    def __init__(self, path, max_tokens=128, local_files_only=True):
        super().__init__()
        from transformers import MBartForConditionalGeneration, MBartTokenizer

        self.tokenizer = MBartTokenizer.from_pretrained(
            path,
            local_files_only=local_files_only,
            legacy=True,
            # This is SentencePiece mBART, not a Mistral regex tokenizer. Some
            # transformers 5 releases mis-detect old local configs without a
            # transformers_version; preserve the pretrained tokenizer unchanged.
            fix_mistral_regex=False,
        )
        model = MBartForConditionalGeneration.from_pretrained(
            path, local_files_only=local_files_only
        )
        self.encoder = model.get_encoder()
        self.dim, self.max_tokens = model.config.d_model, max_tokens
        map_path = Path(path) / "map_ids.pkl"
        mapping = None
        if map_path.exists():
            with open(map_path, "rb") as f:
                id_map = pickle.load(f)
            # SOKE distributes a 19k embedding table with a full 250k tokenizer.
            # Mapping is part of the pretrained asset, not learned from evaluation.
            unknown = id_map.get(self.tokenizer.unk_token_id, self.tokenizer.unk_token_id)
            mapping = torch.full(
                (max(len(self.tokenizer), max(id_map) + 1),), unknown, dtype=torch.long
            )
            for token_id, embedding_id in id_map.items():
                mapping[token_id] = embedding_id
            if mapping.max() >= model.config.vocab_size or mapping.min() < 0:
                raise ValueError("Invalid reduced-vocabulary map")
        elif len(self.tokenizer) > model.config.vocab_size:
            raise ValueError("Reduced-vocabulary mBART requires map_ids.pkl")
        self.register_buffer("id_mapping", mapping)

    def forward(self, texts, languages=None):
        if languages is None or len(languages) != len(texts):
            raise ValueError("Explicit sign-language IDs required for mBART tokenization")
        rows = []
        for text, language in zip(texts, languages.tolist()):
            self.tokenizer.src_lang = ("de_DE", "zh_CN", "en_XX")[language]
            rows.append(self.tokenizer(text, truncation=True, max_length=self.max_tokens))
        tokens = self.tokenizer.pad(rows, padding=True, return_tensors="pt")
        tokens = tokens.to(next(self.encoder.parameters()).device)
        mask = tokens["attention_mask"].bool()
        ids = tokens["input_ids"]
        if self.id_mapping is not None:
            ids = self.id_mapping[ids]
        return self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state, mask
