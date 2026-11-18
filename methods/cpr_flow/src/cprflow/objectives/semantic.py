import torch
import torch.nn.functional as F


def multi_positive_nce(queries, keys, positive, allowed, temperature=0.07):
    logits = (
        F.normalize(queries.float(), dim=-1) @ F.normalize(keys.float(), dim=-1).T / temperature
    )
    positive = positive & allowed
    valid = positive.any(1) & (allowed & ~positive).any(1)
    # Avoid all -inf logsumexp backward on invalid rows.
    safe_allowed = allowed.clone()
    safe_positive = positive.clone()
    safe_allowed[~valid] = True
    safe_positive[~valid] = True
    den = torch.logsumexp(logits.masked_fill(~safe_allowed, float("-inf")), 1)
    num = torch.logsumexp(logits.masked_fill(~safe_positive, float("-inf")), 1)
    return torch.where(valid, den - num, logits.sum(1) * 0), valid


def positive_masks(texts, languages, key_texts=None, key_languages=None):
    key_texts = texts if key_texts is None else key_texts
    key_languages = languages if key_languages is None else key_languages
    device = languages.device
    same_lang = languages[:, None] == key_languages[None, :]
    same_text = torch.tensor([[a == b for b in key_texts] for a in texts], device=device)
    return same_lang & same_text, same_lang
