import torch

from cprflow.models.teacher import SignTeacher, corrupt_spans
from cprflow.objectives.semantic import multi_positive_nce, positive_masks


def test_teacher_clean_masked_and_frozen_input_gradients():
    torch.manual_seed(2)
    teacher = SignTeacher(hidden=16, blocks=2, heads=4, ffn=32, embedding=8)
    x = torch.randn(3, 9, 296)
    lengths = torch.tensor([5, 9, 7])
    times = torch.arange(9).expand(3, 9) / 30
    text = torch.randn(3, 4, 768)
    text_mask = torch.ones(3, 4, dtype=torch.bool)
    language = torch.tensor([0, 0, 0])
    corrupted, mask = corrupt_spans(x, lengths)
    clean = teacher(x, lengths, times)
    noisy = teacher(corrupted, lengths, times, mask)
    keys = teacher.encode_text(text, text_mask, language)
    positive, allowed = positive_masks(["same", "same", "different"], language)
    nce, valid = multi_positive_nce(clean["global"], keys, positive, allowed)
    assert valid.all()
    loss = nce.mean() + (noisy["reconstruction"] - x).square()[mask].mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in teacher.parameters())
    teacher.zero_grad(set_to_none=True)
    teacher.freeze()
    generated = x.clone().requires_grad_()
    teacher(generated, lengths, times)["global"].square().sum().backward()
    assert torch.isfinite(generated.grad).all()
    assert all(p.grad is None for p in teacher.parameters())
