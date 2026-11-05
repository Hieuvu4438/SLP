import pytest
import torch

from methods.temporal_plan.text import TextConditioner


def test_real_tiny_mt5_backbone_stays_frozen_and_eval_but_adapter_learns():
    transformers = pytest.importorskip("transformers")
    config = transformers.MT5Config(vocab_size=32, d_model=16, d_kv=8, d_ff=32,
                                    num_layers=1, num_heads=2, dropout_rate=.2)
    encoder = transformers.MT5EncoderModel(config)
    conditioner = TextConditioner(encoder, output_width=8).train()
    ids = torch.tensor([[1, 2, 3, 0], [1, 4, 0, 0]])
    mask = ids != 0
    languages = torch.tensor([0, 2])
    a, valid = conditioner(ids, mask, languages)
    b, _ = conditioner(ids, mask, languages)
    torch.testing.assert_close(a, b)
    assert not conditioner.encoder.training
    assert not any(p.requires_grad for p in conditioner.encoder.parameters())
    assert a[~valid].count_nonzero() == 0
    a.square().sum().backward()
    assert all(p.grad is None for p in conditioner.encoder.parameters())
    assert conditioner.projection.weight.grad.abs().sum() > 0
    assert conditioner.language.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="language"):
        conditioner(ids, mask, torch.tensor([0, 9]))
    with pytest.raises(ValueError, match="binary"):
        conditioner(ids, torch.full_like(ids, 2), languages)
