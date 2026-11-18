import torch

from cprflow.models.attention import Attention
from cprflow.models.dit import TypedDiT


def inputs():
    return dict(
        z=torch.randn(2, 4, 4, 128),
        r=None,
        t=torch.tensor([0.3, 0.7]),
        text=torch.randn(2, 5, 768),
        text_mask=torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool),
        language=torch.tensor([0, 1]),
        lengths=torch.tensor([5, 8]),
        valid=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool),
        token_times=torch.arange(4)[None].repeat(2, 1).float() / 15,
    )


def nonzero_model(relational=False):
    model = TypedDiT(hidden=24, blocks=2, heads=3, ffn=48, relational=relational).eval()
    for block in model.blocks:
        for mod in block.mods.values():
            torch.nn.init.normal_(mod.weight, std=0.01)
            torch.nn.init.normal_(mod.bias, std=0.01)
    for head in model.outputs:
        torch.nn.init.normal_(head.weight, std=0.03)
    return model


def test_padded_state_text_cannot_affect_valid_output():
    model = nonzero_model()
    inp = inputs()
    a = model(**inp).z
    inp["z"][0, :, 3] = 9999
    inp["text"][0, 3:] = 9999
    b = model(**inp).z
    torch.testing.assert_close(a[0, :, :3], b[0, :, :3], atol=1e-6, rtol=1e-5)
    assert not b[0, :, 3].any()


def test_all_masked_attention_zero_and_finite():
    attn = Attention(24, 3)
    query = torch.randn(2, 3, 24, requires_grad=True)
    output = attn(query, query, torch.zeros(2, 3, 3, dtype=torch.bool))
    assert not output.any() and torch.isfinite(output).all()
    output.sum().backward()
    assert torch.isfinite(query.grad).all()


def test_warm_start_alpha_zero_nontrivial_outputs():
    torch.manual_seed(1)
    plain = nonzero_model()
    relation = nonzero_model(True)
    relation.warm_start(plain.state_dict())
    inp = inputs()
    a = plain(**inp).z
    inp["r"] = torch.randn(2, 4, 33)
    b = relation(**inp, relation_alpha=0).z
    assert a.abs().sum() > 0
    torch.testing.assert_close(a, b, atol=1e-5, rtol=0)
