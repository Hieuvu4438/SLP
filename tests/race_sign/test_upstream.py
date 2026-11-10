"""Compare new adapters against actual author code and licensed SMPL-X assets when supplied."""

import os
from pathlib import Path
import pytest
import torch
from methods.race_sign.adapters import SOKEAdapter, import_upstream
from methods.race_sign.geometry import SMPLXGeometry, pack_geometry
from methods.race_sign.schema import manual_to_6d


def test_real_author_vq_adapter_slices_and_decoding():
    repo = Path(__file__).resolve().parents[2] / "third_party/proposal3_sources/code/SOKE"
    if not repo.exists():
        pytest.skip("Official SOKE source checkout not present")
    vq = import_upstream(repo, "mGPT.archs.mgpt_vq").VQVae
    modules = [
        vq(
            nfeats=d,
            code_num=k,
            code_dim=8,
            output_emb_width=8,
            width=16,
            depth=1,
            down_t=2,
            stride_t=2,
            norm=None,
        ).cpu()
        for d, k in ((43, 4), (45, 6), (45, 7))
    ]
    for module in modules:
        module.quantizer.codebook.copy_(torch.randn_like(module.quantizer.codebook))
    mean, std = torch.randn(133) * 0.05, torch.rand(133) + 0.3
    adapter = SOKEAdapter(*modules, mean, std)
    adapter.train()
    assert all(not m.training for m in modules)
    assert all(not p.requires_grad for p in adapter.parameters())
    pose = torch.randn(12, 133) * 0.1
    tokens = adapter.encode_pose(pose)
    norm = (pose - mean) / std
    expected_inputs = (
        torch.cat((norm[:, :30], norm[:, 120:]), -1),
        norm[:, 30:75],
        norm[:, 75:120],
    )
    for module, token, features in zip(modules, tokens, expected_inputs):
        expected, _ = module.encode(features[None])
        assert torch.equal(token, expected[0])
    # Differing stream lengths: match author decode-then-replicate, not pad-then-decode.
    streams = (tokens[0][:2], tokens[1], tokens[2][:1])
    baseline, codes = adapter.decode_tokens(streams)
    assert baseline.shape == (12, 133) and codes.shape == (3, 3, 8)
    raw_body = modules[0].decode(streams[0])[0]
    torch.testing.assert_close(((baseline - mean) / std)[:8, :30], raw_body[:, :30])
    torch.testing.assert_close(((baseline - mean) / std)[8:, :30], raw_body[-1:, :30].expand(4, -1))
    with pytest.raises(ValueError):
        adapter.decode_tokens((torch.tensor([999]), tokens[1], tokens[2]))


def test_exact_smplx_geometry_and_hand_mean_gradient():
    path = os.environ.get("RACE_SMPLX_PATH")
    if not path:
        pytest.skip("Set RACE_SMPLX_PATH to licensed SMPL-X npz directory")
    import smplx

    geom = SMPLXGeometry(path, chunk_size=2)
    reference = smplx.SMPLX(
        path,
        gender="neutral",
        ext="npz",
        use_pca=False,
        flat_hand_mean=False,
        num_betas=10,
        num_expression_coeffs=10,
    ).requires_grad_(False)
    raw = torch.randn(1, 2, 133) * 0.1
    raw[..., 120:] = 0
    raw[..., 33] = 3.5
    x = manual_to_6d(raw, geom.manual_mean).requires_grad_()
    actual = geom(x)
    n = 2
    expected = reference(
        betas=geom.betas.expand(n, -1),
        global_orient=torch.zeros(n, 3),
        body_pose=torch.cat((torch.zeros(n, 33), raw[..., :30].reshape(n, 30)), -1),
        left_hand_pose=raw[..., 30:75].reshape(n, 45),
        right_hand_pose=raw[..., 75:120].reshape(n, 45),
        jaw_pose=torch.zeros(n, 3),
        expression=torch.zeros(n, 10),
        leye_pose=torch.zeros(n, 3),
        reye_pose=torch.zeros(n, 3),
    )
    wanted = pack_geometry(expected.joints.reshape(1, n, -1, 3))
    for key in ("body", "left", "right", "relations"):
        torch.testing.assert_close(getattr(actual, key), getattr(wanted, key), atol=3e-6, rtol=3e-5)
    (
        actual.left.square().sum() + actual.right.square().sum() + actual.relations.square().sum()
    ).backward()
    assert torch.isfinite(x.grad).all() and x.grad[..., 60:150].abs().sum() > 0


def test_official_mbart_asset_loader_text_to_tokens(tmp_path):
    """Tiny random author-architecture fixture, not pretrained signing quality."""
    import pickle
    import sentencepiece as spm
    from transformers import MBartConfig, MBartForConditionalGeneration, MBartTokenizer
    from methods.race_sign.adapters import _empty_keyword_metadata

    repo = Path(__file__).resolve().parents[2] / "third_party/proposal3_sources/code/SOKE"
    if not repo.exists():
        pytest.skip("Official SOKE source checkout not present")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world signing motion hand\nhello human body language\n" * 20)
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(tmp_path / "sp"),
        vocab_size=40,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    model_path = tmp_path / "mbart"
    model_path.mkdir()
    tokenizer = MBartTokenizer(vocab_file=str(tmp_path / "sp.model"))
    tokenizer.save_pretrained(model_path)
    vocabulary = tokenizer.get_vocab()
    with (model_path / "map_ids.pkl").open("wb") as f:
        pickle.dump({i: i for i in vocabulary.values()}, f)
    config = MBartConfig(
        vocab_size=max(vocabulary.values()) + 1,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=128,
    )
    MBartForConditionalGeneration(config).save_pretrained(model_path)
    module = import_upstream(repo, "mGPT.archs.mgpt_mbart")
    with _empty_keyword_metadata():
        lm = module.Mbart_Based_MLM(
            model_path=str(model_path),
            model_type="mbart_multi",
            num_heads=3,
            motion_codebook_size=96,
            hand_codebook_size=192,
            rhand_codebook_size=192,
            max_length=64,
        )
    gen_path = tmp_path / "generator.pt"
    torch.save({"state_dict": {"lm." + k: v for k, v in lm.state_dict().items()}}, gen_path)
    vq = import_upstream(repo, "mGPT.archs.mgpt_vq").VQVae
    params = dict(
        code_dim=8,
        output_emb_width=8,
        down_t=2,
        stride_t=2,
        width=16,
        depth=1,
        dilation_growth_rate=3,
        norm=None,
        activation="relu",
    )
    state = {}
    for prefix, dims, codes in [("vae.", 43, 96), ("hand_vae.", 45, 192), ("rhand_vae.", 45, 192)]:
        model = vq(nfeats=dims, code_num=codes, **params).cpu()
        model.quantizer.codebook.copy_(torch.randn_like(model.quantizer.codebook))
        state.update({prefix + k: v for k, v in model.state_dict().items()})
    token_path = tmp_path / "tokens.pt"
    torch.save({"state_dict": state}, token_path)
    mean_path, std_path = tmp_path / "mean.pt", tmp_path / "std.pt"
    torch.save(torch.zeros(133), mean_path)
    torch.save(torch.ones(133), std_path)
    adapter = SOKEAdapter.from_assets(
        dict(
            upstream_repo=str(repo),
            model_path=str(model_path),
            tokenizer_checkpoint=str(token_path),
            generator_checkpoint=str(gen_path),
            soke_mean=str(mean_path),
            soke_std=str(std_path),
            vq_params=params,
            max_text_tokens=64,
        )
    )
    features = adapter.text_features("hello world", "how2sign")
    assert features.ndim == 2 and features.shape[1] == 16
    record = adapter.prepare("hello world", "how2sign", "generated_fixture", max_tokens=4)
    assert record["codes"].shape[1:] == (3, 8)
    assert len(record["baseline"]) == len(record["codes"]) * 4
    teacher = adapter.prepare(
        "hello world", "how2sign", "teacher_fixture", target=torch.zeros(10, 133), mode="teacher"
    )
    assert teacher["baseline"].shape == (8, 133)
    assert adapter.lm.num_kws_per_sen == 0 and not adapter.lm.name2kws
