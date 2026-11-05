"""All local verification stays CPU-only and bounds host thread consumption."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch  # noqa: E402
import pytest  # noqa: E402

torch.set_num_threads(2)


@pytest.fixture
def tiny_components():
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from methods.temporal_plan.losses import LossWeights
    from methods.temporal_plan.segments import TeacherConfig
    from methods.temporal_plan.system import ModelConfig, TemporalPlanSystem
    from methods.temporal_plan.text import TextConditioner

    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "hello": 3, "rain": 4, "tomorrow": 5}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="$A [EOS]", special_tokens=[("[EOS]", 2)])
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )
    text = TextConditioner(transformers.MT5EncoderModel(transformers.MT5Config(
        vocab_size=16, d_model=8, d_kv=4, d_ff=16, num_layers=1, num_heads=2, dropout_rate=0,
    )), output_width=7)
    config = ModelConfig(
        max_latents=4, latent_width=2, codec_hidden=8, content_width=3, segment_hidden=12,
        hidden=16, heads=2, planner_layers=1, denoiser_layers=1, mixtures=2, dropout=0,
        diffusion_steps=8, sampling_steps=3, exposure_probability=.25,
        teacher=TeacherConfig(max_duration=3), losses=LossWeights(fk=0),
    )
    return TemporalPlanSystem(config, text, torch.zeros(41, 3)), tokenizer
