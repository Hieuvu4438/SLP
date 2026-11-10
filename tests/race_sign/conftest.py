import json
import pytest
import torch
from methods.race_sign.config import RaceConfig
from methods.race_sign.geometry import SkeletonGeometry


@pytest.fixture(autouse=True)
def small_cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.fixture
def cfg():
    return RaceConfig(
        code_dim=8,
        text_dim=12,
        width=16,
        heads=2,
        layers=1,
        dropout=0,
        text_dropout=0,
        block_size=4,
        blend_frames=1,
        selector_width=16,
        flow_steps=2,
    )


@pytest.fixture
def geometry():
    parents = torch.arange(55) - 1
    rest = torch.arange(55)[:, None] * torch.tensor([[0.005, 0.007, 0.003]])
    return SkeletonGeometry(parents, rest)


@pytest.fixture
def record_factory(cfg):
    def make(index=0, length=8, role="decoder_train"):
        generator = torch.Generator().manual_seed(index + 100)
        raw = torch.randn(length, 133, generator=generator) * 0.1
        base = raw + torch.randn(raw.shape, generator=generator) * 0.02
        return dict(
            sample_id=f"{role}_{index}",
            source_group=f"video_{role}_{index}",
            role=role,
            official_split="train",
            codes=torch.randn(length // 4, 3, cfg.code_dim, generator=generator),
            text=torch.randn(3, cfg.text_dim, generator=generator),
            language=index % 3,
            baseline=base,
            target=raw,
            provenance={
                "fixture": "synthetic; not scientific evidence",
                "preparation": "generated" if role.startswith("selector_") else "teacher",
            },
        )

    return make


@pytest.fixture
def manifest_factory(tmp_path, record_factory):
    def make(role, count=2):
        folder = tmp_path / role
        folder.mkdir(exist_ok=True)
        rows = []
        for i in range(count):
            record = record_factory(i, length=8 + 4 * (i % 2), role=role)
            path = f"{i}.pt"
            torch.save(record, folder / path)
            rows.append(
                dict(
                    path=path,
                    sample_id=record["sample_id"],
                    source_group=record["source_group"],
                    role=role,
                    official_split="train",
                )
            )
        manifest = folder / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return manifest

    return make
