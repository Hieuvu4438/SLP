"""Production assembly requires an explicit manifest and verified local assets."""

from pathlib import Path

from slp.skeleton_fk import SkeletonFK

from .data.manifest import MotionManifest
from .data.smplx_adapter import SOKECodec
from .flow import SourceConfig
from .geometry.fk import SOKEFK
from .geometry.physical_time_features import SequenceFeatures
from .model import CoPlanSO3
from .models.text_encoder import ByteTextEncoder, MBartTextEncoder

ROOT = Path(__file__).resolve().parents[2]


def absolute(path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def build(hp):
    manifest = MotionManifest.load(absolute(hp["manifest"]))
    if not manifest.asset_checksums:
        raise ValueError("Production manifest requires asset checksums")
    manifest.verify_assets(ROOT)
    required_assets = (
        absolute(hp["smplx_dir"]) / "smplx/SMPLX_NEUTRAL.npz",
        absolute(hp["stats"]) / "mean.pt",
        absolute(hp["stats"]) / "std.pt",
    )
    recorded = {(ROOT / p).resolve() for p in manifest.asset_checksums}
    if any(p.resolve() not in recorded for p in required_assets):
        raise ValueError("Loaded model and statistics must be bound to the manifest hashes")
    fk = SkeletonFK(model_dir=absolute(hp["smplx_dir"]), stats=absolute(hp["stats"]))
    codec = SOKECodec(manifest, fk.feat_mean, fk.feat_std, fk.hand_mean)
    features = SequenceFeatures(
        SOKEFK(manifest, codec, fk=fk),
        {"joints": 36, "wrists": 6, "handshape": 30, "palms": 12, "relative": 18},
        weights=tuple(hp.get("physical_weights", [1.0, 0.1, 0.01])),
        group_weights=hp.get("feature_group_weights"),
    )
    text_cfg = hp["text"]
    kind = text_cfg["kind"]
    if kind == "mbart":
        text = MBartTextEncoder(
            str(absolute(text_cfg["path"])),
            max_tokens=text_cfg.get("max_tokens", 128),
            local_files_only=text_cfg.get("local_files_only", True),
        )
    elif kind == "byte_fixture":
        if not hp.get("allow_untrained_text_fixture", False):
            raise ValueError("Byte text encoder requires explicit test-only opt-in")
        text = ByteTextEncoder(text_cfg.get("dim", 64))
    else:
        raise ValueError(f"Unknown text encoder: {kind}")
    model = CoPlanSO3(
        manifest,
        text,
        features,
        source_config=SourceConfig(**hp.get("source", {})),
        **hp.get("architecture", {}),
    )
    return model, codec
