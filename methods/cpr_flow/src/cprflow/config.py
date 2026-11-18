"""Strict defaults, recursive YAML inheritance and safe dotted CLI overrides."""

import copy
import json
import os
from pathlib import Path

import yaml

DEFAULTS = yaml.safe_load(Path(__file__).with_name("defaults.yaml").read_text())


def merge(base, update, prefix=""):
    result = copy.deepcopy(base)
    if not isinstance(update, dict):
        raise ValueError(f"{prefix}: expected mapping")
    for key, value in update.items():
        location = f"{prefix}.{key}".strip(".")
        if key not in base:
            raise ValueError(f"Unknown config key: {location}")
        old = base[key]
        if isinstance(old, dict):
            result[key] = merge(old, value, location)
        else:
            if old is None and value is not None and not isinstance(value, str):
                raise ValueError(f"{location}: expected nullable string")
            if old is not None:
                expected = type(old)
                valid = isinstance(value, expected)
                if expected is float:
                    valid = isinstance(value, (int, float)) and not isinstance(value, bool)
                if expected is int:
                    valid = type(value) is int
                if not valid:
                    raise ValueError(f"{location}: expected {expected.__name__}")
            result[key] = copy.deepcopy(value)
    return result


def load_config(path, overrides=(), world_size=None):
    def read(p, stack):
        p = Path(p).resolve()
        if p in stack:
            raise ValueError("Config inheritance cycle")
        obj = yaml.safe_load(p.read_text()) or {}
        parent = obj.pop("extends", None)
        base = read(p.parent / parent, stack | {p}) if parent else copy.deepcopy(DEFAULTS)
        return merge(base, obj)

    cfg = read(path, set())
    for override in overrides:
        key, sep, raw = override.partition("=")
        if not sep:
            raise ValueError("--set requires dotted.key=value")
        obj = yaml.safe_load(raw)
        for part in reversed(key.split(".")):
            obj = {part: obj}
        cfg = merge(cfg, obj)
    validate(cfg, int(os.environ.get("WORLD_SIZE", 1)) if world_size is None else world_size)
    return cfg


def validate(c, world_size=1):
    from cprflow.types import DATASETS, LANGUAGES, PROFILE

    d, a, g = c["data"], c["ae"], c["geometry"]
    if c["spec_version"] != "cprflow-impl-v1":
        raise ValueError("Unsupported specification")
    if not d["datasets"] or not set(d["datasets"]) <= DATASETS.keys():
        raise ValueError("Only three controlled corpora supported")
    if len(d["q_dataset"]) != len(d["datasets"]) or abs(sum(d["q_dataset"]) - 1) > 1e-6:
        raise ValueError("Dataset probabilities must sum to one")
    if any(x <= 0 for x in d["q_dataset"]):
        raise ValueError("Dataset probabilities must be positive")
    locked = [
        (d["profile"], PROFILE),
        (d["sign_language_ids"], LANGUAGES),
        (d["dims"], {"body": 60, "left": 90, "right": 90, "face": 56}),
        (d["target_fps"], 30.0),
        (d["resample_grid"], "endpoint_linspace_v1"),
        (d["truncate_motion"], False),
        (d["mirror_augmentation"], False),
        (d["random_time_stretch"], False),
        (g["backend"], "smplx_layer_matrix"),
        (g["num_expression"], 50),
        (g["num_betas"], 10),
        (g["face_contour"], False),
        (g["joint_mapper"], None),
        (g["root"], "identity"),
        (g["translation"], "zero"),
        (g["shape"], "neutral_zero"),
        (g["flat_hand_mean"], True),
        (a["stride"], 2),
        (a["latent_dim_per_part"], 128),
        (a["channels"], [128, 256]),
        (a["residual_dilations"], [1, 2]),
        (a["norm"], "channel_layernorm"),
        (a["boundary"], "true_length_replicate"),
        (a["decoder_fusion"], "residual_mlp512_256_512"),
        (a["kl_weight"], 0.0),
        (c["text"]["model_id"], "google/mt5-base"),
        (c["text"]["hidden_dim"], 768),
        (c["text"]["frozen"], True),
        (c["text"]["use_encoder_only"], True),
        (c["text"]["prefix_by_sign_language"], True),
        (c["stats"]["method"], "equal_dataset_equal_clip_moments"),
        (c["stats"]["fit_split"], "train"),
        (c["relation"]["dim"], 33),
        (c["relation"]["pooling"], "valid_pair_mean"),
        (c["flow"]["path"], "linear_noise_to_data"),
        (c["flow"]["t_distribution"], "uniform"),
        (c["flow"]["coupling"], "independent"),
        (c["loss"]["reduction"], "global_mean_of_per_clip_valid_means"),
        (c["train"]["optimizer"], "AdamW"),
        (c["model"]["temporal_attention"], "full_per_stream"),
        (c["model"]["modulation"], "adaln_zero"),
        (c["model"]["dropout"], 0.0),
        (c["sample"]["oracle_inputs"], False),
        (c["eval"]["best_of_n_with_gt"], False),
        (c["eval"]["per_dataset"], True),
        (c["eval"]["split"], "dev"),
        (c["eval"]["recompute_relations_from_geometry"], True),
        (c["guard"]["penetration_weight"], 0.0),
        (c["logging"]["wandb"], False),
        (c["logging"]["jsonl"], True),
        (c["logging"]["tensorboard"], False),
        (c["duration"]["distribution"], "log_normal_frames"),
        (c["duration"]["input_dim"], 800),
        (c["duration"]["round_frames_to"], 2),
        (c["distributed"]["strict_resume_world_size"], True),
        (c["ae"]["local_hand_fk_weight"], 2.0),
        (c["geometry"]["epsilon"], 1e-6),
        (c["eval"]["official_metrics_adapter"], None),
        (c["eval"]["backtranslation_checkpoint"], None),
    ]
    for actual, expected in locked:
        if actual != expected:
            raise ValueError(f"Locked v1 option: expected {expected!r}, received {actual!r}")
    t = c["train"]
    if t["microbatch_per_rank"] * t["accumulate"] * world_size != t["effective_batch"]:
        raise ValueError("microbatch * accumulate * world_size must equal effective_batch")
    if c["ae"]["execution"] not in ("per_clip", "masked_batch"):
        raise ValueError("Unknown AE execution implementation")
    if c["model"]["hidden_dim"] % c["model"]["heads"]:
        raise ValueError("hidden_dim must divide heads")
    if c["run"]["manual_gate_policy"] not in ("required", "operator_waived"):
        raise ValueError("Unknown manual gate policy")
    if c["run"]["manual_gate_policy"] == "operator_waived" and not c["run"]["policy_reference"]:
        raise ValueError("Operator waiver requires its real instruction/reference")
    if (
        not d["require_verified_feature_fps"]
        and c["run"]["manual_gate_policy"] != "operator_waived"
    ):
        raise ValueError(
            "Assumed feature FPS requires an explicit operator-authorized exploratory run"
        )
    if c["run"]["stage"] not in ("parent", "continuation", "standalone"):
        raise ValueError("Unknown training stage")
    if c["run"]["task"] not in ("ae", "duration", "flow", "teacher"):
        raise ValueError("Unknown task")
    if c["run"]["variant"] not in (
        "plain_cfm",
        "relational_cfm",
        "auxiliary_only",
        "capacity_control",
        "diffusion",
        "full",
    ):
        raise ValueError("Unknown variant")
    relational = c["run"]["variant"] in ("relational_cfm", "capacity_control", "full")
    if c["relation"]["enabled"] != relational:
        raise ValueError("variant/relation.enabled mismatch")
    if c["loss"]["cycle"] and not relational:
        raise ValueError("Cycle loss requires relation stream")
    if c["run"]["variant"] == "full" and (not c["loss"]["semantic"] or not c["guard"]["enabled"]):
        raise ValueError("Full variant requires semantic objective and calibrated guard")
    if c["sample"]["duration_source"] != "predicted":
        raise ValueError("Default sampler requires predicted duration; oracle is evaluator-only")
    if c["guard"]["mode"] not in ("semantic", "feature_only", "unrestricted"):
        raise ValueError("Unknown correction mode")
    if c["run"]["variant"] == "full" and c["guard"]["mode"] != "semantic":
        raise ValueError(
            "Full training configuration requires semantic guard; use labeled inference controls"
        )
    if c["sample"]["solver"] not in ("euler", "heun") or c["sample"]["nfe"] < 1:
        raise ValueError("Invalid ODE settings")
    if c["sample"]["cfg"] < 1:
        raise ValueError("CFG scale must be >= 1")
    if t["checkpoint_keep_recent"] < 1:
        raise ValueError("checkpoint_keep_recent must be >=1")
    if t["max_updates"] < 1 or t["microbatch_per_rank"] < 1 or t["accumulate"] < 1:
        raise ValueError("Training counts must be positive")
    if c["model"]["global_part_context_every"] < 1:
        raise ValueError("global context interval must be positive")
    if c["loss"]["auxiliary_every_updates"] != c["loss"]["auxiliary_frequency_compensation"]:
        raise ValueError("Auxiliary frequency compensation must equal the update interval")
    if c["data"]["max_train_frames"] != 1024 or c["data"]["max_generation_frames"] != 2048:
        raise ValueError("Length support changes require a separate versioned profile")
    if c["run"]["variant"] == "diffusion" and any(
        c["loss"][k] for k in ("geom", "cycle", "lag", "semantic")
    ):
        raise ValueError(
            "Matched diffusion uses epsilon objective without CFM endpoint auxiliaries"
        )
    return c


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
