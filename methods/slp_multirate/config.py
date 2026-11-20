"""Strict experiment configuration for the staged pipeline."""

from pathlib import Path
import math

from slp.config import deep_merge, load_yaml, parse_overrides

CONFIG_DIR = Path(__file__).parent / "configs"
PILOTS = ("p0", "p1", "p2", "p3")
VARIANTS = ("A", "B", "C", "D", "onset_mid_offset", "no_anchor")
STAGES = ("length", "body", "anchors", "fine")


def _check_schema(base, layer, prefix=""):
    for key, value in layer.items():
        name = prefix + key
        if key not in base:
            raise ValueError(f"Unknown configuration key {name}")
        default = base[key]
        if isinstance(default, dict) and isinstance(value, dict):
            _check_schema(default, value, name + ".")
            continue
        kind = type(default)
        valid = isinstance(value, kind)
        if kind is int:
            valid = type(value) is int
        if kind is float:
            valid = type(value) in (int, float) and math.isfinite(value)
        if default is None:
            valid = value is None or isinstance(value, str)
        if not valid:
            raise ValueError(f"Invalid type for {name}")


def resolve(pilot=None, variant=None, path=None, overrides=()):
    base = load_yaml(CONFIG_DIR / "base.yaml")
    custom = load_yaml(path) if path else {}
    _check_schema(base, custom)
    selected_pilot = pilot or custom.get("experiment", {}).get("pilot")
    if path and Path(path).stem in PILOTS:
        selected_pilot = pilot or Path(path).stem
    selected_pilot = selected_pilot or "p3"
    selected_variant = variant or custom.get("experiment", {}).get("variant", "D")
    selected_variant = {"E": "onset_mid_offset", "F": "no_anchor"}.get(
        selected_variant, selected_variant
    )
    if selected_pilot not in PILOTS or selected_variant not in VARIANTS:
        raise ValueError("Unknown pilot or variant")
    cfg = base
    layers = [
        load_yaml(CONFIG_DIR / "pilots" / f"{selected_pilot}.yaml"),
        load_yaml(CONFIG_DIR / "variants" / f"{selected_variant}.yaml"),
        custom,
        parse_overrides(list(overrides)),
    ]
    for layer in layers:
        _check_schema(base, layer)
        cfg = deep_merge(cfg, layer)
    cfg["experiment"].update(pilot=selected_pilot, variant=selected_variant)
    return validate(cfg)


def validate(cfg):
    _check_schema(load_yaml(CONFIG_DIR / "base.yaml"), cfg)
    data, text, body, anchors, fine, train, sample = (
        cfg[k] for k in ("data", "text", "body", "anchors", "fine", "train", "sample")
    )
    if not 1 <= data["min_frames"] <= data["max_frames"] or data["fps"] <= 0:
        raise ValueError("Invalid frame range or FPS")
    buckets = data["bucket_boundaries"]
    if (not buckets or any(type(x) is not int or x < 1 for x in buckets)
            or buckets != sorted(set(buckets)) or buckets[-1] < data["max_frames"]):
        raise ValueError("Buckets must increase and cover max_frames")
    if (data["normalize_std_floor"] <= 0 or data["num_workers"] < 0 or
            not text["freeze"] or min(text["max_tokens"], text["mock_width"]) < 1):
        raise ValueError("Invalid data or frozen encoder settings")
    if (text["encoder"] != "mock" and text["revision"] is not None and
            (len(text["revision"]) != 40 or
             any(c not in "0123456789abcdef" for c in text["revision"]))):
        raise ValueError("Use a 40-character immutable text encoder revision")
    for net in (body, anchors, fine):
        if (min(net["width"], net["heads"], net["blocks"]) < 1
                or net["width"] % net["heads"] or not 0 <= net.get("dropout", 0) < 1):
            raise ValueError("Invalid attention dimensions or dropout")
    if body["stride"] not in (1, 2, 4) or fine["state_dim"] != 204:
        raise ValueError("Unsupported body stride or fine state dimension")
    if not 0 < fine["correction_max_rad"] <= math.pi or fine["local_radius"] < 0:
        raise ValueError("Invalid correction cap or local radius")
    if anchors["policy"] not in ("uniform", "event", "onset_mid_offset", "none"):
        raise ValueError("Unknown anchor policy")
    if anchors["confidence"] not in ("uniform", "learned"):
        raise ValueError("Unknown confidence setting")
    if (anchors["count"] < 1 or anchors["sigma_frames"] <= 0 or
            anchors["turn_min_speed_shoulder_widths"] <= 0 or anchors["nms_min_gap_frames"] < 1):
        raise ValueError("Invalid anchor parameters")
    weights = anchors["score_weights"]
    if len(weights) != 3 or any(type(x) not in (float, int) or not math.isfinite(x) or x < 0
                                for x in weights):
        raise ValueError("Expected three nonnegative anchor score weights")
    probs = train["condition_prob"].values()
    if not all(0 <= p <= 1 for p in probs) or abs(sum(probs) - 1) > 1e-6:
        raise ValueError("Condition probabilities must sum to one")
    if (train["precision"] not in ("fp32", "bf16") or train["optimizer"] != "adamw" or
            train["schedule"] != "warmup_cosine" or train["max_grad_norm"] <= 0 or
            min(train["microbatch"], train["gradient_accumulation"], train["keep_last"],
                train["checkpoint_every_updates"]) < 1 or train["val_every_updates"] < 0 or
            train["anchor_jitter_frames"] < 0 or train["anchor_payload_noise_std"] < 0 or
            len(train["betas"]) != 2 or any(not 0 <= x < 1 for x in train["betas"]) or
            train["weight_decay"] < 0 or not 0 <= train["ema_decay"] < 1 or
            not 0 <= train["warmup_ratio"] <= 1 or not 0 < train["min_lr_ratio"] <= 1):
        raise ValueError("Invalid training settings")
    if any(s["lr"] <= 0 or s["total_updates"] < 1 for s in train["stages"].values()):
        raise ValueError("Stage learning rates and update counts must be positive")
    if sample["solver"] != "euler" or sample["cfg"] or min(
            sample["body_nfe"], sample["fine_nfe"], sample["samples_per_text"]) < 1:
        raise ValueError("Expected Euler sampling with positive NFE and CFG disabled")
    loss = cfg["loss"]
    if loss["endpoint_time_weight"] not in ("uniform", "velocity") or any(
            value < 0 for key, value in loss.items() if key != "endpoint_time_weight"):
        raise ValueError("Invalid loss setting")
    ev = cfg["eval"]
    if (min(ev["bootstrap_resamples"], ev["min_usable_gate_clips"]) < 1 or
            min(ev["event_window_radius_frames"], ev["timing_match_tolerance_frames"]) < 0 or
            ev["gate_epsilon"] <= 0 or not 0 <= ev["event_threshold"] <= 1 or
            ev["primary_metric"] not in ("event_tip_mm", "tip_mm", "wrist_mm", "body_mm")):
        raise ValueError("Invalid evaluation settings")
    return cfg
