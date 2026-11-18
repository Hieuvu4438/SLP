"""Text-only generation API. Target motion, gloss and length are absent from public inputs."""

import json
import time
from pathlib import Path

import torch

from cprflow.data.cache import file_hash
from cprflow.geometry.temporal import lengths_mask, pool2
from cprflow.models.ae import MotionAE
from cprflow.models.duration import DurationModel
from cprflow.models.text import FrozenTextEncoder
from cprflow.sampling.ode import ddim_sample, integrate
from cprflow.training.artifacts import load_dependency, normalizer, text_revisions
from cprflow.training.checkpoint import load_checkpoint
from cprflow.training.tasks import make_model, make_teacher
from cprflow.types import GenerationRequest, GenerationResult, LANGUAGES


class MotionGenerator:
    def __init__(
        self,
        model,
        ae,
        duration,
        motion_norm,
        latent_norm,
        encode,
        config,
        metadata=None,
        teacher=None,
        relation_fn=None,
        calibration=None,
        relation_norm=None,
    ):
        self.model = model.eval()
        self.ae = ae.eval()
        self.duration = duration.eval()
        self.motion_norm, self.latent_norm = motion_norm, latent_norm
        self.encode, self.config = encode, config
        self.metadata = metadata or {}
        self.relation_norm = relation_norm
        self.teacher, self.relation_fn, self.calibration = teacher, relation_fn, calibration

    @classmethod
    def from_checkpoint(
        cls, path, device="cpu", guard_mode=None, teacher_checkpoint=None, calibration_path=None
    ):
        state = load_checkpoint(path)
        config = state["config"]
        if config["run"]["task"] != "flow":
            raise ValueError("Sampling requires a trained generator checkpoint")
        from cprflow.training.tasks import provenance

        actual = provenance(config)
        for key, value in state["provenance"].items():
            if key != "config" and actual.get(key) != value:
                raise ValueError(f"Sampling artifact provenance mismatch: {key}")
        model = make_model(config)
        model.load_state_dict(state["ema"]["shadow"])
        model.to(device).requires_grad_(False)
        ae = load_dependency(MotionAE(), config["paths"]["ae_checkpoint"], config, "ae", device)
        dc = config["duration"]
        duration = load_dependency(
            DurationModel(dc["hidden_dim"], dc["sigma_floor"], dc["sigma_max"]),
            config["paths"]["duration_checkpoint"],
            config,
            "duration",
            device,
        )
        mn, ln = normalizer(config, "motion", device), normalizer(config, "latent", device)
        rev, tok = text_revisions(config)
        encoder = FrozenTextEncoder(
            rev, tok, config["text"]["max_tokens"], config["text"]["local_files_only"]
        ).to(device)
        if guard_mode is not None:
            if guard_mode not in ("off", "unrestricted", "feature_only", "semantic"):
                raise ValueError("Unknown inference correction control")
            config["guard"]["enabled"] = guard_mode != "off"
            config["guard"]["mode"] = "semantic" if guard_mode == "off" else guard_mode
        if teacher_checkpoint is not None:
            config["paths"]["teacher_checkpoint"] = teacher_checkpoint
        if calibration_path is not None:
            config["paths"]["teacher_calibration"] = calibration_path
        kwargs = {}
        if config["guard"]["enabled"]:
            if not config["relation"]["enabled"]:
                raise ValueError("Correction requires a co-generated relation plan")
            from cprflow.geometry.body_model import BodyModel
            from cprflow.geometry.relations import relation_operator
            from cprflow.sampling.guard import load_calibration

            mode = config["guard"].get("mode", "semantic")
            teacher = None
            calibration = None
            if mode != "unrestricted":
                teacher = load_dependency(
                    make_teacher(config),
                    config["paths"]["teacher_checkpoint"],
                    config,
                    "teacher",
                    device,
                )
                preprocessing = json.loads(
                    (Path(config["paths"]["manifests"]) / "preprocessing_report.json").read_text()
                )["preprocessing_hash"]
                calibration = load_calibration(
                    config["paths"]["teacher_calibration"],
                    {
                        "teacher_sha256": file_hash(config["paths"]["teacher_checkpoint"]),
                        "asset_sha256": file_hash(config["paths"]["smplx_model"]),
                        "preprocess_sha256": preprocessing,
                    },
                )
            body = BodyModel(
                config["paths"]["smplx_model"], config["geometry"]["fk_chunk_frames"]
            ).to(device)
            rn = normalizer(config, "relation", device)
            kwargs = {
                "teacher": teacher,
                "relation_fn": lambda x, n: relation_operator(x, n, body, rn),
                "calibration": calibration,
            }
        if config["relation"]["enabled"]:
            kwargs["relation_norm"] = normalizer(config, "relation", device)
        return cls(
            model,
            ae,
            duration,
            mn,
            ln,
            encoder,
            config,
            {
                "checkpoint_sha256": file_hash(path),
                "provenance": state["provenance"],
                "synthetic": config["run"]["synthetic"],
                "inference_correction": config["guard"].get("mode", "semantic")
                if config["guard"]["enabled"]
                else "off",
                "inference_teacher_sha256": file_hash(config["paths"]["teacher_checkpoint"])
                if kwargs.get("teacher") is not None
                else None,
                "inference_calibration_sha256": file_hash(config["paths"]["teacher_calibration"])
                if kwargs.get("calibration") is not None
                else None,
            },
            **kwargs,
        )

    def generate(self, request: GenerationRequest, nfe=None, cfg=None, solver=None):
        if not isinstance(request, GenerationRequest):
            raise TypeError("Main API accepts only GenerationRequest")
        return self._generate(request, nfe, cfg, solver)

    @torch.no_grad()
    def _generate(
        self,
        request,
        nfe=None,
        cfg=None,
        solver=None,
        diagnostic_frames=None,
        text_features=None,
        diagnostic_relation=None,
    ):
        config = self.config
        device = next(self.model.parameters()).device

        def clock():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return time.perf_counter()

        start = clock()
        component_seconds = {
            "decoder_seconds": 0.0,
            "fk_relation_seconds": 0.0,
            "guard_teacher_seconds": 0.0,
        }
        text = text_features or self.encode([request.text], [request.sign_language])
        text_seconds = clock() - start
        duration_start = clock()
        language = torch.tensor([LANGUAGES[request.sign_language]], device=device)
        lengths, info = self.duration.predict(
            text.hidden, text.mask, language, config["data"]["max_generation_frames"]
        )
        source = "predicted"
        if request.duration_seconds is not None:
            frames = round(request.duration_seconds * 30) + 1
            if frames < 2 or frames > config["data"]["max_generation_frames"]:
                raise ValueError("unsupported_user_duration")
            lengths = torch.tensor([frames], device=device)
            source = "user"
        if diagnostic_frames is not None:
            lengths = torch.tensor([diagnostic_frames], device=device)
            source = "oracle_diagnostic"
        frames = int(lengths[0])
        if frames > config["data"]["max_generation_frames"]:
            raise ValueError("unsupported_motion_length")
        duration_seconds = clock() - duration_start
        noise_start = clock()
        latent_length = (frames + 1) // 2
        valid = lengths_mask((lengths + 1) // 2)
        rng = torch.Generator(device=device).manual_seed(request.seed)
        z = torch.randn(1, 4, latent_length, 128, generator=rng, device=device)
        r = (
            torch.randn(1, latent_length, 33, generator=rng, device=device)
            if config["relation"]["enabled"]
            else None
        )
        timestamps = torch.arange(frames, device=device, dtype=torch.float32)[None] / 30
        token_times = pool2(timestamps[..., None], lengths)[..., 0]
        conditions = dict(
            text=text.hidden,
            text_mask=text.mask,
            language=language,
            lengths=lengths,
            valid=valid,
            token_times=token_times,
        )
        nfe = nfe or config["sample"]["nfe"]
        cfg = config["sample"]["cfg"] if cfg is None else cfg
        noise_seconds = clock() - noise_start
        ode_start = clock()
        if config["run"]["variant"] == "diffusion":
            from cprflow.objectives.diffusion import CosineSchedule

            d = config["diffusion"]
            schedule = CosineSchedule(
                d["steps"], d["cosine_offset"], d["beta_min"], d["beta_max"], device
            )
            z, r, cost = ddim_sample(self.model, z, conditions, nfe, cfg, schedule)
        else:
            oracle_relation = None
            if diagnostic_relation is not None:
                if r is None:
                    raise ValueError("OracleR diagnostics require a relational checkpoint")
                oracle_relation = torch.nn.functional.interpolate(
                    diagnostic_relation.to(device).T[None],
                    size=latent_length,
                    mode="linear",
                    align_corners=True,
                ).transpose(1, 2)
            z, r, cost = integrate(
                self.model,
                z,
                r,
                conditions,
                nfe,
                cfg,
                solver or config["sample"]["solver"],
                oracle_relation,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        cost["ode_seconds"] = time.perf_counter() - ode_start

        def decode(latent):
            then = clock()
            value = self.motion_norm.inverse(
                self.ae.decode(self.latent_norm.inverse(latent), lengths)
            )
            component_seconds["decoder_seconds"] += clock() - then
            return value

        def relation_fn(physical):
            then = clock()
            value = self.relation_fn(physical, lengths)
            component_seconds["fk_relation_seconds"] += clock() - then
            return value

        guard_start = clock()

        guard_log = {"skip_reason": "disabled", "proposed": 0, "accepted": 0}
        if config["guard"]["enabled"]:
            from cprflow.sampling.guard import guarded_correction

            mode = config["guard"].get("mode", "semantic")
            if r is None or self.relation_fn is None:
                raise ValueError("Correction requires co-generated relations and geometry")
            if mode != "unrestricted" and (self.teacher is None or self.calibration is None):
                raise ValueError("Guard requires calibrated teacher")
            tc = (
                self.teacher.encode_text(text.hidden, text.mask, language)
                if self.teacher is not None
                else None
            )

            def teacher_fn(physical):
                then = clock()
                value = self.teacher(self.motion_norm(physical), lengths, timestamps)
                component_seconds["guard_teacher_seconds"] += clock() - then
                return value

            gc = config["guard"]
            z, guard_log = guarded_correction(
                z,
                r,
                decode,
                relation_fn,
                teacher_fn,
                tc,
                self.calibration,
                valid[:, None, :, None],
                gc["max_steps"],
                gc["max_backtracking"],
                gc["step_size"],
                gc["max_latent_delta_rms"],
                gc["energy_improvement_min"],
                mode=mode,
            )
        guard_seconds = clock() - guard_start if config["guard"]["enabled"] else 0.0
        if r is not None and self.relation_norm is not None:
            from cprflow.geometry.relations import relation_invalid_rates

            cost.update(relation_invalid_rates(self.relation_norm.inverse(r)))
        output = decode(z)[0, :frames].float()
        if not torch.isfinite(output).all():
            raise ValueError("nonfinite_generated_motion")
        from cprflow.geometry.rotations import rotation_6d_to_matrix

        _, deg = rotation_6d_to_matrix(output[:, :246].reshape(-1, 41, 6), True)
        metadata = {
            **self.metadata,
            **cost,
            **component_seconds,
            "text_seconds": text_seconds,
            "duration_prediction_seconds": duration_seconds,
            "noise_setup_seconds": noise_seconds,
            "guard_seconds": guard_seconds,
            "cuda_timing_synchronized": device.type == "cuda",
            "guard": guard_log,
            "duration_source": source,
            "duration_median_frames": float(info["median"][0]),
            "duration_preclamp_frames": float(info["preclamp"][0]),
            "duration_clipped": bool(info["clipped"][0]),
            "degenerate6d_rate": float(deg.float().mean()),
            "total_seconds": time.perf_counter() - start,
            "cfg": cfg,
            "profile": config["data"]["profile"],
            "oracle_relation_protocol": "normalized_time_linear_endpoint_clamped_path_v1"
            if diagnostic_relation is not None
            else None,
        }
        return GenerationResult(output.cpu().numpy(), 30.0, frames, source, request.seed, metadata)
