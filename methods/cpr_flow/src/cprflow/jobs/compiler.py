"""Compile the complete joint3 experiment DAG into immutable configs and argv commands."""

import copy
import sys
from pathlib import Path

import yaml

from cprflow.config import load_config, validate
from cprflow.data.cache import file_hash
from cprflow.jobs.schema import atomic_json, source_digest, validate_plan
from cprflow.types import DATASETS


def create_plan(
    root,
    campaign="joint3_standard",
    preset="standard",
    seeds=(17,),
    device="cuda",
    budget_gib=100,
    operator_waived=False,
    optimized=False,
):
    root = Path(root).resolve()
    state = root / "runs/jobs" / campaign
    target = state / "plan.json"
    if target.exists():
        raise FileExistsError(f"Campaign already exists: {target}; use status/run, not overwrite")
    if preset not in ("standard", "confirmatory"):
        raise ValueError("Unknown preset")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid seeds")
    cfg = load_config(root / "configs/base.yaml")
    out = root / "runs/joint3"
    data = root / "data"
    cache = data / "cache/joint3"
    cfg["run"].update(out_dir=str(out), device=device, synthetic=False)
    if operator_waived:
        cfg["run"].update(
            manual_gate_policy="operator_waived",
            policy_reference="User confirmed research access and explicitly requested no further manual verification; run training/evaluation on 2026-11-10",
        )
        cfg["data"].update(
            require_verified_feature_fps=False,
            source_profile="signspark_body126_hands90_or96_wristdup_face56",
        )
    for key, path in {
        "source_pose": data / "raw_pose",
        "manifests": data / "manifests/joint3",
        "processed": data / "processed",
        "cache": cache,
        "stats": data / "stats/joint3",
        "smplx_model": root / "assets/smplx/SMPLX_NEUTRAL.npz",
        "sources_lock": root / "sources.lock.json",
        "source_timing": root / "source_timing.json",
        "access_record": root / "access_record.json",
        "ae_checkpoint": out / "ae_joint3_s17/checkpoints/best_ema.pt",
        "duration_checkpoint": out / "duration_joint3_s17/checkpoints/best.pt",
    }.items():
        cfg["paths"][key] = str(path)
    if optimized:
        cfg["ae"]["execution"] = "masked_batch"
        cfg["geometry"]["fk_chunk_frames"] = 512
    cfg["data"].update(workers_per_rank=0, pin_memory=True)
    cfg["train"]["checkpoint_keep_recent"] = 1
    configs = {}
    hashes = {}
    jobs = []

    def write_config(name, value):
        validate(value, 1)
        path = state / "configs" / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(value, sort_keys=False))
        hashes[str(path)] = file_hash(path)
        configs[name] = str(path)
        return str(path)

    common = write_config("shared", cfg)

    def cmd(module, *args):
        return [sys.executable, "-m", module, *map(str, args)]

    def add(
        name,
        argv=(),
        deps=(),
        gates=(),
        outputs=(),
        gpu=0,
        slots=1,
        locks=(),
        disk=1,
        priority=0,
        resume=None,
        warm=None,
        attempts=3,
    ):
        job = {
            "id": name,
            "argv": list(argv),
            "deps": list(deps),
            "gates": list(gates),
            "outputs": list(outputs),
            "slots": slots,
            "gpu_gib": gpu,
            "disk_gib": disk,
            "locks": list(locks),
            "max_attempts": attempts,
            "retry_seconds": 60,
            "priority": priority,
        }
        if resume:
            job["resume_dir"] = str(resume)
        if warm:
            job["warm_start"] = str(warm)
        jobs.append(job)
        return name

    def exists(path, **equals):
        return {"path": str(path), **({"equals": equals} if equals else {})}

    def gate(kind, config=common, **extra):
        return {"kind": kind, "config": config, **extra}

    def cli(module, config=common, *args):
        return cmd("cprflow.cli." + module, "--config", config, *args)

    def action(name, *args):
        return cmd("cprflow.jobs.tasks", name, *args)

    env = add(
        "environment",
        action("environment", "--config", common, "--output", state / "environment.json"),
        outputs=[exists(state / "environment.json", status="ready")],
        priority=100,
    )
    audit = add(
        "sources.audit", cli("audit_sources"), deps=[env], locks=["sources-lock"], priority=99
    )
    access = add("gate.access", deps=[audit], gates=[gate("access")], priority=99)
    download = add(
        "data.download", cli("download"), deps=[access], locks=["raw-pose"], disk=16, priority=98
    )
    imported = add(
        "data.import",
        cli("prepare", common, "--phase", "import"),
        deps=[download],
        outputs=[exists(Path(cfg["paths"]["manifests"]) / "source_import_report.json")],
        locks=["manifests"],
        priority=97,
    )
    timing_parent = imported
    if operator_waived:
        timing_parent = add(
            "data.resolve_timing",
            action(
                "resolve-timing", "--config", common, "--output", state / "timing_resolution.json"
            ),
            deps=[imported],
            outputs=[exists(state / "timing_resolution.json")],
            priority=96,
        )
    timing = add("gate.timing", deps=[timing_parent], gates=[gate("timing")], priority=96)
    canonical = add(
        "data.canonicalize",
        cli("prepare", common, "--phase", "canonicalize"),
        deps=[timing],
        outputs=[
            exists(Path(cfg["paths"]["manifests"]) / "preprocessing_report.json", synthetic=False)
        ],
        locks=["manifests"],
        disk=24,
        priority=95,
    )
    render = canonical
    if not operator_waived:
        render = add(
            "data.geometry_audit",
            cli("prepare", common, "--phase", "audit-geometry", "--samples-per-dataset", "64"),
            deps=[canonical],
            outputs=[exists(out / "geometry_audit/report.json", synthetic=False)],
            slots=2,
            priority=94,
        )
    text = add(
        "cache.text",
        cli("cache", common, "--kind", "text", "--splits", "train", "dev", "test"),
        deps=[canonical],
        gpu=4,
        disk=12,
        locks=["text-cache"],
        priority=93,
    )
    g0 = add("gate.geometry", deps=[render], gates=[gate("geometry")], priority=94)
    stats = add(
        "stats.motion_relation",
        cli("fit_stats", common, "--kind", "motion-relation", "--split", "train"),
        gpu=4,
        deps=[g0],
        locks=["stats-index"],
        outputs=[exists(Path(cfg["paths"]["stats"]) / "index.json")],
        priority=92,
    )
    geometry = add(
        "cache.geometry",
        cli("cache", common, "--kind", "geometry", "--splits", "train", "dev", "test"),
        deps=[stats],
        gpu=4,
        disk=12,
        locks=["geometry-cache"],
        priority=91,
    )

    def run_config(template, run_id, seed=17):
        source = load_config(root / f"configs/runs/{template}.yaml")
        value = copy.deepcopy(cfg)
        for key in ("run", "train", "relation", "loss", "data", "guard"):
            # Resolve template values against base, then retain shared paths/runtime settings.
            value[key] = copy.deepcopy(source[key])
        value["run"].update(
            id=run_id,
            out_dir=str(out),
            device=device,
            synthetic=False,
            manual_gate_policy=cfg["run"]["manual_gate_policy"],
            policy_reference=cfg["run"]["policy_reference"],
        )
        value["data"]["require_verified_feature_fps"] = cfg["data"]["require_verified_feature_fps"]
        value["data"]["source_profile"] = cfg["data"]["source_profile"]
        value["data"].update(workers_per_rank=0, pin_memory=True)
        value["seed"] = seed
        value["train"]["checkpoint_keep_recent"] = 1
        if optimized:
            micro = (
                8 if value["run"]["task"] == "ae" else (4 if value["run"]["task"] == "flow" else 1)
            )
            value["train"].update(
                microbatch_per_rank=micro, accumulate=value["train"]["effective_batch"] // micro
            )
        return value

    def train_job(name, value, deps, warm=None, until=None):
        config = write_config(name, value)
        run_dir = out / value["run"]["id"]
        argv = cli("train", config, "--task", value["run"]["task"])
        if until is not None:
            argv += ["--until", str(until)]
        if until is None:
            expected = [
                exists(run_dir / "training_complete.json", synthetic=False),
                exists(run_dir / "checkpoints/best_ema.pt"),
            ]
        else:
            expected = [
                {
                    "path": str(run_dir / f"checkpoints/step_{until}_ema.pt.json"),
                    "minimum": {"successful_optimizer_step": until},
                }
            ]
        job = add(
            name,
            argv,
            deps=deps,
            gates=[gate("geometry", config)],
            outputs=expected,
            gpu=(
                16
                if optimized and value["run"]["task"] == "flow"
                else (12 if value["run"]["task"] == "teacher" else 8)
            ),
            locks=["run:" + value["run"]["id"]],
            resume=run_dir / "checkpoints",
            warm=warm,
            priority=50,
        )
        return job, config, run_dir

    duration, duration_cfg, _ = train_job(
        "train.duration", run_config("duration_joint3", "duration_joint3_s17"), [stats, text]
    )
    smoke_ae_value = run_config("smoke", "smoke_real48_ae")
    smoke_ae_value["train"]["keep_steps"] = [500, 1000]
    smoke_ae, smoke_ae_cfg, smoke_ae_dir = train_job(
        "smoke.ae", smoke_ae_value, [geometry], until=None
    )
    smoke_flow = run_config("smoke", "smoke_real48_cfm")
    smoke_flow["run"].update(task="flow", variant="plain_cfm")
    if optimized:
        smoke_flow["train"].update(microbatch_per_rank=4, accumulate=12)
    smoke_flow["train"].update(
        lr=0.0001, selection="macro_dev_hand_global_dtw_jpe", keep_steps=[500, 1000]
    )
    smoke_flow["paths"].update(
        ae_checkpoint=str(smoke_ae_dir / "checkpoints/best_ema.pt"),
        cache=str(data / "cache/smoke"),
        stats=str(data / "stats/smoke"),
    )
    sm_cfg = write_config("smoke_flow", smoke_flow)
    alias = add(
        "smoke.shared_indexes",
        action(
            "seed-smoke",
            "--config",
            sm_cfg,
            "--main-config",
            common,
            "--output",
            state / "smoke_shared.json",
        ),
        deps=[smoke_ae, text],
        outputs=[exists(state / "smoke_shared.json")],
        locks=["smoke-cache"],
    )
    sm_stats = add(
        "smoke.stats_latent",
        cli("fit_stats", sm_cfg, "--kind", "latent", "--split", "train"),
        gpu=4,
        deps=[alias],
        locks=["smoke-cache"],
    )
    sm_cache = add(
        "smoke.cache",
        cli("cache", sm_cfg, "--kind", "latent-relation", "--splits", "train", "dev"),
        deps=[sm_stats],
        gpu=4,
        locks=["smoke-cache"],
    )
    part, _, sm_dir = train_job("smoke.cfm_first500", smoke_flow, [sm_cache, duration], until=500)
    resumed, _, _ = train_job("smoke.cfm_resume1000", smoke_flow, [part])
    prompts_path = state / "smoke_prompts.jsonl"
    prompts_job = add(
        "smoke.prompts",
        action("prompts", "--config", common, "--output", prompts_path),
        deps=[canonical],
        outputs=[exists(prompts_path)],
    )
    sample = add(
        "smoke.sample",
        cmd(
            "cprflow.cli.sample",
            "--checkpoint",
            sm_dir / "checkpoints/best_ema.pt",
            "--text-file",
            prompts_path,
            "--output",
            out / "smoke_samples",
            "--device",
            device,
            "--nfe",
            "16",
        ),
        deps=[resumed, prompts_job],
        gpu=8,
        outputs=[exists(out / "smoke_samples/sample_report.json")],
    )
    smoke = add(
        "smoke.check",
        action(
            "smoke-check",
            "--ae-log",
            smoke_ae_dir / "train.jsonl",
            "--flow-log",
            sm_dir / "train.jsonl",
            "--sample-report",
            out / "smoke_samples/sample_report.json",
            "--output",
            state / "smoke_pass.json",
        ),
        deps=[sample],
        outputs=[exists(state / "smoke_pass.json", status="engineering_passed", synthetic=False)],
        attempts=1,
    )
    ae, ae_cfg, _ = train_job("train.ae", run_config("ae_joint3", "ae_joint3_s17"), [smoke])
    lat_stats = add(
        "stats.latent",
        cli("fit_stats", ae_cfg, "--kind", "latent", "--split", "train"),
        gpu=4,
        deps=[ae],
        locks=["stats-index"],
    )
    lat_cache = add(
        "cache.latent_relation",
        cli("cache", ae_cfg, "--kind", "latent-relation", "--splits", "train", "dev", "test"),
        deps=[lat_stats],
        gpu=4,
        disk=20,
        locks=["latent-cache"],
    )
    teacher, teacher_cfg, teacher_dir = train_job(
        "train.teacher", run_config("teacher_joint3", "teacher_joint3_s17"), [geometry, text, smoke]
    )
    teacher_path = teacher_dir / "checkpoints/best_ema.pt"
    models = []
    for seed in seeds:
        parent_id = f"cfm_joint3_parent_s{seed}"
        parent, parent_cfg, parent_dir = train_job(
            f"s{seed}.train.parent",
            run_config("baseline_joint3", parent_id, seed),
            [lat_cache, duration],
        )
        parent_checkpoint = parent_dir / "checkpoints/step_200000_ema.pt"
        profile = add(
            f"s{seed}.profile",
            cli("profile", parent_cfg, "--warmup", "100", "--updates", "200"),
            deps=[lat_cache, duration],
            gpu=16,
            slots=2,
            outputs=[exists(out / f"profiles/{parent_id}.json")],
            priority=60,
        )
        # Require the measured real-data profile before admitting the long parent run.
        next(j for j in jobs if j["id"] == parent)["deps"].append(profile)
        for key, template, run_id in [
            ("b1", "baseline_cont_joint3", f"cfm_joint3_s{seed}"),
            ("c0", "relation_joint3", f"relation_joint3_s{seed}"),
        ]:
            job, cpath, directory = train_job(
                f"s{seed}.train.{key}",
                run_config(template, run_id, seed),
                [parent],
                parent_checkpoint,
            )
            models.append((f"{key}_s{seed}", job, directory / "checkpoints/best_ema.pt", [], cpath))
            if key == "c0":
                c0_job, c0_dir, c0_cfg = job, directory, cpath
        full_cfg = run_config("full_joint3", f"full_joint3_s{seed}", seed)
        full_cfg["paths"].update(
            teacher_checkpoint=str(teacher_path),
            teacher_calibration=str(root / "teacher_calibration.json"),
        )
        calibration_cfg = write_config(f"calibration_s{seed}", full_cfg)
        cal = add(
            f"s{seed}.gate.calibration",
            deps=[teacher],
            gates=[gate("calibration", calibration_cfg)],
        )
        for key, template in [("full", "full_joint3"), ("full_control", "full_control_joint3")]:
            value = (
                full_cfg
                if key == "full"
                else run_config(template, f"full_control_joint3_s{seed}", seed)
            )
            deps = [c0_job, cal] if key == "full" else [c0_job]
            job, cpath, directory = train_job(
                f"s{seed}.train.{key}", value, deps, c0_dir / "checkpoints/best_ema.pt"
            )
            models.append((f"{key}_s{seed}", job, directory / "checkpoints/best_ema.pt", [], cpath))
        models.append(
            (
                f"c1_s{seed}",
                c0_job,
                c0_dir / "checkpoints/best_ema.pt",
                ["--guard-mode", "unrestricted"],
                c0_cfg,
            )
        )
        models.append(
            (
                f"c2_s{seed}",
                cal,
                c0_dir / "checkpoints/best_ema.pt",
                [
                    "--guard-mode",
                    "feature_only",
                    "--teacher-checkpoint",
                    str(teacher_path),
                    "--teacher-calibration",
                    str(root / "teacher_calibration.json"),
                ],
                c0_cfg,
            )
        )
        if preset == "confirmatory":
            for key, template in [("b2", "auxonly_joint3"), ("b3", "capacity_joint3")]:
                job, cpath, directory = train_job(
                    f"s{seed}.train.{key}",
                    run_config(template, f"{key}_joint3_s{seed}", seed),
                    [parent],
                    parent_checkpoint,
                )
                models.append(
                    (f"{key}_s{seed}", job, directory / "checkpoints/best_ema.pt", [], cpath)
                )
            diff, diff_cfg, diff_dir = train_job(
                f"s{seed}.train.diffusion_parent",
                run_config("diffusion_joint3", f"diffusion_joint3_parent_s{seed}", seed),
                [lat_cache, duration],
            )
            diff_cont, cp, dd = train_job(
                f"s{seed}.train.diffusion",
                run_config("diffusion_cont_joint3", f"diffusion_joint3_s{seed}", seed),
                [diff],
                diff_dir / "checkpoints/step_200000_ema.pt",
            )
            models.append((f"b0_s{seed}", diff_cont, dd / "checkpoints/best_ema.pt", [], cp))
            for short, dataset in [
                ("csl", "csl_daily"),
                ("how2sign", "how2sign"),
                ("phoenix", "phoenix2014t"),
            ]:
                job, cp, dd = train_job(
                    f"s{seed}.train.mono_{short}",
                    run_config(f"mono_{short}", f"cfm_{dataset}_sharedAE_s{seed}", seed),
                    [lat_cache, duration],
                )
                # Reports are still routed separately; no cross-language pooled BLEU.
                models.append(
                    (f"mono_{short}_s{seed}", job, dd / "checkpoints/best_ema.pt", [], cp)
                )

    def evaluations(label, dependency, checkpoint, extra, config):
        dev_jobs = []
        dev_paths = []
        if label.startswith("c2_"):
            dependency_list = [dependency, label.replace("c2_", "") + ".train.c0"]
        else:
            dependency_list = [dependency]
        evaluation_datasets = load_config(config)["data"]["datasets"]
        for dataset in evaluation_datasets:
            path = out / f"metrics/{label}/dev/{dataset}.json"
            name = f"eval.{label}.dev.{dataset}"
            dev_jobs.append(
                add(
                    name,
                    cmd(
                        "cprflow.cli.evaluate",
                        "--checkpoint",
                        checkpoint,
                        "--dataset",
                        dataset,
                        "--split",
                        "dev",
                        "--device",
                        device,
                        "--output",
                        path,
                        *extra,
                    ),
                    deps=dependency_list,
                    gpu=8,
                    outputs=[
                        exists(path, dataset=dataset, split="dev", oracle=False, synthetic=False)
                    ],
                )
            )
            dev_paths.append(path)
        macro = add(
            f"aggregate.{label}.dev",
            cmd(
                "cprflow.cli.aggregate",
                *dev_paths,
                "--output",
                out / f"metrics/{label}/dev/macro.json",
            ),
            deps=dev_jobs,
            outputs=[exists(out / f"metrics/{label}/dev/macro.json")],
        )
        inputs = state / f"releases/{label}.inputs.json"
        atomic_json(
            inputs,
            {
                label: {
                    "checkpoint": str(checkpoint),
                    "dev_reports": list(map(str, dev_paths)),
                    "expected_datasets": evaluation_datasets,
                    "require_clean_source": not operator_waived,
                    "sampler": {
                        "nfe": 16,
                        "cfg": 1.0,
                        "solver": "ddim" if label.startswith("b0") else "euler",
                    },
                }
            },
        )
        hashes[str(inputs)] = file_hash(inputs)
        release = state / f"releases/{label}.json"
        frozen = add(
            f"release.{label}",
            action("release", "--inputs", inputs, "--output", release),
            deps=[macro],
            gates=[gate("source_snapshot", root=str(root))]
            if operator_waived
            else [gate("clean_git")],
            outputs=[exists(release, protocol_frozen=True)],
        )
        tests = []
        test_paths = []
        for dataset in evaluation_datasets:
            path = out / f"metrics/{label}/test/{dataset}.json"
            tests.append(
                add(
                    f"eval.{label}.test.{dataset}",
                    cmd(
                        "cprflow.cli.evaluate",
                        "--checkpoint",
                        checkpoint,
                        "--dataset",
                        dataset,
                        "--split",
                        "test",
                        "--final-test",
                        "--device",
                        device,
                        "--output",
                        path,
                        *extra,
                    ),
                    deps=[frozen],
                    gates=[gate("source_snapshot", root=str(root))]
                    if operator_waived
                    else [gate("clean_git")],
                    gpu=8,
                    outputs=[
                        exists(path, dataset=dataset, split="test", oracle=False, synthetic=False)
                    ],
                )
            )
            test_paths.append(path)
            spec = root / "integrations/backtranslation.json"
            context = state / f"bt/{label}.{dataset}.context.json"
            result = out / f"metrics/{label}/test/{dataset}.backtranslation.json"
            atomic_json(
                context,
                {
                    "checkpoint": str(checkpoint),
                    "dataset": dataset,
                    "split": "test",
                    "manifest": str(Path(cfg["paths"]["manifests"]) / "test.jsonl"),
                    "output": str(result),
                },
            )
            hashes[str(context)] = file_hash(context)
            add(
                f"bt.{label}.{dataset}",
                action(
                    "external",
                    "--spec",
                    spec,
                    "--context",
                    context,
                    "--output",
                    state / f"bt/{label}.{dataset}.receipt.json",
                ),
                deps=[tests[-1]],
                gates=[gate("external", path=str(spec))],
                gpu=8,
                outputs=[exists(result)],
            )
        add(
            f"aggregate.{label}.test",
            cmd(
                "cprflow.cli.aggregate",
                *test_paths,
                "--output",
                out / f"metrics/{label}/test/macro.json",
            ),
            deps=tests,
            outputs=[exists(out / f"metrics/{label}/test/macro.json")],
        )

    for model in models:
        evaluations(*model)
    # Oracle diagnostics are dev-only, never a selection input or a main aggregate.
    for seed in seeds:
        checkpoint = out / f"relation_joint3_s{seed}/checkpoints/best_ema.pt"
        for kind in ("duration", "relation"):
            for dataset in DATASETS:
                path = out / f"diagnostics/s{seed}/{dataset}_oracle_{kind}.json"
                add(
                    f"oracle.s{seed}.{kind}.{dataset}",
                    cmd(
                        "cprflow.cli.evaluate",
                        "--checkpoint",
                        checkpoint,
                        "--dataset",
                        dataset,
                        "--split",
                        "dev",
                        f"--{kind}-source",
                        "oracle",
                        "--diagnostic-only",
                        "--device",
                        device,
                        "--output",
                        path,
                    ),
                    deps=[f"s{seed}.train.c0"],
                    gpu=8,
                    outputs=[exists(path, oracle=True, split="dev", synthetic=False)],
                )
    plan = {
        "version": 1,
        "id": campaign,
        "root": str(root),
        "state_dir": str(state),
        "global_lock": str(root / "runs/jobs/.global.lock"),
        "max_concurrent": 2,
        "poll_seconds": 10.0,
        "scope": preset,
        "seeds": list(seeds),
        "disk": {
            "min_free_gib": 15.0,
            "budget_gib": float(budget_gib),
            "scan_seconds": 60,
            "tracked_paths": [str(data), str(out)],
        },
        "environment": {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
            "CUDA_VISIBLE_DEVICES": "0",
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONUNBUFFERED": "1",
        },
        "jobs": jobs,
        "config_hashes": hashes,
        "source_digest": source_digest(root),
    }
    validate_plan(plan)
    atomic_json(target, plan)
    return target
