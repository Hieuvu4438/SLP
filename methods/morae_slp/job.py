"""Supervisor: SignRAE jobs, one process at a time, evaluated on DEV only.

    nohup setsid python -u -m methods.morae_slp.job [--plan v2] > <job dir>/supervisor.out 2>&1 &
    python -m methods.morae_slp.job [--plan v2] --status

Plan v1 (outputs/morae_slp/job; finished 2026-11-25, kept as the control):
  jepa   Sign-JEPA on the union of the three train splits
  ae     compact latent (d=64) on the union, AE checkpoint chosen on val
  diag   gate-G1 diagnostics on Phoenix (spectrum, reconstruction ceiling, gloss probe)
  flow   latent flow on Phoenix train + scoring on Phoenix dev (scripts/run.py --split val)
  sweep  guidance grid of the trained flow on 256 dev clips

Plan v2 (outputs/morae_slp/job_v2; PLAN_V2.md): the v1 flow memorised the 7k Phoenix sentences
(dev flow loss with text > without text), so the generator is pre-trained on the union of the
three train splits (56k sentences, mBART fine-tuned, text-token dropout, best-val selection) and
fine-tuned per dataset; sweeps add pose-space ensembles (point estimate of E[pose | text]).
Stages are skipped once done; an interrupted stage restarts from scratch.

A file lock allows a single supervisor, and stages run strictly one after another, so this job
never has more than one training process. Before each stage it only waits until the GPU has
MIN_FREE_MIB free (out-of-memory guard); other jobs on the GPU are not otherwise considered.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from slp.paths import PROJECT_ROOT

JOB = PROJECT_ROOT / "outputs" / "morae_slp" / "job"  # set per plan in main()
PRE = PROJECT_ROOT / "outputs" / "morae_slp" / "pretrain"
DIAG = PROJECT_ROOT / "outputs" / "morae_slp" / "diagnostics"
UNION = "phoenix_smplx,csl_smplx,how2sign_smplx_soke"
EXPERIMENT = "morae_phoenix_main"
MIN_FREE_MIB = 14 * 1024
PY = sys.executable


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(JOB / "job.log", "a") as f:
        f.write(line + "\n")


V2 = {  # dataset -> (experiment name, fine-tune updates)
    "phoenix_smplx": ("morae_v2_phoenix", 20000),
    "csl_smplx": ("morae_v2_csl", 30000),
    "how2sign_smplx_soke": ("morae_v2_how2sign", 40000),
}
V2_COMMON = ["method.text.finetune=true", "method.text.lr_scale=0.1", "method.flow.token_drop=0.1",
             "method.flow.select=flow"]
# Fine-tune stages run next to other users' jobs: half the micro-batch halves the activation peak
# (same optimiser batch of 64; only the frame weighting inside a micro-batch differs slightly).
V2_FT = ["method.flow.microbatch=16"]
OOM_RETRIES = 4


def stages_v2(workers: int) -> list[tuple[str, callable]]:
    ae = PRE / "ae_union_jepa_d64" / "ae.pt"
    union = PRE / "flow_union_v2"
    w = f"method.workers={workers}"
    out = [("flow_union", lambda: ([PY, "-u", "-m", "methods.morae_slp.pretrain", "--stage", "flow",
                                    "--datasets", UNION, "--out", str(union), "--init-ae", str(ae),
                                    w, *V2_COMMON, "method.flow.updates=100000",
                                    "method.flow.snapshot_steps=[40000,70000]"], union)),
           ("zeroshot_phoenix", lambda: ([PY, "-u", "-m", "methods.morae_slp.sweep", "--run",
                                          str(union), "--data", "phoenix_smplx", "--cfg", "1,2.5",
                                          "--ensemble", "1,8", "--limit", "256"], None))]
    for data, (name, updates) in V2.items():
        short = name.removeprefix("morae_v2_")
        out.append((f"ft_{short}", lambda data=data, name=name, updates=updates: (
            [PY, "-u", "scripts/run.py", "--method", "morae_slp", "--data", data, "--name", name,
             "--seeds", "0", "--split", "val", w, *V2_COMMON, *V2_FT, f"method.init_ae={ae}",
             f"method.init_flow={union / 'checkpoints' / 'model.pt'}",
             "method.latent.recalibrate=false", f"method.flow.updates={updates}",
             "method.flow.lr=1.0e-4", "method.flow.warmup_updates=500",
             "method.flow.eval_every=1000", "method.flow.ema_decay=0.999",
             "method.sampler.cfg_scale=2.0"], None)))
        out.append((f"sweep_{short}", lambda name=name: (
            [PY, "-u", "-m", "methods.morae_slp.sweep", "--run", str(latest_run(name)),
             "--cfg", "1,1.5,2.5,3.5", "--ensemble", "1,8", "--limit", "256"], None)))
    out.insert(4, ("export_phoenix", lambda: (
        [PY, "-u", "-m", "methods.morae_slp.sweep", "--run", str(latest_run("morae_v2_phoenix")),
         "--cfg", "1.5,3.5", "--ensemble", "8", "--export"], None)))  # all dev clips, K = 8
    out.insert(5, ("frontier_phoenix", lambda: (
        # Matched-speed comparison with the replica's shrink-to-mean frontier (dev, CPU).
        [PY, "-u", "-m", "methods.signmflow_plus.analysis.score", "--data", "phoenix_smplx",
         "--reference", str(PROJECT_ROOT / "outputs/runs/signmflow_v2_main_phoenix_smplx/"
                            "signmflow/seed0_20261116-185137-965824/predictions.npz"),
         "--arm", f"v2_single_w2={latest_run('morae_v2_phoenix') / 'predictions.npz'}",
         "--arm", f"v2_ens8_w1.5={latest_run('morae_v2_phoenix') / 'sweep/rk45_w1.5_tau1_ens8/predictions.npz'}",
         "--arm", f"v2_ens8_w3.5={latest_run('morae_v2_phoenix') / 'sweep/rk45_w3.5_tau1_ens8/predictions.npz'}",
         "--shrink", "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7",
         "--cache-dir", str(PROJECT_ROOT / "outputs/morae_slp/score_cache"),
         "--out", str(PROJECT_ROOT / "outputs/morae_slp/diagnostics/frontier_v2_phoenix_dev519.json")],
        None)))
    # Control: the same v2 recipe on Phoenix alone (no union pre-training of the flow).
    out.append(("ctrl_phoenix_only", lambda: (
        [PY, "-u", "scripts/run.py", "--method", "morae_slp", "--data", "phoenix_smplx", "--name",
         "morae_v2ctrl_phoenix_only", "--seeds", "0", "--split", "val", w, *V2_COMMON, *V2_FT,
         f"method.init_ae={ae}", "method.flow.updates=30000", "method.sampler.cfg_scale=2.0"],
        None)))
    out.append(("sweep_ctrl", lambda: (
        [PY, "-u", "-m", "methods.morae_slp.sweep", "--run",
         str(latest_run("morae_v2ctrl_phoenix_only")), "--cfg", "1,1.5,2.5", "--ensemble", "1,8",
         "--limit", "256"], None)))
    # The Phoenix-only recipe beat the union-pretrained flow (DTW), so the follow-up queue uses it.
    ctrl = "morae_v2ctrl_phoenix_only"
    out.append(("sweep_tau_phoenix", lambda: (
        [PY, "-u", "-m", "methods.morae_slp.sweep", "--run", str(latest_run(ctrl)),
         "--cfg", "1,1.5,2.5", "--temperature", "0.3,0.5,0.7", "--ensemble", "1", "--limit", "256"],
        None)))
    out.append(("export_ctrl", lambda: (
        [PY, "-u", "-m", "methods.morae_slp.sweep", "--run", str(latest_run(ctrl)),
         "--cfg", "1.5", "--temperature", "0.7", "--ensemble", "1", "--export"], None)))  # all dev
    out.append(("frontier_ctrl", lambda: (
        [PY, "-u", "-m", "methods.signmflow_plus.analysis.score", "--data", "phoenix_smplx",
         "--reference", str(PROJECT_ROOT / "outputs/runs/signmflow_v2_main_phoenix_smplx/"
                            "signmflow/seed0_20261116-185137-965824/predictions.npz"),
         "--arm", f"ctrl_w1.5_t0.7={latest_run(ctrl) / 'sweep/rk45_w1.5_tau0.7/predictions.npz'}",
         "--shrink", "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7",
         "--cache-dir", str(PROJECT_ROOT / "outputs/morae_slp/score_cache"),
         "--out", str(PROJECT_ROOT / "outputs/morae_slp/diagnostics/frontier_ctrl_phoenix_dev519.json")],
        None)))
    for data, short, updates in (("csl_smplx", "csl", 30000), ("how2sign_smplx_soke", "how2sign", 40000)):
        name = f"morae_v2ctrl_{short}_only"
        out.append((f"ctrl_{short}_only", lambda data=data, name=name, updates=updates: (
            [PY, "-u", "scripts/run.py", "--method", "morae_slp", "--data", data, "--name", name,
             "--seeds", "0", "--split", "val", w, *V2_COMMON, *V2_FT, f"method.init_ae={ae}",
             f"method.flow.updates={updates}", "method.sampler.cfg_scale=2.0"], None)))
        out.append((f"sweep_ctrl_{short}", lambda name=name: (
            [PY, "-u", "-m", "methods.morae_slp.sweep", "--run", str(latest_run(name)),
             "--cfg", "1.5,2.5", "--temperature", "0.5,0.7,1", "--ensemble", "1,8",
             "--limit", "256"], None)))
    # Why is CSL / How2Sign weaker than Phoenix? (cheap, dev only): reconstruction ceiling of the
    # shared latent per dataset, and the effect of the predicted durations (oracle lengths).
    for data, short in (("csl_smplx", "csl"), ("how2sign_smplx_soke", "how2sign")):
        out.append((f"diag_recon_{short}", lambda data=data, short=short: (
            [PY, "-u", "-m", "methods.morae_slp.diagnostics", "--data", data, "--ae", str(ae),
             "--recon", "400", "--out", str(DIAG / f"recon_{short}_dev400.json")], None)))
        out.append((f"sweep_gtlen_{short}", lambda short=short: (
            [PY, "-u", "-m", "methods.morae_slp.sweep", "--run",
             str(latest_run(f"morae_v2ctrl_{short}_only")), "--cfg", "1.5", "--temperature", "0.7",
             "--ensemble", "1", "--lengths", "gt", "--limit", "256"], None)))
    return out


# ------------------------------------------------------------------------------------------------
# Plan v3 (PLAN_V3.md §4): defined here, started only on request (`--plan v3`). Phoenix first.
# Baseline = v2 Phoenix-only recipe (morae_v2ctrl_phoenix_only), sampler w = 1.5, tau = 0.7.
REPLICA = {
    "phoenix_smplx": "outputs/runs/signmflow_v2_main_phoenix_smplx/signmflow/"
                     "seed0_20261116-185137-965824/predictions.npz",
    "csl_smplx": "outputs/runs/signmflow_v2_main_csl_smplx/signmflow/"
                 "seed0_20261116-185337-969580/predictions.npz",
    "how2sign_smplx_soke": "outputs/runs/signmflow_v2_main_how2sign_smplx_soke/signmflow/"
                           "seed0_20261118-125753-011324/predictions.npz",
}
EXCLUDED_V3 = {"p1b_fk_target", "p1b_fk_target_sweep",  # C1 was neutral on Phoenix (2026-11-27)
               "csl_fk", "csl_fk_sweep", "csl_fk_euler_dev",  # C2 did not help Phoenix
               "p1c_lr03", "p1c_lr03_sweep",

               "p0_gloss_phoenix", "p0_gloss_gate_phoenix", "p2_gloss_phoenix",
               "p2_gloss_phoenix_sweep", "p2_retrieval_phoenix", "p2_retrieval_phoenix_sweep",
               "p0_gloss_csl", "p0_gloss_gate_csl"}
V3_SAMPLER = ["method.sampler.cfg_scale=1.5", "method.sampler.temperature=0.7"]


def stages_v3(workers: int) -> list[tuple[str, callable]]:
    ae = PRE / "ae_union_jepa_d64" / "ae.pt"
    w = f"method.workers={workers}"
    base = [w, *V2_COMMON, *V2_FT, f"method.init_ae={ae}", "method.flow.updates=30000",
            *V3_SAMPLER]
    common_v3 = [w, *V2_COMMON, *V2_FT, f"method.init_ae={ae}", *V3_SAMPLER]
    gloss_dir = PROJECT_ROOT / "outputs" / "morae_slp" / "gloss"
    ctrl = "morae_v2ctrl_phoenix_only"

    def run(name, data, *extra, seed=0):
        return lambda: ([PY, "-u", "scripts/run.py", "--method", "morae_slp", "--data", data,
                         "--name", name, "--seeds", str(seed), "--split", "val", *base, *extra],
                        None)

    def sweep(name, *args):
        return lambda: ([PY, "-u", "-m", "methods.morae_slp.sweep", "--run",
                         str(latest_run(name)), *args], None)

    def frontier(data, out, *arms, extra=(), clips="confirm"):
        subset = ["--limit", "256"] if clips != "all" else []
        return lambda: ([PY, "-u", "-m", "methods.morae_slp.frontier", "--data", data,
                         "--reference", str(PROJECT_ROOT / REPLICA[data]), "--clips", clips,
                         *subset, *extra,
                         *[x for name, path in arms for x in ("--arm", f"{name}={path()}")],
                         "--out", str(DIAG / out)], None)

    tune = ["--limit", "256"]
    conf = ["--limit", "256", "--confirm", "--export"]
    out = [
        # Real-data smoke test of every v3 option (tiny sizes, CPU) before anything is trained.
        ("p0_smoke_phoenix", lambda: ([PY, "-u", "-m", "methods.morae_slp.smoke", "--data",
                                       "phoenix_smplx", "--ae", str(ae), "--device", "cpu",
                                       "--out", str(JOB / "smoke_phoenix")], None)),
        ("p0_smoke_how2sign", lambda: ([PY, "-u", "-m", "methods.morae_slp.smoke", "--data",
                                        "how2sign_smplx_soke", "--ae", str(ae), "--device", "cpu",
                                        "--out", str(JOB / "smoke_how2sign")], None)),
        # P0.0 replica vs the dataset-only v2 runs on the SAME confirmation clips (+ frontier).
        ("p0_frontier_csl", frontier("csl_smplx", "p0_frontier_csl_confirm.json",
                                     ("v2_csl_only", lambda: latest_run("morae_v2ctrl_csl_only")
                                      / "predictions.npz"), extra=("--cluster-sentences",))),
        ("p0_frontier_how2sign", frontier("how2sign_smplx_soke", "p0_frontier_h2s_confirm.json",
                                          ("v2_h2s_only",
                                           lambda: latest_run("morae_v2ctrl_how2sign_only")
                                           / "predictions.npz"))),
        # P0.1 decoders (C5) on the Phoenix baseline: tune grid, then confirmation exports.
        ("p0_decoders_tune", sweep(ctrl, "--cfg", "1.5", "--temperature", "0.7",
                                   "--ensemble", "8,16", "--decoder", "mean,medoid,dba", *tune)),
        ("p0_decoders_confirm", sweep(ctrl, "--cfg", "1.5", "--temperature", "0.7",
                                      "--ensemble", "8", "--decoder", "medoid,dba", *conf)),
        ("p0_frontier_decoders", frontier(
            "phoenix_smplx", "p0_frontier_phoenix_decoders.json",
            ("medoid8", lambda: latest_run(ctrl) / "sweep/rk45_w1.5_tau0.7_ens8_medoid_confirm/predictions.npz"),
            ("dba8", lambda: latest_run(ctrl) / "sweep/rk45_w1.5_tau0.7_ens8_dba_confirm/predictions.npz"))),
        # P0.2 duration sensitivity (analysis only).
        ("p0_length", sweep(ctrl, "--cfg", "1.5", "--temperature", "0.7",
                            "--length-scale", "0.85,0.9,1.1", *tune)),
        # P0.3 gloss recognizers (train hold-out selection) and their usefulness gate on replica
        # predictions (own vs length-matched swapped sentence).
        ("p0_gloss_phoenix", lambda: ([PY, "-u", "-m", "methods.morae_slp.gloss", "train",
                                       "--data", "phoenix_smplx", "--ae", str(ae),
                                       "--out", str(gloss_dir / "phoenix")], None)),
        ("p0_gloss_gate_phoenix", lambda: ([PY, "-u", "-m", "methods.morae_slp.gloss", "check",
                                            "--data", "phoenix_smplx", "--ae", str(ae),
                                            "--recognizer", str(gloss_dir / "phoenix/recognizer.pt"),
                                            "--predictions", str(PROJECT_ROOT / REPLICA["phoenix_smplx"]),
                                            "--out", str(DIAG / "p0_gloss_gate_phoenix.json")], None)),
        ("p0_gloss_csl", lambda: ([PY, "-u", "-m", "methods.morae_slp.gloss", "train",
                                   "--data", "csl_smplx", "--ae", str(ae),
                                   "--out", str(gloss_dir / "csl")], None)),
        ("p0_gloss_gate_csl", lambda: ([PY, "-u", "-m", "methods.morae_slp.gloss", "check",
                                        "--data", "csl_smplx", "--ae", str(ae),
                                        "--recognizer", str(gloss_dir / "csl/recognizer.pt"),
                                        "--predictions", str(PROJECT_ROOT / REPLICA["csl_smplx"]),
                                        "--out", str(DIAG / "p0_gloss_gate_csl.json")], None)),
        # P0.5 seed noise of the baseline.
        ("p0_seed1", run("morae_v3_phoenix_base_seed1", "phoenix_smplx", seed=1)),
        # Seed noise of the chosen sampler (Euler 5 steps, no guidance) on the whole dev split.
        ("p0_seed1_sweep", sweep("morae_v3_phoenix_base_seed1", "--solver", "euler", "--steps", "5",
                                 "--cfg", "1", "--temperature", "1.0,0.7")),
        # P1.A C1: sampler alone (velocity model, 5 steps: Euler == replan_implied for a velocity
        # model, so only Euler and the re-anchored rule), then the x1 target.
        ("p1a_sampler_only", sweep(ctrl, "--solver", "euler,replan_anchored",
                                   "--cfg", "1.5", "--temperature", "0.7,1", *tune)),
        ("p1a_target", run("morae_v3_phoenix_target", "phoenix_smplx",
                           "method.flow.prediction=target", "method.sampler.solver=replan_implied",
                           "method.sampler.steps=5")),
        ("p1a_target_sweep", sweep("morae_v3_phoenix_target", "--solver",
                                   "replan_implied,replan_anchored", "--cfg", "1,1.5,2.5",
                                   "--temperature", "0.7,1", *tune)),
        # P1.B C2: metric-frame FK loss (2 x 2 with the parameterisation; hands weighted).
        ("p1b_fk_velocity", run("morae_v3_phoenix_fk_velocity", "phoenix_smplx",
                                "method.flow.fk_weight_hand=1.0", "method.flow.fk_weight_body=0.25")),
        ("p1b_fk_target", run("morae_v3_phoenix_fk_target", "phoenix_smplx",
                              "method.flow.prediction=target", "method.sampler.solver=replan_implied",
                              "method.flow.fk_weight_hand=1.0", "method.flow.fk_weight_body=0.25")),
        ("p1b_fk_velocity_sweep", sweep("morae_v3_phoenix_fk_velocity", "--cfg", "1.5",
                                        "--temperature", "0.7,1", *tune)),
        ("p1b_fk_target_sweep", sweep("morae_v3_phoenix_fk_target", "--solver", "replan_implied",
                                      "--cfg", "1,1.5,2.5", "--temperature", "0.7,1", *tune)),
        # P1.C mBART learning rate.
        ("p1c_lr03", run("morae_v3_phoenix_lr03", "phoenix_smplx", "method.text.lr_scale=0.3")),
        ("p1c_lr03_sweep", sweep("morae_v3_phoenix_lr03", "--cfg", "1.5", "--temperature",
                                 "0.7,1", *tune)),
        # P2 content supervision (separate comparison classes): gloss consistency, retrieval.
        ("p2_gloss_phoenix", run("morae_v3_phoenix_gloss", "phoenix_smplx",
                                 f"method.gloss.recognizer={gloss_dir / 'phoenix/recognizer.pt'}",
                                 "method.flow.gloss_weight=0.1")),
        ("p2_gloss_phoenix_sweep", sweep("morae_v3_phoenix_gloss", "--cfg", "1.5",
                                         "--temperature", "0.7,1", *tune)),
        ("p2_retrieval_phoenix", run("morae_v3_phoenix_retrieval", "phoenix_smplx",
                                     "method.retrieval.enabled=true")),
        ("p2_retrieval_phoenix_sweep", sweep("morae_v3_phoenix_retrieval", "--cfg", "1.5",
                                             "--temperature", "0.7,1", *tune)),
    ]
    # Order: the primary Phoenix question (C1) comes as early as possible; seed noise, FK, mBART lr
    # and content supervision follow; CPU-only / other-dataset diagnostics run last.
    order = ["p0_smoke_phoenix", "p0_smoke_how2sign", "p0_frontier_csl", "p0_decoders_tune",
             "p0_decoders_confirm", "p0_frontier_decoders", "p0_length", "p1a_sampler_only",
             "p1a_target", "p1a_target_sweep", "csl_euler_dev", "h2s_euler_dev", "phoenix_euler_dev", "p0_gloss_phoenix",
             "p0_gloss_gate_phoenix", "p1b_fk_velocity", "p1b_fk_velocity_sweep",
             "csl_fk", "csl_fk_sweep", "csl_fk_euler_dev", "p0_seed1", "p0_seed1_sweep",
             "p1b_fk_target", "p1b_fk_target_sweep", "p1c_lr03", "p1c_lr03_sweep",
             "p2_gloss_phoenix", "p2_gloss_phoenix_sweep", "p2_retrieval_phoenix",
             "p2_retrieval_phoenix_sweep", "p0_gloss_csl", "p0_gloss_gate_csl",
             "test_phoenix", "test_csl", "test_how2sign", "csl_euler_frontier",
             "p0_frontier_how2sign"]
    # CSL-Daily (refined ~50k flow, boost plan): Euler 5 steps without CFG (NFE 5) on the whole dev
    # split, then the matched-speed comparison with the replica frontier.
    csl = "morae_v2ctrl_csl_60k_cont"
    out += [("csl_euler_dev", lambda: (
        [PY, "-u", "-m", "methods.morae_slp.sweep", "--run", str(latest_run(csl)), "--solver",
         "euler", "--steps", "5", "--cfg", "1", "--temperature", "0.7", "--export"], None)),
        ("csl_euler_frontier", frontier(
            "csl_smplx", "csl_euler_frontier_dev.json",
            ("csl_euler_w1_t0.7", lambda: latest_run(csl) / "sweep/euler5_w1_tau0.7/predictions.npz"),
            extra=("--cluster-sentences",), clips="all"))]
    # CSL-Daily C2: continue the refined flow (velocity parameterisation) with the metric-frame FK
    # loss, hands weighted; 20k updates warm-started from its best EMA checkpoint.
    def csl_fk():
        run = latest_run("morae_v2ctrl_csl_60k_cont")
        ck = run / "checkpoints"
        base = [a for a in common_v3 if not a.startswith("method.init_ae=")]
        return ([PY, "-u", "scripts/run.py", "--method", "morae_slp", "--data", "csl_smplx",
                 "--name", "morae_v3_csl_fk", "--seeds", "0", "--split", "val", *base,
                 f"method.init_ae={ck / 'ae.pt'}", "method.latent.recalibrate=false",
                 f"method.init_flow={ck / 'model.pt'}", "method.flow.updates=20000",
                 "method.flow.lr=1.0e-4", "method.flow.warmup_updates=500",
                 "method.flow.fk_weight_hand=1.0", "method.flow.fk_weight_body=0.25",
                 "method.flow.snapshot_steps=[10000]"], None)
    out += [("csl_fk", csl_fk),
            ("csl_fk_sweep", sweep("morae_v3_csl_fk", "--cfg", "1,1.5", "--temperature", "0.7,1",
                                   *tune)),
            ("csl_fk_euler_dev", lambda: (
                [PY, "-u", "-m", "methods.morae_slp.sweep", "--run",
                 str(latest_run("morae_v3_csl_fk")), "--solver", "euler", "--steps", "5",
                 "--cfg", "1", "--temperature", "0.7"], None))]

    # Phoenix: Euler 5 steps without guidance (NFE 5) on the whole dev split, two temperatures.
    out.append(("phoenix_euler_dev", lambda: (
        [PY, "-u", "-m", "methods.morae_slp.sweep", "--run",
         str(latest_run("morae_v2ctrl_phoenix_only")), "--solver", "euler", "--steps", "5",
         "--cfg", "1", "--temperature", "0.7,1.0"], None)))

    # How2Sign: Euler 5 steps without guidance on the whole dev split (tau = 1: tau 0.7 freezes it).
    out.append(("h2s_euler_dev", lambda: (
        [PY, "-u", "-m", "methods.morae_slp.sweep", "--run",
         str(latest_run("morae_v2ctrl_how2sign_only")), "--solver", "euler", "--steps", "5",
         "--cfg", "1", "--temperature", "1.0"], None)))

    # TEST evaluation, one run per dataset: the configuration with the best whole-dev result
    # (select_best.py: single-sample, gloss-free; JPE-hand primary; rejects near-static outputs).
    # run.py retrains the chosen recipe and evaluates on test; --allow-dirty is added only when
    # the git tree is dirty (then the number is not reportable). Plan JSON in diagnostics/.
    def test_stage(data):
        def make():
            from . import select_best

            if select_best.settle_test_log():
                log("committed the test access log of the previous test run")
            chosen = select_best.plan(data)
            (DIAG / f"test_plan_{data}.json").write_text(json.dumps(chosen, indent=1))
            log(f"test {data}: {chosen['chosen']} (dirty tree: {chosen['dirty_tree']})")
            return chosen["argv"], None
        return make

    for data, short in (("phoenix_smplx", "phoenix"), ("csl_smplx", "csl"),
                        ("how2sign_smplx_soke", "how2sign")):
        out.append((f"test_{short}", test_stage(data)))
    by_name = dict(out)
    if sorted(order) != sorted(by_name):
        raise RuntimeError(f"v3 stage order out of sync: {set(order) ^ set(by_name)}")
    # C3 (gloss supervision) and C4 (retrieval) belong to other comparison classes (PLAN_V3 §6):
    # implemented and tested, but not part of the default queue. Remove a name from EXCLUDED to
    # run it again.
    return [(name, by_name[name]) for name in order if name not in EXCLUDED_V3]


# ------------------------------------------------------------------------------------------------
# Plan "boost" (CSL-Daily and How2Sign): the dataset-only flows were trained for only 104 / 83 epochs
# (Phoenix 271, replica 300) and their validation loss was still falling at the last step. Run the
# cheap diagnostics (latent reconstruction ceiling, oracle durations), then the same recipe with
# 2x the updates, each followed by a compact single-sample sweep. Runs BEFORE the rest of plan v3.
def stages_boost(workers: int) -> list[tuple[str, callable]]:
    ae = PRE / "ae_union_jepa_d64" / "ae.pt"
    common = [f"method.workers={workers}", *V2_COMMON, *V2_FT, f"method.init_ae={ae}",
              *V3_SAMPLER, "method.flow.eval_every=2000"]

    def train(name, data, updates):
        # EMA snapshots every 20k updates: a killed run can be warm-started from the last one.
        snaps = ",".join(str(k) for k in range(20000, updates, 20000))
        return lambda: ([PY, "-u", "scripts/run.py", "--method", "morae_slp", "--data", data,
                         "--name", name, "--seeds", "0", "--split", "val", *common,
                         f"method.flow.updates={updates}", f"method.flow.snapshot_steps=[{snaps}]"],
                        None)

    def warm_start(name, data, updates, source):
        """Continue a killed run from its best EMA checkpoint (same latent calibration, fresh
        optimiser and a short warm-up; ``updates`` further updates)."""
        def make():
            runs = sorted((PROJECT_ROOT / "outputs" / "runs" / source / "morae_slp")
                          .glob("seed0_*"))
            runs = [r for r in runs if (r / "checkpoints" / "model.pt").is_file()]
            ck = runs[-1] / "checkpoints"
            snaps = ",".join(str(k) for k in range(20000, updates, 20000))
            base = [a for a in common if not a.startswith("method.init_ae=")]
            return ([PY, "-u", "scripts/run.py", "--method", "morae_slp", "--data", data,
                     "--name", name, "--seeds", "0", "--split", "val", *base,
                     f"method.init_ae={ck / 'ae.pt'}", "method.latent.recalibrate=false",
                     f"method.init_flow={ck / 'model.pt'}", f"method.flow.updates={updates}",
                     "method.flow.lr=1.5e-4", "method.flow.warmup_updates=500",
                     f"method.flow.snapshot_steps=[{snaps}]"], None)
        return make

    def sweep(name, *args):
        return lambda: ([PY, "-u", "-m", "methods.morae_slp.sweep", "--run",
                         str(latest_run(name)), "--limit", "256", *args], None)

    def recon(data, short):
        return lambda: ([PY, "-u", "-m", "methods.morae_slp.diagnostics", "--data", data,
                         "--ae", str(ae), "--recon", "400",
                         "--out", str(DIAG / f"recon_{short}_dev400.json")], None)

    out = []
    for data, short, updates in (("csl_smplx", "csl", 60000), ("how2sign_smplx_soke", "how2sign", 80000)):
        out += [(f"boost_recon_{short}", recon(data, short)),
                (f"boost_gtlen_{short}", sweep(f"morae_v2ctrl_{short}_only", "--cfg", "1.5",
                                               "--temperature", "0.7", "--lengths", "gt"))]
    for data, short, updates in (("csl_smplx", "csl", 60000), ("how2sign_smplx_soke", "how2sign", 80000)):
        name = f"morae_v2ctrl_{short}_{updates // 1000}k"
        first = (f"boost_{short}_{updates // 1000}k", train(name, data, updates))
        if short == "csl":
            # The 60k run was SIGKILLed at update 20k (cause unknown, not an OOM): continue with 40k
            # more updates from its EMA checkpoint (effective 60k, restarted lr schedule).
            name = f"{name}_cont"
            first = ("boost_csl_cont", warm_start(name, data, 40000, "morae_v2ctrl_csl_60k"))
        out += [first,
                (f"boost_{short}_sweep", sweep(name, "--cfg", "1,1.5,2.5", "--temperature",
                                               "0.5,0.7,1", "--ensemble", "1")),
                (f"boost_{short}_euler", sweep(name, "--solver", "euler", "--cfg", "1",
                                               "--temperature", "0.7,1"))]
    return out


PLANS = {"v1": ("job", "stages"), "v2": ("job_v2", "stages_v2"), "v3": ("job_v3", "stages_v3"),
         "boost": ("job_boost", "stages_boost")}


def latest_run(name: str) -> Path | None:
    runs = sorted((PROJECT_ROOT / "outputs" / "runs" / name / "morae_slp").glob("seed*_*"),
                  key=lambda r: r.name.split("_", 1)[1])  # newest timestamp, whatever the seed
    done = [r for r in runs if (r / "metrics.json").is_file()]
    return done[-1] if done else None


def stages(workers: int) -> list[tuple[str, callable]]:
    """(name, function returning (argv, output dir to clear before a fresh attempt or None))."""
    jepa_dir, ae_dir = PRE / "jepa_union", PRE / "ae_union_jepa_d64"
    w = f"method.workers={workers}"
    return [
        ("jepa", lambda: ([PY, "-u", "-m", "methods.morae_slp.pretrain", "--stage", "jepa",
                           "--datasets", UNION, "--out", str(jepa_dir), w], jepa_dir)),
        ("ae", lambda: ([PY, "-u", "-m", "methods.morae_slp.pretrain", "--stage", "ae",
                         "--datasets", UNION, "--out", str(ae_dir),
                         "--init-jepa", str(jepa_dir / "jepa.pt"), w], ae_dir)),
        ("diag", lambda: ([PY, "-u", "-m", "methods.morae_slp.diagnostics", "--data",
                           "phoenix_smplx", "--ae", str(ae_dir / "ae.pt"), "--spectrum",
                           "--recon", "519", "--probe",
                           "--out", str(DIAG / "g1_jepa_d64_phoenix.json")], None)),
        ("flow", lambda: ([PY, "-u", "scripts/run.py", "--method", "morae_slp", "--data",
                           "phoenix_smplx", "--name", EXPERIMENT, "--seeds", "0", "--split",
                           "val", f"method.init_ae={ae_dir / 'ae.pt'}", w,
                           "method.flow.snapshot_steps=[20000,40000]"], None)),
        ("sweep", lambda: ([PY, "-u", "-m", "methods.morae_slp.sweep", "--run",
                            str(latest_run(EXPERIMENT)), "--solver", "rk4", "--steps", "5",
                            "--cfg", "1.5,2.5,3.5,4.5,6.5", "--temperature", "1.0",
                            "--limit", "256"], None)),
    ]


def load_state() -> dict:
    path = JOB / "state.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def save_state(state: dict) -> None:
    tmp = JOB / ".state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(JOB / "state.json")


def free_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout
    return max(int(x) for x in out.split())


def wait_for_gpu(poll: int) -> None:
    """Only guards against OOM: wait until the GPU has MIN_FREE_MIB free."""
    last = 0.0
    while (free := free_mib()) < MIN_FREE_MIB:
        if time.time() - last > 1800:
            log(f"waiting for GPU memory: {free} MiB free < {MIN_FREE_MIB}")
            last = time.time()
        time.sleep(poll)


def run_stage(name: str, argv: list[str], clear: Path | None) -> int:
    if clear is not None and clear.exists():
        import shutil
        shutil.rmtree(clear)  # interrupted attempt; pretrain needs an empty directory
    out = JOB / f"{name}.log"
    log(f"start {name}: {' '.join(argv)}")
    with open(out, "a") as f:
        f.write(f"\n===== {now()} {' '.join(argv)}\n")
        f.flush()
        proc = subprocess.Popen(argv, cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1",
                                  "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})

        def stop(signum, _frame):
            proc.terminate()
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                proc.kill()
            log(f"stopped by signal {signum} during {name}")
            sys.exit(1)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        return proc.wait()


def ended_in_oom(name: str) -> bool:
    """The last stage attempt died of CUDA out-of-memory (other jobs share this GPU)."""
    try:
        text = (JOB / f"{name}.log").read_text(errors="replace")
    except OSError:
        return False
    last = text.rsplit("=====", 1)[-1]
    return "OutOfMemoryError" in last or "CUDA out of memory" in last


def status(build) -> None:
    state = load_state()
    for name, _ in build(0):
        print(f"{name:6s} {state.get(name, {}).get('status', 'pending')}  "
              f"{state.get(name, {}).get('finished', '')}")
    experiments = ([EXPERIMENT] if build is stages else [n for n, _ in V2.values()]
                   if build is stages_v2 else [])
    for exp in experiments:
        run = latest_run(exp)
        if run:
            print("flow run:", run)
            print(json.dumps(json.loads((run / "metrics.json").read_text()), indent=2))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--status", action="store_true")
    p.add_argument("--no-wait", action="store_true", help="skip the free-memory check")
    p.add_argument("--min-free-gb", type=float, default=None,
                   help="free GPU memory required before a stage starts (default 14)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--poll", type=int, default=60)
    p.add_argument("--plan", choices=sorted(PLANS), default="v1")
    args = p.parse_args(argv)
    global JOB, MIN_FREE_MIB
    if args.min_free_gb is not None:
        MIN_FREE_MIB = int(args.min_free_gb * 1024)
    folder, builder = PLANS[args.plan]
    JOB = PROJECT_ROOT / "outputs" / "morae_slp" / folder
    build = globals()[builder]
    JOB.mkdir(parents=True, exist_ok=True)
    DIAG.mkdir(parents=True, exist_ok=True)
    if args.status:
        status(build)
        return
    with open(JOB / ".lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("Another SignRAE job supervisor is already running")
        log(f"supervisor pid {os.getpid()} started (plan {args.plan})")
        state = load_state()
        for name, make in build(args.workers):
            if state.get(name, {}).get("status") == "done":
                continue
            if not args.no_wait:
                wait_for_gpu(args.poll)
            argv, clear = make()
            state[name] = {"status": "running", "started": now()}
            save_state(state)
            code = run_stage(name, argv, clear)
            attempts = 0
            while code != 0 and attempts < OOM_RETRIES and ended_in_oom(name):
                attempts += 1
                log(f"{name}: CUDA OOM (another job took the memory); retry {attempts}/{OOM_RETRIES}")
                time.sleep(args.poll)
                if not args.no_wait:
                    wait_for_gpu(args.poll)
                code = run_stage(name, argv, clear)
            state[name].update(status="done" if code == 0 else f"failed({code})",
                               finished=now())
            save_state(state)
            log(f"{name} exit {code}")
            if code != 0:
                log(f"stopping: see {JOB / (name + '.log')}")
                sys.exit(code)
        log(f"all stages done (plan {args.plan})")


if __name__ == "__main__":
    main()
