"""Arm registry for docs/proposal7 (07_lo_trinh_uu_tien.md): prints the commands, never runs them.

    python -m methods.signmflow_plus.experiments list                       # all arms by priority
    python -m methods.signmflow_plus.experiments plan --prio P0,P1          # commands to run, in order
    python -m methods.signmflow_plus.experiments plan --arm E3 --data csl_smplx

Every training arm is the paper configuration (methods/signmflow v2) plus the listed overrides, 300
epochs, scored on the val split with fixed GT lengths (C1) so that the length head does not enter
any comparison. Eval-only arms re-sample a trained arm's checkpoints (``method.init.*``). Score
each arm against the baseline with ``analysis/score.py`` (the shrink-frontier procedure, 07 §1.2).
GPU-hours are the 07 estimates on Phoenix (about 18 GPU-h per 300-epoch run, 1 process).
"""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs" / "runs"
V2_CKPT = {  # the replica's seed-0 runs (methods/signmflow, v2), re-sampled with fixed lengths as B_s0
    "phoenix_smplx": "outputs/runs/signmflow_v2_main_phoenix_smplx/signmflow",
    "csl_smplx": "outputs/runs/signmflow_v2_main_csl_smplx/signmflow",
    "how2sign_smplx_soke": "outputs/runs/signmflow_v2_main_how2sign_smplx_soke/signmflow",
}


@dataclass
class Arm:
    id: str
    prio: str
    what: str
    overrides: list = field(default_factory=list)
    seeds: tuple = (0,)
    from_arm: str | None = None    # eval-only: re-sample the checkpoints of this arm
    from_v2: bool = False          # eval-only: re-sample the methods/signmflow v2 checkpoints
    gpu_h: float = 18.0
    method: str = "signmflow_plus"


FK = ["method.flow.fk_loss.lambda_body=0.1", "method.flow.fk_loss.lambda_hand=1.0"]  # initial scale match:
# check loss_fk_body / loss_fk_hand against 10 * loss_pose in the first epochs and adjust before the sweep
ARMS = [
    # P0 - measurement (no GPU training)
    Arm("E04_mean_pose", "P0", "text-free mean pose (M3 floor)", ["method.mode=mean_pose"], gpu_h=0, method="text_retrieval"),
    Arm("E04_retrieval", "P0", "nearest training sentence, char TF-IDF", gpu_h=0, method="text_retrieval"),
    Arm("B_s0", "P0", "v2 replica seed 0 re-sampled with fixed lengths (the reference)", from_v2=True, gpu_h=0.3),
    # P1 - noise floor and positive control
    Arm("B", "P1", "paper configuration, extra seeds", seeds=(1, 2)),
    Arm("GE", "P1", "E1-gate: geodesic path (no prototypes / correction)", ["method.flow.path=geodesic"], seeds=(0, 1, 2)),
    # P2 - hygiene and position losses
    Arm("E3", "P2", "EMA 0.999 + text_lr 2e-5 + warmup 1000",
        ["method.generator.ema=0.999", "method.generator.text_lr=2.0e-5", "method.generator.warmup_steps=1000"]),
    Arm("E4c", "P2", "lever-arm joint weights", ["method.flow.joint_weights=lever_arm"]),
    Arm("E4a", "P2", "FK loss on x^1 (body + hand)", FK),
    Arm("E4b_v0.5", "P2", "E4a + velocity loss 0.5 lambda_hand", FK + ["method.flow.fk_loss.lambda_vel=0.5"]),
    Arm("E4b_v2", "P2", "E4a + velocity loss 2 lambda_hand", FK + ["method.flow.fk_loss.lambda_vel=2.0"]),
    # P3 - schedules and mechanism
    Arm("E2F_t2.5", "P3", "training-time shift 2.5", ["method.flow.train_shift=2.5"]),
    Arm("E2F_t4", "P3", "training-time shift 4", ["method.flow.train_shift=4.0"]),
    # inference axis of the factorial (free): re-sample the s_train = 1 reference and both shifted runs
    *[Arm(f"E2F_t1_i{s}", "P3", f"inference grid shift {s} on the reference", [f"method.sampler.shift={s}"],
          from_v2=True, gpu_h=0.3) for s in (2.5, 4.0)],
    *[Arm(f"{t}_i{s}", "P3", f"inference grid shift {s} on {t}", [f"method.sampler.shift={s}"],
          from_arm=t, gpu_h=0.3) for t in ("E2F_t2.5", "E2F_t4") for s in (2.5, 4.0)],
    Arm("E1H1_par", "P3", "only if NG > GE: naturalness path, parallel component of eps",
        ["method.flow.path=ng_parallel"]),
    *[Arm(f"E1H1_geo{s}", "P3", f"only if NG > GE: geodesic with SD3 shift {s}",
          ["method.flow.path=geoshift", f"method.flow.shift={s}"]) for s in (2.0, 2.5, 3.0)],
    # P4 - semantics, source, self-conditioning (need M1)
    Arm("E5", "P4", "text dropout 0.1 (CFG training)", ["method.generator.p_uncond=0.1"]),
    *[Arm(f"E5_w{w}", "P4", f"CFG weight {w}", [f"method.sampler.cfg_w={w}"], from_arm="E5", gpu_h=0.5)
      for w in (1.5, 2.0, 3.0)],
    Arm("E7", "P4", "self-conditioning", ["method.flow.self_cond=true"], gpu_h=24),
    *[Arm(f"E6_s{s}", "P4", f"wrapped-normal source at the mean pose, sigma0 {s}",
          ["method.flow.source=wrapped_normal", f"method.flow.source_sigma={s}"]) for s in (0.25, 0.5)],
    # P5 - science only
    Arm("E1lin", "P5", "renormalised linear path (Table 3 MR -> GE confound)", ["method.flow.path=linear_renorm"]),
    Arm("E8", "P5", "6D rotations, linear path (paper's baseline representation)",
        ["method.flow.representation=6d", "method.flow.path=linear"]),
]
BY_ID = {a.id: a for a in ARMS}
PRIO = ["P0", "P1", "P2", "P3", "P4", "P5"]


def lengths_file(data: str) -> str:
    return f"outputs/signmflow_plus/lengths/{data}_val_gt.json"


def experiment_name(arm: Arm, data: str) -> str:
    return f"sp_{data}_{arm.id}"


def latest_ckpt(exp_dir: Path, seed: int) -> Path | None:
    """checkpoints/ of the newest finished run (metrics.json present) of an experiment / seed."""
    runs = sorted(p for p in exp_dir.glob(f"seed{seed}_*") if (p / "metrics.json").exists())
    return (runs[-1] / "checkpoints").relative_to(ROOT) if runs else None


def init_overrides(ck: Path | str, needs_correction: bool) -> list[str]:
    ck = str(ck)
    out = [f"method.init.generator={ck}/generator.pt"]
    if needs_correction:
        out += [f"method.init.prototypes={ck}/prototypes.pt", f"method.init.correction={ck}/correction_final.pt"]
    return out


def commands(arm: Arm, data: str) -> list[str]:
    base = ["python", "scripts/run.py", "--method", arm.method, "--data", data]
    over = list(arm.overrides)
    if arm.method == "signmflow_plus":
        over.append(f"method.sampler.lengths_file={lengths_file(data)}")
    elif "method.mode=mean_pose" in over:
        over.append(f"method.lengths_file={lengths_file(data)}")
    out = []
    for s in arm.seeds:
        o = list(over)
        if arm.from_v2 or arm.from_arm:
            if arm.from_v2:
                ck = latest_ckpt(ROOT / V2_CKPT[data], s) or f"<{V2_CKPT[data]}/seed{s}_*/checkpoints>"
                corr = True
            else:
                src = BY_ID[arm.from_arm]
                ck = latest_ckpt(RUNS / experiment_name(src, data) / "signmflow_plus", s) \
                    or f"<outputs/runs/{experiment_name(src, data)}/signmflow_plus/seed{s}_*/checkpoints>"
                o = list(src.overrides) + o
                corr = not any(x in o for x in ("method.flow.path=geodesic", "method.flow.path=geoshift",
                                                "method.flow.path=linear", "method.flow.path=linear_renorm"))
            o += init_overrides(ck, corr)
        out.append(shlex.join(base + ["--name", experiment_name(arm, data), "--seeds", str(s)] + o))
    return out


def prep_commands(data: str) -> list[str]:
    return [f"python -m methods.signmflow_plus.tools.lengths --data {data} --out {lengths_file(data)}",
            f"python -m methods.signmflow_plus.analysis.evaluator train --data {data}",
            f"python -m methods.signmflow_plus.analysis.evaluator calibrate --data {data}"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("action", choices=["list", "plan"])
    ap.add_argument("--data", default="phoenix_smplx")
    ap.add_argument("--prio", default=",".join(PRIO))
    ap.add_argument("--arm", action="append", default=[])
    a = ap.parse_args(argv)
    arms = [BY_ID[i] for i in a.arm] if a.arm else [x for x in ARMS if x.prio in a.prio.split(",")]
    if a.action == "list":
        for x in arms:
            src = f" (re-samples {x.from_arm or 'v2'})" if (x.from_arm or x.from_v2) else ""
            print(f"{x.prio}  {x.id:<14} seeds {','.join(map(str, x.seeds)):<6} ~{x.gpu_h * len(x.seeds):>5.1f} GPU-h  "
                  f"{x.what}{src}")
        print(f"total ~{sum(x.gpu_h * len(x.seeds) for x in arms):.0f} GPU-h (Phoenix estimates)")
        return
    if any(x.prio == "P0" for x in arms):
        print("# P0 preparation (lengths table, M1 evaluator)")
        print("\n".join(prep_commands(a.data)))
    for x in arms:
        print(f"# {x.prio} {x.id}: {x.what}")
        print("\n".join(commands(x, a.data)))
    print("# then score every arm against B_s0 (07 §1.2):")
    ref = f"outputs/runs/{experiment_name(BY_ID['B_s0'], a.data)}/signmflow_plus/seed0_*/predictions.npz"
    print(f"python -m methods.signmflow_plus.analysis.score --data {a.data} --reference {ref} "
          f"--arm NAME=outputs/runs/<experiment>/<method>/seed<k>_*/predictions.npz ... --text-swap "
          f"--evaluator outputs/signmflow_plus/evaluator/{a.data}/evaluator.pt "
          f"--out outputs/signmflow_plus/scores/{a.data}.json")


if __name__ == "__main__":
    main()
