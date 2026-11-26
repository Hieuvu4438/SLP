"""Choose, per dataset, the configuration with the best DEV result and build its TEST command.

    python -m methods.morae_slp.select_best --data phoenix_smplx [--run-test]

Rule (fixed in advance, dev only; the test split is never read here):
  * candidates = single-sample, gloss-free, retrieval-free configurations whose DEV numbers were
    measured on the WHOLE dev split (``Candidate`` below);
  * validity guard: both low-pass speed ratios must be >= MIN_SPEED (rejects near-static outputs,
    which the DTW metric rewards without being sign production);
  * best = smallest JPE-hand (the config's ``eval.primary_metric``), ties broken by JPE-body.
The test command is the chosen run's own ``scripts/run.py`` command with ``--split test``, a
``<name>_test`` experiment name and the candidate's sampler overrides, so the test run retrains
exactly the dev recipe (``run.py`` always trains, then evaluates). ``run.py`` refuses a TEST
evaluation from a dirty git tree: ``--allow-dirty`` is added only if the tree is dirty, and then the
number is NOT reportable (``code.diff`` of the run records the code).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from slp.paths import PROJECT_ROOT

RUNS = PROJECT_ROOT / "outputs" / "runs"
DIAG = PROJECT_ROOT / "outputs" / "morae_slp" / "diagnostics"
MIN_SPEED = 0.5
PRIMARY = "soke_dtw_jpe_hand"
TIE_BREAK = "soke_dtw_jpe_body"


@dataclass(frozen=True)
class Candidate:
    dataset: str
    label: str
    experiment: str                 # run.py experiment (outputs/runs/<experiment>/morae_slp/...)
    source: str                     # metrics | sweep | frontier
    sampler: tuple[str, ...] = ()   # extra overrides for the TEST command
    where: dict = field(default_factory=dict)  # sweep row filters / frontier file + arm

    def __hash__(self):
        return hash((self.dataset, self.label))


CANDIDATES: tuple[Candidate, ...] = (
    # Phoenix-2014T: dataset-only flow, v3 variants (metrics.json = whole dev at their own sampler).
    Candidate("phoenix_smplx", "baseline w1.5 t0.7", "morae_v2ctrl_phoenix_only", "frontier",
              ("method.sampler.cfg_scale=1.5", "method.sampler.temperature=0.7"),
              {"file": "frontier_ctrl_phoenix_dev519.json", "arm": "ctrl_w1.5_t0.7"}),
    Candidate("phoenix_smplx", "baseline euler w1 t0.7", "morae_v2ctrl_phoenix_only", "sweep",
              ("method.sampler.solver=euler", "method.sampler.steps=5",
               "method.sampler.cfg_scale=1", "method.sampler.temperature=0.7"),
              {"solver": "euler", "cfg_scale": 1.0, "temperature": 0.7}),
    Candidate("phoenix_smplx", "baseline euler w1 t1.0", "morae_v2ctrl_phoenix_only", "sweep",
              ("method.sampler.solver=euler", "method.sampler.steps=5",
               "method.sampler.cfg_scale=1", "method.sampler.temperature=1.0"),
              {"solver": "euler", "cfg_scale": 1.0, "temperature": 1.0}),
    # (seed 1 of the baseline is run only to measure seed noise, never as a candidate)
    Candidate("phoenix_smplx", "C1 target", "morae_v3_phoenix_target", "metrics"),
    Candidate("phoenix_smplx", "C2 fk velocity", "morae_v3_phoenix_fk_velocity", "metrics"),
    Candidate("phoenix_smplx", "C2 fk target", "morae_v3_phoenix_fk_target", "metrics"),
    Candidate("phoenix_smplx", "mBART lr 0.3", "morae_v3_phoenix_lr03", "metrics"),
    # CSL-Daily: refined flow (warm-started continuation), rk4 and Euler-5 without guidance.
    Candidate("csl_smplx", "refined rk4 w1.5 t0.7", "morae_v2ctrl_csl_60k_cont", "metrics"),
    Candidate("csl_smplx", "refined euler w1 t0.7", "morae_v2ctrl_csl_60k_cont", "sweep",
              ("method.sampler.solver=euler", "method.sampler.steps=5",
               "method.sampler.cfg_scale=1", "method.sampler.temperature=0.7"),
              {"solver": "euler", "cfg_scale": 1.0, "temperature": 0.7}),
    # C2 on CSL-Daily: refined flow + metric-frame FK loss (warm-started fine-tune).
    Candidate("csl_smplx", "C2 fk rk4 w1.5 t0.7", "morae_v3_csl_fk", "metrics"),
    Candidate("csl_smplx", "C2 fk euler w1 t0.7", "morae_v3_csl_fk", "sweep",
              ("method.sampler.solver=euler", "method.sampler.steps=5",
               "method.sampler.cfg_scale=1", "method.sampler.temperature=0.7"),
              {"solver": "euler", "cfg_scale": 1.0, "temperature": 0.7}),
    # How2Sign (SOKE protocol): dataset-only flow at its own sampler, plus Euler-5 without guidance.
    Candidate("how2sign_smplx_soke", "dataset-only w2 t1", "morae_v2ctrl_how2sign_only", "metrics"),
    Candidate("how2sign_smplx_soke", "dataset-only euler w1 t1", "morae_v2ctrl_how2sign_only",
              "sweep", ("method.sampler.solver=euler", "method.sampler.steps=5",
                        "method.sampler.cfg_scale=1", "method.sampler.temperature=1.0"),
              {"solver": "euler", "cfg_scale": 1.0, "temperature": 1.0}),
)
DEV_SIZE = {"phoenix_smplx": 519, "csl_smplx": 1077, "how2sign_smplx_soke": 1722}


def latest_run(experiment: str, need: str | None = "metrics.json") -> Path | None:
    runs = sorted((RUNS / experiment / "morae_slp").glob("seed*_*"))
    runs = [r for r in runs if need is None or (r / need).is_file()]
    return runs[-1] if runs else None


def _speeds(metrics: dict) -> tuple[float, float]:
    return metrics.get("soke_speed_ratio_body", 1.0), metrics.get("soke_speed_ratio_hand", 1.0)


def dev_metrics(c: Candidate) -> dict | None:
    """The candidate's whole-dev metrics (shared-evaluator names), or None if not available yet."""
    run = latest_run(c.experiment)
    if run is None:
        return None
    if c.source == "metrics":
        return json.loads((run / "metrics.json").read_text())
    if c.source == "sweep":
        path = run / "sweep.jsonl"
        if not path.is_file():
            return None
        best = None
        for line in path.read_text().splitlines():
            row = json.loads(line)
            keep = (row.get("clips") == DEV_SIZE[c.dataset] and row.get("ensemble", 1) == 1
                    and row.get("decoder", "mean") == "mean"
                    and row.get("lengths", "head") == "head"
                    and row.get("length_scale", 1.0) == 1.0
                    and row.get("subset", "val") == "val"
                    and all(abs(row[k] - v) < 1e-9 if isinstance(v, float) else row[k] == v
                            for k, v in c.where.items()))
            if keep:
                best = row  # the latest matching row
        return best
    if c.source == "frontier":  # four DTW columns of the arm (no kinematic ratios stored as keys)
        data = json.loads((DIAG / c.where["file"]).read_text())
        arm = data["arms"][c.where["arm"]]
        if data.get("n_sequences") != DEV_SIZE[c.dataset]:
            return None
        k = arm["kinematics"]
        return {"soke_dtw_jpe_body": arm["dtw"]["jpe_body"], "soke_dtw_jpe_hand": arm["dtw"]["jpe_hand"],
                "soke_dtw_pa_jpe_body_ua": arm["dtw"]["pa_body_ua"],
                "soke_dtw_pa_jpe_hand": arm["dtw"]["pa_hand"],
                "soke_speed_ratio_body": k["body_speed_ratio"],
                "soke_speed_ratio_hand": k["hand_speed_ratio"]}
    raise ValueError(f"unknown candidate source {c.source!r}")


def rank(dataset: str, candidates=CANDIDATES) -> list[tuple[Candidate, dict | None, str]]:
    """All candidates of a dataset with their dev metrics and a verdict, best first."""
    rows = []
    for c in candidates:
        if c.dataset != dataset:
            continue
        m = dev_metrics(c)
        if m is None:
            rows.append((c, None, "not available"))
        elif min(_speeds(m)) < MIN_SPEED:
            rows.append((c, m, f"rejected: speed ratio < {MIN_SPEED}"))
        else:
            rows.append((c, m, "valid"))
    rows.sort(key=lambda r: (r[2] != "valid", r[1][PRIMARY] if r[1] else 1e9,
                             r[1][TIE_BREAK] if r[1] else 1e9))
    return rows


def choose(dataset: str, candidates=CANDIDATES) -> Candidate:
    rows = rank(dataset, candidates)
    if not rows or rows[0][2] != "valid":
        raise RuntimeError(f"no valid dev candidate for {dataset}: "
                           f"{[(c.label, v) for c, _, v in rows]}")
    return rows[0][0]


def test_argv(command: str, name: str, sampler: tuple[str, ...], dirty: bool,
              python: str = sys.executable) -> list[str]:
    """The chosen run's recorded command, switched to the test split.

    ``scripts/run.py`` takes its ``--options`` first and the ``key=value`` overrides last (the
    overrides are a trailing positional list), so the command is rebuilt in that order."""
    tokens = shlex.split(command)
    if tokens[0].endswith("run.py"):
        tokens = tokens[1:]
    options: dict[str, str] = {}
    overrides: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "--allow-dirty":
            i += 1
        elif t.startswith("--"):
            options[t] = tokens[i + 1]
            i += 2
        else:
            overrides.append(t)
            i += 1
    options["--split"] = "test"
    options["--name"] = f"{name}_test"
    replaced = {s.split("=")[0] for s in sampler}
    overrides = [o for o in overrides if o.split("=")[0] not in replaced] + list(sampler)
    argv = [python, "-u", "scripts/run.py"]
    for key, value in options.items():
        argv += [key, value]
    if dirty:
        argv.append("--allow-dirty")
    return argv + overrides


def git_dirty() -> bool:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True,
                         text=True, check=True).stdout
    return bool(out.strip())


TEST_LOG = "results/test_access_log.jsonl"
ATTRIBUTION = "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"


def settle_test_log(root: Path | None = None) -> bool:
    """``run.py --split test`` appends to the tracked test access log, which makes the git tree
    dirty and would force ``--allow-dirty`` on the next test run. If that log is the ONLY change,
    commit it (locally, no push) so consecutive test runs stay reportable. Returns True if it
    committed; any other pending change is left alone."""
    root = Path(root or PROJECT_ROOT)
    lines = [ln for ln in subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                         capture_output=True, text=True,
                                         check=True).stdout.splitlines() if ln.strip()]
    if not lines or any(ln[3:] != TEST_LOG for ln in lines):
        return False
    subprocess.run(["git", "add", TEST_LOG], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "Record test access log of the previous test run",
                    "-m", ATTRIBUTION, "--", TEST_LOG], cwd=root, check=True)
    return True


def plan(dataset: str, dirty: bool | None = None) -> dict:
    rows = rank(dataset)
    chosen = choose(dataset)
    run = latest_run(chosen.experiment)
    meta = json.loads((run / "meta.json").read_text())
    dirty = git_dirty() if dirty is None else dirty
    name = chosen.experiment
    argv = test_argv(meta["command"], name, chosen.sampler, dirty)
    return {"dataset": dataset, "chosen": chosen.label, "experiment": chosen.experiment,
            "run": str(run), "dirty_tree": dirty, "reportable": not dirty,
            "argv": argv,
            "ranking": [{"label": c.label, "verdict": v,
                         "dev": None if m is None else {k: m[k] for k in (
                             PRIMARY, TIE_BREAK, "soke_dtw_pa_jpe_body_ua", "soke_dtw_pa_jpe_hand",
                             "soke_speed_ratio_body", "soke_speed_ratio_hand") if k in m}}
                        for c, m, v in rows]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, choices=sorted(DEV_SIZE))
    p.add_argument("--out", default=None, help="write the selection / command as JSON")
    p.add_argument("--run-test", action="store_true", help="execute the chosen TEST command")
    args = p.parse_args(argv)
    result = plan(args.data)
    text = json.dumps(result, indent=1)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    if args.run_test:
        return subprocess.call(result["argv"], cwd=PROJECT_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
