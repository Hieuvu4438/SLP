"""Job queue of methods/signmflow/jobs.py: graph consistency and stage-resume overrides."""

from methods.signmflow import jobs as J


def test_graph_is_consistent_and_acyclic():
    J.build((1, 2), ("1", "2", "3", "4", "5", "7", "8"), multilingual=True)
    for j in J.JOBS.values():
        assert all(d in J.JOBS for d in j.deps), j.id
        assert j.is_run or (j.cmd is not None and j.out is not None)
    seen, stack = set(), set()

    def visit(i):
        assert i not in stack, f"cycle at {i}"
        if i in seen:
            return
        stack.add(i)
        for d in J.JOBS[i].deps:
            visit(d)
        stack.discard(i)
        seen.add(i)

    for i in J.JOBS:
        visit(i)
    names = [(j.name, j.seed) for j in J.JOBS.values() if j.is_run]
    assert len(names) == len(set(names)), "two run jobs share an experiment name and seed"


def test_resume_overrides_pick_latest_stage_files(tmp_path, monkeypatch):
    monkeypatch.setattr(J, "RUNS", tmp_path)
    root = tmp_path / "exp" / "signmflow"
    a, b = root / "seed0_20260101-000000", root / "seed0_20260102-000000"
    for d, files in ((a, ["prototypes.pt", "correction.pt"]), (b, ["prototypes.pt", "correction_final.pt", "generator.pt"])):
        (d / "checkpoints").mkdir(parents=True)
        for f in files:
            (d / "checkpoints" / f).write_text("x")
    job = J.Job("x", 0, name="exp", data="csl_smplx", seed=0)
    o = dict(x.split("=", 1) for x in J.resume_overrides(job, []))
    assert o["method.init.prototypes"].startswith(str(b))
    assert o["method.init.correction"].endswith("correction_final.pt")
    assert "method.correction.resume_from" not in o
    assert o["method.generator.resume_from"].startswith(str(b))
    # explicitly provided inits win over resumed ones
    o = J.resume_overrides(job, ["method.init.prototypes=/given.pt"])
    assert not any(x.startswith("method.init.prototypes") for x in o)
    assert J.done_dir("exp", 0) is None
    (b / "metrics.json").write_text("{}")
    assert J.done_dir("exp", 0) == b


def test_default_plan_is_tables_1_2_seed_0():
    J.build()
    runs = sorted(j.id for j in J.JOBS.values() if j.is_run)
    assert runs == ["main_csl_s0", "main_how2sign_s0", "main_phoenix_s0"]  # SignNMFlow-M is opt-in
    J.build(multilingual=True)
    assert {"M_phoenix_s0", "M_csl_s0", "M_how2sign_s0"} <= set(J.JOBS)
    J.build()
    assert J.JOBS["main_how2sign_s0"].data == "how2sign_smplx_soke"
    assert all(j.name.startswith(J.PREFIX) for j in J.JOBS.values() if j.is_run)
    assert all(j.seed == 0 for j in J.JOBS.values() if j.is_run)
