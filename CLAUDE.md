# CLAUDE.md

Research codebase. Read `README.md`, `methods/README.md` and
`docs/experiment_protocol.md` first.

- New method/idea -> `python scripts/new_method.py <name>`, code only inside
  `methods/<name>/`. Never import from another method; shared code goes into
  `src/slp/` with a test.
- Methods implement `fit(train, val)` / `predict(inputs)` only. Never compute
  reported metrics inside a method; never touch the test split in a method.
- Metrics live only in `src/slp/evaluation/metrics.py`, each with a unit test.
- Datasets: loader in `src/slp/data/`, register in `DATASETS`, add
  `configs/data/<name>.yaml`, document in `data/README.md`.
- Run everything through `scripts/run.py`; default split is `val`. Do not run
  `--split test` unless the user asks.
- Run `python -m pytest -q` after changes to `src/slp/` or any method.
- Do not hand-edit files in `results/` or `results/test_access_log.jsonl`.
