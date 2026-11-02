# SLP

Research codebase. Shared, method-agnostic infrastructure lives in
`src/slp/`; every method being explored lives in its own folder under
`methods/`.

## Layout

```
SLP/
├── configs/
│   ├── base.yaml              # global defaults (metrics, eval, ...)
│   ├── data/<dataset>.yaml    # one per dataset
│   └── experiments/           # scripts that reproduce each paper table/figure
├── src/slp/                   # SHARED library (stable, tested)
│   ├── data/                  # dataset loaders + fixed splits
│   ├── evaluation/            # the ONLY place metrics are computed
│   ├── tracking/              # run dirs, provenance (git, config, logs)
│   ├── utils/                 # seeding, logging, git
│   ├── method.py              # BaseMethod interface (fit / predict)
│   ├── registry.py            # auto-discovers methods/<name>/
│   └── config.py              # base -> data -> method -> CLI config merge
├── methods/                   # SANDBOX: one folder per method/idea
│   ├── _template/
│   └── baseline_majority/
├── scripts/
│   ├── run.py                 # train + evaluate any method
│   ├── new_method.py          # scaffold a new method folder
│   └── aggregate.py           # mean ± std over seeds -> results/
├── tests/                     # core tests + auto conformance check per method
├── data/                      # raw/ & processed/ (gitignored), README documents sources
├── outputs/                   # run outputs (gitignored)
├── results/                   # curated, committed tables/figures + test access log
├── notebooks/                 # exploration only
└── docs/                      # research log, ideas, experiment protocol
```

## Quickstart

```bash
pip install -e ".[dev]"          # or: make install
make test                        # all tests
make smoke                       # end-to-end run on toy data

python scripts/new_method.py my_idea          # 1. create methods/my_idea/
# 2. implement methods/my_idea/method.py
python scripts/run.py --method my_idea --name debug --seeds 0      # 3. iterate on val
python scripts/run.py --method my_idea --name exp1 --seeds 0,1,2 method.lr=0.01
python scripts/aggregate.py --name exp1      # 4. mean ± std table -> results/exp1.md
python scripts/run.py --list                 # all methods + status
```

Read [docs/experiment_protocol.md](docs/experiment_protocol.md) before
running experiments that will be reported, and
[methods/README.md](methods/README.md) for the method-folder rules.
