# methods/

Every method / idea we try lives in its **own folder** here. This is the
sandbox: methods can be messy, experimental, and thrown away without touching
the shared library in `src/slp/`.

```
methods/
├── _template/            # copied by `python scripts/new_method.py <name>`
├── baseline_majority/    # trivial sanity baseline (pipeline check)
└── <your_method>/
    ├── README.md         # status, hypothesis, design, results, notes
    ├── config.yaml       # default hyper-parameters (method: ...)
    ├── method.py         # class Method(BaseMethod)
    └── ...               # any extra modules (model.py, losses.py, ...)
```

## Rules

1. **One folder = one method.** Variants of the same idea → same folder with
   config flags; a genuinely different idea → new folder.
2. **Methods never import from other methods.** If code is needed by several
   methods, promote it into `src/slp/` (and add a test). Enforced by
   `tests/test_methods.py`.
3. **Methods never compute final metrics or touch the test split.** They only
   implement `fit(train, val)` and `predict(inputs)`; evaluation is done by
   `slp.evaluation` so every method is scored identically.
4. **Keep `status:` in README.md up to date**: `exploring` → `promising` →
   `final` / `abandoned`. Abandoned methods stay (negative results are
   useful), they are just not run in paper experiments.
5. Folder names must be valid Python identifiers (`snake_case`).

## Promoting a method

When a method is selected as the paper's main method, keep it here (its
folder is what reviewers will read), but clean it up: remove dead flags, move
reusable parts into `src/slp/`, set `status: final`, and make sure its
experiments are scripted in `configs/experiments/`.
