# Experiment definitions

One file per *paper-level* experiment (main table, ablation, sensitivity...).
Each file documents exactly which methods / data / seeds / overrides produce a
table or figure, so results can be regenerated with one command.

Example `main_table.sh`:

```bash
#!/usr/bin/env bash
set -e
NAME=main_table
SEEDS=0,1,2,3,4
for M in baseline_majority my_method; do
  python scripts/run.py --method $M --data toy --name $NAME --seeds $SEEDS
done
python scripts/aggregate.py --name $NAME
```
