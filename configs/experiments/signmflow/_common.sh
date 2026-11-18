# Shared helpers for the SignNMFlow experiment scripts (source, do not run).
# SEEDS: protocol requires >=3 seeds (5 for the main table). SPLIT: val by default;
# set SPLIT=test only for final numbers from a clean, committed tree.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
SEEDS=${SEEDS:-0,1,2}
SPLIT=${SPLIT:-val}
RUN="python scripts/run.py --method signmflow --split ${SPLIT}"
# Latest *finished* run dir (has metrics.json) of an experiment for a seed
last_run() { ls -d outputs/runs/"$1"/signmflow/seed"$2"_*/metrics.json 2>/dev/null | sort | tail -1 | xargs -r dirname; }
