#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

python_bin="${PYTHON_BIN:-python}"
method=articulatory_hierarchical_flow_matching
prefix="${EXPERIMENT_PREFIX:-articulatory_phoenix}"
seeds="${SEEDS:-0,1,2}"
lock=outputs/jobs/.articulatory_hierarchical_flow_matching.lock

case "${1:-}" in
    train)
        shift
        mkdir -p outputs/jobs
        exec 9>"$lock"
        flock -n 9
        for mode in articulatory temporal; do
            "$python_bin" scripts/run.py --method "$method" --data phoenix_smplx \
                --name "${prefix}_${mode}" --seeds "$seeds" "method.subspace.mode=$mode" "$@" \
                method.workers=0
        done
        ;;
    dev)
        shift
        mkdir -p outputs/jobs
        exec flock -n -F "$lock" "$python_bin" configs/experiments/"$method"/dev.py "$@"
        ;;
    roadmap)
        shift
        mkdir -p outputs/jobs
        exec flock -n -F "$lock" \
            "$python_bin" configs/experiments/"$method"/roadmap.py "$@"
        ;;
    *)
        echo "Usage: $0 train [overrides ...] | dev PATH_TO_RUN [options ...] | roadmap --help" >&2
        exit 2
        ;;
esac
