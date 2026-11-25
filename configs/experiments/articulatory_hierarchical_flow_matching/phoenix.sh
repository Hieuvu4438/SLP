#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

python_bin="${PYTHON_BIN:-python}"
method=articulatory_hierarchical_flow_matching
prefix="${EXPERIMENT_PREFIX:-articulatory_phoenix}"
seeds="${SEEDS:-0,1,2}"

case "${1:-}" in
    train)
        shift
        for mode in articulatory temporal; do
            "$python_bin" scripts/run.py --method "$method" --data phoenix_smplx \
                --name "${prefix}_${mode}" --seeds "$seeds" "method.subspace.mode=$mode" "$@"
        done
        ;;
    dev)
        shift
        "$python_bin" configs/experiments/"$method"/dev.py "$@"
        ;;
    *)
        echo "Usage: $0 train [overrides ...] | dev PATH_TO_RUN [options ...]" >&2
        exit 2
        ;;
esac
