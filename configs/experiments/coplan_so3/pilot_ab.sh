#!/usr/bin/env bash
set -Eeuo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
if (( $# != 2 )); then
    echo "Usage: bash $0 RUN_DIR SIGNMFLOW_DEV_PREDICTIONS.npz" >&2
    exit 2
fi
run_dir=$1
baseline=$2
gpu_index=${COPLAN_GPU_INDEX:-0}
pilot_python=${COPLAN_PYTHON:-python}
wait_pid=${COPLAN_WAIT_PID:-}
min_free_mib=16384

mkdir -p -- "$run_dir" outputs/jobs
exec >>"$run_dir/job.log" 2>&1
exec 9>outputs/jobs/.coplan_ab_gpu_"$gpu_index".lock
if ! flock -n 9; then
    echo "A CoPlan A/B job already owns the queue for GPU $gpu_index."
    exit 1
fi

state() {
    printf '%s %s\n' "$(date -Is)" "$*" > "$run_dir/status.tmp"
    mv -- "$run_dir/status.tmp" "$run_dir/status"
    cat -- "$run_dir/status"
}

trap 'rc=$?; if (( rc != 0 )); then state "failed exit=$rc"; fi' EXIT
test -r "$baseline"
test -r outputs/bt/csl_mbart/best.pt
gpu_uuid=$(nvidia-smi -i "$gpu_index" --query-gpu=uuid --format=csv,noheader)

process_start() {
    local stat
    local -a fields
    IFS= read -r stat 2>/dev/null < "/proc/$1/stat" || return 1
    read -r -a fields <<< "${stat##*) }"
    [[ ${fields[0]} != Z && ${fields[0]} != X ]] || return 1
    printf '%s\n' "${fields[19]}"
}

required_checks=3
if [[ -n $wait_pid ]]; then
    if [[ ! $wait_pid =~ ^[1-9][0-9]*$ ]]; then
        echo "COPLAN_WAIT_PID must be a positive process ID."
        exit 2
    fi
    start=$(process_start "$wait_pid") || start=
    while [[ -n $start ]] && current=$(process_start "$wait_pid") && [[ $current == "$start" ]]; do
        state "waiting_pid pid=$wait_pid"
        sleep 30
    done
    state "dependency_exited pid=$wait_pid"
    required_checks=1
fi

ready_checks=0
while (( ready_checks < required_checks )); do
    if stats=$(nvidia-smi -i "$gpu_uuid" --query-gpu=memory.free,utilization.gpu \
        --format=csv,noheader,nounits); then
        IFS=, read -r free_mib utilization <<< "$stats"
        if (( free_mib >= min_free_mib )) && [[ -n $wait_pid || $utilization -le 5 ]]; then
            ready_checks=$((ready_checks + 1))
        else
            ready_checks=0
        fi
        state "waiting_gpu free_mib=$free_mib min_free_mib=$min_free_mib utilization_pct=${utilization// /} ready_checks=$ready_checks/$required_checks"
    else
        ready_checks=0
        state "waiting_gpu nvidia-smi unavailable"
    fi
    if (( ready_checks < required_checks )); then
        sleep 30
    fi
done

export CUDA_VISIBLE_DEVICES="$gpu_uuid"
export PYTHONPATH="src:.${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
state "running_ab gpu=$gpu_uuid"
"$pilot_python" -u -m scripts.coplan_ab_pilot --out "$run_dir" --baseline "$baseline"
state "completed_ab report=$run_dir/report.json"
