#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

PYTHON_BIN="${PYTHON_BIN:-python}"
BT_PYTHON="${BT_PYTHON:-third_party/MaDiS/.venv/bin/python}"
REFERENCE_RUN="${REFERENCE_RUN:-outputs/runs/motionhiflow_phoenix_opt/motionhiflow_slp/seed0_20261123-110901-538484}"
DURATION_REFERENCE="${DURATION_REFERENCE:-$REFERENCE_RUN/predictions.npz}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-motionhiflow_phoenix_improve}"
MICROBATCH="${MICROBATCH:-16}"
SAMPLE_BATCH="${SAMPLE_BATCH:-16}"
FK_WEIGHT="${FK_WEIGHT:-0.02}"

scales_for() {
    case "$1" in
        three) echo '[0.3,0.6,1.0]' ;;
        single) echo '[1.0]' ;;
        *) return 2 ;;
    esac
}

train() {
    local name=$1
    local scales=$2
    shift 2
    "$PYTHON_BIN" -u scripts/run.py --method motionhiflow_slp --data phoenix_smplx \
        --name "${EXPERIMENT_PREFIX}_${name}" --seeds 0 --split val \
        "method.init_codec=$REFERENCE_RUN/checkpoints/codec.pt" \
        "method.sampler.lengths_from_npz=$DURATION_REFERENCE" \
        "method.flow.scales=$scales" "method.flow.microbatch=$MICROBATCH" \
        "method.sampler.batch_size=$SAMPLE_BATCH" method.flow.updates=24000 \
        'method.flow.snapshot_steps=[8000,16000,24000]' "$@"
}

completed_run() {
    local run completed=""
    for run in "outputs/runs/${EXPERIMENT_PREFIX}_$1/motionhiflow_slp"/seed0_*/; do
        if [[ -s "$run/metrics.json" || -s "$run/selection.json" ]]; then
            completed="${run%/}"
        fi
    done
    if [[ -z "$completed" ]]; then
        echo "No completed run for ${EXPERIMENT_PREFIX}_$1" >&2
        return 1
    fi
    printf '%s\n' "$completed"
}

select_dev() {
    if [[ ! -s "$1/selection.json" ]]; then
        "$PYTHON_BIN" -u configs/experiments/motionhiflow_slp/select_dev.py "$1"
    fi
}

finish() {
    local three single winner fk scorer
    three=$(completed_run three)
    single=$(completed_run single)
    select_dev "$three"
    select_dev "$single"
    winner=$("$PYTHON_BIN" - "$three/selection.json" "$single/selection.json" <<'PY'
import json
import sys

def key(path):
    with open(path) as f:
        d = json.load(f)
    score = d["snapshot_selection"][d["checkpoint"]]
    return score[d["selection_metric"]]

print(min(zip(("three", "single"), sys.argv[1:]), key=lambda arm: key(arm[1]))[0])
PY
)
    echo "Selected $winner on the Phoenix tuning broadcasts"
    if ! fk=$(completed_run "fk_$winner" 2>/dev/null); then
        train "fk_$winner" "$(scales_for "$winner")" "method.flow.endpoint_fk_weight=$FK_WEIGHT"
        fk=$(completed_run "fk_$winner")
    fi
    select_dev "$fk"
    scorer="outputs/jobs/$EXPERIMENT_PREFIX/backtranslation"
    mkdir -p "$scorer"
    ln -sfn "$(realpath outputs/bt/phoenix_mbart/best.pt)" "$scorer/best.pt"
    "$BT_PYTHON" -u scripts/train_backtranslator.py --source phoenix --arch mbart \
        --out "$scorer" --split val --score_name metrics.json --score \
        "original=$REFERENCE_RUN/predictions.npz" "three=$three/predictions_selected.npz" \
        "single=$single/predictions_selected.npz" "fk_$winner=$fk/predictions_selected.npz" @gt
    "$PYTHON_BIN" - "$scorer/metrics.json" "$winner" "$three" "$single" "$fk" <<'PY'
import json
import sys
from pathlib import Path

score_path, winner, *run_paths = sys.argv[1:]
score_path = Path(score_path)
bt = json.loads(score_path.read_text())
if any(result["n"] != 519 for result in bt.values()):
    raise ValueError("Back-translation did not score all 519 Phoenix dev clips")
summary = {"split": "Phoenix-2014T dev", "seed": 0, "selected_schedule": winner,
           "arms": {}, "backtranslation": bt}
for arm, path in zip(("three", "single", f"fk_{winner}"), run_paths, strict=True):
    selected = json.loads((Path(path) / "selection.json").read_text())
    selected.pop("tune_clip_ids")
    selected.pop("gate_clip_ids")
    summary["arms"][arm] = selected
folder = score_path.parent.parent
baseline = folder / "baselines.json"
if baseline.is_file():
    summary["baselines"] = json.loads(baseline.read_text())
report = folder / "report.json"
temp = report.with_suffix(".tmp")
temp.write_text(json.dumps(summary, indent=2) + "\n")
temp.replace(report)
print("Report", report, flush=True)
PY
}

case "${1:-}" in
    ablate)
        train three "$(scales_for three)"
        train single "$(scales_for single)"
        ;;
    three|single) train "$1" "$(scales_for "$1")" ;;
    finish) finish ;;
    *) echo "Usage: bash configs/experiments/motionhiflow_slp/phoenix.sh {ablate|three|single|finish}" >&2; exit 2 ;;
esac
