#!/usr/bin/env bash
# SignRAE (methods/morae_slp) experiment phases of methods/morae_slp/PLAN.md section 7.
# One process at a time, seed 0, dev split only. Every step is skipped when its output exists.
#
#   bash configs/experiments/morae_slp/run.sh g0          # CPU diagnostics, no training
#   bash configs/experiments/morae_slp/run.sh g1          # Sign-JEPA + latents + diagnostics
#   bash configs/experiments/morae_slp/run.sh g2          # Phoenix: jepa vs ae vs raw latents
#   bash configs/experiments/morae_slp/run.sh sweep       # sampler grid of the g2 runs (dev)
#   bash configs/experiments/morae_slp/run.sh ablations   # M1 / D1 / W1 / T1 / A2 (Phoenix)
#   bash configs/experiments/morae_slp/run.sh scale       # CSL-Daily then How2Sign
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

PY="${PYTHON_BIN:-python}"
ROOT="${MORAE_ROOT:-outputs/morae_slp}"
PRE="$ROOT/pretrain"
DIAG="$ROOT/diagnostics"
UNION="phoenix_smplx,csl_smplx,how2sign_smplx_soke"
PREFIX="${EXPERIMENT_PREFIX:-morae}"
mkdir -p "$PRE" "$DIAG"

pretrain() {  # pretrain <stage> <out> [args...]
    local stage=$1 out=$2
    shift 2
    local file="$out/jepa.pt"
    [[ $stage == ae ]] && file="$out/ae.pt"
    if [[ -s $file ]]; then echo "skip $file"; return; fi
    rm -rf "$out"
    "$PY" -u -m methods.morae_slp.pretrain --stage "$stage" --datasets "${DATASETS:-$UNION}" \
        --out "$out" "$@"
}

diagnose() {  # diagnose <ae.pt> <data> <out.json>
    if [[ -s $3 ]]; then echo "skip $3"; return; fi
    "$PY" -u -m methods.morae_slp.diagnostics --data "$2" --ae "$1" --spectrum --recon 519 \
        --probe --out "$3"
}

latest_run() {  # latest completed run dir of experiment $1
    local run found=""
    for run in outputs/runs/"$1"/morae_slp/seed0_*/; do
        [[ -s "$run/metrics.json" ]] && found="${run%/}"
    done
    [[ -n $found ]] && printf '%s\n' "$found"
}

train_flow() {  # train_flow <name> <data> [overrides...]
    local name="${PREFIX}_$1" data=$2
    shift 2
    if latest_run "$name" >/dev/null; then echo "skip $name"; return; fi
    "$PY" -u scripts/run.py --method morae_slp --data "$data" --name "$name" --seeds 0 \
        --split val "$@"
}

flow_updates() {
    case "$1" in
        phoenix_smplx) echo 60000 ;;
        csl_smplx) echo 100000 ;;
        how2sign_smplx_soke) echo 150000 ;;
    esac
}

case "${1:-}" in
g0)
    for d in phoenix_smplx csl_smplx; do
        out="$DIAG/g0_rawcp_$d.json"
        [[ -s $out ]] || "$PY" -u -m methods.morae_slp.diagnostics --data "$d" --raw-cp \
            --device cpu --out "$out"
    done
    ;;
g1)
    pretrain jepa "$PRE/jepa_union"
    for dim in 64 32; do
        pretrain ae "$PRE/ae_union_jepa_d$dim" --init-jepa "$PRE/jepa_union/jepa.pt" \
            "method.latent.dim=$dim"
        diagnose "$PRE/ae_union_jepa_d$dim/ae.pt" phoenix_smplx "$DIAG/g1_jepa_d${dim}_phoenix.json"
    done
    # Reconstruction control (paper Table 3): same encoder architecture trained with the bottleneck.
    pretrain ae "$PRE/ae_union_ae_d64" method.latent.kind=ae method.latent.dim=64 \
        method.ae.updates=60000
    diagnose "$PRE/ae_union_ae_d64/ae.pt" phoenix_smplx "$DIAG/g1_ae_d64_phoenix.json"
    ;;
g2)
    u=$(flow_updates phoenix_smplx)
    train_flow g2_jepa_d64 phoenix_smplx "method.init_ae=$PRE/ae_union_jepa_d64/ae.pt" \
        "method.flow.updates=$u" 'method.flow.snapshot_steps=[20000,40000]'
    train_flow g2_ae_d64 phoenix_smplx "method.init_ae=$PRE/ae_union_ae_d64/ae.pt" \
        method.latent.kind=ae "method.flow.updates=$u"
    train_flow g2_raw phoenix_smplx method.latent.kind=raw "method.flow.updates=$u"
    ;;
sweep)
    for arm in g2_jepa_d64 g2_ae_d64 g2_raw; do
        run=$(latest_run "${PREFIX}_$arm") || { echo "no run for $arm"; continue; }
        [[ -s $run/sweep.jsonl ]] && { echo "skip sweep $run"; continue; }
        "$PY" -u -m methods.morae_slp.sweep --run "$run" --solver rk4 --steps 5,8 \
            --cfg 1.5,2.5,3.5,4.5,6.5 --temperature 1.0,0.8,0.6
    done
    ;;
ablations)
    u=$(flow_updates phoenix_smplx)
    # M1 masking unit: paper joint-only and hand-heavy JEPA on the union.
    pretrain jepa "$PRE/jepa_union_jointonly" 'method.jepa.mask.unit_probs={joint: 1.0, chain: 0.0, hand: 0.0}'
    pretrain ae "$PRE/ae_union_jointonly_d64" --init-jepa "$PRE/jepa_union_jointonly/jepa.pt"
    diagnose "$PRE/ae_union_jointonly_d64/ae.pt" phoenix_smplx "$DIAG/m1_jointonly_phoenix.json"
    train_flow m1_jointonly phoenix_smplx "method.init_ae=$PRE/ae_union_jointonly_d64/ae.pt" \
        "method.flow.updates=$u"
    pretrain jepa "$PRE/jepa_union_handheavy" \
        'method.jepa.mask.unit_probs={joint: 0.2, chain: 0.4, hand: 0.4}'
    pretrain ae "$PRE/ae_union_handheavy_d64" --init-jepa "$PRE/jepa_union_handheavy/jepa.pt"
    train_flow m1_handheavy phoenix_smplx "method.init_ae=$PRE/ae_union_handheavy_d64/ae.pt" \
        "method.flow.updates=$u"
    # D1 Phoenix-only self-supervision.
    DATASETS=phoenix_smplx pretrain jepa "$PRE/jepa_phoenix" method.jepa.updates=30000
    DATASETS=phoenix_smplx pretrain ae "$PRE/ae_phoenix_d64" --init-jepa "$PRE/jepa_phoenix/jepa.pt"
    train_flow d1_phoenix_only phoenix_smplx "method.init_ae=$PRE/ae_phoenix_d64/ae.pt" \
        "method.flow.updates=$u"
    # W1 latent width.
    train_flow w1_d32 phoenix_smplx "method.init_ae=$PRE/ae_union_jepa_d32/ae.pt" \
        method.latent.dim=32 "method.flow.updates=$u"
    # T1 paper-faithful sentence-only conditioning.
    train_flow t1_adaln phoenix_smplx "method.init_ae=$PRE/ae_union_jepa_d64/ae.pt" \
        method.flow.text_mode=adaln "method.flow.updates=$u"
    # A2 decoder trained with SOKE joint-position loss.
    pretrain ae "$PRE/ae_union_jepa_d64_fk" --init-jepa "$PRE/jepa_union/jepa.pt" \
        method.ae.fk_weight=0.05
    diagnose "$PRE/ae_union_jepa_d64_fk/ae.pt" phoenix_smplx "$DIAG/a2_fk_phoenix.json"
    train_flow a2_fk phoenix_smplx "method.init_ae=$PRE/ae_union_jepa_d64_fk/ae.pt" \
        "method.flow.updates=$u"
    ;;
scale)
    for d in csl_smplx how2sign_smplx_soke; do
        diagnose "$PRE/ae_union_jepa_d64/ae.pt" "$d" "$DIAG/g1_jepa_d64_$d.json"
        train_flow "scale_${d}" "$d" "method.init_ae=$PRE/ae_union_jepa_d64/ae.pt" \
            "method.flow.updates=$(flow_updates "$d")"
    done
    ;;
*)
    sed -n '2,10p' "$0"
    exit 2
    ;;
esac
