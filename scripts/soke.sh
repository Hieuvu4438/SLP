#!/usr/bin/env bash
# Always run upstream from its repository root (its assets use relative paths).
set -euo pipefail
slp_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
soke_repo="$slp_root/third_party/SOKE"
soke_python="$soke_repo/.venv/bin/python"
soke_action="${1:-check}"
if (( $# )); then shift; fi
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
case "$soke_action" in
  check) exec "$soke_python" "$slp_root/scripts/check_soke.py" "$@" ;;
  smoke) exec "$soke_python" "$slp_root/scripts/check_soke.py" --smoke "$@" ;;
  train-deto|test-deto|train-soke|test-soke|tokenize) ;;
  *) echo "Usage: bash scripts/soke.sh {check|smoke|train-deto|test-deto|train-soke|test-soke|tokenize} [args]" >&2; exit 2 ;;
esac
soke_config=soke
case "$soke_action" in
  *-deto) soke_config=deto ;;
esac
case "$soke_action" in
  train-*) soke_module=train ;;
  test-*) soke_module=test ;;
  tokenize) soke_module=scripts.get_motion_code ;;
esac
if [[ "$soke_action" == test-soke && ! -s "$soke_repo/experiments/mgpt/SOKE/checkpoints/last.ckpt" ]]; then
  echo 'Missing trained SOKE generator: experiments/mgpt/SOKE/checkpoints/last.ckpt (upstream issue #2).' >&2
  exit 1
fi
cd -- "$soke_repo"
exec "$soke_python" -m "$soke_module" \
  --cfg "$slp_root/configs/soke/$soke_config.local.yaml" \
  --cfg_assets "$slp_root/configs/soke/assets.local.yaml" \
  --use_gpus "${SOKE_GPUS:-0}" --nodebug "$@"
