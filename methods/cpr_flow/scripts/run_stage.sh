#!/usr/bin/env bash
set -euo pipefail
stage="${1:?usage: run_stage.sh ae|duration|baseline|relation|teacher|diffusion|full [CLI options]}"
shift
case "$stage" in
  ae) run=ae_joint3; task=ae ;;
  duration) run=duration_joint3; task=duration ;;
  baseline) run=baseline_joint3; task=flow ;;
  relation) run=relation_joint3; task=flow ;;
  teacher) run=teacher_joint3; task=teacher ;;
  diffusion) run=diffusion_joint3; task=flow ;;
  full) run=full_joint3; task=flow ;;
  *) exit 2 ;;
esac
python -m cprflow.cli.train --config "configs/runs/${run}.yaml" --task "$task" "$@"
