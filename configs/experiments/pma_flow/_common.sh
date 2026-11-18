# Shared helpers for the PMA-Flow experiment scripts (source, do not run).
# SEEDS: seed 0 for the gated phases P1-P3 (proposal 5 Sec. 5); 3 seeds only for the final config.
# SPLIT: val. Use SPLIT=test only when the user asks, from a clean committed tree.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
SEEDS=${SEEDS:-0}
SPLIT=${SPLIT:-val}
DATA=${DATA:-phoenix_smplx}
RUN="python scripts/run.py --method pma_flow --split ${SPLIT} --data ${DATA}"
# Latest *finished* run dir (has metrics.json) of an experiment for a seed
last_run() { ls -d outputs/runs/"$1"/pma_flow/seed"$2"_*/metrics.json 2>/dev/null | sort | tail -1 | xargs -r dirname; }
# SignNMFlow-GE inside this code base: no FK loss, no hygiene, uniform t, no CFG, fp32, Eq. 53 sampler
REPLICA="method.pipeline=uniform_flow method.flow.loss.pos=0 method.flow.loss.vel=0 \
 method.flow.t_sampling=uniform method.flow.cfg_drop=0 method.flow.align_pairs=false \
 method.train.ema_decay=null method.train.select=last method.train.freeze_text_epochs=0 \
 method.train.text_lr=2.0e-4 method.train.warmup_steps=0 method.train.grad_clip=null method.train.amp=none \
 method.sampler.kind=replan"
