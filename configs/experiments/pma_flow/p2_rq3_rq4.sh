#!/usr/bin/env bash
# P2 (proposal 5): RQ3 - anchored residual flow (alpha curve, two source scales) and RQ4 - CFG.
# Run only if gate G1/G2 passed. The full-data Stage A comes from P1's anchor_only run; the first
# run here adds the cross-fitted training anchors, the others re-use all of it (init.anchor_dir).
source "$(dirname "$0")/_common.sh"
for S in ${SEEDS//,/ }; do
  A0=$(last_run pma_anchor_only_${DATA} $S)
  $RUN --name pma_flow_${DATA} --seeds $S ${A0:+method.init.anchor_dir=$A0/checkpoints}   # sigma = auto
  A=$(last_run pma_flow_${DATA} $S)
  python -m methods.pma_flow.sweep --run "$A"
  $RUN --name pma_flow_s05_${DATA} --seeds $S method.flow.sigma_scale=0.5 \
       method.init.anchor_dir=$A/checkpoints
  python -m methods.pma_flow.sweep --run "$(last_run pma_flow_s05_${DATA} $S)"
  # RQ4: CFG is inference-only (the flow net was trained with cfg_drop = 0.1)
  for W in 1.5 2.5 4; do
    $RUN --name pma_flow_cfg${W}_${DATA} --seeds $S method.sampler.cfg_w=$W \
         method.init.anchor_dir=$A/checkpoints method.init.flow=$A/checkpoints/flow_net.pt
  done
  # NFE (sampler) sensitivity of the same model
  for N in 1 10; do
    $RUN --name pma_flow_nfe${N}_${DATA} --seeds $S method.sampler.steps=$N \
         method.init.anchor_dir=$A/checkpoints method.init.flow=$A/checkpoints/flow_net.pt
  done
done
