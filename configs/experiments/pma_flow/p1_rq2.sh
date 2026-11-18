#!/usr/bin/env bash
# P1 (proposal 5): RQ2 - joint-space loss + training hygiene vs the matched SignNMFlow-GE, and the
# Stage-A regression. Phoenix dev, seed 0; 4 runs (~1 GPU each; at most 2 in parallel).
#   replica      SignNMFlow-GE re-implemented in this code base (sanity vs methods/signmflow GE)
#   hygiene      + EMA, val selection, logit-normal t, mBART lr/freeze, warm-up, clip, bf16, CFG drop
#   hygiene_fk   + FK position / velocity loss                        (RQ2 = hygiene_fk vs replica)
#   anchor_only  Stage A alone (alpha = 0 reference; its anchor_net.pt is re-used by P2)
source "$(dirname "$0")/_common.sh"
for S in ${SEEDS//,/ }; do
  $RUN --name pma_replica_${DATA} --seeds $S $REPLICA
  $RUN --name pma_hygiene_${DATA} --seeds $S method.pipeline=uniform_flow \
       method.flow.loss.pos=0 method.flow.loss.vel=0
  $RUN --name pma_hygiene_fk_${DATA} --seeds $S method.pipeline=uniform_flow
  $RUN --name pma_anchor_only_${DATA} --seeds $S method.pipeline=anchor_only
done
