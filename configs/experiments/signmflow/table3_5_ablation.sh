#!/usr/bin/env bash
# Table 3 (MR / GE / NG) and Table 5 (velocity vs target prediction), CSL-Daily.
#   row 1  6D + linear          (CFM baseline)          target
#   row 2  quat + linear        (+MR)                   target
#   row 3  quat + geodesic      (+MR +GE)               target
#   row 4  quat + naturalness   (+MR +GE +NG, full)     target  = signmflow_main_csl_smplx (not re-trained)
#   Table 5: CFM / Ours with velocity prediction.
source "$(dirname "$0")/_common.sh"
D=csl_smplx
$RUN --data $D --name signmflow_abl_cfm6d_target  --seeds $SEEDS method.flow.representation=6d method.flow.path=linear
$RUN --data $D --name signmflow_abl_mr_linear     --seeds $SEEDS method.flow.path=linear
$RUN --data $D --name signmflow_abl_mr_geodesic   --seeds $SEEDS method.flow.path=geodesic
$RUN --data $D --name signmflow_abl_cfm6d_velocity --seeds $SEEDS method.flow.representation=6d method.flow.path=linear method.flow.prediction=velocity
for S in ${SEEDS//,/ }; do   # Ours-velocity reuses the prototypes / correction of the full run
  CK=$(last_run signmflow_main_csl_smplx $S)/checkpoints
  $RUN --data $D --name signmflow_abl_full_velocity --seeds $S method.flow.prediction=velocity \
    method.init.prototypes=$CK/prototypes.pt method.init.correction=$CK/correction_final.pt
done
for N in cfm6d_target mr_linear mr_geodesic cfm6d_velocity full_velocity; do
  python scripts/aggregate.py --name signmflow_abl_$N
done
