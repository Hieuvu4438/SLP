#!/usr/bin/env bash
# Table 4: number of prototypes K and kernel bandwidth sigma (CSL-Daily).
# K=500 / sigma=0.5 is the full model (signmflow_main_csl_smplx).
source "$(dirname "$0")/_common.sh"
D=csl_smplx
for K in 250 750; do
  $RUN --data $D --name signmflow_measure_K${K} --seeds $SEEDS method.measure.K=$K
  python scripts/aggregate.py --name signmflow_measure_K${K}
done
for S in ${SEEDS//,/ }; do   # sigma only changes the measure: reuse the K=500 prototypes
  CK=$(last_run signmflow_main_csl_smplx $S)/checkpoints
  for SIG in 0.25 0.75; do
    $RUN --data $D --name signmflow_measure_sigma${SIG} --seeds $S method.measure.sigma=$SIG \
      method.init.prototypes=$CK/prototypes.pt
  done
done
for SIG in 0.25 0.75; do python scripts/aggregate.py --name signmflow_measure_sigma${SIG}; done
