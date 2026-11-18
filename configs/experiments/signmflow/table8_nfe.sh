#!/usr/bin/env bash
# Table 8: number of function evaluations at sampling (CSL-Daily); re-uses the trained full model.
source "$(dirname "$0")/_common.sh"
D=csl_smplx
for S in ${SEEDS//,/ }; do
  CK=$(last_run signmflow_main_csl_smplx $S)/checkpoints
  for NFE in 5 10 20; do
    $RUN --data $D --name signmflow_nfe${NFE} --seeds $S method.sampler.steps=$NFE \
      method.init.prototypes=$CK/prototypes.pt method.init.correction=$CK/correction_final.pt \
      method.init.generator=$CK/generator.pt
  done
done
for NFE in 5 10 20; do python scripts/aggregate.py --name signmflow_nfe${NFE}; done
