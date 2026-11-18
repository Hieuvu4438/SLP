#!/usr/bin/env bash
# Tables 1 & 2: SignNMFlow on Phoenix-2014T, CSL-Daily, How2Sign (single-dataset models),
# and SignNMFlow-M (one model trained on all three, evaluated on each).
source "$(dirname "$0")/_common.sh"
SEEDS=${SEEDS:-0,1,2,3,4}
for D in phoenix_smplx csl_smplx how2sign_smplx; do
  $RUN --data $D --name signmflow_main_${D} --seeds $SEEDS
  python scripts/aggregate.py --name signmflow_main_${D}
done
# SignNMFlow-M: train once per seed (on the Phoenix run), reuse the checkpoints for the others.
$RUN --data phoenix_smplx --name signmflow_M_phoenix_smplx --seeds $SEEDS method.multilingual=true
for S in ${SEEDS//,/ }; do
  CK=$(last_run signmflow_M_phoenix_smplx $S)/checkpoints
  for D in csl_smplx how2sign_smplx; do
    $RUN --data $D --name signmflow_M_${D} --seeds $S method.multilingual=true \
      method.init.prototypes=$CK/prototypes.pt method.init.correction=$CK/correction_final.pt \
      method.init.generator=$CK/generator.pt
  done
done
for D in phoenix_smplx csl_smplx how2sign_smplx; do python scripts/aggregate.py --name signmflow_M_${D}; done
