#!/usr/bin/env bash
# B-T BLEU-4 column (Tables 1-2) for every seed of an experiment, using the repo's mBART
# back-translators (outputs/bt/<source>_mbart). SOKE's back-translator is not public, so these
# numbers are comparable across our runs but not directly to the paper's B-T column.
#   bash configs/experiments/signmflow/bt_score.sh signmflow_main_phoenix_smplx phoenix
source "$(dirname "$0")/_common.sh"
NAME=$1; SRC=$2
PY=${BT_PYTHON:-third_party/MaDiS/.venv/bin/python}   # needs sacrebleu
SPECS=""
for d in outputs/runs/$NAME/signmflow/seed*; do [ -f $d/predictions.npz ] && SPECS="$SPECS $(basename $d)=$d/predictions.npz"; done
$PY scripts/train_backtranslator.py --arch mbart --source $SRC --out outputs/bt/${SRC}_mbart \
  --score $SPECS --score_name $NAME
