#!/usr/bin/env bash
# =============================================================================
# Pure Wanda HYBRID calibration: OKVQA text -> T5, CC3M images -> ViT,
# then prune + merge + four-benchmark eval.
#
# Thin wrapper over the parameterized engine
#   run_pure_wanda_split_text_t5_cc3m_image_vit_then_fourbench_eval.sh
# with PRUNER=wanda and TEXT_SOURCE=okvqa. It first materialises the OKVQA
# text-only calibration json (list of strings) if it is missing.
#
# Override paths via env: OKVQA_JSON (raw annotations), OKVQA_TEXT_JSON (output),
# NUM_DATA, SEED, CC3M_ROOT/CC3M_IMAGES, etc. (all forwarded to the engine).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${BASE:-${AUTODL_TMP:-/data/data2/mfs}}"

OKVQA_JSON="${OKVQA_JSON:-$BASE/datasets/okvqa/annotations/okvqa_train.json}"
export OKVQA_TEXT_JSON="${OKVQA_TEXT_JSON:-$BASE/okvqa_text_128.json}"
NUM_DATA="${NUM_DATA:-128}"
SEED="${SEED:-42}"

if [[ ! -f "$OKVQA_TEXT_JSON" ]]; then
  echo ">>> building OKVQA text calib: $OKVQA_TEXT_JSON"
  python "$SCRIPT_DIR/build_okvqa_text_calib.py" \
    --okvqa_json "$OKVQA_JSON" --out "$OKVQA_TEXT_JSON" \
    --num "$NUM_DATA" --seed "$SEED"
else
  echo ">>> reusing existing OKVQA text calib: $OKVQA_TEXT_JSON"
fi

PRUNER=wanda TEXT_SOURCE=okvqa NUM_DATA="$NUM_DATA" SEED="$SEED" \
  bash "$SCRIPT_DIR/run_pure_wanda_split_text_t5_cc3m_image_vit_then_fourbench_eval.sh" "$@"
