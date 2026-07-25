#!/usr/bin/env bash
# =============================================================================
# SparseGPT HYBRID calibration: OKVQA text -> T5, CC3M images -> ViT,
# then prune + merge + four-benchmark eval.
#
# Same wrapper as the Wanda one but PRUNER=sparsegpt. The engine then adds the
# SparseGPT-only flags used by the sparsegpt split scripts:
#   --importance_scope {llm_only|vit_only_encode}
#   --score_method "$SCORE_METHOD"            (default MEZO-GradOnly_sum)
#   --max_sparsity_per_layer "$MAX_SPARSITY_PER_LAYER"  (default 0.6)
#   --num_data_first_stage "$NUM_DATA_FIRST_STAGE"
#
# Override paths via env: OKVQA_JSON, OKVQA_TEXT_JSON, NUM_DATA, SEED,
# SCORE_METHOD, MAX_SPARSITY_PER_LAYER, CC3M_ROOT/CC3M_IMAGES, etc.
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

PRUNER=sparsegpt TEXT_SOURCE=okvqa NUM_DATA="$NUM_DATA" SEED="$SEED" \
  bash "$SCRIPT_DIR/run_pure_wanda_split_text_t5_cc3m_image_vit_then_fourbench_eval.sh" "$@"
