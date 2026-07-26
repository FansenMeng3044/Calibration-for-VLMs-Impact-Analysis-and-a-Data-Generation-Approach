#!/usr/bin/env bash
# =============================================================================
# Pure Wanda: fixed CC3M-caption T5-only checkpoint + four image-only ViT
# checkpoints, then merge and run four-benchmark evaluation for each pair.
#
# This script never prunes. It only:
#   1) selects an existing CC3M-caption T5-only checkpoint;
#   2) merges it with each of the four specified ViT-only checkpoints;
#   3) evaluates every merged checkpoint on MMBench / MMMU / OKVQA / MathVista.
#
# Default T5 search order:
#   pure_wanda_cc3m_caption_t5only*20260621*025350_seed42.pth
#   pure_wanda_cc3m_caption_t5only*20260621*023721_seed42.pth
#
# Override paths with:
#   T5_CC3M_CAPTION=/path/t5.pth
#   VIT_MMBENCH=/path/vit.pth
#   VIT_MMMU=/path/vit.pth
#   VIT_OKVQA=/path/vit.pth
#   VIT_MATHVISTA=/path/vit.pth
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE="${BASE:-/data/data2/mfs}"
CKPT_DIR="${CKPT_DIR:-$BASE/2/ECoFLaP/LAVIS/pruned_checkpoint}"
VIT_DIR="${VIT_DIR:-$CKPT_DIR}"

_first_existing_glob() {
  local pattern
  for pattern in "$@"; do
    # Intentionally allow glob expansion here.
    for path in $pattern; do
      [[ -f "$path" ]] || continue
      echo "$path"
      return 0
    done
  done
  return 1
}

T5_CC3M_CAPTION="${T5_CC3M_CAPTION:-}"
if [[ -z "$T5_CC3M_CAPTION" ]]; then
  if ! T5_CC3M_CAPTION="$(_first_existing_glob \
    "$CKPT_DIR/pure_wanda_cc3m_caption_t5only"*20260621*025350_seed42.pth \
    "$CKPT_DIR/pure_wanda_cc3m_caption_t5only"*20260621*023721_seed42.pth \
    "$REPO_ROOT/pruned_checkpoint/pure_wanda_cc3m_caption_t5only"*20260621*025350_seed42.pth \
    "$REPO_ROOT/pruned_checkpoint/pure_wanda_cc3m_caption_t5only"*20260621*023721_seed42.pth)"; then
    echo "[FATAL] Cannot find CC3M-caption T5-only checkpoint." >&2
    echo "        Set T5_CC3M_CAPTION=/path/to/pure_wanda_cc3m_caption_t5only_*.pth" >&2
    exit 1
  fi
fi

SOURCES=(mmbench mmmu okvqa mathvista)
VIT_CKPTS=(
  "${VIT_MMBENCH:-$VIT_DIR/pure_wanda_split_mmbench_image_vitonly_20260625_000939_seed42.pth}"
  "${VIT_MMMU:-$VIT_DIR/pure_wanda_split_mmmu_image_vitonly_20260625_000939_seed42.pth}"
  "${VIT_OKVQA:-$VIT_DIR/pure_wanda_split_okvqa_image_vitonly_20260625_000939_seed42.pth}"
  "${VIT_MATHVISTA:-$VIT_DIR/pure_wanda_split_mathvista_image_vitonly_20260625_154315_seed42.pth}"
)

OUT_DIR="${OUT_DIR:-$REPO_ROOT/pruned_checkpoint/cc3m_caption_t5_with_four_source_vit}"
OUT_JSONL="${OUT_JSONL:-$REPO_ROOT/lavis/output/BLIP2/cc3m_caption_t5_four_vit_fourbench_metrics.jsonl}"
APPEND_METRICS="${APPEND_METRICS:-0}"
mkdir -p "$OUT_DIR" "$(dirname "$OUT_JSONL")"

[[ -f "$T5_CC3M_CAPTION" ]] || { echo "[FATAL] T5 ckpt not found: $T5_CC3M_CAPTION" >&2; exit 1; }
for i in "${!SOURCES[@]}"; do
  [[ -f "${VIT_CKPTS[$i]}" ]] || {
    echo "[FATAL] ViT ckpt not found (${SOURCES[$i]}): ${VIT_CKPTS[$i]}" >&2
    exit 1
  }
done

echo "[INFO] repo root:      $REPO_ROOT"
echo "[INFO] checkpoint dir: $CKPT_DIR"
echo "[INFO] T5 checkpoint:  $T5_CC3M_CAPTION"
echo "[INFO] merged out dir: $OUT_DIR"
echo "[INFO] metrics jsonl:  $OUT_JSONL"

if [[ "$APPEND_METRICS" != "1" ]]; then
  : > "$OUT_JSONL"
fi

for i in "${!SOURCES[@]}"; do
  src="${SOURCES[$i]}"
  vit="${VIT_CKPTS[$i]}"
  t5_stem="$(basename "$T5_CC3M_CAPTION" .pth)"
  vit_stem="$(basename "$vit" .pth)"
  merged="$OUT_DIR/merged_${t5_stem}__vit_${vit_stem}.pth"

  echo ""
  echo "############################################################"
  echo "## text=T5 CC3M-caption  visual=${src}"
  echo "## vit=$vit"
  echo "############################################################"

  CKPT_T5_ONLY="$T5_CC3M_CAPTION" \
  CKPT_VIT_ONLY="$vit" \
  MERGED_CKPT="$merged" \
  RUN_MERGE=1 \
  RUN_EVAL=1 \
  LAVIS_METRICS_JSONL="$OUT_JSONL" \
  LAVIS_EVAL_CALIB_TAG="cc3m_caption_t5__${src}_vit" \
  EVAL_JOB_OKVQA="okvqa_eval_cc3m_caption_t5__${src}_vit" \
    bash "$SCRIPT_DIR/run_ecoflap_split_merge_eval_fourbench.sh" "$@"
done

echo ""
echo "########## DONE: CC3M-caption T5 + four ViT checkpoints ##########"
echo "  merged checkpoints: $OUT_DIR"
echo "  metrics jsonl:      $OUT_JSONL"
