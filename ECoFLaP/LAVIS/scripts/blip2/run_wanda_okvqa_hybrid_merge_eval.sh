#!/usr/bin/env bash
# =============================================================================
# Wanda OKVQA-text HYBRID, from EXISTING checkpoints -- no pruning.
#
# Factorized masks are separable: the T5 mask depends only on the text, the ViT
# mask only on the images. So the hybrid matrix (text = OKVQA, image = each of 5
# sources) is just: merge the ONE OKVQA-pruned T5 with each source's ViT, then
# run the four-benchmark eval. This driver does exactly that -- it never prunes.
#
# Override any path via env; defaults are the checkpoints you provided.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE="${BASE:-/data/data2/mfs}"

# --- the single OKVQA-pruned T5 (shared across all 5 rows) -------------------
T5_OKVQA="${T5_OKVQA:-$BASE/2/ECoFLaP/LAVIS/pruned_checkpoint/pure_wanda_split_okvqa_text_t5only_20260625_000939_seed42.pth}"

# --- per visual source: the ViT-only checkpoint -----------------------------
VIT_DIR="${VIT_DIR:-$BASE/2/LAVIS_backup/pruned_checkpoint}"
SOURCES=(mmbench mmmu okvqa mathvista cc3m)
VIT_CKPTS=(
  "${VIT_MMBENCH:-$VIT_DIR/pure_wanda_split_mmbench_image_vitonly_20260724_hybrid_split_seed42.pth}"
  "${VIT_MMMU:-$VIT_DIR/pure_wanda_split_mmmu_image_vitonly_20260724_hybrid_split_seed42.pth}"
  "${VIT_OKVQA:-$VIT_DIR/pure_wanda_split_okvqa_image_vitonly_20260724_hybrid_split_seed42.pth}"
  "${VIT_MATHVISTA:-$VIT_DIR/pure_wanda_split_mathvista_image_vitonly_20260724_hybrid_split_seed42.pth}"
  "${VIT_CC3M:-$VIT_DIR/pure_wanda_split_cc3m_image_vitonly_20260725_hybrid_split_cc3mc4_seed42.pth}"
)

OUT_DIR="${OUT_DIR:-$REPO_ROOT/pruned_checkpoint/hybrid_wanda_okvqa}"
OUT_JSONL="${OUT_JSONL:-$REPO_ROOT/lavis/output/BLIP2/hybrid_wanda_okvqa_fourbench.jsonl}"
mkdir -p "$OUT_DIR" "$(dirname "$OUT_JSONL")"

# --- preflight: fail fast if any checkpoint is missing ----------------------
[[ -f "$T5_OKVQA" ]] || { echo "[FATAL] T5 ckpt not found: $T5_OKVQA" >&2; exit 1; }
for i in "${!SOURCES[@]}"; do
  [[ -f "${VIT_CKPTS[$i]}" ]] || { echo "[FATAL] ViT ckpt not found (${SOURCES[$i]}): ${VIT_CKPTS[$i]}" >&2; exit 1; }
done

echo "[INFO] T5 (OKVQA text): $T5_OKVQA"
echo "[INFO] merged out dir:  $OUT_DIR"
echo "[INFO] metrics jsonl:   $OUT_JSONL"

# fresh combined metrics file (comment out to append instead)
: > "$OUT_JSONL"

for i in "${!SOURCES[@]}"; do
  src="${SOURCES[$i]}"
  vit="${VIT_CKPTS[$i]}"
  merged="$OUT_DIR/merged_wanda_vis-${src}_txt-okvqa.pth"
  echo ""
  echo "############################################################"
  echo "## hybrid row: visual=${src}  text=okvqa"
  echo "############################################################"
  CKPT_T5_ONLY="$T5_OKVQA" \
  CKPT_VIT_ONLY="$vit" \
  MERGED_CKPT="$merged" \
  RUN_MERGE=1 RUN_EVAL=1 \
  LAVIS_METRICS_JSONL="$OUT_JSONL" \
  EVAL_JOB_OKVQA="okvqa_eval_hybrid_wanda_vis-${src}_txt-okvqa" \
    bash "$SCRIPT_DIR/run_ecoflap_split_merge_eval_fourbench.sh" "$@"
done

echo ""
echo "########## ALL 5 HYBRID ROWS DONE ##########"
echo "  merged ckpts: $OUT_DIR"
echo "  metrics:      $OUT_JSONL"
echo "  -> each row is tagged by its ViT ckpt stem inside the jsonl."
