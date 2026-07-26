#!/usr/bin/env bash
# =============================================================================
# SparseGPT: CC3M text/caption -> T5-only pruning, then merge that T5 with four
# existing image-only ViT checkpoints and run four-benchmark evaluation.
#
# This script is for the hybrid question:
#   text side  = CC3M caption/text calibrated SparseGPT T5
#   image side = each existing SparseGPT ViT-only checkpoint below
#
# It performs:
#   1) optional T5-only SparseGPT pruning using CC3M text/caption;
#   2) merge T5 with each ViT-only checkpoint;
#   3) evaluate every merged checkpoint on MMBench / MMMU / OKVQA / MathVista.
#
# If the CC3M-text T5 checkpoint already exists, skip pruning with:
#   RUN_PRUNE_T5=0 T5_CC3M_TEXT=/path/to/t5.pth bash this_script.sh
#
# Override default ViT checkpoint paths with:
#   VIT_MMBENCH=...
#   VIT_MMMU=...
#   VIT_OKVQA=...
#   VIT_MATHVISTA=...
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE="${BASE:-${AUTODL_TMP:-/data/data2/mfs}}"
CKPT_DIR="${CKPT_DIR:-$BASE/2/ECoFLaP/LAVIS/pruned_checkpoint}"
VIT_DIR="${VIT_DIR:-$CKPT_DIR}"
cd "$REPO_ROOT" || exit 1

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

SEED="${SEED:-42}"
NUM_DATA="${NUM_DATA:-128}"
PRUNING_CALIB_BATCH="${PRUNING_CALIB_BATCH:-${BS:-8}}"
JOB_STAMP="${JOB_STAMP:-20260726_cc3m_text_sparsegpt}"
RUN_PRUNE_T5="${RUN_PRUNE_T5:-1}"
APPEND_METRICS="${APPEND_METRICS:-0}"

MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$BASE/model_cache}"
ECOFLAP_ENV="${ECOFLAP_ENV:-${MODEL_CACHE_ROOT}/ecoflap_model_env.sh}"
if [[ -f "$ECOFLAP_ENV" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "$ECOFLAP_ENV"
  set -u
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$MODEL_CACHE_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_HOME="${TORCH_HOME:-$MODEL_CACHE_ROOT/torch}"
mkdir -p "$HF_HOME/hub" "$HF_HOME/transformers" "$TORCH_HOME/hub/checkpoints" 2>/dev/null || true

_resolve_hub_snapshot_dir() {
  local repo_dir="$1"
  [[ -d "$repo_dir/snapshots" ]] || { echo ""; return 0; }
  find "$repo_dir/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | head -1
}

HUB_ROOT="$HUGGINGFACE_HUB_CACHE"
if [[ -z "${BERT_BASE_UNCASED_SNAPSHOT:-}" ]]; then
  BERT_BASE_UNCASED_SNAPSHOT="$(_resolve_hub_snapshot_dir "$HUB_ROOT/models--bert-base-uncased")"
fi
if [[ -z "${FLAN_T5_XL_SNAPSHOT:-}" ]]; then
  FLAN_T5_XL_SNAPSHOT="$(_resolve_hub_snapshot_dir "$HUB_ROOT/models--google--flan-t5-xl")"
fi
export BERT_BASE_UNCASED_SNAPSHOT
export FLAN_T5_XL_SNAPSHOT

BLIP2_PRETRAINED="${BLIP2_PRETRAINED:-$TORCH_HOME/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
if [[ ! -f "$BLIP2_PRETRAINED" ]]; then
  BLIP2_PRETRAINED="$MODEL_CACHE_ROOT/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth"
fi
export BLIP2_PRETRAINED

if [[ -z "${EVA_VIT_G_PTH:-}" ]]; then
  if [[ -f "$TORCH_HOME/hub/checkpoints/eva_vit_g.pth" ]]; then
    export EVA_VIT_G_PTH="$TORCH_HOME/hub/checkpoints/eva_vit_g.pth"
  elif [[ -f "$MODEL_CACHE_ROOT/torch/hub/checkpoints/eva_vit_g.pth" ]]; then
    export EVA_VIT_G_PTH="$MODEL_CACHE_ROOT/torch/hub/checkpoints/eva_vit_g.pth"
  fi
fi

CC3M_CALIB_JSON="${CC3M_CALIB_JSON:-$BASE/CC3M_calib_128/cc3m_calib_128.json}"
CFG_T5_BASE="${CFG_T5_BASE:-$REPO_ROOT/lavis/projects/blip2/eval/t5_c4_text_prune_calib.yaml}"
T5_SPEC="${T5_SPEC:-${T5_PRUNE_SPEC:-24-0.5-1.0-1.0}}"
SCORE_METHOD="${SCORE_METHOD:-MEZO-GradOnly_sum}"
MAX_SPARSITY_PER_LAYER="${MAX_SPARSITY_PER_LAYER:-0.6}"
NUM_DATA_FIRST_STAGE="${NUM_DATA_FIRST_STAGE:-$NUM_DATA}"

T5_CC3M_TEXT="${T5_CC3M_TEXT:-$REPO_ROOT/pruned_checkpoint/sparsegpt_split_cc3m_text_t5only_${JOB_STAMP}_seed${SEED}.pth}"

SOURCES=(mmbench mmmu okvqa mathvista)
VIT_CKPTS=(
  "${VIT_MMBENCH:-$VIT_DIR/sparsegpt_split_mmbench_t5okvqa_image_vitonly_20260726_okvqa_text_hybrid_sparsegpt_seed42.pth}"
  "${VIT_MMMU:-$VIT_DIR/sparsegpt_split_mmmu_t5okvqa_image_vitonly_20260726_okvqa_text_hybrid_sparsegpt_seed42.pth}"
  "${VIT_OKVQA:-$VIT_DIR/sparsegpt_split_okvqa_t5okvqa_image_vitonly_20260726_okvqa_text_hybrid_sparsegpt_seed42.pth}"
  "${VIT_MATHVISTA:-$VIT_DIR/sparsegpt_split_mathvista_t5okvqa_image_vitonly_20260726_okvqa_text_hybrid_sparsegpt_seed42.pth}"
)

OUT_DIR="${OUT_DIR:-$REPO_ROOT/pruned_checkpoint/sparsegpt_cc3m_text_t5_with_four_source_vit}"
OUT_JSONL="${OUT_JSONL:-$REPO_ROOT/lavis/output/BLIP2/sparsegpt_cc3m_text_t5_four_vit_fourbench_metrics.jsonl}"
mkdir -p "$OUT_DIR" "$(dirname "$OUT_JSONL")"

prune_t5_cc3m_text() {
  echo ""
  echo "############################################################"
  echo "## prune T5-only: SparseGPT with CC3M text/caption calibration"
  echo "############################################################"
  [[ -f "$BLIP2_PRETRAINED" ]] || { echo "[FATAL] BLIP2_PRETRAINED not found: $BLIP2_PRETRAINED" >&2; exit 1; }
  [[ -f "$CFG_T5_BASE" ]] || { echo "[FATAL] CFG_T5_BASE not found: $CFG_T5_BASE" >&2; exit 1; }
  [[ -f "$CC3M_CALIB_JSON" ]] || { echo "[FATAL] CC3M_CALIB_JSON not found: $CC3M_CALIB_JSON" >&2; exit 1; }
  if [[ ! -d "${FLAN_T5_XL_SNAPSHOT:-}" ]]; then
    echo "[FATAL] flan-t5-xl snapshot not found. Set HF_HOME or FLAN_T5_XL_SNAPSHOT." >&2
    exit 1
  fi

  python - "$CC3M_CALIB_JSON" "$NUM_DATA" <<'PY'
import json
import sys

path, n = sys.argv[1], int(sys.argv[2])
with open(path, "r", encoding="utf-8") as f:
    rows = json.load(f)
if not isinstance(rows, list) or len(rows) < n:
    raise SystemExit(f"[FATAL] {path} must contain at least {n} rows")
for i, row in enumerate(rows[:n]):
    if not isinstance(row, dict):
        raise SystemExit(f"[FATAL] row {i} is not a dict")
    if not any(str(row.get(k, "")).strip() for k in ("text", "caption", "text_input", "output")):
        raise SystemExit(f"[FATAL] row {i} has no text/caption/text_input/output for T5 text-only calibration")
print(f"[OK] CC3M text calibration rows checked: {n}")
PY

  local jid="sparsegpt_split_cc3m_text_t5only_${JOB_STAMP}_seed${SEED}"
  python evaluate_blip.py \
    --cfg-path "$CFG_T5_BASE" \
    --options "model.pretrained=${BLIP2_PRETRAINED}" \
    --prune_calib_mode t5_c4_text \
    --importance_scope llm_only \
    --c4_calib_json "$CC3M_CALIB_JSON" \
    --no_prune_vit \
    --pruning_method blipt5_sparsegpt_pruner \
    --t5_prune_spec "$T5_SPEC" \
    --score_method "$SCORE_METHOD" \
    --max_sparsity_per_layer "$MAX_SPARSITY_PER_LAYER" \
    --num_data "$NUM_DATA" \
    --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
    --prunining_dataset_batch_size "$PRUNING_CALIB_BATCH" \
    --job_id "$jid" \
    --save_pruned_model

  local produced="$REPO_ROOT/pruned_checkpoint/sparsegpt_split_cc3m_text_t5only_${JOB_STAMP}_seed${SEED}.pth"
  [[ -f "$produced" ]] || {
    echo "[FATAL] expected SparseGPT CC3M-text T5 checkpoint was not produced: $produced" >&2
    exit 1
  }
  T5_CC3M_TEXT="$produced"
}

if [[ "$RUN_PRUNE_T5" == "1" ]]; then
  prune_t5_cc3m_text
else
  echo "[INFO] RUN_PRUNE_T5=0, reuse T5 checkpoint: $T5_CC3M_TEXT"
fi

[[ -f "$T5_CC3M_TEXT" ]] || {
  echo "[FATAL] T5 checkpoint not found: $T5_CC3M_TEXT" >&2
  echo "        Either set RUN_PRUNE_T5=1 to create it or pass T5_CC3M_TEXT=/path/to/t5.pth." >&2
  exit 1
}

for i in "${!SOURCES[@]}"; do
  [[ -f "${VIT_CKPTS[$i]}" ]] || {
    echo "[FATAL] ViT ckpt not found (${SOURCES[$i]}): ${VIT_CKPTS[$i]}" >&2
    exit 1
  }
done

echo ""
echo "[INFO] repo root:      $REPO_ROOT"
echo "[INFO] T5 checkpoint:  $T5_CC3M_TEXT"
echo "[INFO] merged out dir: $OUT_DIR"
echo "[INFO] metrics jsonl:  $OUT_JSONL"

if [[ "$APPEND_METRICS" != "1" ]]; then
  : > "$OUT_JSONL"
fi

for i in "${!SOURCES[@]}"; do
  src="${SOURCES[$i]}"
  vit="${VIT_CKPTS[$i]}"
  t5_stem="$(basename "$T5_CC3M_TEXT" .pth)"
  vit_stem="$(basename "$vit" .pth)"
  merged="$OUT_DIR/merged_${t5_stem}__vit_${vit_stem}.pth"

  echo ""
  echo "############################################################"
  echo "## text=SparseGPT CC3M-text T5  visual=${src}"
  echo "## vit=$vit"
  echo "############################################################"

  CKPT_T5_ONLY="$T5_CC3M_TEXT" \
  CKPT_VIT_ONLY="$vit" \
  MERGED_CKPT="$merged" \
  RUN_MERGE=1 \
  RUN_EVAL=1 \
  LAVIS_METRICS_JSONL="$OUT_JSONL" \
  LAVIS_EVAL_CALIB_TAG="sparsegpt_cc3m_text_t5__${src}_vit" \
  EVAL_JOB_OKVQA="okvqa_eval_sparsegpt_cc3m_text_t5__${src}_vit" \
    bash "$SCRIPT_DIR/run_ecoflap_split_merge_eval_fourbench.sh" "$@"
done

echo ""
echo "########## DONE: SparseGPT CC3M-text T5 + four ViT checkpoints ##########"
echo "  T5 checkpoint:      $T5_CC3M_TEXT"
echo "  merged checkpoints: $OUT_DIR"
echo "  metrics jsonl:      $OUT_JSONL"
