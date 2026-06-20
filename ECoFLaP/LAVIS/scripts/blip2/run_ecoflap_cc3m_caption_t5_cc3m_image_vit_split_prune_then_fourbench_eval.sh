#!/usr/bin/env bash
# =============================================================================
# ECoFLaP split pruning with matched CC3M modalities.
#
#   T5 side: CC3M captions only -> t5_c4_text path -> prune T5 only.
#   ViT side: CC3M images only -> vit_image_only path -> prune ViT only.
#
# This is intentionally parallel to:
#   run_ecoflap_c4_t5_cc3m_vit_split_prune_then_fourbench_eval.sh
#
# The only methodological change is replacing C4 text calibration with the
# caption field from the same 128 CC3M rows used by the ViT image calibration.
# `build_text_only_dataloader()` already accepts JSON dicts with a `caption`
# field, so no model/pruner logic needs to change.
#
# Usage from ECoFLaP/LAVIS:
#   bash scripts/blip2/run_ecoflap_cc3m_caption_t5_cc3m_image_vit_split_prune_then_fourbench_eval.sh
#
# Useful overrides:
#   BASE=/data/data2/mfs
#   CC3M_ROOT=/data/data2/mfs/CC3M_calib_128
#   CC3M_JSON=/data/data2/mfs/CC3M_calib_128/cc3m_calib_128.json
#   RUN_PRUNE=1 RUN_EVAL=1
#   RUN_PRUNE_T5=1 RUN_PRUNE_VIT=1
#   JOB_STAMP=...
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

BASE="${BASE:-${AUTODL_TMP:-/data/data2/mfs}}"
AUTODL_TMP="$BASE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$BASE/model_cache}"
ECOFLAP_ENV="${ECOFLAP_ENV:-${MODEL_CACHE_ROOT}/ecoflap_model_env.sh}"
if [[ -f "${ECOFLAP_ENV}" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${ECOFLAP_ENV}"
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
mkdir -p "${HF_HOME}/hub" "${HF_HOME}/transformers" "${TORCH_HOME}/hub/checkpoints" 2>/dev/null || true

_resolve_hub_snapshot_dir() {
  local repo_dir="$1"
  [[ -d "$repo_dir/snapshots" ]] || { echo ""; return 0; }
  find "$repo_dir/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | head -1
}

HUB_ROOT="${HUGGINGFACE_HUB_CACHE}"
if [[ -z "${BERT_BASE_UNCASED_SNAPSHOT:-}" ]]; then
  BERT_BASE_UNCASED_SNAPSHOT="$(_resolve_hub_snapshot_dir "$HUB_ROOT/models--bert-base-uncased")"
  BERT_BASE_UNCASED_SNAPSHOT="${BERT_BASE_UNCASED_SNAPSHOT:-$HF_HOME/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594}"
fi
if [[ -z "${FLAN_T5_XL_SNAPSHOT:-}" ]]; then
  FLAN_T5_XL_SNAPSHOT="$(_resolve_hub_snapshot_dir "$HUB_ROOT/models--google--flan-t5-xl")"
  FLAN_T5_XL_SNAPSHOT="${FLAN_T5_XL_SNAPSHOT:-$HF_HOME/hub/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001}"
fi
export BERT_BASE_UNCASED_SNAPSHOT
export FLAN_T5_XL_SNAPSHOT

BLIP2_PRETRAINED="${BLIP2_PRETRAINED:-$TORCH_HOME/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
if [[ ! -f "${BLIP2_PRETRAINED}" ]]; then
  BLIP2_PRETRAINED="$BASE/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth"
fi
export BLIP2_PRETRAINED

_resolve_eva_vit_g() {
  local candidates=(
    "${EVA_VIT_G_PTH:-}"
    "${TORCH_HOME}/hub/checkpoints/eva_vit_g.pth"
    "${MODEL_CACHE_ROOT}/torch/hub/checkpoints/eva_vit_g.pth"
    "${HOME}/.cache/torch/hub/checkpoints/eva_vit_g.pth"
    /root/autodl-tmp/cache_moved/torch/hub/checkpoints/eva_vit_g.pth
  )
  local p
  for p in "${candidates[@]}"; do
    [[ -n "$p" && -f "$p" ]] || continue
    echo "$p"
    return 0
  done
  return 1
}
if EVA_VIT_G_PTH="$(_resolve_eva_vit_g || true)" && [[ -n "$EVA_VIT_G_PTH" ]]; then
  export EVA_VIT_G_PTH
else
  echo "[FATAL] eva_vit_g.pth not found. Set EVA_VIT_G_PTH or TORCH_HOME." >&2
  exit 1
fi

CC3M_ROOT="${CC3M_ROOT:-$BASE/CC3M_calib_128}"
CC3M_JSON="${CC3M_JSON:-$CC3M_ROOT/cc3m_calib_128.json}"
CC3M_IMAGES="${CC3M_IMAGES:-$CC3M_ROOT/images}"

# CFG_T5 only loads the BLIP2 model and gives evaluate_blip a valid config.
# The actual T5 text-only dataloader comes from --c4_calib_json, which we point
# to CC3M_JSON here.  The variable name is historical in evaluate_blip.py.
CFG_T5="${CFG_T5:-$REPO_ROOT/lavis/projects/blip2/eval/t5_c4_text_prune_calib.yaml}"
CC3M_CFG="${CC3M_CFG:-lavis/projects/blip2/eval/cc_prefix_derivative_compute_cc3m_calib128.yaml}"

T5_SPEC="${T5_SPEC:-${T5_PRUNE_SPEC:-24-0.5-1.0-1.0}}"
VIT_SPEC="${VIT_SPEC:-${VIT_PRUNE_SPEC:-39-0.5-1.0-1.0}}"
NUM_DATA="${NUM_DATA:-128}"
NUM_DATA_FIRST_STAGE="${NUM_DATA_FIRST_STAGE:-128}"
PRUNING_CALIB_BATCH="${PRUNING_CALIB_BATCH:-8}"
SCORE_METHOD="${SCORE_METHOD:-MEZO-GradOnly_sum}"
SPARSITY_GRANULARITY="${SPARSITY_GRANULARITY:-block}"
MAX_SPARSITY_PER_LAYER="${MAX_SPARSITY_PER_LAYER:-0.6}"
MASTER_PORT_VIT="${MASTER_PORT_VIT:-29517}"

export ECOFLAP_BENCH_ROOT="${ECOFLAP_BENCH_ROOT:-$BASE}"
export MMBENCH_ROOT="${MMBENCH_ROOT:-$BASE/MMBench_eval}"
export MMMU_ROOT="${MMMU_ROOT:-$BASE/MMMU_single_image}"
export MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
export MMMU_SPLIT="${MMMU_SPLIT:-test}"
export MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-$BASE/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
export MATHVISTA_IMAGES_DIR="${MATHVISTA_IMAGES_DIR:-$BASE/MathVista_eval_testmini_mc/images}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
MASTER_PORT_EVAL="${MASTER_PORT_EVAL:-29600}"

SEED="${SEED:-${LAVIS_DISTRIBUTED_SAMPLER_SEED:-42}}"
export LAVIS_DISTRIBUTED_SAMPLER_SEED="$SEED"
JOB_STAMP="${JOB_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PRUNE_T5="${RUN_PRUNE_T5:-$RUN_PRUNE}"
RUN_PRUNE_VIT="${RUN_PRUNE_VIT:-$RUN_PRUNE}"

JID_T5="ecoflap_cc3m_caption_t5only_${JOB_STAMP}_seed${SEED}"
JID_VIT="ecoflap_cc3m_image_vitonly_${JOB_STAMP}_seed${SEED}"
CKPT_T5="$REPO_ROOT/pruned_checkpoint/${JID_T5}.pth"
CKPT_VIT="$REPO_ROOT/pruned_checkpoint/${JID_VIT}.pth"
MERGED_CKPT="${MERGED_CKPT:-$REPO_ROOT/pruned_checkpoint/merged_ecoflap_cc3m_caption_t5_${JID_T5}__image_vit_${JID_VIT}.pth}"

METRICS_JSONL="${LAVIS_METRICS_JSONL:-$REPO_ROOT/lavis/output/BLIP2/ecoflap_cc3m_caption_t5_image_vit_split_${JOB_STAMP}_seed${SEED}.jsonl}"
SUMMARY_MD="${SUMMARY_MD:-$REPO_ROOT/lavis/output/BLIP2/ecoflap_cc3m_caption_t5_image_vit_split_${JOB_STAMP}_seed${SEED}.md}"
SUMMARY_TSV="${SUMMARY_TSV:-$REPO_ROOT/lavis/output/BLIP2/ecoflap_cc3m_caption_t5_image_vit_split_${JOB_STAMP}_seed${SEED}.tsv}"
EVAL_TAG="ecoflap_cc3m_caption_t5_image_vit_${JOB_STAMP}_seed${SEED}"
EVAL_JOB_OKVQA="${EVAL_JOB_OKVQA:-okvqa_eval_${EVAL_TAG}_fullval}"

mkdir -p "$REPO_ROOT/pruned_checkpoint" "$REPO_ROOT/lavis/output/BLIP2" "$REPO_ROOT/training_statistics"

_preflight() {
  if [[ ! -f "$BLIP2_PRETRAINED" ]]; then
    echo "[FATAL] BLIP2_PRETRAINED not found: $BLIP2_PRETRAINED" >&2
    exit 1
  fi
  if [[ ! -d "$BERT_BASE_UNCASED_SNAPSHOT" ]]; then
    echo "[FATAL] bert-base-uncased snapshot not found: $BERT_BASE_UNCASED_SNAPSHOT" >&2
    exit 1
  fi
  if [[ ! -d "$FLAN_T5_XL_SNAPSHOT" ]]; then
    echo "[FATAL] flan-t5-xl snapshot not found: $FLAN_T5_XL_SNAPSHOT" >&2
    exit 1
  fi
  if [[ ! -d "$REPO_ROOT/lavis/datasets" ]]; then
    echo "[FATAL] missing lavis/datasets: $REPO_ROOT/lavis/datasets" >&2
    exit 1
  fi
  if [[ ! -f "$CC3M_JSON" ]]; then
    echo "[FATAL] CC3M_JSON not found: $CC3M_JSON" >&2
    exit 1
  fi
  if [[ ! -d "$CC3M_IMAGES" ]]; then
    echo "[FATAL] CC3M_IMAGES not found: $CC3M_IMAGES" >&2
    exit 1
  fi
  python - "$CC3M_JSON" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    rows = json.load(f)
if not isinstance(rows, list) or not rows:
    raise SystemExit(f"[FATAL] {path} is not a non-empty JSON list")
missing = [i for i, row in enumerate(rows[:128]) if not isinstance(row, dict) or not str(row.get("caption", "")).strip()]
if missing:
    raise SystemExit(f"[FATAL] first 128 CC3M rows missing non-empty caption at indices: {missing[:10]}")
missing_img = [i for i, row in enumerate(rows[:128]) if not isinstance(row, dict) or "image" not in row]
if missing_img:
    raise SystemExit(f"[FATAL] first 128 CC3M rows missing image at indices: {missing_img[:10]}")
print(f"[OK] CC3M rows checked: {min(128, len(rows))}, caption+image fields present")
PY
}

run_prune_t5_cc3m_caption() {
  echo ""
  echo ">>> [prune 1/2] ECoFLaP Wanda+MEZO+block | CC3M captions only | prune T5"
  local extra=()
  if [[ "${T5_CC3M_CAPTION_ENCODER_ONLY:-0}" == "1" ]]; then
    extra+=(--t5_c4_encoder_only)
  fi
  python evaluate_blip.py \
    --cfg-path "$CFG_T5" \
    --options "model.pretrained=${BLIP2_PRETRAINED}" \
    --prune_calib_mode t5_c4_text \
    --c4_calib_json "$CC3M_JSON" \
    --no_prune_vit \
    --pruning_method blipt5_wanda_pruner \
    --score_method "$SCORE_METHOD" \
    --sparsity_ratio_granularity "$SPARSITY_GRANULARITY" \
    --max_sparsity_per_layer "$MAX_SPARSITY_PER_LAYER" \
    --t5_prune_spec "$T5_SPEC" \
    --num_data "$NUM_DATA" \
    --prunining_dataset_batch_size "$PRUNING_CALIB_BATCH" \
    --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
    --job_id "$JID_T5" \
    --save_pruned_model \
    "${extra[@]}"
  [[ -f "$CKPT_T5" ]] || { echo "[FATAL] missing T5 ckpt after pruning: $CKPT_T5" >&2; exit 1; }
}

run_prune_vit_cc3m_image_only() {
  echo ""
  echo ">>> [prune 2/2] ECoFLaP Wanda+MEZO+block | CC3M images only | prune ViT"
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$MASTER_PORT_VIT" evaluate_blip.py \
    --cfg-path "$CC3M_CFG" \
    --options "model.pretrained=${BLIP2_PRETRAINED}" \
    --prune_calib_mode vit_image_only \
    --no_prune_t5 \
    --pruning_method blipt5_wanda_pruner \
    --score_method "$SCORE_METHOD" \
    --sparsity_ratio_granularity "$SPARSITY_GRANULARITY" \
    --max_sparsity_per_layer "$MAX_SPARSITY_PER_LAYER" \
    --vit_prune_spec "$VIT_SPEC" \
    --num_data "$NUM_DATA" \
    --prunining_dataset_batch_size "$PRUNING_CALIB_BATCH" \
    --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
    --job_id "$JID_VIT" \
    --save_pruned_model
  [[ -f "$CKPT_VIT" ]] || { echo "[FATAL] missing ViT ckpt after pruning: $CKPT_VIT" >&2; exit 1; }
}

echo "========== ECoFLaP split: CC3M captions -> T5 + CC3M images -> ViT =========="
echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] CC3M_JSON=$CC3M_JSON"
echo "[INFO] CC3M_IMAGES=$CC3M_IMAGES"
echo "[INFO] CFG_T5=$CFG_T5"
echo "[INFO] CC3M_CFG=$CC3M_CFG"
echo "[INFO] T5 ckpt: $CKPT_T5"
echo "[INFO] ViT ckpt: $CKPT_VIT"
echo "[INFO] Merged:   $MERGED_CKPT"
echo "[INFO] RUN_PRUNE=$RUN_PRUNE (T5=$RUN_PRUNE_T5 ViT=$RUN_PRUNE_VIT) RUN_EVAL=$RUN_EVAL"
echo "============================================================================="

_preflight

if [[ "$RUN_PRUNE_T5" == "1" ]]; then
  run_prune_t5_cc3m_caption
else
  echo "[INFO] RUN_PRUNE_T5=0, using existing T5 ckpt: $CKPT_T5"
  [[ -f "$CKPT_T5" ]] || { echo "[FATAL] missing T5 ckpt: $CKPT_T5" >&2; exit 1; }
fi

if [[ "$RUN_PRUNE_VIT" == "1" ]]; then
  run_prune_vit_cc3m_image_only
else
  echo "[INFO] RUN_PRUNE_VIT=0, using existing ViT ckpt: $CKPT_VIT"
  [[ -f "$CKPT_VIT" ]] || { echo "[FATAL] missing ViT ckpt: $CKPT_VIT" >&2; exit 1; }
fi

if [[ "$RUN_EVAL" != "1" ]]; then
  echo "[INFO] RUN_EVAL=0, skip merge and benchmark evaluation"
  echo "[INFO] T5: $CKPT_T5"
  echo "[INFO] ViT: $CKPT_VIT"
  exit 0
fi

: > "$METRICS_JSONL"

export CKPT_T5_ONLY="$CKPT_T5"
export CKPT_VIT_ONLY="$CKPT_VIT"
export MERGED_CKPT
export RUN_MERGE=1
export RUN_EVAL=1
export EVAL_BATCH_SIZE
export MASTER_PORT="$MASTER_PORT_EVAL"
export EVAL_JOB_OKVQA
export LAVIS_METRICS_JSONL="$METRICS_JSONL"
export LAVIS_EVAL_CALIB_TAG="$EVAL_TAG"

echo ""
echo ">>> [merge + four benchmark eval] run_ecoflap_split_merge_eval_fourbench.sh"
bash "$SCRIPT_DIR/run_ecoflap_split_merge_eval_fourbench.sh"

if [[ -f "$REPO_ROOT/scripts/blip2/collect_ecoflap_eval_summary.py" ]]; then
  echo ""
  echo ">>> [summary] collect_ecoflap_eval_summary.py"
  python "$REPO_ROOT/scripts/blip2/collect_ecoflap_eval_summary.py" \
    --repo-root "$REPO_ROOT" \
    --metrics-jsonl "$METRICS_JSONL" \
    --out-md "$SUMMARY_MD" \
    --out-tsv "$SUMMARY_TSV" \
    --suites "${EVAL_TAG}:${EVAL_JOB_OKVQA}" || true
fi

echo ""
echo "========== ALL DONE =========="
echo "  T5 caption-only: $CKPT_T5"
echo "  ViT image-only:  $CKPT_VIT"
echo "  Merged:          $MERGED_CKPT"
echo "  Metrics:         $METRICS_JSONL"
echo "  Summary:         $SUMMARY_MD"
