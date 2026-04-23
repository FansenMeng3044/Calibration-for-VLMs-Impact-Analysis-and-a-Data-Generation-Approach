#!/usr/bin/env bash
set -euo pipefail

# 通用：在任意 LAVIS 仓库根目录下跑 MathVista testmini 多选题 eval（mathvista_mc_eval.py）
#
# 必填/常用环境变量：
#   LAVIS_REPO_ROOT  例如 /root/autodl-tmp/LAVIS_backup 或 .../ECoFLaP/LAVIS
#   CKPT_PATH        剪枝权重 .pth
# 可选：
#   EVAL_JSON / IMAGES_DIR / EVAL_BATCH_SIZE
#   LAVIS_EVAL_CALIB_TAG / LAVIS_METRICS_BENCHMARK / LAVIS_METRICS_JSONL
#   DATE_TAG + LAVIS_DISTRIBUTED_SAMPLER_SEED：仅在未设 CKPT_PATH 时拼默认 okvqa_cf_0.5_mathvista_overall_${DATE_TAG}_s${SEED}.pth

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache_moved/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export BERT_BASE_UNCASED_SNAPSHOT="${BERT_BASE_UNCASED_SNAPSHOT:-/root/autodl-tmp/cache_moved/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594}"
export FLAN_T5_XL_SNAPSHOT="${FLAN_T5_XL_SNAPSHOT:-/root/autodl-tmp/cache_moved/huggingface/hub/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001}"

AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
LAVIS_REPO_ROOT="${LAVIS_REPO_ROOT:-$AUTODL_TMP/LAVIS_backup}"
export LAVIS_DISTRIBUTED_SAMPLER_SEED="${LAVIS_DISTRIBUTED_SAMPLER_SEED:-0}"
export DATE_TAG="${DATE_TAG:-0327}"
SEED="${LAVIS_DISTRIBUTED_SAMPLER_SEED}"
DATE_TAG_FULL="${DATE_TAG}_s${SEED}"

export CKPT_PATH="${CKPT_PATH:-$LAVIS_REPO_ROOT/pruned_checkpoint/okvqa_cf_0.5_mathvista_overall_${DATE_TAG_FULL}.pth}"
export EVAL_JSON="${EVAL_JSON:-$AUTODL_TMP/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
export IMAGES_DIR="${IMAGES_DIR:-$AUTODL_TMP/MathVista_eval_testmini_mc/images}"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "[FATAL] checkpoint not found: $CKPT_PATH"
  exit 1
fi
if [[ ! -f "$EVAL_JSON" ]]; then
  echo "[FATAL] eval JSON not found: $EVAL_JSON"
  exit 1
fi

export LAVIS_METRICS_BENCHMARK="${LAVIS_METRICS_BENCHMARK:-MathVista_MC}"
export LAVIS_EVAL_CALIB_TAG="${LAVIS_EVAL_CALIB_TAG:-MathVistaoverall_seed${SEED}_${DATE_TAG_FULL}}"

SUMMARY_DIR="$LAVIS_REPO_ROOT/lavis/output/BLIP2"
mkdir -p "$SUMMARY_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
export LAVIS_METRICS_JSONL="${LAVIS_METRICS_JSONL:-$SUMMARY_DIR/mathvista_mc_eval_${LAVIS_EVAL_CALIB_TAG}_${STAMP}.jsonl}"
: > "$LAVIS_METRICS_JSONL"

cd "$LAVIS_REPO_ROOT"

echo "[INFO] LAVIS_REPO_ROOT=$LAVIS_REPO_ROOT"
echo "[INFO] ckpt=$CKPT_PATH"
echo "[INFO] eval_json=$EVAL_JSON"

python scripts/blip2/mathvista_mc_eval.py \
  --eval_json "$EVAL_JSON" \
  --images_dir "$IMAGES_DIR" \
  --ckpt "$CKPT_PATH" \
  --batch_size "${EVAL_BATCH_SIZE:-2}" \
  --device cuda

echo "[DONE] metrics: $LAVIS_METRICS_JSONL"
