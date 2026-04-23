#!/usr/bin/env bash
# =============================================================================
# LAVIS_backup：用 run_lavisbackup_prune_t5_c4_llm_only.sh 剪出的单份权重，
# 依次跑四基准：MMBench / OKVQA（val overall）/ MMMU / MathVista_MC。
#
# 权重为完整 state_dict（仅 T5 被稀疏化时 ViT 与预训练一致）；评测时同一 pth
# 同时作为 --t5_pruned_checkpoint 与 --vit_pruned_checkpoint（与 fourcalib 脚本一致）。
#
# 用法（在 LAVIS_backup 根目录）:
#   cd /root/autodl-tmp/LAVIS_backup
#   bash scripts/blip2/run_lavisbackup_t5_c4_ckpt_fourbench_eval.sh
#
# 指定剪枝时的 JOB_ID（默认与 prune 脚本一致）或直传整路径:
#   JOB_ID=my_job bash scripts/blip2/run_lavisbackup_t5_c4_ckpt_fourbench_eval.sh
#   CKPT=/path/to/model.pth bash scripts/blip2/run_lavisbackup_t5_c4_ckpt_fourbench_eval.sh
#
# 可选: 追加指标到 jsonl（便于汇总）:
#   export LAVIS_METRICS_JSONL=$PWD/lavis/output/BLIP2/t5_c4_fourbench_metrics.jsonl
#
# 环境变量（与 run_lavisbackup_fourcalib_prune_eval_fourbench.sh 对齐）:
#   MMBENCH_ROOT MMMU_ROOT MMBENCH_SPLIT MMMU_SPLIT
#   MATHVISTA_EVAL_JSON MATHVISTA_IMAGES_DIR EVAL_BATCH_SIZE MASTER_PORT
#   CUDA_VISIBLE_DEVICES AUTODL_TMP
# HuggingFace：默认离线 + 本地 hub snapshot（勿联网拉 bert / flan-t5）
#   HF_HOME BERT_BASE_UNCASED_SNAPSHOT FLAN_T5_XL_SNAPSHOT HF_ENDPOINT
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- HuggingFace / Transformers：离线，BERT & Flan-T5 从本地 snapshot（与 prune 脚本一致）---
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache_moved/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

_resolve_hub_snapshot_dir() {
  local repo_dir="$1"
  [[ -d "$repo_dir/snapshots" ]] || { echo ""; return 0; }
  find "$repo_dir/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | head -1
}

HUB_ROOT="${HUGGINGFACE_HUB_CACHE}"
if [[ -z "${BERT_BASE_UNCASED_SNAPSHOT:-}" ]]; then
  BERT_BASE_UNCASED_SNAPSHOT="$(_resolve_hub_snapshot_dir "$HUB_ROOT/models--bert-base-uncased")"
fi
if [[ -z "${FLAN_T5_XL_SNAPSHOT:-}" ]]; then
  FLAN_T5_XL_SNAPSHOT="$(_resolve_hub_snapshot_dir "$HUB_ROOT/models--google--flan-t5-xl")"
fi
export BERT_BASE_UNCASED_SNAPSHOT
export FLAN_T5_XL_SNAPSHOT

if [[ ! -d "${BERT_BASE_UNCASED_SNAPSHOT}" ]]; then
  echo "[FATAL] 未找到 bert-base-uncased 本地 snapshot。请设置 HF_HOME 或 export BERT_BASE_UNCASED_SNAPSHOT=.../hub/models--bert-base-uncased/snapshots/<hash>" >&2
  exit 1
fi
if [[ ! -d "${FLAN_T5_XL_SNAPSHOT}" ]]; then
  echo "[FATAL] 未找到 google/flan-t5-xl 本地 snapshot。请设置 HF_HOME 或 export FLAN_T5_XL_SNAPSHOT=..." >&2
  exit 1
fi

echo "[INFO] HF_HOME=$HF_HOME HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "[INFO] BERT_BASE_UNCASED_SNAPSHOT=$BERT_BASE_UNCASED_SNAPSHOT"
echo "[INFO] FLAN_T5_XL_SNAPSHOT=$FLAN_T5_XL_SNAPSHOT"

JOB_ID="${JOB_ID:-lavisbackup_t5_c4_llm_only}"
CKPT="${CKPT:-$REPO_ROOT/pruned_checkpoint/${JOB_ID}.pth}"

MMBENCH_ROOT="${MMBENCH_ROOT:-/root/autodl-tmp/MMBench_eval}"
MMMU_ROOT="${MMMU_ROOT:-/root/autodl-tmp/MMMU_single_image}"
MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
MMMU_SPLIT="${MMMU_SPLIT:-test}"

AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
export MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-$AUTODL_TMP/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
export MATHVISTA_IMAGES_DIR="${MATHVISTA_IMAGES_DIR:-$AUTODL_TMP/MathVista_eval_testmini_mc/images}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
MASTER_PORT="${MASTER_PORT:-29750}"

EVAL_TAG="${EVAL_TAG:-t5_c4_${JOB_ID}}"
export LAVIS_EVAL_CALIB_TAG="${LAVIS_EVAL_CALIB_TAG:-$EVAL_TAG}"
OKVQA_JOB_ID="${OKVQA_JOB_ID:-okvqa_eval_${EVAL_TAG}_fullval}"

if [[ ! -f "$CKPT" ]]; then
  echo "[FATAL] 找不到权重: $CKPT" >&2
  echo "  请先跑剪枝或设置 CKPT=... / JOB_ID=...（默认 JOB_ID=$JOB_ID）" >&2
  exit 1
fi

echo "[INFO] CKPT=$(readlink -f "$CKPT")"
echo "[INFO] LAVIS_EVAL_CALIB_TAG=$LAVIS_EVAL_CALIB_TAG OKVQA_JOB_ID=$OKVQA_JOB_ID"

echo ""
echo ">>> MMBench ($MMBENCH_ROOT, split=$MMBENCH_SPLIT)"
export LAVIS_METRICS_BENCHMARK="MMBench"
python scripts/blip2/mmmu_eval_by_discipline.py \
  --mmmu_root "$MMBENCH_ROOT" \
  --split "$MMBENCH_SPLIT" \
  --ckpt "$CKPT" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --device cuda \
  --overall_only

echo ""
echo ">>> OKVQA full val (--job_id $OKVQA_JOB_ID)"
python -m torch.distributed.run --nproc_per_node=1 --master_port="$MASTER_PORT" evaluate_blip.py \
  --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
  --t5_pruned_checkpoint "$CKPT" \
  --vit_pruned_checkpoint "$CKPT" \
  --job_id "$OKVQA_JOB_ID"

echo ""
echo ">>> MMMU ($MMMU_ROOT, split=$MMMU_SPLIT)"
export LAVIS_METRICS_BENCHMARK="MMMU"
python scripts/blip2/mmmu_eval_by_discipline.py \
  --mmmu_root "$MMMU_ROOT" \
  --split "$MMMU_SPLIT" \
  --ckpt "$CKPT" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --device cuda \
  --overall_only

echo ""
echo ">>> MathVista MC"
if [[ ! -f "$MATHVISTA_EVAL_JSON" ]]; then
  echo "[WARN] 跳过 MathVista：缺少 $MATHVISTA_EVAL_JSON"
else
  export LAVIS_METRICS_BENCHMARK="MathVista_MC"
  python scripts/blip2/mathvista_mc_eval.py \
    --eval_json "$MATHVISTA_EVAL_JSON" \
    --images_dir "$MATHVISTA_IMAGES_DIR" \
    --ckpt "$CKPT" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --device cuda
fi

echo ""
echo "[INFO] 四基准评测流程结束。"
