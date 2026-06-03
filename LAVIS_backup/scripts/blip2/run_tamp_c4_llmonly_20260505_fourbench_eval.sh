#!/usr/bin/env bash
# =============================================================================
# 用 C4 TAMP llm-only 权重跑四基准：MMBench / OKVQA / MMMU / MathVista MC
#
# 默认权重:
#   pruned_checkpoint/tamp_c4_llmonly_20260505_202326.pth
#
# 用法:
#   cd /data/data2/mfs/2/LAVIS_backup
#   bash scripts/blip2/run_tamp_c4_llmonly_20260505_fourbench_eval.sh
#
# 换权重:
#   CKPT=/path/to/other.pth bash scripts/blip2/...
#
# 跳过某项:
#   SKIP_MMBENCH=1 bash scripts/blip2/...
#
# 环境变量:
#   BASE, CKPT, EVAL_TAG, OKVQA_JOB_ID, MASTER_PORT, EVAL_BATCH_SIZE
#   MMBENCH_ROOT, MMMU_ROOT, MATHVISTA_EVAL_JSON, MATHVISTA_IMAGES_DIR
#   LAVIS_METRICS_JSONL（默认写入 lavis/output/BLIP2/）
#   SKIP_MMBENCH / SKIP_OKVQA / SKIP_MMMU / SKIP_MATHVISTA =1
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

BASE="${BASE:-/data/data2/mfs}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export HF_HOME="${HF_HOME:-$BASE/model_cache/huggingface}"
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
  echo "[FATAL] 未找到 bert-base-uncased 本地 snapshot。请设置 HF_HOME 或 BERT_BASE_UNCASED_SNAPSHOT。" >&2
  exit 1
fi
if [[ ! -d "${FLAN_T5_XL_SNAPSHOT}" ]]; then
  echo "[FATAL] 未找到 flan-t5-xl 本地 snapshot。请设置 HF_HOME 或 FLAN_T5_XL_SNAPSHOT。" >&2
  exit 1
fi

CKPT="${CKPT:-$REPO_ROOT/pruned_checkpoint/tamp_c4_llmonly_20260505_202326.pth}"
EVAL_TAG="${EVAL_TAG:-t5_c4_tamp_tamp_c4_llmonly_20260505_202326}"
OKVQA_JOB_ID="${OKVQA_JOB_ID:-okvqa_eval_t5_c4_tamp_tamp_c4_llmonly_20260505_202326_fullval}"

MMBENCH_ROOT="${MMBENCH_ROOT:-$BASE/MMBench_eval}"
MMMU_ROOT="${MMMU_ROOT:-$BASE/MMMU_single_image}"
MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
MMMU_SPLIT="${MMMU_SPLIT:-test}"
MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-$BASE/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
MATHVISTA_IMAGES_DIR="${MATHVISTA_IMAGES_DIR:-$BASE/MathVista_eval_testmini_mc/images}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
MASTER_PORT="${MASTER_PORT:-29750}"

SUMMARY_DIR="$REPO_ROOT/lavis/output/BLIP2"
mkdir -p "$SUMMARY_DIR"
export LAVIS_METRICS_JSONL="${LAVIS_METRICS_JSONL:-$SUMMARY_DIR/tamp_c4_llmonly_20260505_fourbench_metrics.jsonl}"
export LAVIS_EVAL_CALIB_TAG="${LAVIS_EVAL_CALIB_TAG:-$EVAL_TAG}"

if [[ ! -f "$CKPT" ]]; then
  echo "[FATAL] 找不到权重: $CKPT" >&2
  exit 1
fi

echo "========== C4 TAMP llmonly 四基准评测 =========="
echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] CKPT=$(readlink -f "$CKPT")"
echo "[INFO] LAVIS_EVAL_CALIB_TAG=$LAVIS_EVAL_CALIB_TAG"
echo "[INFO] OKVQA_JOB_ID=$OKVQA_JOB_ID"
echo "[INFO] LAVIS_METRICS_JSONL=$LAVIS_METRICS_JSONL"
echo "[INFO] MMBENCH_ROOT=$MMBENCH_ROOT MMMU_ROOT=$MMMU_ROOT"
echo "================================================"

if [[ "${SKIP_MMBENCH:-0}" != "1" ]]; then
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
else
  echo "[SKIP] MMBench"
fi

if [[ "${SKIP_OKVQA:-0}" != "1" ]]; then
  echo ""
  echo ">>> OKVQA full val (--job_id $OKVQA_JOB_ID)"
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$MASTER_PORT" evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
    --t5_pruned_checkpoint "$CKPT" \
    --vit_pruned_checkpoint "$CKPT" \
    --job_id "$OKVQA_JOB_ID"
else
  echo "[SKIP] OKVQA"
fi

if [[ "${SKIP_MMMU:-0}" != "1" ]]; then
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
else
  echo "[SKIP] MMMU"
fi

if [[ "${SKIP_MATHVISTA:-0}" != "1" ]]; then
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
else
  echo "[SKIP] MathVista"
fi

echo ""
echo "[INFO] 四基准评测结束。"
echo "[INFO] OKVQA 结果目录: $REPO_ROOT/lavis/output/BLIP2/OKVQA/$OKVQA_JOB_ID/"
echo "[INFO] 指标 jsonl: $LAVIS_METRICS_JSONL"
