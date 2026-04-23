#!/usr/bin/env bash
# =============================================================================
# UKMP（ukmp_prune.py）：当前 **仅** 用 OKVQA overall 一份 bin 跑四项评测；
# MME / MMMU 两套 run_one_ckpt_suite 已注释。
#
# 默认权重（可用 CKPT_OKVQA_ONLY 覆盖）:
#   pruned_checkpoint/ukmp_prune/okvqa_overall-128data-taylor+knowledge-param_first-param_norm-0.5-blockwise-global-select_loss-multimodal/pytorch_model.bin
#
# 权重格式：pruned_checkpoint/ukmp_prune/<job_id>/pytorch_model.bin
#   （torch.save({"model": nn.Module}, ...)）
#
# 默认 GPU：1（CUDA_VISIBLE_DEVICES=1）。本脚本顺序执行，单进程脚本无 torch.distributed 端口；
# 若你自行并行多卡，请为每个进程设不同 CUDA_VISIBLE_DEVICES。
#
# OKVQA：仅全量 val 一次（overall），使用 evaluate_blip2_pruned.py + okvqa_zeroshot_flant5xl_eval_overall.yaml
# MMBench / MMMU：mmmu_eval_by_discipline.py --overall_only
# MME：mme_eval_yes_no.py
#
# 推理 batch：默认 2（EVAL_BATCH_SIZE）
#
# 用法:
#   cd /root/autodl-tmp/UKMP-main/LAVIS
#   bash scripts/structured_blip2/run_ukmp_eval_mme_okvqa_mmmu_calib_ckpts.sh
#
# 覆盖权重路径示例:
#   CKPT_MME=/path/to/pytorch_model.bin CKPT_OKVQA_OVERALL=... CKPT_MMMU=... bash ...
#
# 若表里两行数字相同，多为三个 bin 误指同一路径；三套文件都存在时会自动校验路径/inode。
# 跳过校验: SKIP_CKPT_DISTINCT_CHECK=1
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# -----------------------------------------------------------------------------
# 可改配置
# -----------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

export MMBENCH_ROOT="${MMBENCH_ROOT:-/root/autodl-tmp/MMBench_eval}"
export MME_ROOT="${MME_ROOT:-/root/autodl-tmp/MME_eval}"
export MMMU_ROOT="${MMMU_ROOT:-/root/autodl-tmp/MMMU_single_image}"

# 仅 OKVQA overall calibration（与下面注释掉的三套逻辑二选一）
CKPT_OKVQA_ONLY="${CKPT_OKVQA_ONLY:-$REPO_ROOT/pruned_checkpoint/ukmp_prune/okvqa_overall-128data-taylor+knowledge-param_first-param_norm-0.5-blockwise-global-select_loss-multimodal/pytorch_model.bin}"
if [[ ! -f "$CKPT_OKVQA_ONLY" ]]; then
  echo "[FATAL] 找不到 UKMP OKVQA overall 权重: $CKPT_OKVQA_ONLY"
  exit 1
fi
echo "[INFO] 本次只跑 OKVQA calibration 权重四项评测: $CKPT_OKVQA_ONLY"

# # --- 原三套默认路径（恢复 MME/MMMU 评测时用）---
# UKMP_PRUNE_SUBDIR="${UKMP_PRUNE_SUBDIR:-pruned_checkpoint/ukmp_prune}"
# JOB_MME=...
# CKPT_MME=... CKPT_MMMU=... CKPT_OKVQA_OVERALL=...
# assert_three_ckpts_distinct "$CKPT_MME" "$CKPT_OKVQA_OVERALL" "$CKPT_MMMU"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"

export MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
export MME_SPLIT="${MME_SPLIT:-test}"
export MMMU_SPLIT="${MMMU_SPLIT:-test}"

# HuggingFace（与仓库内其它脚本一致，可按机器改）
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache_moved/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export BERT_BASE_UNCASED_SNAPSHOT="${BERT_BASE_UNCASED_SNAPSHOT:-/root/autodl-tmp/cache_moved/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594}"
export FLAN_T5_XL_SNAPSHOT="${FLAN_T5_XL_SNAPSHOT:-/root/autodl-tmp/cache_moved/huggingface/hub/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001}"

SUMMARY_DIR="$REPO_ROOT/lavis/output/BLIP2"
mkdir -p "$SUMMARY_DIR"
SUMMARY_STAMP="$(date +%Y%m%d_%H%M%S)"
export LAVIS_METRICS_JSONL="${LAVIS_METRICS_JSONL:-$SUMMARY_DIR/ukmp_eval_metrics_${SUMMARY_STAMP}.jsonl}"
: > "$LAVIS_METRICS_JSONL"
SUMMARY_MD="$SUMMARY_DIR/ukmp_eval_summary_${SUMMARY_STAMP}.md"
SUMMARY_TSV="$SUMMARY_DIR/ukmp_eval_summary_${SUMMARY_STAMP}.tsv"

check_ckpt() {
  local c="$1"
  if [[ ! -f "$c" ]]; then
    echo "[WARN] 跳过：找不到权重文件: $c"
    return 1
  fi
  return 0
}

# -----------------------------------------------------------------------------
run_one_ckpt_suite() {
  local tag="$1"
  local ckpt="$2"
  local fullval_job="$3"

  echo ""
  echo "#####################################################################"
  echo "# 校准标签: $tag"
  echo "# CKPT: $ckpt"
  echo "#####################################################################"

  if ! check_ckpt "$ckpt"; then
    return 0
  fi

  export LAVIS_EVAL_CALIB_TAG="$tag"

  echo ""
  echo ">>> [$tag] MMBench（$MMBENCH_ROOT, split=$MMBENCH_SPLIT）overall only"
  export LAVIS_METRICS_BENCHMARK="MMBench"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMBENCH_ROOT" \
    --split "$MMBENCH_SPLIT" \
    --ckpt "$ckpt" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --device cuda \
    --overall_only

  echo ""
  echo ">>> [$tag] MME yes/no（$MME_ROOT, split=$MME_SPLIT）"
  python scripts/blip2/mme_eval_yes_no.py \
    --mme_root "$MME_ROOT" \
    --split "$MME_SPLIT" \
    --ckpt "$ckpt" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --device cuda

  echo ""
  echo ">>> [$tag] OKVQA 全量 val（overall） job_id=$fullval_job"
  python -u evaluate_blip2_pruned.py \
    --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
    --job_id "$fullval_job" \
    --pruned_ckpt "$ckpt"

  echo ""
  echo ">>> [$tag] MMMU（$MMMU_ROOT, split=$MMMU_SPLIT）overall only"
  export LAVIS_METRICS_BENCHMARK="MMMU"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMMU_ROOT" \
    --split "$MMMU_SPLIT" \
    --ckpt "$ckpt" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --device cuda \
    --overall_only
}

# -----------------------------------------------------------------------------
# run_one_ckpt_suite "MME_calib" "$CKPT_MME" "okvqa_eval_ukmp_calibMME_fullval"

run_one_ckpt_suite "OKVQA_train_overall_calib" "$CKPT_OKVQA_ONLY" "okvqa_eval_ukmp_calibOKVQAoverall_fullval"

# run_one_ckpt_suite "MMMU_overall_calib" "$CKPT_MMMU" "okvqa_eval_ukmp_calibMMMUoverall_fullval"

echo ""
echo "========== 全部完成 =========="
echo "Per-job 输出: $REPO_ROOT/lavis/output/BLIP2/"
echo "OKVQA: lavis/output/BLIP2/OKVQA/okvqa_eval_ukmp_calib*_fullval/"

echo ""
echo "========== 整体结果总表（Markdown）=========="
python "$SCRIPT_DIR/collect_ukmp_eval_summary.py" \
  --repo-root "$REPO_ROOT" \
  --metrics-jsonl "$LAVIS_METRICS_JSONL" \
  --out-md "$SUMMARY_MD" \
  --out-tsv "$SUMMARY_TSV"
echo "已写入: $SUMMARY_MD"
echo "TSV:    $SUMMARY_TSV"
echo "指标行: $LAVIS_METRICS_JSONL"
