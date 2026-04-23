#!/usr/bin/env bash
# =============================================================================
# ECoFLaP（Wanda）：三套剪枝 — OKVQA train overall / MMBench calib / MMMU overall
# 输出 pruned_checkpoint/<job_id>.pth，job_id 带日期（默认月日 0324）；再各跑 MMBench + OKVQA full val + MMMU（不跑 MME）。
#
# 默认 job_id（可用环境变量 PRUNE_JOB_* 覆盖）:
#   okvqa_cf_0.5_overall_<DATE_TAG>
#   okvqa_cf_0.5_MMBench_<DATE_TAG>
#   okvqa_cf_0.5_MMMU_overall_<DATE_TAG>
#
# 需存在:
#   lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_mmbench.yaml（本仓库已加）
#   /root/autodl-tmp/MMBench_calibration/ 与 MMMU_calibration 等数据
#
# 环境变量:
#   DATE_TAG=0324              # 默认 $(date +%m%d)
#   RUN_PRUNE=1 RUN_EVAL=1     # 默认均为 1
#   PRUNING_CALIB_BATCH=8      # 与 run_ecoflap_eval_mme_okvqa_mmmu_calib_ckpts.sh 一致
#   CUDA_VISIBLE_DEVICES=0
#   MASTER_PORT=29900
#
# 用法:
#   cd /root/autodl-tmp/ECoFLaP/LAVIS
#   bash scripts/blip2/run_ecoflap_prune_eval_okvqa_mmbench_mmu_dated.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export MMBENCH_ROOT="${MMBENCH_ROOT:-/root/autodl-tmp/MMBench_eval}"
export MME_ROOT="${MME_ROOT:-/root/autodl-tmp/MME_eval}"
export MMMU_ROOT="${MMMU_ROOT:-/root/autodl-tmp/MMMU_single_image}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
export MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
export MME_SPLIT="${MME_SPLIT:-test}"
export MMMU_SPLIT="${MMMU_SPLIT:-test}"

RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
PRUNING_CALIB_BATCH="${PRUNING_CALIB_BATCH:-8}"

DATE_TAG="${DATE_TAG:-$(date +%m%d)}"

PRUNE_JOB_OKVQA="${PRUNE_JOB_OKVQA:-okvqa_cf_0.5_overall_${DATE_TAG}}"
PRUNE_JOB_MMBENCH="${PRUNE_JOB_MMBENCH:-okvqa_cf_0.5_MMBench_${DATE_TAG}}"
PRUNE_JOB_MMU="${PRUNE_JOB_MMU:-okvqa_cf_0.5_MMMU_overall_${DATE_TAG}}"

CKPT_OKVQA="${CKPT_OKVQA:-$REPO_ROOT/pruned_checkpoint/${PRUNE_JOB_OKVQA}.pth}"
CKPT_MMBENCH="${CKPT_MMBENCH:-$REPO_ROOT/pruned_checkpoint/${PRUNE_JOB_MMBENCH}.pth}"
CKPT_MMU="${CKPT_MMU:-$REPO_ROOT/pruned_checkpoint/${PRUNE_JOB_MMU}.pth}"

TAG_OKVQA="${TAG_OKVQA:-OKVQAtrain_overall_calib_${DATE_TAG}}"
TAG_MMBENCH="${TAG_MMBENCH:-MMBench_calib_${DATE_TAG}}"
TAG_MMU="${TAG_MMU:-MMMU_overall_calib_${DATE_TAG}}"
EVAL_JOB_OKVQA="${EVAL_JOB_OKVQA:-okvqa_eval_ecoflapOKVQAoverall_${DATE_TAG}_fullval}"
EVAL_JOB_MMBENCH="${EVAL_JOB_MMBENCH:-okvqa_eval_ecoflapMMBench_${DATE_TAG}_fullval}"
EVAL_JOB_MMU="${EVAL_JOB_MMU:-okvqa_eval_ecoflapMMMUoverall_${DATE_TAG}_fullval}"

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
export LAVIS_METRICS_JSONL="${LAVIS_METRICS_JSONL:-$SUMMARY_DIR/ecoflap_eval_metrics_${SUMMARY_STAMP}.jsonl}"
: > "$LAVIS_METRICS_JSONL"
SUMMARY_MD="$SUMMARY_DIR/ecoflap_eval_summary_okvqa_mmbench_mmu_${DATE_TAG}_${SUMMARY_STAMP}.md"
SUMMARY_TSV="$SUMMARY_DIR/ecoflap_eval_summary_okvqa_mmbench_mmu_${DATE_TAG}_${SUMMARY_STAMP}.tsv"

MASTER_PORT="${MASTER_PORT:-29900}"

echo "[INFO] DATE_TAG=$DATE_TAG  (ECoFLaP Wanda 剪枝)"
echo "[INFO] 剪枝 job_id -> pth:"
echo "  OKVQA overall: ${PRUNE_JOB_OKVQA}.pth"
echo "  MMBench calib: ${PRUNE_JOB_MMBENCH}.pth"
echo "  MMMU overall:  ${PRUNE_JOB_MMU}.pth"

check_ckpt() {
  local c="$1"
  if [[ ! -f "$c" ]]; then
    echo "[WARN] 跳过：找不到权重: $c"
    return 1
  fi
  return 0
}

assert_three_ckpts_distinct() {
  local a="$1" b="$2" c="$3"
  if [[ "${SKIP_CKPT_DISTINCT_CHECK:-0}" == "1" ]]; then
    echo "[INFO] 已跳过三套权重互异性检查（SKIP_CKPT_DISTINCT_CHECK=1）"
    return 0
  fi
  if [[ ! -f "$a" ]] || [[ ! -f "$b" ]] || [[ ! -f "$c" ]]; then
    echo "[WARN] 三套权重未齐，跳过互异性检查。"
    return 0
  fi
  local ra rb rc ia ib ic
  ra=$(readlink -f "$a"); rb=$(readlink -f "$b"); rc=$(readlink -f "$c")
  if [[ "$ra" == "$rb" ]] || [[ "$ra" == "$rc" ]] || [[ "$rb" == "$rc" ]]; then
    echo "[FATAL] 三套 CKPT 解析路径必须两两不同。"; exit 1
  fi
  ia=$(stat -c '%d:%i' "$ra"); ib=$(stat -c '%d:%i' "$rb"); ic=$(stat -c '%d:%i' "$rc")
  if [[ "$ia" == "$ib" ]] || [[ "$ia" == "$ic" ]] || [[ "$ib" == "$ic" ]]; then
    echo "[FATAL] 三套 CKPT inode 必须两两不同。"; exit 1
  fi
  echo "[INFO] 已确认 OKVQA / MMBench / MMMU 三套权重路径与 inode 两两不同。"
}

wanda_prune() {
  local cfg="$1"
  local jid="$2"
  local P=$MASTER_PORT
  MASTER_PORT=$((MASTER_PORT + 1))
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$P" evaluate_blip.py \
    --cfg-path "$cfg" \
    --pruning_method 'blipt5_wanda_pruner' --save_pruned_model \
    --score_method MEZO-GradOnly_sum --sparsity_ratio_granularity block \
    --max_sparsity_per_layer 0.6 --prunining_dataset_batch_size "${PRUNING_CALIB_BATCH}" \
    --t5_prune_spec 24-0.5-1.0-1.0 --vit_prune_spec 39-0.5-1.0-1.0 \
    --job_id "$jid"
}

if [[ "$RUN_PRUNE" == "1" ]]; then
  echo ""
  echo "========== RUN_PRUNE=1：三套 Wanda 剪枝（GPU $CUDA_VISIBLE_DEVICES）=========="
  wanda_prune "lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_okvqa_train_overall.yaml" "$PRUNE_JOB_OKVQA"
  wanda_prune "lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_mmbench.yaml" "$PRUNE_JOB_MMBENCH"
  wanda_prune "lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_mmmu_overall.yaml" "$PRUNE_JOB_MMU"
  echo "[INFO] 剪枝完成。"
else
  echo "[INFO] RUN_PRUNE=0，跳过剪枝。"
fi

run_one_ckpt_eval_three() {
  local tag="$1"
  local ckpt="$2"
  local fullval_job="$3"

  echo ""
  if ! check_ckpt "$ckpt"; then
    return 0
  fi

  echo "#####################################################################"
  echo "# 校准标签: $tag"
  echo "# CKPT: $(readlink -f "$ckpt")"
  echo "#####################################################################"

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
  echo ">>> [$tag] OKVQA 全量 val（overall） job_id=$fullval_job"
  local P_FULL=$MASTER_PORT
  MASTER_PORT=$((MASTER_PORT + 1))
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$P_FULL" evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
    --t5_pruned_checkpoint "$ckpt" \
    --vit_pruned_checkpoint "$ckpt" \
    --job_id "$fullval_job"

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

if [[ "$RUN_EVAL" == "1" ]]; then
  echo ""
  echo "========== RUN_EVAL=1：三套 pth → MMBench + OKVQA + MMMU =========="

  if [[ ! -f "$CKPT_OKVQA" ]] || [[ ! -f "$CKPT_MMBENCH" ]] || [[ ! -f "$CKPT_MMU" ]]; then
    echo "[FATAL] 评测需要三个 pth 均存在。"
    [[ -f "$CKPT_OKVQA" ]] || echo "  缺: $CKPT_OKVQA"
    [[ -f "$CKPT_MMBENCH" ]] || echo "  缺: $CKPT_MMBENCH"
    [[ -f "$CKPT_MMU" ]] || echo "  缺: $CKPT_MMU"
    exit 1
  fi

  assert_three_ckpts_distinct "$CKPT_OKVQA" "$CKPT_MMBENCH" "$CKPT_MMU"

  run_one_ckpt_eval_three "$TAG_OKVQA" "$CKPT_OKVQA" "$EVAL_JOB_OKVQA"
  run_one_ckpt_eval_three "$TAG_MMBENCH" "$CKPT_MMBENCH" "$EVAL_JOB_MMBENCH"
  run_one_ckpt_eval_three "$TAG_MMU" "$CKPT_MMU" "$EVAL_JOB_MMU"

  echo ""
  echo "========== 整体结果总表 =========="
  python "$SCRIPT_DIR/collect_ecoflap_eval_summary.py" \
    --repo-root "$REPO_ROOT" \
    --metrics-jsonl "$LAVIS_METRICS_JSONL" \
    --out-md "$SUMMARY_MD" \
    --out-tsv "$SUMMARY_TSV" \
    --suites \
      "${TAG_OKVQA}:${EVAL_JOB_OKVQA}" \
      "${TAG_MMBENCH}:${EVAL_JOB_MMBENCH}" \
      "${TAG_MMU}:${EVAL_JOB_MMU}"

  echo "MD:   $SUMMARY_MD"
  echo "TSV:  $SUMMARY_TSV"
  echo "JSONL: $LAVIS_METRICS_JSONL"
else
  echo "[INFO] RUN_EVAL=0，跳过评测。"
fi

echo ""
echo "========== 结束（DATE_TAG=$DATE_TAG）=========="
