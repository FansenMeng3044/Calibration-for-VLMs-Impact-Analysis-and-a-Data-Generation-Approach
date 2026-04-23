#!/usr/bin/env bash
# =============================================================================
# LAVIS_backup（TAMP 剪枝权重）：默认跑 **OKVQA overall + MMMU overall** 两套评测（MME 剪枝/评测仍注释）。
#
# 默认只做评测（假定剪枝已完成）。若需先剪枝，设 RUN_PRUNE=1。
#
# 约定权重路径（与 tamp 剪枝 job_id 一致，可用环境变量覆盖）:
#   MME              -> pruned_checkpoint/okvqa_cf_0.5_MME.pth
#   OKVQA overall    -> pruned_checkpoint/okvqa_cf_0.5_overall.pth
#   MMMU overall     -> pruned_checkpoint/okvqa_cf_0.5_MMMU_overall.pth
#
# GPU: 默认 cuda:2（整脚本顺序执行，单卡即可）
# torch.distributed 各次 evaluate_blip 使用递增 master_port，避免冲突
#
# OKVQA: 仅全量 val 一次（overall）；不再跑 11 类。
# MMBench/MMMU: mmmu_eval_by_discipline.py 单遍全量，--overall_only（只打印 Overall）。
#
# 结束时会生成「整体结果」总表（Markdown + 可选 TSV）：
#   lavis/output/BLIP2/lavisbackup_eval_summary_<时间戳>.md
#   并在终端 cat 该文件。指标来自 LAVIS_METRICS_JSONL + OKVQA 的 evaluate.txt。
#
# 用法:
#   cd /root/autodl-tmp/LAVIS_backup
#   bash scripts/blip2/run_lavisbackup_eval_mme_okvqa_mmmu_calib_ckpts.sh
#
# 若 OKVQA/MMMU 两行数字完全相同，多半是两套 CKPT 误指向同一路径；脚本会在两个文件都存在时检查互异。
# 需跳过检查: SKIP_CKPT_DISTINCT_CHECK=1
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# -----------------------------------------------------------------------------
# 可改配置
# -----------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# 数据与 parquet 根目录
export MMBENCH_ROOT="${MMBENCH_ROOT:-/root/autodl-tmp/MMBench_eval}"
export MME_ROOT="${MME_ROOT:-/root/autodl-tmp/MME_eval}"
export MMMU_ROOT="${MMMU_ROOT:-/root/autodl-tmp/MMMU_single_image}"

# 剪枝权重（MME 评测已注释）
# CKPT_MME="${CKPT_MME:-$REPO_ROOT/pruned_checkpoint/okvqa_cf_0.5_MME.pth}"
CKPT_OKVQA_OVERALL="${CKPT_OKVQA_OVERALL:-$REPO_ROOT/pruned_checkpoint/okvqa_cf_0.5_overall.pth}"
CKPT_MMMU_OVERALL="${CKPT_MMMU_OVERALL:-$REPO_ROOT/pruned_checkpoint/okvqa_cf_0.5_MMMU_overall.pth}"

# 推理 batch（MMBench/MME/MMMU 脚本）；过小略慢，可缓解 24G 卡 OOM
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"

# MMBench 请用 dev（test 常无 GT）；MME/MMMU split 可按需改环境变量
export MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
export MME_SPLIT="${MME_SPLIT:-test}"
export MMMU_SPLIT="${MMMU_SPLIT:-test}"

# 可选：先跑三次剪枝（同一 GPU，端口 29800–29802）
RUN_PRUNE="${RUN_PRUNE:-0}"

# HuggingFace / 本地 snapshot（与仓库内 blip2.py / blip2_t5.py 一致）
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

T5_SPEC="24-0.5-1.0-1.0"
VIT_SPEC="39-0.5-1.0-1.0"
PRUNE_METHOD="blipt5_tamp_pruner"

# 汇总输出（整体结果总表）
SUMMARY_DIR="$REPO_ROOT/lavis/output/BLIP2"
mkdir -p "$SUMMARY_DIR"
SUMMARY_STAMP="$(date +%Y%m%d_%H%M%S)"
export LAVIS_METRICS_JSONL="${LAVIS_METRICS_JSONL:-$SUMMARY_DIR/lavisbackup_eval_metrics_${SUMMARY_STAMP}.jsonl}"
: > "$LAVIS_METRICS_JSONL"
SUMMARY_MD="$SUMMARY_DIR/lavisbackup_eval_summary_${SUMMARY_STAMP}.md"
SUMMARY_TSV="$SUMMARY_DIR/lavisbackup_eval_summary_${SUMMARY_STAMP}.tsv"

# master_port 递增（仅 evaluate_blip / distributed 用）
# 注意：禁止写 P=$(next_port) —— 在 bash 里命令替换会开子 shell，子 shell 里改的 MASTER_PORT 不会回写到父进程，
# 会导致每次 OKVQA 的 torch.distributed.run 都用同一个 port。下面在调用处用「两行展开」递增。
MASTER_PORT="${MASTER_PORT:-29700}"

check_ckpt() {
  local c="$1"
  if [[ ! -f "$c" ]]; then
    echo "[WARN] 跳过：找不到权重文件: $c"
    return 1
  fi
  return 0
}

# assert_three_ckpts_distinct() { ... }  # 恢复 MME 评测时使用

assert_two_ckpts_distinct() {
  local o="$1" mm="$2"
  if [[ "${SKIP_CKPT_DISTINCT_CHECK:-0}" == "1" ]]; then
    echo "[INFO] 已跳过两套权重互异性检查（SKIP_CKPT_DISTINCT_CHECK=1）"
    return 0
  fi
  if [[ ! -f "$o" ]] || [[ ! -f "$mm" ]]; then
    echo "[WARN] OKVQA/MMMU 权重有缺失，跳过互异性检查。"
    return 0
  fi
  local rp_o rp_mm i_o i_mm
  rp_o=$(readlink -f "$o")
  rp_mm=$(readlink -f "$mm")
  i_o=$(stat -c '%d:%i' "$rp_o")
  i_mm=$(stat -c '%d:%i' "$rp_mm")
  if [[ "$rp_o" == "$rp_mm" ]]; then
    echo "[FATAL] OKVQA 与 MMMU 两个 CKPT 解析后路径相同: $rp_o"
    exit 1
  fi
  if [[ "$i_o" == "$i_mm" ]]; then
    echo "[FATAL] OKVQA 与 MMMU 两个 CKPT 硬链接同 inode（同一文件）"
    exit 1
  fi
  echo "[INFO] 已确认 OKVQA_overall 与 MMMU_overall 权重路径/inode 互不相同。"
}

# -----------------------------------------------------------------------------
# 可选：三次剪枝
# -----------------------------------------------------------------------------
if [[ "$RUN_PRUNE" == "1" ]]; then
  echo "========== RUN_PRUNE=1：OKVQA overall + MMMU overall 剪枝（MME 已注释）（GPU $CUDA_VISIBLE_DEVICES）=========="
  # P1=$MASTER_PORT; MASTER_PORT=$((MASTER_PORT + 1))
  # python -m torch.distributed.run --nproc_per_node=1 --master_port="$P1" evaluate_blip.py \
  #   --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_mme.yaml \
  #   --pruning_method "$PRUNE_METHOD" --save_pruned_model \
  #   --t5_prune_spec "$T5_SPEC" --vit_prune_spec "$VIT_SPEC" \
  #   --job_id 'okvqa_cf_0.5_MME'

  P2=$MASTER_PORT; MASTER_PORT=$((MASTER_PORT + 1))
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$P2" evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_okvqa_train_overall.yaml \
    --pruning_method "$PRUNE_METHOD" --save_pruned_model \
    --t5_prune_spec "$T5_SPEC" --vit_prune_spec "$VIT_SPEC" \
    --job_id 'okvqa_cf_0.5_overall'

  P3=$MASTER_PORT; MASTER_PORT=$((MASTER_PORT + 1))
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$P3" evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_mmmu_overall.yaml \
    --pruning_method "$PRUNE_METHOD" --save_pruned_model \
    --t5_prune_spec "$T5_SPEC" --vit_prune_spec "$VIT_SPEC" \
    --job_id 'okvqa_cf_0.5_MMMU_overall'

  echo "[INFO] 剪枝完成，继续评测..."
fi

echo "[INFO] 两套评测权重（MME 已跳过）:"
echo "  CKPT_OKVQA_OVERALL=$CKPT_OKVQA_OVERALL"
echo "  CKPT_MMMU_OVERALL=$CKPT_MMMU_OVERALL"
if [[ ! -f "$CKPT_OKVQA_OVERALL" ]]; then
  echo "[FATAL] 找不到 OKVQA 权重: $CKPT_OKVQA_OVERALL"
  exit 1
fi
if [[ ! -f "$CKPT_MMMU_OVERALL" ]]; then
  echo "[FATAL] 找不到 MMMU 权重: $CKPT_MMMU_OVERALL"
  exit 1
fi
echo "[INFO] 解析后路径:"
echo "  OKVQA: $(readlink -f "$CKPT_OKVQA_OVERALL")"
echo "  MMMU:  $(readlink -f "$CKPT_MMMU_OVERALL")"
assert_two_ckpts_distinct "$CKPT_OKVQA_OVERALL" "$CKPT_MMMU_OVERALL"

# -----------------------------------------------------------------------------
# 单套权重：MMBench + MME + OKVQA（仅全量 val overall）+ MMMU（单遍 overall_only）
# -----------------------------------------------------------------------------
run_one_ckpt_suite() {
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
  echo ">>> [$tag] MME yes/no（$MME_ROOT, split=$MME_SPLIT）"
  python scripts/blip2/mme_eval_yes_no.py \
    --mme_root "$MME_ROOT" \
    --split "$MME_SPLIT" \
    --ckpt "$ckpt" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --device cuda

  echo ""
  echo ">>> [$tag] OKVQA 全量 val（overall） job_id=$fullval_job"
  local P_FULL
  P_FULL=$MASTER_PORT
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

# -----------------------------------------------------------------------------
# OKVQA overall + MMMU overall（MME 仍注释）
# -----------------------------------------------------------------------------
# run_one_ckpt_suite "MME_calib" "$CKPT_MME" "okvqa_eval_calibMME_fullval"
run_one_ckpt_suite "OKVQA_train_overall_calib" "$CKPT_OKVQA_OVERALL" "okvqa_eval_calibOKVQAoverall_fullval"

run_one_ckpt_suite "MMMU_overall_calib" "$CKPT_MMMU_OVERALL" "okvqa_eval_calibMMMUoverall_fullval"

echo ""
echo "========== 全部完成 =========="
echo "Per-job 日志与 evaluate.txt 一般在: $REPO_ROOT/lavis/output/BLIP2/"
echo "OKVQA 全量 val 目录示例: lavis/output/BLIP2/OKVQA/okvqa_eval_calib*_fullval/"

echo ""
echo "========== 整体结果总表（Markdown）=========="
python "$SCRIPT_DIR/collect_lavisbackup_eval_summary.py" \
  --repo-root "$REPO_ROOT" \
  --metrics-jsonl "$LAVIS_METRICS_JSONL" \
  --out-md "$SUMMARY_MD" \
  --out-tsv "$SUMMARY_TSV"
echo "已写入: $SUMMARY_MD"
echo "TSV:    $SUMMARY_TSV"
echo "指标行: $LAVIS_METRICS_JSONL"
