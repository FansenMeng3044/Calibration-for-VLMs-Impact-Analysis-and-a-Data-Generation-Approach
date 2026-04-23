#!/usr/bin/env bash
# =============================================================================
# ECoFLaP（Wanda 剪枝 .pth）：当前 **仅** 用 OKVQA train-overall 一份权重跑四项评测；
# MME / MMMU 两套 run_one_ckpt_suite 已注释（需恢复时取消注释并恢复下方 CKPT 解析块）。
#
# 默认 OKVQA 侧 calibration 权重（当前盘片为 okvqa_ghlc；可用 CKPT_OKVQA_ONLY 覆盖）:
#   pruned_checkpoint/okvqa_ghlc-blipt5_wanda_pruner_0.5-1.0-1.0_MEZO-GradOnly_sum0.6_block_bs8.pth
#
# 默认 GPU：物理卡 0（export CUDA_VISIBLE_DEVICES=0），单进程顺序跑。
# master_port 从 29900 递增，与剪枝段 29810+ 分离。
#
# 权重路径（evaluate_blip 保存为 pruned_checkpoint/<job_id>.pth）:
#   MME: 与 ecoflap_zeroth.py --calib_source mme 一致 → job_prefix okvqa_mme
#   OKVQA 全量 train calib:
#     - 本脚本 RUN_PRUNE=1 → job_id okvqa_okvqaoverall-${WANDA_SUFFIX}（默认只用此文件）
#     - 不再静默用 okvqa_ghlc 代替 overall；若必须用 ghlc，请设 ALLOW_OKVQA_GHLC_FALLBACK=1（不推荐）
#   MMMU: 本脚本 RUN_PRUNE=1 或你手动指定的 job_id okvqa_mmmu-${WANDA_SUFFIX}
#   WANDA_SUFFIX 须与剪枝时 job_id 完全一致（末尾 _bs8 对应 PRUNING_CALIB_BATCH=8；若剪枝用别的 batch 请改 WANDA_SUFFIX）
#
# 结束生成: lavis/output/BLIP2/ecoflap_eval_summary_<时间戳>.md (+ .tsv + metrics jsonl)
#
# 用法:
#   cd /root/autodl-tmp/ECoFLaP/LAVIS
#   bash scripts/blip2/run_ecoflap_eval_mme_okvqa_mmmu_calib_ckpts.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

WANDA_SUFFIX="${WANDA_SUFFIX:-blipt5_wanda_pruner_0.5-1.0-1.0_MEZO-GradOnly_sum0.6_block_bs8}"

MMBENCH_ROOT="${MMBENCH_ROOT:-/root/autodl-tmp/MMBench_eval}"
MME_ROOT="${MME_ROOT:-/root/autodl-tmp/MME_eval}"
MMMU_ROOT="${MMMU_ROOT:-/root/autodl-tmp/MMMU_single_image}"

MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
MME_SPLIT="${MME_SPLIT:-test}"
MMMU_SPLIT="${MMMU_SPLIT:-test}"
# 评测推理 batch：MMBench/MME/MMMU 脚本 + OKVQA 的 yaml（zeroshot / 由模板重写的 per-category）
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
# 剪枝时 --prunining_dataset_batch_size，写进 job_id 的 _bs*；默认 8 与 WANDA_SUFFIX 中 bs8 一致
PRUNING_CALIB_BATCH="${PRUNING_CALIB_BATCH:-8}"

RUN_PRUNE="${RUN_PRUNE:-0}"

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
SUMMARY_MD="$SUMMARY_DIR/ecoflap_eval_summary_${SUMMARY_STAMP}.md"
SUMMARY_TSV="$SUMMARY_DIR/ecoflap_eval_summary_${SUMMARY_STAMP}.tsv"

MASTER_PORT="${MASTER_PORT:-29900}"
next_port() {
  local p=$MASTER_PORT
  MASTER_PORT=$((MASTER_PORT + 1))
  echo "$p"
}

check_ckpt() {
  local c="$1"
  if [[ ! -f "$c" ]]; then
    echo "[WARN] 跳过：找不到权重: $c"
    return 1
  fi
  return 0
}

# 避免 CKPT_* 指向同一路径/硬链接 → 总表里两行数字完全一样
assert_three_ckpts_distinct() {
  local m="$1" o="$2" mm="$3"
  if [[ "${SKIP_CKPT_DISTINCT_CHECK:-0}" == "1" ]]; then
    echo "[INFO] 已跳过三套权重互异性检查（SKIP_CKPT_DISTINCT_CHECK=1）"
    return 0
  fi
  if [[ ! -f "$m" ]] || [[ ! -f "$o" ]] || [[ ! -f "$mm" ]]; then
    echo "[WARN] 有权重缺失，跳过互异性检查；若两行指标相同，请核对 CKPT 是否误指向同一文件。"
    return 0
  fi
  local rp_m rp_o rp_mm i_m i_o i_mm
  rp_m=$(readlink -f "$m")
  rp_o=$(readlink -f "$o")
  rp_mm=$(readlink -f "$mm")
  i_m=$(stat -c '%d:%i' "$rp_m")
  i_o=$(stat -c '%d:%i' "$rp_o")
  i_mm=$(stat -c '%d:%i' "$rp_mm")
  if [[ "$rp_m" == "$rp_o" ]] || [[ "$rp_m" == "$rp_mm" ]] || [[ "$rp_o" == "$rp_mm" ]]; then
    echo "[FATAL] 三个 CKPT 解析后路径有重复（会出现两两相同的准确率）："
    echo "  MME=$rp_m"
    echo "  OKVQA_overall=$rp_o"
    echo "  MMMU_overall=$rp_mm"
    exit 1
  fi
  if [[ "$i_m" == "$i_o" ]] || [[ "$i_m" == "$i_mm" ]] || [[ "$i_o" == "$i_mm" ]]; then
    echo "[FATAL] 三个 CKPT 中有硬链接同 inode（等价同一文件）："
    echo "  MME=$i_m $rp_m"
    echo "  OKVQA=$i_o $rp_o"
    echo "  MMMU=$i_mm $rp_mm"
    exit 1
  fi
  echo "[INFO] 已确认三个权重路径/inode 互不相同。"
}

# ---------- 可选：三次 Wanda 剪枝（与默认 ckpt 文件名一致）----------
if [[ "$RUN_PRUNE" == "1" ]]; then
  echo "========== RUN_PRUNE=1（ECoFLaP Wanda，GPU=$CUDA_VISIBLE_DEVICES）=========="
  MASTER_PORT="${MASTER_PORT:-29810}"
  J_MME="okvqa_mme-${WANDA_SUFFIX}"
  J_OK="okvqa_okvqaoverall-${WANDA_SUFFIX}"
  J_MM="okvqa_mmmu-${WANDA_SUFFIX}"

  P=$(next_port)
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$P" evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_mme.yaml \
    --pruning_method 'blipt5_wanda_pruner' --save_pruned_model \
    --score_method MEZO-GradOnly_sum --sparsity_ratio_granularity block \
    --max_sparsity_per_layer 0.6 --prunining_dataset_batch_size "${PRUNING_CALIB_BATCH:-8}" \
    --t5_prune_spec 24-0.5-1.0-1.0 --vit_prune_spec 39-0.5-1.0-1.0 \
    --job_id "$J_MME"

  P=$(next_port)
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$P" evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_okvqa_train_overall.yaml \
    --pruning_method 'blipt5_wanda_pruner' --save_pruned_model \
    --score_method MEZO-GradOnly_sum --sparsity_ratio_granularity block \
    --max_sparsity_per_layer 0.6 --prunining_dataset_batch_size "${PRUNING_CALIB_BATCH:-8}" \
    --t5_prune_spec 24-0.5-1.0-1.0 --vit_prune_spec 39-0.5-1.0-1.0 \
    --job_id "$J_OK"

  P=$(next_port)
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$P" evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_mmmu_overall.yaml \
    --pruning_method 'blipt5_wanda_pruner' --save_pruned_model \
    --score_method MEZO-GradOnly_sum --sparsity_ratio_granularity block \
    --max_sparsity_per_layer 0.6 --prunining_dataset_batch_size "${PRUNING_CALIB_BATCH:-8}" \
    --t5_prune_spec 24-0.5-1.0-1.0 --vit_prune_spec 39-0.5-1.0-1.0 \
    --job_id "$J_MM"

  echo "[INFO] 剪枝完成。"
fi

# --- 仅 OKVQA train-overall 权重（四项评测）；三套并行时请恢复 MME/MMMU 的 CKPT 解析与 assert_three_ckpts_distinct ---
CKPT_OKVQA_ONLY="${CKPT_OKVQA_ONLY:-$REPO_ROOT/pruned_checkpoint/okvqa_ghlc-blipt5_wanda_pruner_0.5-1.0-1.0_MEZO-GradOnly_sum0.6_block_bs8.pth}"
if [[ ! -f "$CKPT_OKVQA_ONLY" ]]; then
  echo "[FATAL] 找不到 OKVQA calibration 权重: $CKPT_OKVQA_ONLY"
  exit 1
fi
echo "[INFO] 本次只跑 OKVQA calibration 权重四项评测: $CKPT_OKVQA_ONLY"

# # --- 原三套权重解析（恢复 MME/MMMU 评测时取消注释）---
# CKPT_MME="${CKPT_MME:-$REPO_ROOT/pruned_checkpoint/okvqa_mme-${WANDA_SUFFIX}.pth}"
# if [[ -z "${CKPT_OKVQA_OVERALL:-}" ]]; then
#   _OKVQA_PRUNE="$REPO_ROOT/pruned_checkpoint/okvqa_okvqaoverall-${WANDA_SUFFIX}.pth"
#   ...
# fi
# CKPT_MMMU_OVERALL="${CKPT_MMMU_OVERALL:-$REPO_ROOT/pruned_checkpoint/okvqa_mmmu-${WANDA_SUFFIX}.pth}"
# assert_three_ckpts_distinct "$CKPT_MME" "$CKPT_OKVQA_OVERALL" "$CKPT_MMMU_OVERALL"

# 评测段统一从 29900 起，避免与剪枝端口混用
MASTER_PORT=29900

run_one_ckpt_suite() {
  local tag="$1"
  local ckpt="$2"
  local fullval_job="$3"

  echo ""
  echo "#####################################################################"
  echo "# $tag  |  $ckpt"
  echo "#####################################################################"

  if ! check_ckpt "$ckpt"; then
    return 0
  fi

  export LAVIS_EVAL_CALIB_TAG="$tag"

  echo ""
  echo ">>> [$tag] MMBench ($MMBENCH_ROOT, $MMBENCH_SPLIT) overall only"
  export LAVIS_METRICS_BENCHMARK="MMBench"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMBENCH_ROOT" --split "$MMBENCH_SPLIT" \
    --ckpt "$ckpt" --batch_size "$EVAL_BATCH_SIZE" --device cuda \
    --overall_only

  echo ""
  echo ">>> [$tag] MME yes/no ($MME_ROOT, $MME_SPLIT)"
  python scripts/blip2/mme_eval_yes_no.py \
    --mme_root "$MME_ROOT" --split "$MME_SPLIT" \
    --ckpt "$ckpt" --batch_size "$EVAL_BATCH_SIZE" --device cuda

  echo ""
  echo ">>> [$tag] OKVQA 全量 val overall  job_id=$fullval_job"
  P_FULL=$(next_port)
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$P_FULL" evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
    --t5_pruned_checkpoint "$ckpt" --vit_pruned_checkpoint "$ckpt" \
    --job_id "$fullval_job"

  echo ""
  echo ">>> [$tag] MMMU ($MMMU_ROOT, $MMMU_SPLIT) overall only"
  export LAVIS_METRICS_BENCHMARK="MMMU"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMMU_ROOT" --split "$MMMU_SPLIT" \
    --ckpt "$ckpt" --batch_size "$EVAL_BATCH_SIZE" --device cuda \
    --overall_only
}

# run_one_ckpt_suite "MME_calib" "$CKPT_MME" "okvqa_eval_ecoflapMME_fullval"
run_one_ckpt_suite "OKVQA_train_overall_calib" "$CKPT_OKVQA_ONLY" "okvqa_eval_ecoflapOKVQAoverall_fullval"
# run_one_ckpt_suite "MMMU_overall_calib" "$CKPT_MMMU_OVERALL" "okvqa_eval_ecoflapMMMUoverall_fullval"

echo ""
echo "========== 整体结果总表 =========="
python "$SCRIPT_DIR/collect_ecoflap_eval_summary.py" \
  --repo-root "$REPO_ROOT" \
  --metrics-jsonl "$LAVIS_METRICS_JSONL" \
  --out-md "$SUMMARY_MD" \
  --out-tsv "$SUMMARY_TSV"
echo "MD:  $SUMMARY_MD"
echo "TSV: $SUMMARY_TSV"
echo "JSONL: $LAVIS_METRICS_JSONL"
