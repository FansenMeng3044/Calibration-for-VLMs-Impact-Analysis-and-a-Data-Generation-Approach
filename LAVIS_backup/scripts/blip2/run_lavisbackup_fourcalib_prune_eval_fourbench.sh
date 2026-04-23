#!/usr/bin/env bash
# =============================================================================
# LAVIS_backup：四套 calibration 剪枝（T5-only，evaluate_blip 默认不剪 ViT）+
# 每套权重各跑 MMBench / OKVQA overall / MMMU / MathVista_MC 评测。
#
# 默认对 LAVIS_DISTRIBUTED_SAMPLER_SEED = 0、30、42 各跑完整一轮（四套剪枝 × 四套 eval × 3 seed）。
# 可通过 SAMPLER_SEEDS / SEEDS 覆盖；单 seed： SEEDS=42 bash ...
#
# Calibration 与剪枝产物命名:
#   pruned_checkpoint/okvqa_cf_0.5_calibOKVQAtrain_${STAMP}.pth
#   ... STAMP = ${DATE_TAG}_seed${SEED}
#
# 环境变量:
#   DATE_TAG              默认 $(date +%m%d)
#   SAMPLER_SEEDS / SEEDS 默认 "0 30 42"（空格分隔）
#   RUN_PRUNE=1 RUN_EVAL=1
#   MASTER_PORT           每轮 seed 从该基址递增（默认 29700）
#   CUDA_VISIBLE_DEVICES MMBENCH_ROOT MMMU_ROOT ...
#   MATHVISTA_EVAL_JSON MATHVISTA_IMAGES_DIR EVAL_BATCH_SIZE
#
# 用法:
#   cd /root/autodl-tmp/LAVIS_backup
#   bash scripts/blip2/run_lavisbackup_fourcalib_prune_eval_fourbench.sh
#
# 只跑 seed 42 一轮:
#   SEEDS=42 bash scripts/blip2/run_lavisbackup_fourcalib_prune_eval_fourbench.sh
#
# 只评测某 seed（pth 已存在）:
#   RUN_PRUNE=0 SEEDS=42 DATE_TAG=0409 bash .../run_lavisbackup_fourcalib_prune_eval_fourbench.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

export DATE_TAG="${DATE_TAG:-$(date +%m%d)}"
# 多 seed：默认 0 30 42；单 seed 可 SEEDS=42
SAMPLER_SEEDS="${SAMPLER_SEEDS:-${SEEDS:-0 30 42}}"

export MMBENCH_ROOT="${MMBENCH_ROOT:-/root/autodl-tmp/MMBench_eval}"
export MMMU_ROOT="${MMMU_ROOT:-/root/autodl-tmp/MMMU_single_image}"
export MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
export MMMU_SPLIT="${MMMU_SPLIT:-test}"

AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
export MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-$AUTODL_TMP/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
export MATHVISTA_IMAGES_DIR="${MATHVISTA_IMAGES_DIR:-$AUTODL_TMP/MathVista_eval_testmini_mc/images}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"

RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

T5_SPEC="${T5_SPEC:-24-0.5-1.0-1.0}"
VIT_SPEC="${VIT_SPEC:-39-0.5-1.0-1.0}"
PRUNE_METHOD="${PRUNE_METHOD:-blipt5_tamp_pruner}"

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

MASTER_PORT_BASE="${MASTER_PORT:-29700}"
MASTER_PORT="$MASTER_PORT_BASE"

echo "[INFO] DATE_TAG=$DATE_TAG"
echo "[INFO] SAMPLER_SEEDS=$SAMPLER_SEEDS"
echo "[INFO] T5-only 剪枝；PRUNE_METHOD=$PRUNE_METHOD RUN_PRUNE=$RUN_PRUNE RUN_EVAL=$RUN_EVAL"

check_ckpt() {
  local c="$1"
  if [[ ! -f "$c" ]]; then
    echo "[WARN] 跳过：找不到权重: $c"
    return 1
  fi
  return 0
}

assert_four_ckpts_distinct() {
  local a="$1" b="$2" c="$3" d="$4"
  if [[ "${SKIP_CKPT_DISTINCT_CHECK:-0}" == "1" ]]; then
    echo "[INFO] 已跳过四套权重互异性检查（SKIP_CKPT_DISTINCT_CHECK=1）"
    return 0
  fi
  local paths=("$a" "$b" "$c" "$d")
  local i j
  for i in 0 1 2 3; do
    for j in 0 1 2 3; do
      [[ "$i" -lt "$j" ]] || continue
      if [[ ! -f "${paths[$i]}" ]] || [[ ! -f "${paths[$j]}" ]]; then
        echo "[WARN] 未齐四套 pth，跳过互异性检查。"
        return 0
      fi
    done
  done
  local -a resolved inodes
  for p in "${paths[@]}"; do
    resolved+=("$(readlink -f "$p")")
    inodes+=("$(stat -c '%d:%i' "$(readlink -f "$p")")")
  done
  for i in 0 1 2 3; do
    for j in 0 1 2 3; do
      [[ "$i" -lt "$j" ]] || continue
      if [[ "${resolved[$i]}" == "${resolved[$j]}" ]]; then
        echo "[FATAL] 四套 CKPT 解析路径必须两两不同。"
        exit 1
      fi
      if [[ "${inodes[$i]}" == "${inodes[$j]}" ]]; then
        echo "[FATAL] 四套 CKPT inode 必须两两不同（勿硬链同一文件）。"
        exit 1
      fi
    done
  done
  echo "[INFO] 已确认四套权重路径与 inode 两两不同。"
}

run_one_ckpt_eval_four() {
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
  echo ">>> [$tag] MMBench ($MMBENCH_ROOT, split=$MMBENCH_SPLIT)"
  export LAVIS_METRICS_BENCHMARK="MMBench"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMBENCH_ROOT" \
    --split "$MMBENCH_SPLIT" \
    --ckpt "$ckpt" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --device cuda \
    --overall_only

  echo ""
  echo ">>> [$tag] OKVQA full val job_id=$fullval_job"
  local P_FULL
  P_FULL=$MASTER_PORT
  MASTER_PORT=$((MASTER_PORT + 1))
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$P_FULL" evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
    --t5_pruned_checkpoint "$ckpt" \
    --vit_pruned_checkpoint "$ckpt" \
    --job_id "$fullval_job"

  echo ""
  echo ">>> [$tag] MMMU ($MMMU_ROOT, split=$MMMU_SPLIT)"
  export LAVIS_METRICS_BENCHMARK="MMMU"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMMU_ROOT" \
    --split "$MMMU_SPLIT" \
    --ckpt "$ckpt" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --device cuda \
    --overall_only

  echo ""
  echo ">>> [$tag] MathVista MC"
  if [[ ! -f "$MATHVISTA_EVAL_JSON" ]]; then
    echo "[WARN] 跳过 MathVista：缺少 $MATHVISTA_EVAL_JSON"
  else
    export LAVIS_METRICS_BENCHMARK="MathVista_MC"
    python scripts/blip2/mathvista_mc_eval.py \
      --eval_json "$MATHVISTA_EVAL_JSON" \
      --images_dir "$MATHVISTA_IMAGES_DIR" \
      --ckpt "$ckpt" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda
  fi
}

run_one_sampler_seed() {
  local SEED="$1"
  export SEED
  export LAVIS_DISTRIBUTED_SAMPLER_SEED="$SEED"

  local STAMP="${DATE_TAG}_seed${SEED}"

  # 多 seed 时每轮仅由 STAMP 生成命名（勿 export PRUNE_JOB_* / TAG_* 到外层，以免串 seed）
  local PRUNE_JOB_OKVQA="okvqa_cf_0.5_calibOKVQAtrain_${STAMP}"
  local PRUNE_JOB_MMBENCH="okvqa_cf_0.5_calibMMBench_${STAMP}"
  local PRUNE_JOB_MMU="okvqa_cf_0.5_calibMMMU_${STAMP}"
  local PRUNE_JOB_MV="okvqa_cf_0.5_calibMathVista_${STAMP}"

  local CKPT_OKVQA="$REPO_ROOT/pruned_checkpoint/${PRUNE_JOB_OKVQA}.pth"
  local CKPT_MMBENCH="$REPO_ROOT/pruned_checkpoint/${PRUNE_JOB_MMBENCH}.pth"
  local CKPT_MMU="$REPO_ROOT/pruned_checkpoint/${PRUNE_JOB_MMU}.pth"
  local CKPT_MV="$REPO_ROOT/pruned_checkpoint/${PRUNE_JOB_MV}.pth"

  local TAG_OKVQA="calibOKVQAtrain_${STAMP}"
  local TAG_MMBENCH="calibMMBench_${STAMP}"
  local TAG_MMU="calibMMMU_${STAMP}"
  local TAG_MV="calibMathVista_${STAMP}"

  local EVAL_JOB_OKVQA="okvqa_eval_${TAG_OKVQA}_fullval"
  local EVAL_JOB_MMBENCH="okvqa_eval_${TAG_MMBENCH}_fullval"
  local EVAL_JOB_MMU="okvqa_eval_${TAG_MMU}_fullval"
  local EVAL_JOB_MV="okvqa_eval_${TAG_MV}_fullval"

  local SUMMARY_STAMP
  SUMMARY_STAMP="$(date +%Y%m%d_%H%M%S)"
  export LAVIS_METRICS_JSONL="$SUMMARY_DIR/lavisbackup_fourcalib_fourbench_${STAMP}_${SUMMARY_STAMP}.jsonl"
  : > "$LAVIS_METRICS_JSONL"
  local SUMMARY_MD="$SUMMARY_DIR/lavisbackup_fourcalib_fourbench_${STAMP}_${SUMMARY_STAMP}.md"
  local SUMMARY_TSV="$SUMMARY_DIR/lavisbackup_fourcalib_fourbench_${STAMP}_${SUMMARY_STAMP}.tsv"

  echo ""
  echo "########################################################################"
  echo "#  LAVIS_DISTRIBUTED_SAMPLER_SEED=$SEED  STAMP=$STAMP"
  echo "########################################################################"
  echo "[INFO] 四套剪枝 pth:"
  echo "  OKVQA train: $CKPT_OKVQA"
  echo "  MMBench:     $CKPT_MMBENCH"
  echo "  MMMU:        $CKPT_MMU"
  echo "  MathVista:   $CKPT_MV"
  echo "[INFO] metrics jsonl: $LAVIS_METRICS_JSONL"

  if [[ "$RUN_PRUNE" == "1" ]]; then
    echo ""
    echo "========== RUN_PRUNE=1：seed=$SEED 四套剪枝 =========="

    local P
    P=$MASTER_PORT; MASTER_PORT=$((MASTER_PORT + 1))
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$P" evaluate_blip.py \
      --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_okvqa_train_overall.yaml \
      --pruning_method "$PRUNE_METHOD" --save_pruned_model \
      --t5_prune_spec "$T5_SPEC" --vit_prune_spec "$VIT_SPEC" \
      --job_id "$PRUNE_JOB_OKVQA"

    P=$MASTER_PORT; MASTER_PORT=$((MASTER_PORT + 1))
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$P" evaluate_blip.py \
      --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_mmbench.yaml \
      --pruning_method "$PRUNE_METHOD" --save_pruned_model \
      --t5_prune_spec "$T5_SPEC" --vit_prune_spec "$VIT_SPEC" \
      --job_id "$PRUNE_JOB_MMBENCH"

    P=$MASTER_PORT; MASTER_PORT=$((MASTER_PORT + 1))
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$P" evaluate_blip.py \
      --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_mmmu_overall.yaml \
      --pruning_method "$PRUNE_METHOD" --save_pruned_model \
      --t5_prune_spec "$T5_SPEC" --vit_prune_spec "$VIT_SPEC" \
      --job_id "$PRUNE_JOB_MMU"

    P=$MASTER_PORT; MASTER_PORT=$((MASTER_PORT + 1))
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$P" evaluate_blip.py \
      --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_mathvista.yaml \
      --pruning_method "$PRUNE_METHOD" --save_pruned_model \
      --t5_prune_spec "$T5_SPEC" --vit_prune_spec "$VIT_SPEC" \
      --job_id "$PRUNE_JOB_MV"

    echo "[INFO] seed=$SEED 剪枝完成。"
  else
    echo "[INFO] RUN_PRUNE=0，跳过 seed=$SEED 剪枝。"
  fi

  if [[ "$RUN_EVAL" == "1" ]]; then
    echo ""
    echo "========== RUN_EVAL=1：seed=$SEED 四套 pth × 四评测 =========="

    if [[ ! -f "$CKPT_OKVQA" ]] || [[ ! -f "$CKPT_MMBENCH" ]] || [[ ! -f "$CKPT_MMU" ]] || [[ ! -f "$CKPT_MV" ]]; then
      echo "[FATAL] seed=$SEED 评测需要四个 pth 均存在。"
      [[ -f "$CKPT_OKVQA" ]] || echo "  缺: $CKPT_OKVQA"
      [[ -f "$CKPT_MMBENCH" ]] || echo "  缺: $CKPT_MMBENCH"
      [[ -f "$CKPT_MMU" ]] || echo "  缺: $CKPT_MMU"
      [[ -f "$CKPT_MV" ]] || echo "  缺: $CKPT_MV"
      exit 1
    fi

    assert_four_ckpts_distinct "$CKPT_OKVQA" "$CKPT_MMBENCH" "$CKPT_MMU" "$CKPT_MV"

    run_one_ckpt_eval_four "$TAG_OKVQA" "$CKPT_OKVQA" "$EVAL_JOB_OKVQA"
    run_one_ckpt_eval_four "$TAG_MMBENCH" "$CKPT_MMBENCH" "$EVAL_JOB_MMBENCH"
    run_one_ckpt_eval_four "$TAG_MMU" "$CKPT_MMU" "$EVAL_JOB_MMU"
    run_one_ckpt_eval_four "$TAG_MV" "$CKPT_MV" "$EVAL_JOB_MV"

    echo ""
    echo "========== seed=$SEED 汇总表 =========="
    python "$SCRIPT_DIR/collect_lavisbackup_eval_summary.py" \
      --repo-root "$REPO_ROOT" \
      --metrics-jsonl "$LAVIS_METRICS_JSONL" \
      --out-md "$SUMMARY_MD" \
      --out-tsv "$SUMMARY_TSV" \
      --suites \
        "${TAG_OKVQA}:${EVAL_JOB_OKVQA}" \
        "${TAG_MMBENCH}:${EVAL_JOB_MMBENCH}" \
        "${TAG_MMU}:${EVAL_JOB_MMU}" \
        "${TAG_MV}:${EVAL_JOB_MV}"

    echo "已写入: $SUMMARY_MD"
    echo "TSV:    $SUMMARY_TSV"
    echo "指标行: $LAVIS_METRICS_JSONL"
  else
    echo "[INFO] RUN_EVAL=0，跳过 seed=$SEED 评测。"
  fi

  echo ""
  echo "========== 完成 seed=$SEED STAMP=$STAMP =========="
}

for _seed in $SAMPLER_SEEDS; do
  run_one_sampler_seed "$_seed"
done

echo ""
echo "[DONE] 全部 seed 跑完: $SAMPLER_SEEDS"
