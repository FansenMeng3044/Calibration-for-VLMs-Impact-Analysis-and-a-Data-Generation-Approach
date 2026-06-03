#!/usr/bin/env bash
# 仅 OKVQA overall calibration 剪枝（DistributedSampler seed=0）
# 产出重命名 pth，并用该 pth 跑 MMBench/MMMU/OKVQA overall eval。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LAVIS_DISTRIBUTED_SAMPLER_SEED="${LAVIS_DISTRIBUTED_SAMPLER_SEED:-0}"

export MMBENCH_ROOT="${MMBENCH_ROOT:-/root/autodl-tmp/MMBench_eval}"
export MMMU_ROOT="${MMMU_ROOT:-/root/autodl-tmp/MMMU_single_image}"
export MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
export MMMU_SPLIT="${MMMU_SPLIT:-test}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"

RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
DATE_TAG="${DATE_TAG:-$(date +%m%d)}"

# 剪枝 job_id（evaluate_blip 会写 pruned_checkpoint/<job_id>.pth）
PRUNE_JOB_OKVQA="${PRUNE_JOB_OKVQA:-okvqa_cf_0.5_overall_${DATE_TAG}_seed0}"
CKPT_OKVQA="${CKPT_OKVQA:-$REPO_ROOT/pruned_checkpoint/${PRUNE_JOB_OKVQA}.pth}"

# 单权重评测标签/job
TAG_OKVQA="${TAG_OKVQA:-OKVQAtrain_overall_calib_${DATE_TAG}_seed0}"
EVAL_JOB_OKVQA="${EVAL_JOB_OKVQA:-okvqa_eval_calibOKVQAoverall_${DATE_TAG}_seed0_fullval}"

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
MASTER_PORT="${MASTER_PORT:-29700}"

SUMMARY_DIR="$REPO_ROOT/lavis/output/BLIP2"
mkdir -p "$SUMMARY_DIR"
SUMMARY_STAMP="$(date +%Y%m%d_%H%M%S)"
export LAVIS_METRICS_JSONL="${LAVIS_METRICS_JSONL:-$SUMMARY_DIR/lavisbackup_eval_metrics_${SUMMARY_STAMP}.jsonl}"
: > "$LAVIS_METRICS_JSONL"
SUMMARY_MD="$SUMMARY_DIR/lavisbackup_eval_summary_okvqa_only_seed0_${DATE_TAG}_${SUMMARY_STAMP}.md"
SUMMARY_TSV="$SUMMARY_DIR/lavisbackup_eval_summary_okvqa_only_seed0_${DATE_TAG}_${SUMMARY_STAMP}.tsv"

echo "[INFO] LAVIS_DISTRIBUTED_SAMPLER_SEED=$LAVIS_DISTRIBUTED_SAMPLER_SEED"
echo "[INFO] DATE_TAG=$DATE_TAG"
echo "[INFO] OKVQA-only CKPT: $CKPT_OKVQA"

if [[ "$RUN_PRUNE" == "1" ]]; then
  echo "========== RUN_PRUNE=1：只剪 OKVQA overall calibration =========="
  P=$MASTER_PORT; MASTER_PORT=$((MASTER_PORT + 1))
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$P" evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_okvqa_train_overall.yaml \
    --pruning_method "$PRUNE_METHOD" --save_pruned_model \
    --t5_prune_spec "$T5_SPEC" --vit_prune_spec "$VIT_SPEC" \
    --job_id "$PRUNE_JOB_OKVQA"
fi

if [[ ! -f "$CKPT_OKVQA" ]]; then
  echo "[FATAL] 找不到权重: $CKPT_OKVQA"
  exit 1
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  export LAVIS_EVAL_CALIB_TAG="$TAG_OKVQA"

  echo ">>> [$TAG_OKVQA] MMBench overall only"
  export LAVIS_METRICS_BENCHMARK="MMBench"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMBENCH_ROOT" --split "$MMBENCH_SPLIT" \
    --ckpt "$CKPT_OKVQA" --batch_size "$EVAL_BATCH_SIZE" --device cuda --overall_only

  echo ">>> [$TAG_OKVQA] OKVQA overall full val"
  P_FULL=$MASTER_PORT; MASTER_PORT=$((MASTER_PORT + 1))
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$P_FULL" evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
    --t5_pruned_checkpoint "$CKPT_OKVQA" --vit_pruned_checkpoint "$CKPT_OKVQA" \
    --job_id "$EVAL_JOB_OKVQA"

  echo ">>> [$TAG_OKVQA] MMMU overall only"
  export LAVIS_METRICS_BENCHMARK="MMMU"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMMU_ROOT" --split "$MMMU_SPLIT" \
    --ckpt "$CKPT_OKVQA" --batch_size "$EVAL_BATCH_SIZE" --device cuda --overall_only

  python "$SCRIPT_DIR/collect_lavisbackup_eval_summary.py" \
    --repo-root "$REPO_ROOT" --metrics-jsonl "$LAVIS_METRICS_JSONL" \
    --out-md "$SUMMARY_MD" --out-tsv "$SUMMARY_TSV" \
    --suites "${TAG_OKVQA}:${EVAL_JOB_OKVQA}"

  echo "[DONE] MD:  $SUMMARY_MD"
  echo "[DONE] TSV: $SUMMARY_TSV"
  echo "[DONE] JSONL: $LAVIS_METRICS_JSONL"
fi
