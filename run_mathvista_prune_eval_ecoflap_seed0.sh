#!/usr/bin/env bash
set -euo pipefail

# --------------------
# ECoFLaP: Wanda 剪枝 + 评测（MMBench/OKVQA overall/MMMU overall）
# 使用 MathVista calibration
# DistributedSampler seed：0（通过 LAVIS_DISTRIBUTED_SAMPLER_SEED）
# --------------------

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LAVIS_DISTRIBUTED_SAMPLER_SEED="${LAVIS_DISTRIBUTED_SAMPLER_SEED:-0}"
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

export DATE_TAG="${DATE_TAG:-$(date +%m%d)}"
export SEED="${LAVIS_DISTRIBUTED_SAMPLER_SEED}"
export DATE_TAG_FULL="${DATE_TAG}_s${SEED}"

REPO_ROOT="/root/autodl-tmp/ECoFLaP/LAVIS"
cd "$REPO_ROOT"

export MMBENCH_ROOT="${MMBENCH_ROOT:-/root/autodl-tmp/MMBench_eval}"
export MMMU_ROOT="${MMMU_ROOT:-/root/autodl-tmp/MMMU_single_image}"
export MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
export MMMU_SPLIT="${MMMU_SPLIT:-test}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"

export MATHVISTA_CFG_PATH="lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_mathvista.yaml"

export PRUNING_CALIB_BATCH="${PRUNING_CALIB_BATCH:-8}"

export CKPT_JOB_ID="okvqa_cf_0.5_mathvista_overall_${DATE_TAG_FULL}"
export CKPT_PATH="$REPO_ROOT/pruned_checkpoint/${CKPT_JOB_ID}.pth"

export LAVIS_EVAL_CALIB_TAG="MathVistaoverall_seed${SEED}_${DATE_TAG_FULL}"
export EVAL_JOB_OKVQA="okvqa_eval_calibMathVistaoverall_seed${SEED}_${DATE_TAG_FULL}_fullval"

SUMMARY_DIR="$REPO_ROOT/lavis/output/BLIP2"
mkdir -p "$SUMMARY_DIR"
SUMMARY_STAMP="$(date +%Y%m%d_%H%M%S)"
export LAVIS_METRICS_JSONL="$SUMMARY_DIR/ecoflap_eval_metrics_mathvista_seed${SEED}_${DATE_TAG_FULL}_${SUMMARY_STAMP}.jsonl"
: > "$LAVIS_METRICS_JSONL"

# 1) 剪枝
P_PRUNE="${P_PRUNE:-29910}"
echo "[INFO] ECoFLaP pruning: job_id=$CKPT_JOB_ID"
python -m torch.distributed.run --nproc_per_node=1 --master_port="$P_PRUNE" evaluate_blip.py \
  --cfg-path "$MATHVISTA_CFG_PATH" \
  --pruning_method 'blipt5_wanda_pruner' --save_pruned_model \
  --score_method MEZO-GradOnly_sum --sparsity_ratio_granularity block \
  --max_sparsity_per_layer 0.6 --prunining_dataset_batch_size "${PRUNING_CALIB_BATCH}" \
  --t5_prune_spec 24-0.5-1.0-1.0 --vit_prune_spec 39-0.5-1.0-1.0 \
  --job_id "$CKPT_JOB_ID"

echo "[INFO] ckpt: $CKPT_PATH"
if [[ ! -f "$CKPT_PATH" ]]; then
  echo "[FATAL] ckpt not found: $CKPT_PATH"
  exit 1
fi

# 2) MMBench
export LAVIS_METRICS_BENCHMARK="MMBench"
echo "[INFO] Eval MMBench (overall)"
python scripts/blip2/mmmu_eval_by_discipline.py \
  --mmmu_root "$MMBENCH_ROOT" \
  --split "$MMBENCH_SPLIT" \
  --ckpt "$CKPT_PATH" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --device cuda \
  --overall_only

# 3) OKVQA overall
P_OKVQA="${P_OKVQA:-29911}"
echo "[INFO] Eval OKVQA overall"
python -m torch.distributed.run --nproc_per_node=1 --master_port="$P_OKVQA" evaluate_blip.py \
  --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
  --t5_pruned_checkpoint "$CKPT_PATH" \
  --vit_pruned_checkpoint "$CKPT_PATH" \
  --job_id "$EVAL_JOB_OKVQA"

# 4) MMMU
export LAVIS_METRICS_BENCHMARK="MMMU"
echo "[INFO] Eval MMMU (overall)"
python scripts/blip2/mmmu_eval_by_discipline.py \
  --mmmu_root "$MMMU_ROOT" \
  --split "$MMMU_SPLIT" \
  --ckpt "$CKPT_PATH" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --device cuda \
  --overall_only

# 5) 汇总
SUMMARY_MD="$SUMMARY_DIR/ecoflap_eval_summary_mathvista_seed${SEED}_${DATE_TAG_FULL}_${SUMMARY_STAMP}.md"
SUMMARY_TSV="$SUMMARY_DIR/ecoflap_eval_summary_mathvista_seed${SEED}_${DATE_TAG_FULL}_${SUMMARY_STAMP}.tsv"
python scripts/blip2/collect_ecoflap_eval_summary.py \
  --repo-root "$REPO_ROOT" \
  --metrics-jsonl "$LAVIS_METRICS_JSONL" \
  --out-md "$SUMMARY_MD" \
  --out-tsv "$SUMMARY_TSV" \
  --suites "$LAVIS_EVAL_CALIB_TAG:$EVAL_JOB_OKVQA"

echo "[DONE] ECoFLaP done."
