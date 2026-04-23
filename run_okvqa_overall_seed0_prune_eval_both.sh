#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LAVIS_DISTRIBUTED_SAMPLER_SEED=0
DATE_TAG="${DATE_TAG:-$(date +%m%d)_s0}"

# 强制离线，避免 huggingface.co 联网重试
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

# 避免 libgomp: Invalid value for environment variable OMP_NUM_THREADS
# OpenMP 通常要求 OMP_NUM_THREADS >= 1；如果你环境里是空/0/非数字则回退到 1。
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]] || [[ "${OMP_NUM_THREADS:-0}" -le 0 ]]; then
  export OMP_NUM_THREADS=1
fi

run_lavisbackup() {
  cd /root/autodl-tmp/LAVIS_backup
  export CKPT_JOB_ID="okvqa_cf_0.5_overall_${DATE_TAG}"
  export CKPT_PATH="/root/autodl-tmp/LAVIS_backup/pruned_checkpoint/${CKPT_JOB_ID}.pth"
  export SUMMARY_STAMP="$(date +%Y%m%d_%H%M%S)"
  export LAVIS_METRICS_JSONL="/root/autodl-tmp/LAVIS_backup/lavis/output/BLIP2/lavisbackup_eval_metrics_${SUMMARY_STAMP}.jsonl"
  mkdir -p /root/autodl-tmp/LAVIS_backup/lavis/output/BLIP2
  : > "$LAVIS_METRICS_JSONL"

  python -m torch.distributed.run --nproc_per_node=1 --master_port=29710 evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_okvqa_train_overall.yaml \
    --pruning_method blipt5_tamp_pruner --save_pruned_model \
    --t5_prune_spec 24-0.5-1.0-1.0 --vit_prune_spec 39-0.5-1.0-1.0 \
    --job_id "$CKPT_JOB_ID"

  export LAVIS_EVAL_CALIB_TAG="OKVQAoverall_seed0_${DATE_TAG}"
  export LAVIS_METRICS_BENCHMARK="MMBench"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root /root/autodl-tmp/MMBench_eval --split dev --ckpt "$CKPT_PATH" \
    --batch_size 2 --device cuda --overall_only

  python -m torch.distributed.run --nproc_per_node=1 --master_port=29711 evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
    --t5_pruned_checkpoint "$CKPT_PATH" --vit_pruned_checkpoint "$CKPT_PATH" \
    --job_id "okvqa_eval_calibOKVQAoverall_seed0_${DATE_TAG}_fullval"

  export LAVIS_METRICS_BENCHMARK="MMMU"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root /root/autodl-tmp/MMMU_single_image --split test --ckpt "$CKPT_PATH" \
    --batch_size 2 --device cuda --overall_only
}

run_ecoflap() {
  cd /root/autodl-tmp/ECoFLaP/LAVIS
  export CKPT_JOB_ID="okvqa_cf_0.5_overall_${DATE_TAG}"
  export CKPT_PATH="/root/autodl-tmp/ECoFLaP/LAVIS/pruned_checkpoint/${CKPT_JOB_ID}.pth"
  export SUMMARY_STAMP="$(date +%Y%m%d_%H%M%S)"
  export LAVIS_METRICS_JSONL="/root/autodl-tmp/ECoFLaP/LAVIS/lavis/output/BLIP2/ecoflap_eval_metrics_${SUMMARY_STAMP}.jsonl"
  mkdir -p /root/autodl-tmp/ECoFLaP/LAVIS/lavis/output/BLIP2
  : > "$LAVIS_METRICS_JSONL"

  python -m torch.distributed.run --nproc_per_node=1 --master_port=29910 evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_okvqa_train_overall.yaml \
    --pruning_method blipt5_wanda_pruner --save_pruned_model \
    --score_method MEZO-GradOnly_sum --sparsity_ratio_granularity block \
    --max_sparsity_per_layer 0.6 --prunining_dataset_batch_size 8 \
    --t5_prune_spec 24-0.5-1.0-1.0 --vit_prune_spec 39-0.5-1.0-1.0 \
    --job_id "$CKPT_JOB_ID"

  export LAVIS_EVAL_CALIB_TAG="OKVQAoverall_seed0_${DATE_TAG}"
  export LAVIS_METRICS_BENCHMARK="MMBench"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root /root/autodl-tmp/MMBench_eval --split dev --ckpt "$CKPT_PATH" \
    --batch_size 2 --device cuda --overall_only

  python -m torch.distributed.run --nproc_per_node=1 --master_port=29911 evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
    --t5_pruned_checkpoint "$CKPT_PATH" --vit_pruned_checkpoint "$CKPT_PATH" \
    --job_id "okvqa_eval_ecoflapOKVQAoverall_seed0_${DATE_TAG}_fullval"

  export LAVIS_METRICS_BENCHMARK="MMMU"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root /root/autodl-tmp/MMMU_single_image --split test --ckpt "$CKPT_PATH" \
    --batch_size 2 --device cuda --overall_only
}

echo "[INFO] DATE_TAG=$DATE_TAG, sampler_seed=0"
echo ">>> LAVIS_backup"
run_lavisbackup
echo ">>> ECoFLaP"
run_ecoflap
echo "[DONE]"
