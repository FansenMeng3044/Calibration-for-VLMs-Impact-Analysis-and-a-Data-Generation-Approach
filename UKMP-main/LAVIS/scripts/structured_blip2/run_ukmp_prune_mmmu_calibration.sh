#!/usr/bin/env bash
# =============================================================================
# UKMP 剪枝：以 MMMU（dev+validation 单图）为 calibration
#
# 前置:
#   1. 已生成 MMMU calibration 数据:
#        python scripts/blip2/mmmu_to_calibration_format.py \
#          --mmmu_root /root/autodl-tmp/MMMU_single_image \
#          --out_dir /root/autodl-tmp/MMMU_calibration
#   2. 数据盘/镜像（可选）: export HF_HOME=... HF_ENDPOINT=...
#
# 用法:
#   cd /root/autodl-tmp/UKMP-main/LAVIS
#   bash scripts/structured_blip2/run_ukmp_prune_mmmu_calibration.sh
#
# 输出: pruned_checkpoint/ukmp_prune/okvqa_MMMU-128data-.../pytorch_model.bin
# =============================================================================

set -e
cd "$(dirname "$0")/../.."
ROOT="$PWD"
if [ "$(basename "$ROOT")" != "LAVIS" ]; then
  echo "[ERROR] Run from UKMP-main/LAVIS. Current: $ROOT"
  exit 1
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

MMMU_JOB_ID="okvqa_MMMU-128data-taylor+knowledge-param_first-param_norm-0.5-blockwise-global-select_loss-multimodal"

echo "[Step1] UKMP prune with MMMU calibration -> $MMMU_JOB_ID"
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.run \
  --nproc_per_node=1 --master_port=18085 \
  ukmp_prune.py \
  --cfg-path lavis/projects/blip2/prune/cc595k_prefix_derivative_compute_okvqa_mmmu.yaml \
  --device cuda \
  --save_ckpt_log_name ukmp_prune \
  --job_id "$MMMU_JOB_ID" \
  --pruning_ratio 0.5 \
  --granularity block \
  --pruner_type taylor+knowledge \
  --taylor param_first \
  --num_examples 128 \
  --channel_per_step 1000 \
  --global_pruning \
  --imp_normalizer param \
  --select_loss \
  --entropy_importance \
  --multimodal

MMMU_CKPT="pruned_checkpoint/ukmp_prune/${MMMU_JOB_ID}/pytorch_model.bin"
if [ ! -f "$MMMU_CKPT" ]; then
  echo "[ERROR] Pruned checkpoint not found: $MMMU_CKPT"
  exit 1
fi
echo "[DONE] MMMU calibration prune saved: $MMMU_CKPT"
