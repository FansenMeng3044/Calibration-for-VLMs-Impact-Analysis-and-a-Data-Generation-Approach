#!/usr/bin/env bash
# =============================================================================
# UKMP 剪枝：6 个 MMMU 领域各自作为 calibration（每类 dev+validation 各取 90 条，共 90 条/类）
# 与 MMMU eval 同源：MMMU 单图多选题，dev+validation。
#
# 步骤:
#   Step0 生成 6 个 calibration JSON（每类 90 条，图片共用 MMMU_calibration/images）
#   Step1 按 6 个领域分别跑 UKMP 剪枝，得到 6 个 checkpoint
#
# 用法:
#   cd /root/autodl-tmp/UKMP-main/LAVIS
#   bash scripts/structured_blip2/run_ukmp_prune_mmmu_6disciplines.sh
#
# 输出:
#   pruned_checkpoint/ukmp_prune/okvqa_MMMU_Art_Design-90data-.../pytorch_model.bin
#   ... 共 6 个
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

MMMU_ROOT="${MMMU_ROOT:-/root/autodl-tmp/MMMU_single_image}"
CALIB_DIR="${CALIB_DIR:-/root/autodl-tmp/MMMU_calibration}"
MAX_PER_DISC=90

# -----------------------------------------------------------------------------
# Step0: 生成 6 个领域 JSON，每类 90 条（dev+validation），图片导出到 CALIB_DIR/images
# -----------------------------------------------------------------------------
echo "[Step0] Generating 6 discipline calibration JSONs (max ${MAX_PER_DISC} per discipline, dev+validation)"
python scripts/blip2/mmmu_to_calibration_format.py \
  --mmmu_root "$MMMU_ROOT" \
  --out_dir "$CALIB_DIR" \
  --splits dev validation \
  --by_discipline \
  --max_per_discipline $MAX_PER_DISC

for f in "$CALIB_DIR"/mmmu_calibration_train_Art_Design.json \
         "$CALIB_DIR"/mmmu_calibration_train_Business.json \
         "$CALIB_DIR"/mmmu_calibration_train_Science.json \
         "$CALIB_DIR"/mmmu_calibration_train_Health_Medicine.json \
         "$CALIB_DIR"/mmmu_calibration_train_Humanities_Social_Science.json \
         "$CALIB_DIR"/mmmu_calibration_train_Tech_Engineering.json; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] Missing $f"
    exit 1
  fi
done
echo "[Step0] Done."

# -----------------------------------------------------------------------------
# Step1: 6 次 UKMP 剪枝（每类一个 calibration）
# -----------------------------------------------------------------------------
DISCIPLINES=(Art_Design Business Science Health_Medicine Humanities_Social_Science Tech_Engineering)
MASTER_PORT=18086

for i in "${!DISCIPLINES[@]}"; do
  DISC="${DISCIPLINES[$i]}"
  CFG="lavis/projects/blip2/prune/cc595k_prefix_derivative_compute_okvqa_mmmu_${DISC}.yaml"
  JOB_ID="okvqa_MMMU_${DISC}-90data-taylor+knowledge-param_first-param_norm-0.5-blockwise-global-select_loss-multimodal"
  echo "[Step1] Pruning with calibration: $DISC -> $JOB_ID"
  CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.run \
    --nproc_per_node=1 --master_port=$((MASTER_PORT + i)) \
    ukmp_prune.py \
    --cfg-path "$CFG" \
    --device cuda \
    --save_ckpt_log_name ukmp_prune \
    --job_id "$JOB_ID" \
    --pruning_ratio 0.5 \
    --granularity block \
    --pruner_type taylor+knowledge \
    --taylor param_first \
    --num_examples 90 \
    --channel_per_step 1000 \
    --global_pruning \
    --imp_normalizer param \
    --select_loss \
    --entropy_importance \
    --multimodal
  CKPT="pruned_checkpoint/ukmp_prune/${JOB_ID}/pytorch_model.bin"
  if [ ! -f "$CKPT" ]; then
    echo "[ERROR] Pruned checkpoint not found: $CKPT"
    exit 1
  fi
  echo "[Step1] Saved: $CKPT"
done

echo "[DONE] All 6 MMMU-discipline calibration prunes finished."
