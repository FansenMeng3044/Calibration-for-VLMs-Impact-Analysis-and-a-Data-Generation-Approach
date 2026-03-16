#!/usr/bin/env bash
# =============================================================================
# 完整流程：6 种 MMMU calibration 剪枝 + 每个 checkpoint 跑一次 MMMU eval
#
# Step0: 生成 6 个领域 calibration JSON（每类 90 条，dev+validation）
# Step1: 按 6 个领域分别 UKMP 剪枝 -> 6 个 checkpoint
# Step2: 用每个 checkpoint 各跑一次 MMMU test eval，结果按类别存 log
#
# 用法:
#   cd /root/autodl-tmp/UKMP-main/LAVIS
#   bash scripts/structured_blip2/run_mmmu_6calib_prune_and_eval.sh
#
# 可选环境变量:
#   MMMU_ROOT, CALIB_DIR, EVAL_LOG_DIR, BATCH_SIZE_EVAL, CUDA_DEVICE
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
EVAL_LOG_DIR="${EVAL_LOG_DIR:-/root/autodl-tmp}"
BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-4}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_PER_DISC=90

DISCIPLINES=(Art_Design Business Science Health_Medicine Humanities_Social_Science Tech_Engineering)
MASTER_PORT=18086

# -----------------------------------------------------------------------------
# Step0: 生成 6 个领域 calibration JSON（每类 90 条）
# -----------------------------------------------------------------------------
echo "========== Step0: Generate 6 discipline calibration JSONs (max ${MAX_PER_DISC} per discipline) =========="
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
# Step1: 6 次 UKMP 剪枝（每类一个 calibration -> 6 个 checkpoint）
# -----------------------------------------------------------------------------
echo "========== Step1: UKMP prune per discipline (6 checkpoints) =========="
CKPT_LIST=()
for i in "${!DISCIPLINES[@]}"; do
  DISC="${DISCIPLINES[$i]}"
  CFG="lavis/projects/blip2/prune/cc595k_prefix_derivative_compute_okvqa_mmmu_${DISC}.yaml"
  JOB_ID="okvqa_MMMU_${DISC}-90data-taylor+knowledge-param_first-param_norm-0.5-blockwise-global-select_loss-multimodal"
  echo "[Step1] Pruning: $DISC -> $JOB_ID"
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u -m torch.distributed.run \
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
  CKPT_LIST+=("$CKPT")
  echo "[Step1] Saved: $CKPT"
done
echo "[Step1] All 6 prunes done."

# -----------------------------------------------------------------------------
# Step2: 每个 checkpoint 跑一次 MMMU test eval（一个类别/ckpt 跑一次）
# -----------------------------------------------------------------------------
echo "========== Step2: MMMU eval with each of the 6 checkpoints =========="
for i in "${!DISCIPLINES[@]}"; do
  DISC="${DISCIPLINES[$i]}"
  CKPT="${CKPT_LIST[$i]}"
  EVAL_LOG="${EVAL_LOG_DIR}/mmmu_eval_ukmp_${DISC}.log"
  echo "[Step2] Eval checkpoint ($DISC): $CKPT -> $EVAL_LOG"
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMMU_ROOT" \
    --split test \
    --ckpt "$CKPT" \
    --batch_size "$BATCH_SIZE_EVAL" \
    --device cuda \
    2>&1 | tee "$EVAL_LOG"
  echo "[Step2] Eval ($DISC) done. Log: $EVAL_LOG"
done

echo "========== DONE =========="
echo "6 checkpoints: pruned_checkpoint/ukmp_prune/okvqa_MMMU_*-90data-.../pytorch_model.bin"
echo "6 eval logs:   ${EVAL_LOG_DIR}/mmmu_eval_ukmp_Art_Design.log, ... Tech_Engineering.log"
