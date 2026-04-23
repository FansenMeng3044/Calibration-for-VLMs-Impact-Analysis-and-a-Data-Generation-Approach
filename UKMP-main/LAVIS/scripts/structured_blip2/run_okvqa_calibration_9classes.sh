#!/usr/bin/env bash
# =============================================================================
# 用剩余 9 个类轮流做 calibration，每个类剪枝一次得到新 pth，再跑同样的 11 类 eval
#
# 用法:
#   cd /root/autodl-tmp/UKMP-main/LAVIS
#   bash scripts/structured_blip2/run_okvqa_calibration_9classes.sh
#
# 环境: conda 环境 ecoflap，Huggingface 镜像 HF_ENDPOINT
# 外层循环: 9 个 calibration 类 (VT, BCP, OMC, SR, PEL, PA, ST, WC, Other)
# 内层: 每个 calibration 的 pth 跑 11 类 eval（由 run_okvqa_eval_by_category.py 完成）
#
# 输出:
#   剪枝: pruned_checkpoint/ukmp_prune/okvqa_<LABEL>-128data-.../pytorch_model.bin
#   评估: lavis/output/BLIP2/OKVQA/okvqa_eval_calib<LABEL>_<VT|BCP|...>/result/
# =============================================================================

set -e
cd "$(dirname "$0")/../.."
ROOT="$PWD"
if [ "$(basename "$ROOT")" != "LAVIS" ]; then
  echo "[ERROR] Run from UKMP-main/LAVIS. Current: $ROOT"
  exit 1
fi

# -----------------------------------------------------------------------------
# 激活 conda 环境 ecoflap
# -----------------------------------------------------------------------------
if [ -n "$CONDA_PREFIX" ]; then
  echo "[INFO] Conda env already active: $CONDA_PREFIX"
else
  if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ecoflap
    echo "[INFO] Conda env ecoflap activated."
  else
    echo "[WARN] Conda not found or not in PATH; continuing without conda."
  fi
fi

# -----------------------------------------------------------------------------
# Huggingface 镜像（国内）
# -----------------------------------------------------------------------------
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "[INFO] HF_ENDPOINT=$HF_ENDPOINT"

# -----------------------------------------------------------------------------
# 避免 libgomp: Invalid value for environment variable OMP_NUM_THREADS
# -----------------------------------------------------------------------------
export OMP_NUM_THREADS=1

# -----------------------------------------------------------------------------
# 【原 CF / GHLC 校准命令】仅作保留，已注释。需要时取消注释并注释掉下方 9 类循环。
# -----------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.run \
#   --nproc_per_node=1 --master_port=18083 \
#   ukmp_prune.py \
#   --cfg-path lavis/projects/blip2/prune/cc595k_prefix_derivative_compute_okvqa.yaml \
#   ... (CF 1000data)
#
# GHLC_JOB_ID="okvqa_GHLC-128data-taylor+knowledge-..."
# CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.run ... --job_id "$GHLC_JOB_ID" ...
# export PRUNED_CKPT="$GHLC_CKPT"
# python scripts/structured_blip2/run_okvqa_eval_by_category.py

# -----------------------------------------------------------------------------
# 复用已有 flan-t5-xl 缓存（与 run_ukmp_prune_okvqa_ghlc_calibration.sh 一致）
# -----------------------------------------------------------------------------
DEFAULT_HUB="/root/.cache/huggingface/hub"
FLAN_DIR="$DEFAULT_HUB/models--google--flan-t5-xl"
SAFETENSORS_BAK=""
if [ -d "$FLAN_DIR" ] && [ -d "$FLAN_DIR/blobs" ]; then
  SIZE_GB=0
  for f in "$FLAN_DIR/blobs"/*; do [ -f "$f" ] && SIZE_GB=$((SIZE_GB + $(stat -c%s "$f"))); done
  SIZE_GB=$((SIZE_GB / 1024 / 1024 / 1024))
  if [ "$SIZE_GB" -ge 10 ]; then
    export HF_HOME="/root/.cache/huggingface"
    export HUGGINGFACE_HUB_CACHE="$DEFAULT_HUB"
    export TRANSFORMERS_OFFLINE=1
    for rev in "$FLAN_DIR/snapshots"/*; do
      if [ -f "$rev/pytorch_model.bin.index.json" ]; then
        export FLAN_T5_XL_SNAPSHOT="$rev"
        if [ -f "$rev/model.safetensors.index.json" ]; then
          mv "$rev/model.safetensors.index.json" "$rev/model.safetensors.index.json.bak"
          SAFETENSORS_BAK="$rev/model.safetensors.index.json.bak"
        fi
        break
      fi
    done
    if [ -n "$FLAN_T5_XL_SNAPSHOT" ]; then
      echo "[INFO] Reusing existing flan-t5-xl cache: $FLAN_T5_XL_SNAPSHOT"
    fi
  fi
fi
restore_safetensors() {
  if [ -n "$SAFETENSORS_BAK" ] && [ -f "$SAFETENSORS_BAK" ]; then
    mv "$SAFETENSORS_BAK" "${SAFETENSORS_BAK%.bak}" 2>/dev/null || true
  fi
}
trap restore_safetensors EXIT

# -----------------------------------------------------------------------------
# 9 个 calibration 类（除去已用过的 CF、GHLC）: (LABEL, 数据目录名)
# -----------------------------------------------------------------------------
CALIB_DATASETS_ROOT="/root/autodl-tmp/datasets"
CALIB_YAML="lavis/configs/datasets/okvqa/calibration.yaml"

# 外层循环: 9 个类轮流做 calibration
for CALIB_LABEL in VT BCP OMC SR PEL PA ST WC Other; do
  case "$CALIB_LABEL" in
    VT)   CALIB_DIR="Vehicles_and_Transportation" ;;
    BCP)  CALIB_DIR="Brands_Companies_and_Products" ;;
    OMC)  CALIB_DIR="Objects_Material_and_Clothing" ;;
    SR)   CALIB_DIR="Sports_and_Recreation" ;;
    PEL)  CALIB_DIR="People_and_Everyday_life" ;;
    PA)   CALIB_DIR="Plants_and_Animals" ;;
    ST)   CALIB_DIR="Science_and_Technology" ;;
    WC)   CALIB_DIR="Weather_and_Climate" ;;
    Other) CALIB_DIR="Other" ;;
    *)    echo "[ERROR] Unknown calibration label: $CALIB_LABEL"; exit 1 ;;
  esac

  TRAIN_JSON="${CALIB_DATASETS_ROOT}/okvqa_by_category/${CALIB_DIR}/okvqa_train.json"
  if [ ! -f "$TRAIN_JSON" ]; then
    echo "[WARN] Calibration data not found: $TRAIN_JSON , skip $CALIB_LABEL"
    continue
  fi

  echo ""
  echo "========== Calibration class: $CALIB_LABEL ($CALIB_DIR) =========="

  # ----- Step 1: 写入当前类的 calibration.yaml -----
  cat > "$CALIB_YAML" << EOF
# Auto-generated for calibration class: $CALIB_LABEL ($CALIB_DIR)
datasets:
  prefix_okvqa_calibration:
    data_type: images
    build_info:
      annotations:
        train:
          url:
              - ${TRAIN_JSON}
          storage:
              - ${TRAIN_JSON}
      images:
          storage: ${CALIB_DATASETS_ROOT}/okvqa_official/images
EOF
  echo "[Step1] Written $CALIB_YAML -> $CALIB_DIR"

  # ----- Step 2: UKMP 剪枝，得到新 pth -----
  JOB_ID="okvqa_${CALIB_LABEL}-128data-taylor+knowledge-param_first-param_norm-0.5-blockwise-global-select_loss-multimodal"
  echo "[Step2] UKMP prune with $CALIB_LABEL calibration -> $JOB_ID"
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -u -m torch.distributed.run \
    --nproc_per_node=1 --master_port=18085 \
    ukmp_prune.py \
    --cfg-path lavis/projects/blip2/prune/cc595k_prefix_derivative_compute_okvqa.yaml \
    --device cuda \
    --save_ckpt_log_name ukmp_prune \
    --job_id "$JOB_ID" \
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

  CKPT="pruned_checkpoint/ukmp_prune/${JOB_ID}/pytorch_model.bin"
  if [ ! -f "$CKPT" ]; then
    echo "[ERROR] Pruned checkpoint not found: $CKPT"
    exit 1
  fi
  echo "[Step2] Saved: $CKPT"

  # ----- Step 3: 用该 pth 跑 11 类 eval（内层由 run_okvqa_eval_by_category.py 完成）-----
  echo "[Step3] Running 11-category OK-VQA eval with calib=$CALIB_LABEL pruned model"
  export PRUNED_CKPT="$CKPT"
  export EVAL_RUN_PREFIX="calib${CALIB_LABEL}"
  python scripts/structured_blip2/run_okvqa_eval_by_category.py

  echo "[DONE] Calibration $CALIB_LABEL: prune + 11-class eval finished."
done

echo ""
echo "[ALL DONE] All 9 calibration classes: prune + 11-class eval finished."
