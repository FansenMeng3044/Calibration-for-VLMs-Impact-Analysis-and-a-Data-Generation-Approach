#!/usr/bin/env bash
# =============================================================================
# GHLC (Geography_History_Language_and_Culture) 校准剪枝 + 11 类 OK-VQA eval
#
# 用法:
#   cd /root/autodl-tmp/UKMP-main/LAVIS
#   bash scripts/structured_blip2/run_ukmp_prune_okvqa_ghlc_calibration.sh
#
# 前置: calibration.yaml 已指向 GHLC（当前已改好）。若要用回 CF 校准，改
#       lavis/configs/datasets/okvqa/calibration.yaml 注释掉 GHLC 段、恢复 CF 段。
#
# 输出:
#   Step1 剪枝: pruned_checkpoint/ukmp_prune/okvqa_GHLC-128data-.../pytorch_model.bin
#   Step2 评估: lavis/output/BLIP2/OKVQA/okvqa_eval_*/result/
# =============================================================================

set -e
cd "$(dirname "$0")/../.."
ROOT="$PWD"
if [ "$(basename "$ROOT")" != "LAVIS" ]; then
  echo "[ERROR] Run from UKMP-main/LAVIS. Current: $ROOT"
  exit 1
fi

# -----------------------------------------------------------------------------
# 【原 CF 校准命令】仅作保留，已注释。需要时取消注释并注释掉下方 GHLC 段。
# -----------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.run \
#   --nproc_per_node=1 --master_port=18083 \
#   ukmp_prune.py \
#   --cfg-path lavis/projects/blip2/prune/cc595k_prefix_derivative_compute_okvqa.yaml \
#   --device cuda \
#   --save_ckpt_log_name ukmp_prune \
#   --job_id "okvqa_CF-1000data-taylor+knowledge-param_first-param_norm-0.5-blockwise-global-select_loss-multimodal" \
#   --pruning_ratio 0.5 \
#   --granularity block \
#   --pruner_type taylor+knowledge \
#   --taylor param_first \
#   --num_examples 1000 \
#   --channel_per_step 1000 \
#   --global_pruning \
#   --imp_normalizer param \
#   --select_loss \
#   --entropy_importance \
#   --multimodal

# -----------------------------------------------------------------------------
# Step1: GHLC 校准剪枝（新 pth 存到 pruned_checkpoint/ukmp_prune/okvqa_GHLC-.../）
# -----------------------------------------------------------------------------
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# -----------------------------------------------------------------------------
# Calibration source selector
# -----------------------------------------------------------------------------
# 默认按脚本名使用 GHLC calibration（依赖你已提前把 calibration.yaml 指到 GHLC）。
# 当 CALIB_SOURCE=mmbench/mme 时，会临时覆盖 UKMP 的:
#   lavis/configs/datasets/okvqa/calibration.yaml
# 分别指向:
#   /root/autodl-tmp/MMBench_calibration/
#   /root/autodl-tmp/MME_calibration/
CALIB_SOURCE="${CALIB_SOURCE:-ghlc}"
MMBENCH_TRAIN_JSON="/root/autodl-tmp/MMBench_calibration/mmbench_calibration_train.json"
MMBENCH_IMAGES_ROOT="/root/autodl-tmp/MMBench_calibration/images"
MME_TRAIN_JSON="/root/autodl-tmp/MME_calibration/mmE_calibration_train.json"
MME_IMAGES_ROOT="/root/autodl-tmp/MME_calibration/images"

if [ "$CALIB_SOURCE" = "mmbench" ] || [ "$CALIB_SOURCE" = "mme" ]; then
  CALIB_YAML="lavis/configs/datasets/okvqa/calibration.yaml"
  if [ "$CALIB_SOURCE" = "mmbench" ]; then
    echo "[INFO] Overwriting $CALIB_YAML for CALIB_SOURCE=mmbench"
    _bak_suffix="backup_before_mmbench"
    _train_json="$MMBENCH_TRAIN_JSON"
    _images_root="$MMBENCH_IMAGES_ROOT"
  else
    echo "[INFO] Overwriting $CALIB_YAML for CALIB_SOURCE=mme"
    _bak_suffix="backup_before_mme"
    _train_json="$MME_TRAIN_JSON"
    _images_root="$MME_IMAGES_ROOT"
  fi

  if [ -f "$CALIB_YAML" ] && [ ! -f "$CALIB_YAML.$_bak_suffix" ]; then
    cp -a "$CALIB_YAML" "$CALIB_YAML.$_bak_suffix"
  fi

  cat > "$CALIB_YAML" << EOF
datasets:
  prefix_okvqa_calibration:
    data_type: images
    build_info:
      annotations:
        train:
          url:
              - $_train_json
          storage:
              - $_train_json
      images:
          storage: $_images_root
EOF
fi

# 复用已有 flan-t5-xl 缓存，避免重新下载（与 run_okvqa_eval_by_category.py 一致）
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
# 退出时恢复 safetensors index（若被重命名过）
restore_safetensors() {
  if [ -n "$SAFETENSORS_BAK" ] && [ -f "$SAFETENSORS_BAK" ]; then
    mv "$SAFETENSORS_BAK" "${SAFETENSORS_BAK%.bak}" 2>/dev/null || true
  fi
}
trap restore_safetensors EXIT

if [ "$CALIB_SOURCE" = "mmbench" ]; then
  JOB_ID="okvqa_MMBench-128data-taylor+knowledge-param_first-param_norm-0.5-blockwise-global-select_loss-multimodal"
  export EVAL_RUN_PREFIX="calibMMBench"
elif [ "$CALIB_SOURCE" = "mme" ]; then
  JOB_ID="okvqa_MME-128data-taylor+knowledge-param_first-param_norm-0.5-blockwise-global-select_loss-multimodal"
  export EVAL_RUN_PREFIX="calibMME"
else
  JOB_ID="okvqa_GHLC-128data-taylor+knowledge-param_first-param_norm-0.5-blockwise-global-select_loss-multimodal"
fi

echo "[Step1] UKMP prune with CALIB_SOURCE=$CALIB_SOURCE -> $JOB_ID"
# torch.distributed.run rendezvous port（并行多任务时必须不同）
MASTER_PORT="${UKMP_MASTER_PORT:-18088}"

# UKMP prune 是分布式数据并行的：使用多 GPU 时把世界规模也同步到 cfg。
# 例如：GPU1+GPU2 -> nproc_per_node=2，run.world_size=2
CUDA_VISIBLE_DEVICES="${UKMP_CUDA_VISIBLE_DEVICES:-1,2}"
NPROC_PER_NODE="${UKMP_NPROC_PER_NODE:-2}"
WORLD_SIZE="${UKMP_WORLD_SIZE:-2}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python -u -m torch.distributed.run \
  --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
  ukmp_prune.py \
  --cfg-path lavis/projects/blip2/prune/cc595k_prefix_derivative_compute_okvqa.yaml \
  --device cuda \
  --save_ckpt_log_name ukmp_prune \
  --job_id "$JOB_ID" \
  --options run.world_size="$WORLD_SIZE" \
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

PRUNED_CKPT="pruned_checkpoint/ukmp_prune/${JOB_ID}/pytorch_model.bin"
if [ ! -f "$PRUNED_CKPT" ]; then
  echo "[ERROR] Pruned checkpoint not found: $PRUNED_CKPT"
  exit 1
fi
echo "[Step1] Saved: $PRUNED_CKPT"

# -----------------------------------------------------------------------------
# Step2: 用 GHLC 剪枝模型跑 11 类 OK-VQA eval（与之前 CF 跑法相同）
# -----------------------------------------------------------------------------
echo "[Step2] Running 11-category OK-VQA eval with CALIB_SOURCE=$CALIB_SOURCE pruned model"
export PRUNED_CKPT="$PRUNED_CKPT"
python scripts/structured_blip2/run_okvqa_eval_by_category.py

echo "[DONE] Calibration prune ($CALIB_SOURCE) + 11-category eval finished."
