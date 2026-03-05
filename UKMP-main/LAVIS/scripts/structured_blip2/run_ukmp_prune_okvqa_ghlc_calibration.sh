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

GHLC_JOB_ID="okvqa_GHLC-128data-taylor+knowledge-param_first-param_norm-0.5-blockwise-global-select_loss-multimodal"

echo "[Step1] UKMP prune with GHLC calibration -> $GHLC_JOB_ID"
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.run \
  --nproc_per_node=1 --master_port=18084 \
  ukmp_prune.py \
  --cfg-path lavis/projects/blip2/prune/cc595k_prefix_derivative_compute_okvqa.yaml \
  --device cuda \
  --save_ckpt_log_name ukmp_prune \
  --job_id "$GHLC_JOB_ID" \
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

GHLC_CKPT="pruned_checkpoint/ukmp_prune/${GHLC_JOB_ID}/pytorch_model.bin"
if [ ! -f "$GHLC_CKPT" ]; then
  echo "[ERROR] Pruned checkpoint not found: $GHLC_CKPT"
  exit 1
fi
echo "[Step1] Saved: $GHLC_CKPT"

# -----------------------------------------------------------------------------
# Step2: 用 GHLC 剪枝模型跑 11 类 OK-VQA eval（与之前 CF 跑法相同）
# -----------------------------------------------------------------------------
echo "[Step2] Running 11-category OK-VQA eval with GHLC pruned model"
export PRUNED_CKPT="$GHLC_CKPT"
python scripts/structured_blip2/run_okvqa_eval_by_category.py

echo "[DONE] GHLC calibration prune + 11-category eval finished."
