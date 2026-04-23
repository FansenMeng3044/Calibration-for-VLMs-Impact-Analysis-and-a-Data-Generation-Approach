#!/usr/bin/env bash
# =============================================================================
# MathVista 标定 — 四套「单侧剪枝」串行脚本
#
# 1) ECoFLaP   | blipt5_wanda_pruner + MEZO-GradOnly_sum + block | 只剪 ViT (--no_prune_t5)
# 2) ECoFLaP   | 同上                                              | 只剪 T5 (--no_prune_vit)
# 3) LAVIS_backup | blipt5_tamp_pruner (→ wanda+amia+density_sum+layer) | 只剪 ViT
# 4) LAVIS_backup | 同上                                              | 只剪 T5
#
# 与 run_mathvista_prune_eval_suite.sh Phase A 的 spec / batch 默认一致。
#
# 可选环境变量（默认见下）：
#   RUN_SPLIT_1 … RUN_SPLIT_4     每项为 0 则跳过该步（默认全 1）
#   LAVIS_DISTRIBUTED_SAMPLER_SEED / SEED   采样器种子（默认 30）
#   PRUNING_CALIB_BATCH           ECo Wanda 标定 batch（默认 8）
#   T5_PRUNE_SPEC / VIT_PRUNE_SPEC
#   MASTER_PORT_START             四套 master_port 依次为 START..START+3（默认 29821）
#   JOB_STAMP                     拼进 job_id，避免与旧 ckpt 混淆（默认时间戳）
#   AUTODL_TMP / CUDA_VISIBLE_DEVICES
# =============================================================================

set -euo pipefail

AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
ECOFLAP_ROOT="$AUTODL_TMP/ECoFLaP/LAVIS"
LB_ROOT="$AUTODL_TMP/LAVIS_backup"
MATHVISTA_CFG="lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_mathvista.yaml"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$AUTODL_TMP/cache_moved/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export BERT_BASE_UNCASED_SNAPSHOT="${BERT_BASE_UNCASED_SNAPSHOT:-$AUTODL_TMP/cache_moved/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594}"
export FLAN_T5_XL_SNAPSHOT="${FLAN_T5_XL_SNAPSHOT:-$AUTODL_TMP/cache_moved/huggingface/hub/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001}"

SEED="${LAVIS_DISTRIBUTED_SAMPLER_SEED:-${SEED:-30}}"
export LAVIS_DISTRIBUTED_SAMPLER_SEED="$SEED"

PRUNING_CALIB_BATCH="${PRUNING_CALIB_BATCH:-8}"
T5_SPEC="${T5_PRUNE_SPEC:-24-0.5-1.0-1.0}"
VIT_SPEC="${VIT_PRUNE_SPEC:-39-0.5-1.0-1.0}"

MASTER_PORT_START="${MASTER_PORT_START:-29821}"
JOB_STAMP="${JOB_STAMP:-$(date +%Y%m%d_%H%M%S)}"

RUN_SPLIT_1="${RUN_SPLIT_1:-1}"
RUN_SPLIT_2="${RUN_SPLIT_2:-1}"
RUN_SPLIT_3="${RUN_SPLIT_3:-1}"
RUN_SPLIT_4="${RUN_SPLIT_4:-1}"

run_eco_vit_only() {
  echo "========== 1/4 ECoFLaP | Wanda+MEZO | 只剪 ViT (--no_prune_t5) =========="
  (
    cd "$ECOFLAP_ROOT"
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$1" evaluate_blip.py \
      --cfg-path "$MATHVISTA_CFG" \
      --pruning_method blipt5_wanda_pruner \
      --save_pruned_model \
      --score_method MEZO-GradOnly_sum \
      --sparsity_ratio_granularity block \
      --max_sparsity_per_layer 0.6 \
      --prunining_dataset_batch_size "$PRUNING_CALIB_BATCH" \
      --t5_prune_spec "$T5_SPEC" \
      --vit_prune_spec "$VIT_SPEC" \
      --no_prune_t5 \
      --job_id "split_vit_wanda_${JOB_STAMP}_s${SEED}"
  )
}

run_eco_t5_only() {
  echo "========== 2/4 ECoFLaP | Wanda+MEZO | 只剪 T5 (--no_prune_vit) =========="
  (
    cd "$ECOFLAP_ROOT"
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$1" evaluate_blip.py \
      --cfg-path "$MATHVISTA_CFG" \
      --pruning_method blipt5_wanda_pruner \
      --save_pruned_model \
      --score_method MEZO-GradOnly_sum \
      --sparsity_ratio_granularity block \
      --max_sparsity_per_layer 0.6 \
      --prunining_dataset_batch_size "$PRUNING_CALIB_BATCH" \
      --t5_prune_spec "$T5_SPEC" \
      --vit_prune_spec "$VIT_SPEC" \
      --no_prune_vit \
      --job_id "split_t5_wanda_${JOB_STAMP}_s${SEED}"
  )
}

run_lb_vit_only() {
  echo "========== 3/4 LAVIS_backup | TAMP | 只剪 ViT (--no_prune_t5) =========="
  (
    cd "$LB_ROOT"
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$1" evaluate_blip.py \
      --cfg-path "$MATHVISTA_CFG" \
      --pruning_method blipt5_tamp_pruner \
      --save_pruned_model \
      --t5_prune_spec "$T5_SPEC" \
      --vit_prune_spec "$VIT_SPEC" \
      --no_prune_t5 \
      --job_id "split_vit_tamp_${JOB_STAMP}_s${SEED}"
  )
}

run_lb_t5_only() {
  echo "========== 4/4 LAVIS_backup | TAMP | 只剪 T5 (--no_prune_vit) =========="
  (
    cd "$LB_ROOT"
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$1" evaluate_blip.py \
      --cfg-path "$MATHVISTA_CFG" \
      --pruning_method blipt5_tamp_pruner \
      --save_pruned_model \
      --t5_prune_spec "$T5_SPEC" \
      --vit_prune_spec "$VIT_SPEC" \
      --no_prune_vit \
      --job_id "split_t5_tamp_${JOB_STAMP}_s${SEED}"
  )
}

echo "========== MathVista 单侧剪枝 ×4 | SEED=${SEED} | STAMP=${JOB_STAMP} =========="
echo "  ECo:   $ECOFLAP_ROOT/pruned_checkpoint/split_*_${JOB_STAMP}_s${SEED}.pth"
echo "  LB:    $LB_ROOT/pruned_checkpoint/split_*_${JOB_STAMP}_s${SEED}.pth"

P1="$MASTER_PORT_START"
P2=$((MASTER_PORT_START + 1))
P3=$((MASTER_PORT_START + 2))
P4=$((MASTER_PORT_START + 3))

if [[ "$RUN_SPLIT_1" == "1" ]]; then run_eco_vit_only "$P1"; else echo "[SKIP] step 1"; fi
if [[ "$RUN_SPLIT_2" == "1" ]]; then run_eco_t5_only "$P2"; else echo "[SKIP] step 2"; fi
if [[ "$RUN_SPLIT_3" == "1" ]]; then run_lb_vit_only "$P3"; else echo "[SKIP] step 3"; fi
if [[ "$RUN_SPLIT_4" == "1" ]]; then run_lb_t5_only "$P4"; else echo "[SKIP] step 4"; fi

echo "========== 完成 =========="
