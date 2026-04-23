#!/usr/bin/env bash
# =============================================================================
# 合并跑 LAVIS_backup（TAMP）与 ECoFLaP（Wanda）的 dated 三校准剪枝+评测脚本，
# 并对 DistributedSampler 的 seed 各跑一遍（默认 42 与 30）。
#
# 依赖：两处仓库的 lavis/runners/runner_base.py 已支持
#   环境变量 LAVIS_DISTRIBUTED_SAMPLER_SEED → torch.utils.data.DistributedSampler(seed=...)
#
# 为避免同一 DATE_TAG 下第二次覆盖 pth，每次循环会设置：
#   DATE_TAG="${BASE_DATE_TAG}_s<seed>"   例如 0324_s42、0324_s30
#
# 环境变量（可选）:
#   BASE_DATE_TAG=0324     # 默认 $(date +%m%d)；最终 DATE_TAG=0324_s42 等
#   SAMPLER_SEEDS="42 30"  # 空格分隔
#   RUN_PRUNE / RUN_EVAL / CUDA_VISIBLE_DEVICES 等会传给子脚本
#   默认 CUDA_VISIBLE_DEVICES=0（单卡）；只有一张卡却设为 2 时 PyTorch 会报 No CUDA GPUs
#
# 用法:
#   cd /root/autodl-tmp
#   bash run_prune_eval_lavisbackup_and_ecoflap_sampler_seeds.sh
# =============================================================================

set -euo pipefail

# 单卡容器常见只有 device 0；LAVIS_backup 子脚本曾默认 2 会暴露 0 张卡给 PyTorch
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAVIS_BACKUP="$WORKSPACE/LAVIS_backup"
ECOFLAP_LAVIS="$WORKSPACE/ECoFLaP/LAVIS"

SAMPLER_SEEDS="${SAMPLER_SEEDS:-42 30}"
BASE_DATE_TAG="${BASE_DATE_TAG:-${DATE_TAG:-$(date +%m%d)}}"

if [[ ! -d "$LAVIS_BACKUP" ]] || [[ ! -d "$ECOFLAP_LAVIS" ]]; then
  echo "[FATAL] 需要目录: $LAVIS_BACKUP 与 $ECOFLAP_LAVIS"
  exit 1
fi

for SEED in $SAMPLER_SEEDS; do
  export LAVIS_DISTRIBUTED_SAMPLER_SEED="$SEED"
  export DATE_TAG="${BASE_DATE_TAG}_s${SEED}"
  echo ""
  echo "#####################################################################"
  echo "# LAVIS_DISTRIBUTED_SAMPLER_SEED=$SEED  DATE_TAG=$DATE_TAG"
  echo "#####################################################################"

  echo ""
  echo ">>> [LAVIS_backup / TAMP] ..."
  (cd "$LAVIS_BACKUP" && bash scripts/blip2/run_lavisbackup_prune_eval_okvqa_mmbench_mmu_dated.sh)

  echo ""
  echo ">>> [ECoFLaP / Wanda] ..."
  (cd "$ECOFLAP_LAVIS" && bash scripts/blip2/run_ecoflap_prune_eval_okvqa_mmbench_mmu_dated.sh)
done

echo ""
echo "========== 全部完成（BASE_DATE_TAG=$BASE_DATE_TAG, seeds=$SAMPLER_SEEDS）=========="
