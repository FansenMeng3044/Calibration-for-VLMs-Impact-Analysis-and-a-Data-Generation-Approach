#!/usr/bin/env bash
# =============================================================================
# 用「纯 Wanda / 联合剪枝」产出的单份 pruned_checkpoint/*.pth 跑四基准：
#   MMBench / MMMU / OKVQA overall / MathVista MC
#
# 依赖：与 run_ecoflap_split_merge_eval_fourbench.sh 相同（数据路径、HF 本地缓存等）
#
# 用法（在 ECoFLaP/LAVIS 根目录）:
#   export JOINT_SINGLE_CKPT=/root/autodl-tmp/ECoFLaP/LAVIS/pruned_checkpoint/pure_wanda_joint_cc3m128.pth
#   bash scripts/blip2/run_wanda_pruned_fourbench_eval.sh
#
# 或只给 job_id（默认找 pruned_checkpoint/<JOB_ID>.pth）:
#   JOB_ID=pure_wanda_joint_cc3m128 bash scripts/blip2/run_wanda_pruned_fourbench_eval.sh
#
# 覆盖数据根目录（可选）:
#   MMBENCH_ROOT=... MMMU_ROOT=... MATHVISTA_EVAL_JSON=... JOB_ID=... bash ...
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -n "${JOINT_SINGLE_CKPT:-}" ]]; then
  :
elif [[ -n "${CKPT:-}" ]]; then
  JOINT_SINGLE_CKPT="$CKPT"
elif [[ -n "${JOB_ID:-}" ]]; then
  JOINT_SINGLE_CKPT="$REPO_ROOT/pruned_checkpoint/${JOB_ID}.pth"
else
  echo "[FATAL] 请设置以下之一: JOINT_SINGLE_CKPT=/path/to.pth 或 CKPT=... 或 JOB_ID=my_job" >&2
  exit 1
fi

if [[ ! -f "$JOINT_SINGLE_CKPT" ]]; then
  echo "[FATAL] 找不到权重: $JOINT_SINGLE_CKPT" >&2
  exit 1
fi

export JOINT_SINGLE_CKPT
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache_moved/huggingface}"

mkdir -p "$REPO_ROOT/training_statistics"
export LAVIS_METRICS_JSONL="${LAVIS_METRICS_JSONL:-$REPO_ROOT/training_statistics/pure_wanda_fourbench_metrics.jsonl}"

echo "[INFO] 四基准评测，单权重: $JOINT_SINGLE_CKPT"
echo "[INFO] LAVIS_METRICS_JSONL=$LAVIS_METRICS_JSONL"
exec bash "$SCRIPT_DIR/run_ecoflap_split_merge_eval_fourbench.sh"
