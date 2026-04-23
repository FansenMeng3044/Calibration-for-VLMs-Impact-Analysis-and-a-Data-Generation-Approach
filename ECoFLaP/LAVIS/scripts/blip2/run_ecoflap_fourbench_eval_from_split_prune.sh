#!/usr/bin/env bash
# =============================================================================
# 使用「分开剪枝」产出的 T5-only + ViT-only 权重 → merge（默认）→ 四基准评测
#
# 默认读取（与 scripts/run_prune_*_c4_128 / run_prune_vit_cc3m_128 的 JOB_ID 一致）:
#   pruned_checkpoint/ecoflap_separate_t5_only.pth
#   pruned_checkpoint/ecoflap_vit_encode_proxy.pth
#
# 实际逻辑在 run_ecoflap_split_merge_eval_fourbench.sh（MMBench / MMMU / OKVQA / MathVista）。
#
# 用法:
#   cd /root/autodl-tmp/ECoFLaP/LAVIS
#   bash scripts/blip2/run_ecoflap_fourbench_eval_from_split_prune.sh
#
# 已有 merged pth、只想评测且跳过 merge:
#   RUN_MERGE=0 bash scripts/blip2/run_ecoflap_fourbench_eval_from_split_prune.sh
#
# 自定义两侧 ckpt:
#   CKPT_T5_ONLY=/path/a.pth CKPT_VIT_ONLY=/path/b.pth bash ...
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CKPT_T5_ONLY="${CKPT_T5_ONLY:-$REPO_ROOT/pruned_checkpoint/ecoflap_separate_t5_only.pth}"
export CKPT_VIT_ONLY="${CKPT_VIT_ONLY:-$REPO_ROOT/pruned_checkpoint/ecoflap_vit_encode_proxy.pth}"

for _f in "$CKPT_T5_ONLY" "$CKPT_VIT_ONLY"; do
  if [[ ! -f "$_f" ]]; then
    echo "[FATAL] 找不到权重: $_f" >&2
    echo "请先跑: bash scripts/run_prune_t5_c4_128.sh  与  bash scripts/run_prune_vit_cc3m_128.sh" >&2
    echo "或设置 CKPT_T5_ONLY / CKPT_VIT_ONLY" >&2
    exit 1
  fi
done

echo "[INFO] T5-only:  $CKPT_T5_ONLY"
echo "[INFO] ViT-only: $CKPT_VIT_ONLY"
echo "[INFO] 调用 run_ecoflap_split_merge_eval_fourbench.sh（RUN_MERGE=${RUN_MERGE:-1} RUN_EVAL=${RUN_EVAL:-1}）"

exec bash "$SCRIPT_DIR/run_ecoflap_split_merge_eval_fourbench.sh"
