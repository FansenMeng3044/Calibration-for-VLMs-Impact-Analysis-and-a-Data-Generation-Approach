#!/usr/bin/env bash
# =============================================================================
# SparseGPT「单侧分开剪枝」→ 合并 → MMBench / MMMU / OKVQA / MathVista 四评测
#
# 与 scripts/blip2/run_sparsegpt_unimodal_split_no_granularity.sh 的默认 JOB_ID 对齐:
#   JOB_VIT=sparsegpt_vit_image_only_nogran  → pruned_checkpoint/<JOB_VIT>.pth
#   JOB_T5=sparsegpt_t5_c4_nogran           → pruned_checkpoint/<JOB_T5>.pth
#
# 用法（在 ECoFLaP_official/LAVIS 根）:
#   bash scripts/blip2/run_sparsegpt_split_merge_eval_fourbench.sh
#
# 自定义两侧权重:
#   CKPT_VIT_ONLY=/path/vit.pth CKPT_T5_ONLY=/path/t5.pth bash ...
#
# 已有合并文件、只跑评测:
#   RUN_MERGE=0 MERGED_CKPT=/path/merged.pth bash ...
#
# 数据路径（与 run_ecoflap_split_merge_eval_fourbench.sh 相同）:
#   MMBENCH_ROOT / MMMU_ROOT / MATHVISTA_EVAL_JSON
#   或一键: export ECOFLAP_BENCH_ROOT=/data/data2/mfs（其下 MMBench_eval、MMMU_single_image、MathVista_eval_testmini_mc/）
# OKVQA 标注路径写在 lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml
#   或用 OKVQA_CFG=... 指向你自己的 yaml。
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

JOB_VIT="${JOB_VIT:-sparsegpt_vit_image_only_nogran}"
JOB_T5="${JOB_T5:-sparsegpt_t5_c4_nogran}"

export CKPT_VIT_ONLY="${CKPT_VIT_ONLY:-$REPO_ROOT/pruned_checkpoint/${JOB_VIT}.pth}"
export CKPT_T5_ONLY="${CKPT_T5_ONLY:-$REPO_ROOT/pruned_checkpoint/${JOB_T5}.pth}"
export MERGED_CKPT="${MERGED_CKPT:-$REPO_ROOT/pruned_checkpoint/merged_sparsegpt_${JOB_VIT}__${JOB_T5}.pth}"

for _f in "$CKPT_VIT_ONLY" "$CKPT_T5_ONLY"; do
  if [[ ! -f "$_f" ]]; then
    echo "[FATAL] 找不到权重: $_f" >&2
    echo "请先跑:" >&2
    echo "  bash scripts/blip2/run_sparsegpt_unimodal_split_no_granularity.sh vit" >&2
    echo "  bash scripts/blip2/run_sparsegpt_unimodal_split_no_granularity.sh t5" >&2
    echo "或设置 CKPT_VIT_ONLY / CKPT_T5_ONLY / JOB_VIT / JOB_T5" >&2
    exit 1
  fi
done

echo "[INFO] ViT ckpt: $CKPT_VIT_ONLY"
echo "[INFO] T5  ckpt: $CKPT_T5_ONLY"
echo "[INFO] Merge →: $MERGED_CKPT (RUN_MERGE=${RUN_MERGE:-1})"

exec bash "$SCRIPT_DIR/run_ecoflap_split_merge_eval_fourbench.sh"
