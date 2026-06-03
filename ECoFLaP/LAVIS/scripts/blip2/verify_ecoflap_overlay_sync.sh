#!/usr/bin/env bash
# 校验「工作树」与「参考树」在 ECoFLaP 剪枝 overlay 相关文件上是否一致（cmp 字节级）。
#
# 用法:
#   bash scripts/blip2/verify_ecoflap_overlay_sync.sh
#   export REF_LAVIS=/data/data2/mfs/2/ECoFLaP_official/LAVIS
#   export TREE=/data/data2/mfs/2/LAVIS_backup
#   bash scripts/blip2/verify_ecoflap_overlay_sync.sh
#
# 退出码: 0 全部一致；1 有缺失或差异。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 默认：工作树 = 本脚本所在 LAVIS 根；参考树 = ECoFLaP_official（可 export 覆盖）
TREE="${TREE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
REF_LAVIS="${REF_LAVIS:-/data/data2/mfs/2/ECoFLaP_official/LAVIS}"

RELS=(
  evaluate_blip.py
  lavis/compression/__init__.py
  lavis/compression/unimodal_prune.py
  lavis/compression/pruners/sparsegpt_pruner.py
  lavis/compression/pruners/wanda_pruner.py
  lavis/compression/pruners/utils.py
  lavis/models/blip2_models/blip2.py
  lavis/models/blip2_models/blip2_t5.py
  lavis/models/eva_vit.py
  lavis/projects/blip2/eval/cc_prefix_derivative_compute_cc3m_calib128.yaml
  lavis/projects/blip2/eval/t5_c4_text_prune_calib.yaml
  lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml
  lavis/configs/datasets/okvqa/defaults.yaml
  scripts/blip2/run_sparsegpt_unimodal_split_no_granularity.sh
  scripts/blip2/merge_ecoflap_split_prune_ckpts.py
  scripts/blip2/run_ecoflap_split_merge_eval_fourbench.sh
  scripts/blip2/run_sparsegpt_split_merge_eval_fourbench.sh
  scripts/blip2/mmmu_eval_by_discipline.py
  scripts/blip2/mathvista_mc_eval.py
  scripts/blip2/load_blip2_t5_split_ckpts.py
  scripts/blip2/setup_lavisbackup_full_from_official_lavis.sh
  scripts/blip2/overlay_ecoflap_changes_from_old_lavis.sh
  scripts/blip2/verify_ecoflap_overlay_sync.sh
  scripts/run_prune_vit_cc3m_128.sh
  scripts/run_prune_t5_c4_128.sh
)

echo "[INFO] REF_LAVIS=$REF_LAVIS"
echo "[INFO] TREE=    $TREE"
echo ""

bad=0
for r in "${RELS[@]}"; do
  if [[ ! -e "${REF_LAVIS%/}/$r" ]]; then echo "[ERR] 参考树缺文件: $r"; bad=1; continue; fi
  if [[ ! -e "${TREE%/}/$r" ]]; then echo "[ERR] 工作树缺文件: $r"; bad=1; continue; fi
  if ! cmp -s "${REF_LAVIS%/}/$r" "${TREE%/}/$r"; then
    echo "[ERR] 不一致: $r"
    bad=1
  else
    echo "[OK]  $r"
  fi
done

if [[ ! -d "${TREE%/}/lavis/datasets" ]]; then
  echo "[ERR] 工作树缺 lavis/datasets（不是完整 LAVIS）"
  bad=1
else
  echo "[OK]  lavis/datasets/"
fi

if [[ "$bad" -ne 0 ]]; then
  echo ""
  echo "[FAIL] 请先 rsync 或: OLD_LAVIS=... NEW_LAVIS=... bash scripts/blip2/overlay_ecoflap_changes_from_old_lavis.sh" >&2
  exit 1
fi
echo ""
echo "[OK] overlay 相关文件与参考树一致。"
