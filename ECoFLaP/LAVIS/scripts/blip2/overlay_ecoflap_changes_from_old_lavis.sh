#!/usr/bin/env bash
# 将「旧 LAVIS 工作树」里与 ECoFLaP / 剪枝 / 校准相关的改动，迁到「官方克隆的新 LAVIS」上。
# 用法（先 dry-run）：
#   export OLD_LAVIS=/data/data2/mfs/2/ECoFLaP/LAVIS
#   export NEW_LAVIS=/path/to/ECoFLaP_official/LAVIS
#   bash scripts/blip2/overlay_ecoflap_changes_from_old_lavis.sh --dry-run
# 确认后去掉 --dry-run。
#
# 只同步列出的相对路径，避免用残缺 lavis/ 整树覆盖官方完整源码。
set -euo pipefail

DRY=()
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY=(--dry-run --itemize-changes)
fi

OLD_LAVIS="${OLD_LAVIS:?export OLD_LAVIS=/你的/旧/LAVIS根}"
NEW_LAVIS="${NEW_LAVIS:?export NEW_LAVIS=/你的/新/LAVIS根}"

# 与 verify_ecoflap_overlay_sync.sh 列表保持一致（旧树 → 新树 增量同步用）
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
  scripts/blip2/verify_ecoflap_overlay_sync.sh
  scripts/blip2/setup_official_ecoflap_lavis.sh
  scripts/blip2/overlay_ecoflap_changes_from_old_lavis.sh
  scripts/run_prune_vit_cc3m_128.sh
  scripts/run_prune_t5_c4_128.sh
)

for rel in "${RELS[@]}"; do
  src="${OLD_LAVIS%/}/${rel}"
  dst="${NEW_LAVIS%/}/${rel}"
  if [[ ! -e "$src" ]]; then
    echo "[WARN] 跳过（旧树不存在）: $src" >&2
    continue
  fi
  mkdir -p "$(dirname "$dst")"
  rsync -a "${DRY[@]}" "$src" "$dst"
done

echo ""
echo "[NEXT] cd \"${NEW_LAVIS}\" && pip install -e .   # 仅 py 改动时通常可省略"
echo "[NEXT] 抽查: diff -u \"${NEW_LAVIS}/evaluate_blip.py\" \"${OLD_LAVIS}/evaluate_blip.py\" | head -80"
