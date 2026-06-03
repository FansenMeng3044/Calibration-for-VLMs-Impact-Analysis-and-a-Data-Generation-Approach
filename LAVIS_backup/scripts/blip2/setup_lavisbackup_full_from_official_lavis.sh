#!/usr/bin/env bash
# =============================================================================
# 用「完整官方 LAVIS + 已与 ECoFLaP_official 对齐的 overlay」重建 LAVIS_backup
#
# 做法：以 SOURCE_LAVIS（默认 ECoFLaP_official/LAVIS）为源做 rsync，保证含 lavis/datasets/ 等完整树；
#       旧残缺 LAVIS_backup 先改名为 LAVIS_backup_partial_<时间戳>，再把其中 run_lavisbackup* 等脚本并回新树。
#
# 默认排除 pruned_checkpoint/（常几十 GB）；需要一并拷贝时: export SYNC_PRUNED_CHECKPOINT=1
#
# 用法:
#   bash scripts/blip2/setup_lavisbackup_full_from_official_lavis.sh
#
# 自定义:
#   export LAVIS_BACKUP=/data/data2/mfs/2/LAVIS_backup
#   export SOURCE_LAVIS=/data/data2/mfs/2/ECoFLaP_official/LAVIS
#   export NO_MOVE_OLD=1          # 不移动旧目录（目标需不存在或空）
#   export SYNC_PRUNED_CHECKPOINT=1
#   bash scripts/blip2/setup_lavisbackup_full_from_official_lavis.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

LAVIS_BACKUP="${LAVIS_BACKUP:-/data/data2/mfs/2/LAVIS_backup}"
SOURCE_LAVIS="${SOURCE_LAVIS:-$REPO_ROOT}"
TS="$(date +%Y%m%d_%H%M%S)"
PARTIAL_ARCHIVE="${LAVIS_BACKUP}_partial_${TS}"

if [[ ! -d "${SOURCE_LAVIS}/lavis/datasets" ]]; then
  echo "[FATAL] SOURCE_LAVIS 不是完整 LAVIS（缺少 lavis/datasets）: ${SOURCE_LAVIS}" >&2
  exit 1
fi

RSYNC_EXCLUDES=(
  --exclude=.git/
  --exclude=__pycache__/
  --exclude="*.py[cod]"
  --exclude=.pytest_cache/
  --exclude=output/
)
if [[ "${SYNC_PRUNED_CHECKPOINT:-0}" != "1" ]]; then
  RSYNC_EXCLUDES+=(--exclude=pruned_checkpoint/)
fi

if [[ -d "${LAVIS_BACKUP}" ]] && [[ "${NO_MOVE_OLD:-0}" != "1" ]]; then
  if [[ -f "${LAVIS_BACKUP}/setup.py" ]] || [[ -f "${LAVIS_BACKUP}/pyproject.toml" ]]; then
    echo "[INFO] 移动旧树 → ${PARTIAL_ARCHIVE}"
    mv "${LAVIS_BACKUP}" "${PARTIAL_ARCHIVE}"
  else
    echo "[WARN] ${LAVIS_BACKUP} 存在但不是 LAVIS 根（无 setup.py），仍改名为 partial" >&2
    mv "${LAVIS_BACKUP}" "${PARTIAL_ARCHIVE}"
  fi
elif [[ -d "${LAVIS_BACKUP}" ]]; then
  echo "[FATAL] ${LAVIS_BACKUP} 已存在且 NO_MOVE_OLD=1，请先清空或改名。" >&2
  exit 1
fi

mkdir -p "${LAVIS_BACKUP}"
echo "[INFO] rsync 完整树: ${SOURCE_LAVIS}/ → ${LAVIS_BACKUP}/"
rsync -a --delete "${RSYNC_EXCLUDES[@]}" "${SOURCE_LAVIS}/" "${LAVIS_BACKUP}/"

# 旧树里的 lavisbackup 专用脚本 / 统计（若刚做了 partial 归档）
if [[ -d "${PARTIAL_ARCHIVE}/scripts/blip2" ]]; then
  echo "[INFO] 合并旧 scripts/blip2/run_lavisbackup* …"
  shopt -s nullglob
  for f in "${PARTIAL_ARCHIVE}/scripts/blip2"/run_lavisbackup*.sh \
           "${PARTIAL_ARCHIVE}/scripts/blip2"/collect_lavisbackup*.py; do
    [[ -f "$f" ]] || continue
    install -D -m 0644 "$f" "${LAVIS_BACKUP}/scripts/blip2/$(basename "$f")"
  done
  shopt -u nullglob
fi
if [[ -d "${PARTIAL_ARCHIVE}/training_statistics" ]]; then
  rsync -a "${PARTIAL_ARCHIVE}/training_statistics/" "${LAVIS_BACKUP}/training_statistics/" 2>/dev/null || true
fi
if [[ -f "${PARTIAL_ARCHIVE}/mmmu_results_lavisbackup.csv" ]]; then
  cp -a "${PARTIAL_ARCHIVE}/mmmu_results_lavisbackup.csv" "${LAVIS_BACKUP}/" 2>/dev/null || true
fi

if [[ ! -d "${LAVIS_BACKUP}/lavis/datasets" ]]; then
  echo "[FATAL] 同步后仍缺 lavis/datasets" >&2
  exit 1
fi

echo ""
echo "[OK] 新 LAVIS_backup: ${LAVIS_BACKUP}"
echo "     旧树备份:      ${PARTIAL_ARCHIVE}（可核对后自行删除）"
if [[ "${SYNC_PRUNED_CHECKPOINT:-0}" != "1" ]] && [[ -d "${PARTIAL_ARCHIVE}/pruned_checkpoint" ]]; then
  echo "     pruned_checkpoint 未同步；需要可: export SYNC_PRUNED_CHECKPOINT=1 重跑，或手动:"
  echo "       rsync -a \"${PARTIAL_ARCHIVE}/pruned_checkpoint/\" \"${LAVIS_BACKUP}/pruned_checkpoint/\""
fi
echo ""
echo "下一步（在 conda ecoflap 中）:"
echo "  cd \"${LAVIS_BACKUP}\" && pip install -e ."
echo "  source /data/data2/mfs/model_cache/ecoflap_model_env.sh   # 或你本机 HF 环境"
echo "  export PYTHONPATH=\"\$(pwd):\${PYTHONPATH:-}\""
