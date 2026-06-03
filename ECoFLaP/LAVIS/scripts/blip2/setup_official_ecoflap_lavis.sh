#!/usr/bin/env bash
# =============================================================================
# 官方 / 标准做法：完整 ECoFLaP 仓库（内含完整 LAVIS 子树）→ 锁 commit → 可编辑安装
#
# 论文与 README 指向：https://github.com/ylsung/ECoFLaP
# 其中 BLIP-2 实验说明：cd LAVIS && pip install -e .
#
# 用法示例（装到独立目录，避免覆盖你当前残缺拷贝）：
#   export INSTALL_ROOT=/data/data2/mfs/2
#   export ECOFLAP_DIRNAME=ECoFLaP_official
#   bash scripts/blip2/setup_official_ecoflap_lavis.sh
#
# 可选：锁到指定 commit（默认：2024-02-16 作者在 main 上的 tip，便于复现）
#   export ECOFLAP_GIT_REF=59dac0a894d36aeb050bcb64ad4f1fb411407959
#
# 装完后在本机再配：HF_HOME、BLIP2 pth、CC3M/C4 校准路径（见 run_sparsegpt_* 脚本）。
# =============================================================================
set -euo pipefail

ECOFLAP_URL="${ECOFLAP_URL:-https://github.com/ylsung/ECoFLaP.git}"
ECOFLAP_GIT_REF="${ECOFLAP_GIT_REF:-59dac0a894d36aeb050bcb64ad4f1fb411407959}"
INSTALL_ROOT="${INSTALL_ROOT:-${HOME}/src}"
ECOFLAP_DIRNAME="${ECOFLAP_DIRNAME:-ECoFLaP}"

DEST="${INSTALL_ROOT}/${ECOFLAP_DIRNAME}"

mkdir -p "${INSTALL_ROOT}"

if [[ -d "${DEST}/.git" ]]; then
  echo "[INFO] 已存在 git 仓库: ${DEST}，执行 fetch + checkout ${ECOFLAP_GIT_REF}"
  git -C "${DEST}" fetch origin
  git -C "${DEST}" checkout "${ECOFLAP_GIT_REF}"
else
  echo "[INFO] 完整克隆 ${ECOFLAP_URL} → ${DEST}（便于 checkout 任意 commit）"
  rm -rf "${DEST}"
  git clone "${ECOFLAP_URL}" "${DEST}"
  git -C "${DEST}" checkout "${ECOFLAP_GIT_REF}"
fi

LAVIS_ROOT="${DEST}/LAVIS"
if [[ ! -f "${LAVIS_ROOT}/setup.py" ]] && [[ ! -f "${LAVIS_ROOT}/pyproject.toml" ]]; then
  echo "[FATAL] 未找到 ${LAVIS_ROOT}/setup.py 或 pyproject.toml，仓库结构异常。" >&2
  exit 1
fi

echo "[INFO] 可编辑安装 LAVIS（需在 conda/venv 中已激活目标环境，如 ecoflap）"
echo "       cd \"${LAVIS_ROOT}\" && pip install -e ."
echo ""
echo "若你使用 NumPy 2.x，官方 requirements 里的 opencv==4.5.5.64 可能不兼容，二选一："
echo "  A) pip install \"numpy>=1.23,<2\" --force-reinstall"
echo "  B) pip install -U \"opencv-python-headless>=4.10\"（可能与 salesforce-lavis 元数据冲突，可 pip uninstall salesforce-lavis）"
echo ""
echo "下一步:"
echo "  cd \"${LAVIS_ROOT}\""
echo "  export PYTHONPATH=\"\$(pwd):\${PYTHONPATH:-}\""
echo "  # 再跑 evaluate_blip / 作者 scripts/blip2/*.py 或你自己的 SparseGPT 脚本（若尚未合入官方仓库，用 git cherry-pick / patch 叠到该树上）"
