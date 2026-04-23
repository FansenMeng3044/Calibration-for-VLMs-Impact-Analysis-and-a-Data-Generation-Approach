#!/bin/bash
# 在 LAVIS_backup 根目录下执行此脚本，创建 conda 环境 lavis_backup 并安装依赖
# 用法: bash scripts/conda_env_create.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

echo "=== 创建 conda 环境 lavis_backup (Python 3.8) ==="
conda env create -f environment.yml --force 2>/dev/null || conda env create -f environment.yml

echo ""
echo "=== 激活环境并安装 LAVIS_backup 依赖 (pip install -e .) ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lavis_backup
pip install -e .

echo ""
echo "=== 安装完成。使用前请执行: conda activate lavis_backup ==="
