#!/usr/bin/env bash
# =============================================================================
# 四基准 eval 冒烟测试（路径 / HF / 各入口能否跑通）
#
# - MMBench / MMMU：mmmu_eval_by_discipline.py + --max_samples（可走预训练，可不传 ckpt）
# - MathVista：mathvista_mc_eval.py 必须提供 --ckpt（脚本限制）
# - OKVQA：evaluate_blip 全量很慢；这里用 smoke_load_blip2_full_ckpt.py 验证同一条 ckpt 能否 load_model
#
# 用法（ECoFLaP/LAVIS 根目录）:
#   # 只测数据路径 + 预训练 MMBench/MMMU 各 4 条（不传 ckpt）
#   bash scripts/blip2/test_four_eval_smoke.sh
#
#   # 含剪枝权重：MathVista + OKVTA smoke 会用到
#   CKPT=/path/to/pruned.pth bash scripts/blip2/test_four_eval_smoke.sh
#
# 环境变量:
#   MAX_SAMPLES=4          MMBench/MMMU/MathVista 子集大小
#   MMBENCH_ROOT MMMU_ROOT MATHVISTA_EVAL_JSON
#   HF_HOME                与正式 eval 一致
#   SKIP_MMBENCH=1 等      跳过某一环
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MAX_SAMPLES="${MAX_SAMPLES:-4}"
CKPT="${CKPT:-}"

MMBENCH_ROOT="${MMBENCH_ROOT:-/root/autodl-tmp/MMBench_eval}"
MMMU_ROOT="${MMMU_ROOT:-/root/autodl-tmp/MMMU_single_image}"
MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-/root/autodl-tmp/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache_moved/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "========== [0] 路径检查 =========="
fail=0
check_dir() {
  if [[ ! -d "$1" ]]; then echo "[FAIL] 目录不存在: $1"; fail=1; else echo "[OK]   $1"; fi
}
check_file() {
  if [[ ! -f "$1" ]]; then echo "[FAIL] 文件不存在: $1"; fail=1; else echo "[OK]   $1"; fi
}

check_dir "$MMBENCH_ROOT"
check_dir "$MMMU_ROOT"
check_file "$MATHVISTA_EVAL_JSON"
if [[ -n "$CKPT" ]]; then
  check_file "$CKPT"
fi
if [[ "$fail" != "0" ]]; then
  echo "[FATAL] 请先准备数据或设置 MMBENCH_ROOT/MMMU_ROOT/MATHVISTA_EVAL_JSON/CKPT" >&2
  exit 1
fi

MM_ARGS=()
if [[ -n "$CKPT" ]]; then
  MM_ARGS+=(--ckpt "$CKPT")
fi

run_mmbench() {
  echo ""
  echo "========== [1] MMBench smoke (split=dev, max_samples=$MAX_SAMPLES) =========="
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMBENCH_ROOT" \
    --split dev \
    "${MM_ARGS[@]}" \
    --batch_size 2 \
    --device cuda \
    --max_samples "$MAX_SAMPLES" \
    --overall_only
}

run_mmmu() {
  echo ""
  echo "========== [2] MMMU smoke (split=test, max_samples=$MAX_SAMPLES) =========="
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMMU_ROOT" \
    --split test \
    "${MM_ARGS[@]}" \
    --batch_size 2 \
    --device cuda \
    --max_samples "$MAX_SAMPLES" \
    --overall_only
}

run_mathvista() {
  echo ""
  echo "========== [3] MathVista MC smoke (max_samples=$MAX_SAMPLES) =========="
  if [[ -z "$CKPT" ]]; then
    echo "[SKIP] MathVista 必须提供 CKPT=...（mathvista_mc_eval 限制），跳过。"
    return 0
  fi
  python scripts/blip2/mathvista_mc_eval.py \
    --eval_json "$MATHVISTA_EVAL_JSON" \
    --ckpt "$CKPT" \
    --batch_size 2 \
    --device cuda \
    --max_samples "$MAX_SAMPLES"
}

run_okvqa_smoke() {
  echo ""
  echo "========== [4] OKVQA 等价加载 smoke（load_model + 可选 forward）=========="
  if [[ -z "$CKPT" ]]; then
    echo "[SKIP] 未设置 CKPT，跳过（仅测 load 时请设置 CKPT=...）。"
    return 0
  fi
  python scripts/blip2/smoke_load_blip2_full_ckpt.py --ckpt "$CKPT" --forward
}

if [[ "${SKIP_MMBENCH:-0}" != "1" ]]; then run_mmbench; else echo "[SKIP] MMBench"; fi
if [[ "${SKIP_MMMU:-0}" != "1" ]]; then run_mmmu; else echo "[SKIP] MMMU"; fi
if [[ "${SKIP_MATHVISTA:-0}" != "1" ]]; then run_mathvista; else echo "[SKIP] MathVista"; fi
if [[ "${SKIP_OKVQA_SMOKE:-0}" != "1" ]]; then run_okvqa_smoke; else echo "[SKIP] OKVQA smoke"; fi

echo ""
echo "========== 冒烟测试流程结束 =========="
echo "说明: OKVQA 正式评测仍用 evaluate_blip + okvqa_zeroshot_flant5xl_eval_overall.yaml，"
echo "      全量较慢；此处 [4] 只验证同一份 ckpt 能否被 load_model 并前向一步。"
