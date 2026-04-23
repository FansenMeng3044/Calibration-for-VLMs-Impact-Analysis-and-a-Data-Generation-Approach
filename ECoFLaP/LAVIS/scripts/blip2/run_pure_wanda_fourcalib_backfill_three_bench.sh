#!/usr/bin/env bash
# =============================================================================
# 已有四份「纯 Wanda × 四 calibration」剪枝权重时，只补跑其中三项评测（跳过 OKVQA）。
#
# 默认权重:
#   pruned_checkpoint/pure_wanda_calib_{mmbench,mmmu,okvqa,mathvista}.pth
#
# 执行顺序（默认）：先对四个权重各跑一遍 **MathVista**，再四遍 **MMMU**，再四遍 **MMBench**。
# 可改顺序: EVAL_BENCH_ORDER=mmbench,mmmu,mathvista
#
# 会设置 LAVIS_METRICS_JSONL，指标追加到 jsonl。
#
# 用法:
#   bash scripts/blip2/run_pure_wanda_fourcalib_backfill_three_bench.sh
#
# 只补部分 calibration（仍按 EVAL_BENCH_ORDER 分三轮）:
#   CALIBS="mmbench okvqa" bash scripts/blip2/run_pure_wanda_fourcalib_backfill_three_bench.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$REPO_ROOT" || exit 1

JOB_PREFIX="${JOB_PREFIX:-pure_wanda_calib}"
CALIBS="${CALIBS:-mmbench mmmu okvqa mathvista}"
# 逗号分隔：先跑完该基准下全部 calibration，再下一基准
EVAL_BENCH_ORDER="${EVAL_BENCH_ORDER:-mathvista,mmmu,mmbench}"

mkdir -p "$REPO_ROOT/training_statistics"
export LAVIS_METRICS_JSONL="${LAVIS_METRICS_JSONL:-$REPO_ROOT/training_statistics/pure_wanda_fourbench_metrics.jsonl}"
echo "[INFO] LAVIS_METRICS_JSONL=$LAVIS_METRICS_JSONL"
echo "[INFO] EVAL_BENCH_ORDER=$EVAL_BENCH_ORDER"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache_moved/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MASTER_PORT_EVAL_BASE="${MASTER_PORT_EVAL_BASE:-29600}"

IFS_OLD=$IFS
IFS=','
read -ra BENCH_ROUNDS <<< "$EVAL_BENCH_ORDER"
IFS=$IFS_OLD

idx=0
for bench in "${BENCH_ROUNDS[@]}"; do
  bench="${bench//[[:space:]]/}"
  [[ -z "$bench" ]] && continue

  echo ""
  echo "############################ 基准轮次: $bench ############################"

  for calib in $CALIBS; do
    STEM="${JOB_PREFIX}_${calib}"
    CKPT="$REPO_ROOT/pruned_checkpoint/${STEM}.pth"
    if [[ ! -f "$CKPT" ]]; then
      echo "[FATAL] 找不到权重: $CKPT" >&2
      exit 1
    fi

    export JOINT_SINGLE_CKPT="$CKPT"
    export MASTER_PORT=$((MASTER_PORT_EVAL_BASE + idx * 10))
    export EVAL_JOB_OKVQA="${EVAL_JOB_OKVQA:-okvqa_eval_${STEM}}"

    export SKIP_OKVQA=1
    export SKIP_MMBENCH=1
    export SKIP_MMMU=1
    export SKIP_MATHVISTA=1

    case "$bench" in
      mathvista) export SKIP_MATHVISTA=0 ;;
      mmmu) export SKIP_MMMU=0 ;;
      mmbench) export SKIP_MMBENCH=0 ;;
      *)
        echo "[FATAL] EVAL_BENCH_ORDER 中未知项: $bench（允许: mathvista, mmmu, mmbench）" >&2
        exit 1
        ;;
    esac

    echo ""
    echo "-------- $bench | calib=$calib | $CKPT | MASTER_PORT=$MASTER_PORT --------"

    bash "$SCRIPT_DIR/run_ecoflap_split_merge_eval_fourbench.sh"

    idx=$((idx + 1))
  done
done

echo ""
echo "[INFO] 补跑结束。可执行: python scripts/blip2/collect_pure_wanda_fourbench_summary.py"
