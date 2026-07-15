#!/usr/bin/env bash
# =============================================================================
# Part 2 -- calibration mechanism exploration, end to end.
#
# Stage A (GPU): extract the raw Wanda statistic for every dataset, once.
# Stage B (CPU): statistic geometry / structure / accuracy-link analysis.
# Stage C (CPU): mask-space analysis from the actual pruned checkpoints.
#
# You provide, per dataset, a (calibration json, images dir). The 4 eval
# benchmarks double as their own reference distributions, so extract stats for
# all of them plus any pretraining set (cc3m).
#
# Required env (edit to your paths), one line per dataset "label|calib_json|images_dir":
#   DATASETS=$'MMBench|/p/mmbench.json|/p/mmbench_images\nOKVQA|/p/okvqa.json|/p/okvqa_images\n...'
# Optional:
#   DENSE_CKPT   dense blip2 pth (default: pretrained)
#   ACC_CSV      accuracy matrix csv (rows=calib, cols=eval) for the link test
#   JOINT_CKPTS  "label=pth,label=pth,..." pruned joint checkpoints for Stage C
#   MAX_SAMPLES  default 128    BATCH_SIZE default 8
#   OUT_ROOT     default $BASE/part2_calibration_mechanism
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

BASE="${BASE:-${AUTODL_TMP:-/data/data2/mfs}}"
OUT_ROOT="${OUT_ROOT:-$BASE/part2_calibration_mechanism}"
STATS_ROOT="$OUT_ROOT/stats"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DENSE_CKPT="${DENSE_CKPT:-}"
ACC_CSV="${ACC_CSV:-}"
JOINT_CKPTS="${JOINT_CKPTS:-}"

: "${DATASETS:?set DATASETS with lines 'label|calib_json|images_dir'}"
mkdir -p "$STATS_ROOT"

echo "########## Stage A: extract Wanda statistics (GPU) ##########"
STATS_ARGS=()
while IFS='|' read -r label cjson cimg; do
  [[ -z "${label// }" ]] && continue
  echo ">>> extract [$label]"
  extra=()
  [[ -n "$DENSE_CKPT" ]] && extra+=(--ckpt "$DENSE_CKPT")
  python scripts/blip2/extract_wanda_statistics.py \
    --label "$label" --calib_json "$cjson" --images_dir "$cimg" \
    --out_dir "$STATS_ROOT/$label" \
    --max_samples "$MAX_SAMPLES" --batch_size "$BATCH_SIZE" "${extra[@]}"
  STATS_ARGS+=(--stats "$label=$STATS_ROOT/$label")
done <<< "$DATASETS"

echo "########## Stage B: statistic analysis (CPU) ##########"
acc_arg=()
[[ -n "$ACC_CSV" ]] && acc_arg=(--accuracy_csv "$ACC_CSV")
for grp in all text visual; do
  python scripts/blip2/analyze_calibration_statistics.py \
    "${STATS_ARGS[@]}" \
    --out_dir "$OUT_ROOT/stat_analysis_group_${grp}" \
    --component both --group "$grp" "${acc_arg[@]}"
done

if [[ -n "$JOINT_CKPTS" ]]; then
  echo "########## Stage C: mask-space analysis from checkpoints (CPU) ##########"
  ckpt_args=()
  IFS=',' read -ra pairs <<< "$JOINT_CKPTS"
  for kv in "${pairs[@]}"; do ckpt_args+=(--ckpt "$kv"); done
  python scripts/blip2/analyze_calibration_mask_mechanism.py \
    "${ckpt_args[@]}" --out_dir "$OUT_ROOT/mask_mechanism" "${acc_arg[@]}"
fi

echo "########## DONE ##########"
echo "outputs under: $OUT_ROOT"
