#!/usr/bin/env bash
# =============================================================================
# Four-eval T5 layer-wise hidden-state fidelity suite.
#
# Purpose:
#   For each evaluation dataset, run the dense BLIP2-T5 model and every
#   calibration-pruned checkpoint on the exact same eval rows.  Extract pooled
#   T5 encoder hidden states at every layer, then plot one fidelity curve per
#   calibration checkpoint.
#
# Required:
#   PRUNED_CKPTS="MMBench=/path/mmbench.pth,MMMU=/path/mmmu.pth,OKVQA=/path/okvqa.pth,MathVista=/path/mathvista.pth,CC3M=/path/cc3m.pth"
#
#   EVAL_SPECS=$'MMBench|/path/mmbench.json_or.parquet|/path/mmbench_images|question|multimodal|512
#   MMMU|/path/mmmu.json_or.parquet|/path/mmmu_images|question|multimodal|512
#   OKVQA|/path/okvqa.json|/path/okvqa_images|question|multimodal|512
#   MathVista|/path/mathvista.json|/path/mathvista_images|question|multimodal|512'
#
# Optional:
#   DENSE_CKPT=/path/blip2_pretrained_flant5xl.pth
#   OUT_ROOT=/path/out
#   T5_LAYER_PARTS="both visual text"  # default: both
#   MAX_SAMPLES=512
#   BATCH_SIZE=4
#   SAVE_DTYPE=float32                # float32 or float16
#   FORCE=1                           # re-extract even if npz already exists
#   NO_PLOTS=1                        # write CSV only
#   FP32=1                            # run extraction in fp32
#
# Main plots:
#   $OUT_ROOT/MMBench/t5_layer_both/t5_layer_fidelity_both.png
#   $OUT_ROOT/MMMU/t5_layer_both/t5_layer_fidelity_both.png
#   $OUT_ROOT/OKVQA/t5_layer_both/t5_layer_fidelity_both.png
#   $OUT_ROOT/MathVista/t5_layer_both/t5_layer_fidelity_both.png
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DENSE_CKPT="${DENSE_CKPT:-/data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
OUT_ROOT="${OUT_ROOT:-/data/data2/mfs/t5_layer_fidelity_fourbench}"
MAX_SAMPLES="${MAX_SAMPLES:-512}"
BATCH_SIZE="${BATCH_SIZE:-4}"
T5_LAYER_PARTS="${T5_LAYER_PARTS:-both}"
SAVE_DTYPE="${SAVE_DTYPE:-float32}"
FORCE="${FORCE:-0}"
NO_PLOTS="${NO_PLOTS:-0}"
FP32="${FP32:-0}"

: "${PRUNED_CKPTS:?set PRUNED_CKPTS as comma-separated label=/path/checkpoint.pth}"
: "${EVAL_SPECS:?set EVAL_SPECS with lines 'label|annotation|images_dir|text_field|input_mode|max_samples'}"

mkdir -p "$OUT_ROOT"

extract_one() {
  local label="$1"
  local ckpt="$2"
  local eval_json="$3"
  local images_dir="$4"
  local text_field="$5"
  local input_mode="$6"
  local max_samples="$7"
  local out_dir="$8"

  if [[ "$FORCE" != "1" && -f "$out_dir/t5_layer_hidden_states.npz" ]]; then
    echo "[SKIP] existing T5 layer states: $out_dir"
    return 0
  fi

  local args=(
    python scripts/blip2/extract_t5_layer_hidden_states.py
    --label "$label"
    --input_mode "$input_mode"
    --calib_json "$eval_json"
    --text_field "$text_field"
    --ckpt "$ckpt"
    --out_dir "$out_dir"
    --max_samples "$max_samples"
    --batch_size "$BATCH_SIZE"
    --parts "$T5_LAYER_PARTS"
    --save_dtype "$SAVE_DTYPE"
  )
  if [[ "$input_mode" != "text_only" ]]; then
    args+=(--images_dir "$images_dir")
  fi
  if [[ "$FP32" == "1" ]]; then
    args+=(--fp32)
  fi
  echo ">>> [extract layers] $label -> $out_dir"
  "${args[@]}"
}

analyze_eval() {
  local eval_label="$1"
  local dense_dir="$2"
  local pruned_root="$3"
  local out_eval="$4"

  IFS=',' read -ra ckpt_pairs <<< "$PRUNED_CKPTS"
  local emb_args=()
  for pair in "${ckpt_pairs[@]}"; do
    [[ -z "${pair// }" ]] && continue
    local calib_label="${pair%%=*}"
    emb_args+=(--emb "$calib_label=$pruned_root/$calib_label")
  done

  for part in $T5_LAYER_PARTS; do
    local plot_args=()
    [[ "$NO_PLOTS" == "1" ]] && plot_args+=(--no_plots)
    echo ">>> [analyze layers] eval=$eval_label part=$part"
    python scripts/blip2/analyze_t5_layer_fidelity.py \
      --part "$part" \
      --eval_label "$eval_label" \
      --dense "$dense_dir" \
      "${emb_args[@]}" \
      --out_dir "$out_eval/t5_layer_${part}" \
      "${plot_args[@]}"
  done
}

while IFS='|' read -r eval_label eval_json images_dir text_field input_mode eval_max_samples _rest; do
  [[ -z "${eval_label// }" ]] && continue
  text_field="${text_field:-auto}"
  input_mode="${input_mode:-multimodal}"
  eval_max_samples="${eval_max_samples:-$MAX_SAMPLES}"

  out_eval="$OUT_ROOT/$eval_label"
  dense_dir="$out_eval/dense"
  pruned_root="$out_eval/pruned"
  mkdir -p "$dense_dir" "$pruned_root"

  echo "========== eval T5 layer fidelity: $eval_label =========="
  extract_one "dense_${eval_label}" "$DENSE_CKPT" "$eval_json" "$images_dir" \
    "$text_field" "$input_mode" "$eval_max_samples" "$dense_dir"

  IFS=',' read -ra ckpt_pairs <<< "$PRUNED_CKPTS"
  for pair in "${ckpt_pairs[@]}"; do
    [[ -z "${pair// }" ]] && continue
    calib_label="${pair%%=*}"
    ckpt_path="${pair#*=}"
    extract_one "${calib_label}_on_${eval_label}" "$ckpt_path" "$eval_json" "$images_dir" \
      "$text_field" "$input_mode" "$eval_max_samples" "$pruned_root/$calib_label"
  done

  analyze_eval "$eval_label" "$dense_dir" "$pruned_root" "$out_eval"
done <<< "$EVAL_SPECS"

echo ">>> [aggregate] collect T5 layer fidelity CSVs"
python - "$OUT_ROOT" $T5_LAYER_PARTS <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
parts = sys.argv[2:]
rows = []
for part in parts:
    for path in root.glob(f"*/t5_layer_{part}/t5_layer_fidelity_{part}.csv"):
        eval_label = path.parents[1].name
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = dict(row)
                row["eval"] = eval_label
                row["part"] = part
                rows.append(row)
out = root / "t5_layer_fidelity_summary_all_evals.csv"
if rows:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
else:
    out.write_text("", encoding="utf-8")
print("[OK] wrote", out)
PY

echo "[OK] outputs under: $OUT_ROOT"
