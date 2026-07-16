#!/usr/bin/env bash
# =============================================================================
# Four-eval teacher-forced T5 decoder hidden-state + logit fidelity suite.
#
# Purpose:
#   For each evaluation dataset, run the dense BLIP2-T5 model and every
#   calibration-pruned checkpoint on the same eval rows.  Use the ground-truth
#   answer/caption as teacher-forced decoder targets, then compare:
#     1) T5 decoder hidden states layer-by-layer.
#     2) Compact logit summaries: gold-token logprob, top-1 agreement, top-k overlap.
#
# Required:
#   PRUNED_CKPTS="MMBench=/path/mmbench.pth,MMMU=/path/mmmu.pth,OKVQA=/path/okvqa.pth,MathVista=/path/mathvista.pth,CC3M=/path/cc3m.pth"
#
#   EVAL_SPECS=$'MMBench|/path/mmbench.json_or.parquet|/path/mmbench_images|question|multimodal|512|answer
#   MMMU|/path/mmmu.json_or.parquet|/path/mmmu_images|question|multimodal|512|answer
#   OKVQA|/path/okvqa.json|/path/okvqa_images|question|multimodal|512|answer
#   MathVista|/path/mathvista.json|/path/mathvista_images|question|multimodal|512|answer'
#
# EVAL_SPECS columns:
#   label|annotation|images_dir|text_field|input_mode|max_samples|output_field
#   output_field is optional; auto tries text_output/answer/caption/text/question.
#
# Optional:
#   DENSE_CKPT=/path/blip2_pretrained_flant5xl.pth
#   OUT_ROOT=/path/out
#   MAX_SAMPLES=512
#   BATCH_SIZE=2
#   MAX_OUTPUT_LEN=32
#   TOP_K=10
#   SAVE_DTYPE=float32                # float32 or float16
#   FORCE=1                           # re-extract even if npz already exists
#   NO_PLOTS=1                        # write CSV only
#   FP32=1                            # run extraction in fp32
#
# Main outputs:
#   $OUT_ROOT/MMBench/decoder_logits/t5_decoder_layer_fidelity.png
#   $OUT_ROOT/MMBench/decoder_logits/t5_logit_fidelity.csv
#   $OUT_ROOT/t5_decoder_layer_fidelity_summary_all_evals.csv
#   $OUT_ROOT/t5_logit_fidelity_summary_all_evals.csv
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DENSE_CKPT="${DENSE_CKPT:-/data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
OUT_ROOT="${OUT_ROOT:-/data/data2/mfs/t5_decoder_logits_fidelity_fourbench}"
MAX_SAMPLES="${MAX_SAMPLES:-512}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN:-32}"
TOP_K="${TOP_K:-10}"
SAVE_DTYPE="${SAVE_DTYPE:-float32}"
FORCE="${FORCE:-0}"
NO_PLOTS="${NO_PLOTS:-0}"
FP32="${FP32:-0}"

: "${PRUNED_CKPTS:?set PRUNED_CKPTS as comma-separated label=/path/checkpoint.pth}"
: "${EVAL_SPECS:?set EVAL_SPECS with lines 'label|annotation|images_dir|text_field|input_mode|max_samples|output_field'}"

mkdir -p "$OUT_ROOT"

extract_one() {
  local label="$1"
  local ckpt="$2"
  local eval_json="$3"
  local images_dir="$4"
  local text_field="$5"
  local input_mode="$6"
  local max_samples="$7"
  local output_field="$8"
  local out_dir="$9"

  if [[ "$FORCE" != "1" && -f "$out_dir/t5_decoder_logits.npz" ]]; then
    echo "[SKIP] existing decoder/logit states: $out_dir"
    return 0
  fi

  local args=(
    python scripts/blip2/extract_t5_decoder_logits.py
    --label "$label"
    --input_mode "$input_mode"
    --calib_json "$eval_json"
    --text_field "$text_field"
    --output_field "$output_field"
    --ckpt "$ckpt"
    --out_dir "$out_dir"
    --max_samples "$max_samples"
    --batch_size "$BATCH_SIZE"
    --max_output_len "$MAX_OUTPUT_LEN"
    --top_k "$TOP_K"
    --save_dtype "$SAVE_DTYPE"
  )
  if [[ "$input_mode" != "text_only" ]]; then
    args+=(--images_dir "$images_dir")
  fi
  if [[ "$FP32" == "1" ]]; then
    args+=(--fp32)
  fi
  echo ">>> [extract decoder/logits] $label -> $out_dir"
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

  local plot_args=()
  [[ "$NO_PLOTS" == "1" ]] && plot_args+=(--no_plots)
  echo ">>> [analyze decoder/logits] eval=$eval_label"
  python scripts/blip2/analyze_t5_decoder_logits_fidelity.py \
    --eval_label "$eval_label" \
    --dense "$dense_dir" \
    "${emb_args[@]}" \
    --out_dir "$out_eval/decoder_logits" \
    "${plot_args[@]}"
}

while IFS='|' read -r eval_label eval_json images_dir text_field input_mode eval_max_samples output_field _rest; do
  [[ -z "${eval_label// }" ]] && continue
  text_field="${text_field:-auto}"
  input_mode="${input_mode:-multimodal}"
  eval_max_samples="${eval_max_samples:-$MAX_SAMPLES}"
  output_field="${output_field:-auto}"

  out_eval="$OUT_ROOT/$eval_label"
  dense_dir="$out_eval/dense_decoder_logits"
  pruned_root="$out_eval/pruned_decoder_logits"
  mkdir -p "$dense_dir" "$pruned_root"

  echo "========== eval decoder/logit fidelity: $eval_label =========="
  extract_one "dense_${eval_label}" "$DENSE_CKPT" "$eval_json" "$images_dir" \
    "$text_field" "$input_mode" "$eval_max_samples" "$output_field" "$dense_dir"

  IFS=',' read -ra ckpt_pairs <<< "$PRUNED_CKPTS"
  for pair in "${ckpt_pairs[@]}"; do
    [[ -z "${pair// }" ]] && continue
    calib_label="${pair%%=*}"
    ckpt_path="${pair#*=}"
    extract_one "${calib_label}_on_${eval_label}" "$ckpt_path" "$eval_json" "$images_dir" \
      "$text_field" "$input_mode" "$eval_max_samples" "$output_field" "$pruned_root/$calib_label"
  done

  analyze_eval "$eval_label" "$dense_dir" "$pruned_root" "$out_eval"
done <<< "$EVAL_SPECS"

echo ">>> [aggregate] collect decoder/logit fidelity CSVs"
python - "$OUT_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])

def collect(pattern, out_name):
    rows = []
    for path in root.glob(pattern):
        eval_label = path.parents[1].name
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = dict(row)
                row["eval"] = eval_label
                rows.append(row)
    out = root / out_name
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

collect("*/decoder_logits/t5_decoder_layer_fidelity.csv", "t5_decoder_layer_fidelity_summary_all_evals.csv")
collect("*/decoder_logits/t5_logit_fidelity.csv", "t5_logit_fidelity_summary_all_evals.csv")
collect("*/decoder_logits/t5_decoder_final_layer_fidelity.csv", "t5_decoder_final_layer_fidelity_summary_all_evals.csv")
PY

echo "[OK] outputs under: $OUT_ROOT"
