#!/usr/bin/env bash
# =============================================================================
# Four-eval LLM-input embedding semantic + fidelity suite.
#
# Purpose:
#   For each evaluation dataset, run the dense BLIP2-T5 model and every
#   calibration-pruned checkpoint on the exact same eval rows, extract the
#   embedding that enters the T5 LLM, then compare pruned-vs-dense cosine
#   similarity.
#
# Required:
#   EVAL_SPECS=$'MMBench|/path/mmbench.json_or.parquet|/path/mmbench_images|question|multimodal|512
#   MMMU|/path/mmmu.json_or.parquet|/path/mmmu_images|question|multimodal|512
#   OKVQA|/path/okvqa.json|/path/okvqa_images|question|multimodal|512
#   MathVista|/path/mathvista.json|/path/mathvista_images|question|multimodal|512'
#
# Optional:
#   CALIB_SPECS=$'MMBench|/path/mmbench_calib.json|/path/mmbench_images|question|multimodal|128
#   MMMU|/path/mmmu_calib.json|/path/mmmu_images|question|multimodal|128
#   OKVQA|/path/okvqa_calib.json|/path/okvqa_images|question|multimodal|128
#   MathVista|/path/mathvista_calib.json|/path/mathvista_images|question|multimodal|128
#   CC3M|/path/cc3m_calib.json|/path/cc3m_images|caption|multimodal|128'
#
#   DENSE_CKPT=/path/blip2_pretrained_flant5xl.pth
#   OUT_ROOT=/path/out
#   RUN_FIDELITY=1                    # set 0 for semantic-only runs
#   PRUNED_CKPTS="calibA=/path/a.pth,calibB=/path/b.pth"
#   PARTS="visual both text"              # fidelity parts; default: visual both
#   SEMANTIC_PARTS="both text visual"     # semantic parts; default: both
#   MAX_SAMPLES=512             # used when the eval spec omits max_samples
#   CALIB_MAX_SAMPLES=128       # used when the calib spec omits max_samples
#   BATCH_SIZE=8
#   FORCE=1                     # re-extract even if npz already exists
#   NO_PLOTS=1                  # write CSV only
#
# Notes:
#   - Dense and pruned extraction for one eval must use the same annotation,
#     images_dir, max_samples, and ordering.
#   - If a checkpoint is T5-only pruned, LLM-input embedding may stay very close
#     to dense because this embedding is before T5 encoder pruning takes effect.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DENSE_CKPT="${DENSE_CKPT:-/data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
OUT_ROOT="${OUT_ROOT:-/data/data2/mfs/llm_embedding_fidelity_fourbench}"
MAX_SAMPLES="${MAX_SAMPLES:-512}"
CALIB_MAX_SAMPLES="${CALIB_MAX_SAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RUN_FIDELITY="${RUN_FIDELITY:-1}"
PARTS="${PARTS:-visual both}"
SEMANTIC_PARTS="${SEMANTIC_PARTS:-both}"
FORCE="${FORCE:-0}"
NO_PLOTS="${NO_PLOTS:-0}"
PRUNED_CKPTS="${PRUNED_CKPTS:-}"

: "${EVAL_SPECS:?set EVAL_SPECS with lines 'label|annotation|images_dir|text_field|input_mode|max_samples'}"
if [[ "$RUN_FIDELITY" == "1" && -z "$PRUNED_CKPTS" ]]; then
  echo "[WARN] PRUNED_CKPTS is empty; disabling fidelity and running semantic only if CALIB_SPECS is set."
  RUN_FIDELITY=0
fi

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

  if [[ "$FORCE" != "1" && -f "$out_dir/llm_input_embeddings.npz" ]]; then
    echo "[SKIP] existing embeddings: $out_dir"
    return 0
  fi

  local args=(
    python scripts/blip2/extract_llm_input_embeddings.py
    --label "$label"
    --input_mode "$input_mode"
    --calib_json "$eval_json"
    --text_field "$text_field"
    --ckpt "$ckpt"
    --out_dir "$out_dir"
    --max_samples "$max_samples"
    --batch_size "$BATCH_SIZE"
  )
  if [[ "$input_mode" != "text_only" ]]; then
    args+=(--images_dir "$images_dir")
  fi
  echo ">>> [extract] $label -> $out_dir"
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

  for part in $PARTS; do
    local plot_args=()
    [[ "$NO_PLOTS" == "1" ]] && plot_args+=(--no_plots)
    echo ">>> [analyze] eval=$eval_label part=$part"
    python scripts/blip2/analyze_llm_embeddings.py \
      --mode fidelity \
      --part "$part" \
      --dense "$dense_dir" \
      "${emb_args[@]}" \
      --out_dir "$out_eval/fidelity_${part}" \
      "${plot_args[@]}"
  done
}

run_semantic_if_requested() {
  if [[ -z "${CALIB_SPECS:-}" ]]; then
    echo "[INFO] CALIB_SPECS is empty; skip calibration-vs-eval semantic similarity."
    return 0
  fi

  local semantic_root="$OUT_ROOT/semantic_dense"
  local calib_root="$semantic_root/calib"
  local eval_root="$semantic_root/eval"
  mkdir -p "$calib_root" "$eval_root"

  echo "========== semantic: dense calibration embeddings =========="
  while IFS='|' read -r calib_label calib_json calib_images text_field input_mode calib_max_samples _rest; do
    [[ -z "${calib_label// }" ]] && continue
    text_field="${text_field:-auto}"
    input_mode="${input_mode:-multimodal}"
    calib_max_samples="${calib_max_samples:-$CALIB_MAX_SAMPLES}"
    extract_one "calib_${calib_label}" "$DENSE_CKPT" "$calib_json" "$calib_images" \
      "$text_field" "$input_mode" "$calib_max_samples" "$calib_root/$calib_label"
  done <<< "$CALIB_SPECS"

  echo "========== semantic: dense eval embeddings =========="
  while IFS='|' read -r eval_label eval_json images_dir text_field input_mode eval_max_samples _rest; do
    [[ -z "${eval_label// }" ]] && continue
    text_field="${text_field:-auto}"
    input_mode="${input_mode:-multimodal}"
    eval_max_samples="${eval_max_samples:-$MAX_SAMPLES}"
    extract_one "eval_${eval_label}" "$DENSE_CKPT" "$eval_json" "$images_dir" \
      "$text_field" "$input_mode" "$eval_max_samples" "$eval_root/$eval_label"
  done <<< "$EVAL_SPECS"

  local calib_labels=()
  local eval_labels=()
  local emb_args=()
  while IFS='|' read -r calib_label _rest; do
    [[ -z "${calib_label// }" ]] && continue
    calib_labels+=("$calib_label")
    emb_args+=(--emb "$calib_label=$calib_root/$calib_label")
  done <<< "$CALIB_SPECS"
  while IFS='|' read -r eval_label _rest; do
    [[ -z "${eval_label// }" ]] && continue
    eval_labels+=("$eval_label")
    emb_args+=(--emb "$eval_label=$eval_root/$eval_label")
  done <<< "$EVAL_SPECS"

  local calib_csv
  local eval_csv
  calib_csv="$(IFS=','; echo "${calib_labels[*]}")"
  eval_csv="$(IFS=','; echo "${eval_labels[*]}")"

  local plot_args=()
  [[ "$NO_PLOTS" == "1" ]] && plot_args+=(--no_plots)
  for part in $SEMANTIC_PARTS; do
    echo ">>> [semantic analyze] part=$part calibs=$calib_csv evals=$eval_csv"
    python scripts/blip2/analyze_llm_embeddings.py \
      --mode semantic \
      --part "$part" \
      "${emb_args[@]}" \
      --calibs "$calib_csv" \
      --evals "$eval_csv" \
      --out_dir "$semantic_root/semantic_${part}" \
      "${plot_args[@]}"
  done
}

run_semantic_if_requested

if [[ "$RUN_FIDELITY" == "1" ]]; then
  while IFS='|' read -r eval_label eval_json images_dir text_field input_mode eval_max_samples _rest; do
    [[ -z "${eval_label// }" ]] && continue
    text_field="${text_field:-auto}"
    input_mode="${input_mode:-multimodal}"
    eval_max_samples="${eval_max_samples:-$MAX_SAMPLES}"

    out_eval="$OUT_ROOT/$eval_label"
    dense_dir="$out_eval/dense"
    pruned_root="$out_eval/pruned"
    mkdir -p "$dense_dir" "$pruned_root"

    echo "========== eval fidelity: $eval_label =========="
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

  echo ">>> [aggregate] collect fidelity CSVs"
  python - "$OUT_ROOT" $PARTS <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
parts = sys.argv[2:]
rows = []
for part in parts:
    for path in root.glob(f"*/fidelity_{part}/llm_input_fidelity_{part}.csv"):
        eval_label = path.parents[1].name
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = dict(row)
                row["eval"] = eval_label
                row["part"] = part
                rows.append(row)
out = root / "fidelity_summary_all_evals.csv"
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
else
  echo "[INFO] RUN_FIDELITY=0; skip pruned-vs-dense fidelity."
fi

echo "[OK] outputs under: $OUT_ROOT"
