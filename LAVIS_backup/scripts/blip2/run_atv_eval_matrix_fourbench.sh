#!/usr/bin/env bash
# =============================================================================
# Evaluate an ATV validation model matrix on the same four benchmarks:
# MMBench / OKVQA / MMMU / MathVista.
#
# This script does not prune. It consumes existing checkpoints and writes:
#   - one per-seed metrics JSONL for MMBench/MMMU/MathVista
#   - OKVQA evaluate.txt paths per model
#   - one per-seed eval provenance CSV mapping methods to checkpoint paths
#   - eval_results.csv inside REPORT_DIR via validate_atv_migration.py
#
# Supported model labels in MODELS:
#   dense  : full precision model. Leave DENSE_CKPT empty to use the LAVIS
#            dense BLIP2-T5 loader.
#   atv    : CKPT_ATV
#   atv_alpha0 : CKPT_ATV_ALPHA0, defaulting to the alpha-sweep checkpoint name
#   atv_alpha4 : CKPT_ATV_ALPHA4, defaulting to the alpha-sweep checkpoint name
#   naive  : CKPT_NAIVE or CKPT_WANDA
#   wanda  : CKPT_WANDA or CKPT_NAIVE
#   tamp   : CKPT_TAMP
#
# Example:
#   cd /data/data2/mfs/2/LAVIS_backup
#   BASE=/data/data2/mfs \
#   ATV_ROOT=/data/data2/mfs/ATV-Pruning \
#   REPORT_DIR=/data/data2/mfs/atv_validation_report_xxx \
#   CKPT_ATV=/path/to/atv.pth \
#   CKPT_NAIVE=/path/to/naive_wanda.pth \
#   CKPT_TAMP=/path/to/tamp.pth \
#   bash scripts/blip2/run_atv_eval_matrix_fourbench.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

BASE="${BASE:-/data/data2/mfs}"
ATV_ROOT="${ATV_ROOT:-$BASE/ATV-Pruning}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
REPORT_DIR="${REPORT_DIR:-$BASE/atv_validation_report_eval_matrix_${STAMP}_seed${SEED}}"
MODELS="${MODELS:-dense atv atv_alpha0 atv_alpha4 naive tamp}"
VALIDATE_STRICT="${VALIDATE_STRICT:-0}"
if [[ "$VALIDATE_STRICT" == "1" ]]; then
  REQUIRE_MODEL_CKPTS="${REQUIRE_MODEL_CKPTS:-1}"
else
  REQUIRE_MODEL_CKPTS="${REQUIRE_MODEL_CKPTS:-0}"
fi

MMBENCH_ROOT="${MMBENCH_ROOT:-$BASE/MMBench_eval}"
MMMU_ROOT="${MMMU_ROOT:-$BASE/MMMU_single_image}"
MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
MMMU_SPLIT="${MMMU_SPLIT:-test}"
MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-$BASE/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
MATHVISTA_IMAGES_DIR="${MATHVISTA_IMAGES_DIR:-$BASE/MathVista_eval_testmini_mc/images}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29900}"

DENSE_CKPT="${DENSE_CKPT:-}"
CKPT_ATV="${CKPT_ATV:-}"
CKPT_ATV_ALPHA0="${CKPT_ATV_ALPHA0:-$REPO_ROOT/pruned_checkpoint/atv_cc3m_t5only_alpha0_${STAMP}_seed${SEED}.pth}"
CKPT_ATV_ALPHA4="${CKPT_ATV_ALPHA4:-$REPO_ROOT/pruned_checkpoint/atv_cc3m_t5only_alpha4_${STAMP}_seed${SEED}.pth}"
CKPT_NAIVE="${CKPT_NAIVE:-${CKPT_WANDA:-}}"
CKPT_WANDA="${CKPT_WANDA:-$CKPT_NAIVE}"
CKPT_TAMP="${CKPT_TAMP:-}"

mkdir -p "$REPORT_DIR"
export LAVIS_METRICS_JSONL="${LAVIS_METRICS_JSONL:-$REPORT_DIR/eval_matrix_fourbench_metrics_seed${SEED}.jsonl}"
EVAL_PROVENANCE_CSV="${EVAL_PROVENANCE_CSV:-$REPORT_DIR/eval_provenance_seed${SEED}.csv}"
: > "$LAVIS_METRICS_JSONL"
printf 'seed,method,calibration,alpha,ckpt,metrics_jsonl,okvqa_eval_txt\n' > "$EVAL_PROVENANCE_CSV"

OKVQA_EVIDENCE_ARGS=()

csv_cell() {
  local value="${1:-}"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

append_eval_provenance() {
  local method="$1" tag="$2" ckpt="$3" okvqa_eval_txt="$4"
  local alpha=""
  if [[ "$tag" =~ alpha([0-9p.+-]+) ]]; then
    alpha="${BASH_REMATCH[1]//p/.}"
  elif [[ "$method" == "atv" ]]; then
    alpha="1"
  fi
  {
    csv_cell "$SEED"; printf ','
    csv_cell "$method"; printf ','
    csv_cell "$tag"; printf ','
    csv_cell "$alpha"; printf ','
    csv_cell "$ckpt"; printf ','
    csv_cell "$LAVIS_METRICS_JSONL"; printf ','
    csv_cell "$okvqa_eval_txt"; printf '\n'
  } >> "$EVAL_PROVENANCE_CSV"
}

warn_or_die_missing_ckpt() {
  local label="$1" ckpt="$2"
  if [[ -n "$ckpt" && -f "$ckpt" ]]; then
    return 0
  fi
  if [[ "$REQUIRE_MODEL_CKPTS" == "1" ]]; then
    echo "[FATAL] missing checkpoint for $label: ${ckpt:-<empty>}" >&2
    exit 1
  fi
  echo "[WARN] skip $label because checkpoint is missing: ${ckpt:-<empty>}" >&2
  return 1
}

eval_one() {
  local method="$1" tag="$2" ckpt="$3" okvqa_job="$4" master_port="$5"
  local ckpt_args=()
  if [[ -n "$ckpt" ]]; then
    [[ -f "$ckpt" ]] || { echo "[FATAL] checkpoint missing for $tag: $ckpt" >&2; exit 1; }
    ckpt_args=(--ckpt "$ckpt")
  fi

  export LAVIS_EVAL_CALIB_TAG="$tag"
  echo ""
  echo "========== four-benchmark eval | method=$method tag=$tag =========="

  export LAVIS_METRICS_BENCHMARK="MMBench"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMBENCH_ROOT" --split "$MMBENCH_SPLIT" "${ckpt_args[@]}" \
    --batch_size "$EVAL_BATCH_SIZE" --device cuda --overall_only

  local okvqa_ckpt_args=()
  if [[ -n "$ckpt" ]]; then
    okvqa_ckpt_args=(--t5_pruned_checkpoint "$ckpt" --vit_pruned_checkpoint "$ckpt")
  fi
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$master_port" evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
    "${okvqa_ckpt_args[@]}" \
    --job_id "$okvqa_job"
  local okvqa_eval_txt="$REPO_ROOT/lavis/output/BLIP2/OKVQA/$okvqa_job/evaluate.txt"
  OKVQA_EVIDENCE_ARGS+=(--okvqa_eval_txt "$tag=$okvqa_eval_txt")

  export LAVIS_METRICS_BENCHMARK="MMMU"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMMU_ROOT" --split "$MMMU_SPLIT" "${ckpt_args[@]}" \
    --batch_size "$EVAL_BATCH_SIZE" --device cuda --overall_only

  if [[ -f "$MATHVISTA_EVAL_JSON" ]]; then
    export LAVIS_METRICS_BENCHMARK="MathVista_MC"
    python scripts/blip2/mathvista_mc_eval.py \
      --eval_json "$MATHVISTA_EVAL_JSON" --images_dir "$MATHVISTA_IMAGES_DIR" \
      "${ckpt_args[@]}" --batch_size "$EVAL_BATCH_SIZE" --device cuda
  else
    echo "[WARN] skip MathVista because file is missing: $MATHVISTA_EVAL_JSON"
  fi
  append_eval_provenance "$method" "$tag" "$ckpt" "$okvqa_eval_txt"
}

model_i=0
for model in $MODELS; do
  model_i=$((model_i + 1))
  case "$model" in
    dense)
      eval_one dense "dense_fullprec_${STAMP}" "$DENSE_CKPT" "okvqa_eval_dense_${STAMP}_seed${SEED}_fullval" "$((MASTER_PORT_BASE + model_i))"
      ;;
    atv)
      warn_or_die_missing_ckpt atv "$CKPT_ATV" || continue
      eval_one atv "cc3m_atv_t5only_${STAMP}" "$CKPT_ATV" "okvqa_eval_atv_${STAMP}_seed${SEED}_fullval" "$((MASTER_PORT_BASE + model_i))"
      ;;
    atv_alpha0|atv0)
      warn_or_die_missing_ckpt atv_alpha0 "$CKPT_ATV_ALPHA0" || continue
      eval_one atv "cc3m_atv_alpha0_t5only_${STAMP}" "$CKPT_ATV_ALPHA0" "okvqa_eval_atv_alpha0_${STAMP}_seed${SEED}_fullval" "$((MASTER_PORT_BASE + model_i))"
      ;;
    atv_alpha4|atv4)
      warn_or_die_missing_ckpt atv_alpha4 "$CKPT_ATV_ALPHA4" || continue
      eval_one atv "cc3m_atv_alpha4_t5only_${STAMP}" "$CKPT_ATV_ALPHA4" "okvqa_eval_atv_alpha4_${STAMP}_seed${SEED}_fullval" "$((MASTER_PORT_BASE + model_i))"
      ;;
    naive|wanda)
      warn_or_die_missing_ckpt "$model" "$CKPT_WANDA" || continue
      eval_one wanda "cc3m_naive_wanda_t5only_${STAMP}" "$CKPT_WANDA" "okvqa_eval_naive_wanda_${STAMP}_seed${SEED}_fullval" "$((MASTER_PORT_BASE + model_i))"
      ;;
    tamp|amia)
      warn_or_die_missing_ckpt tamp "$CKPT_TAMP" || continue
      eval_one tamp "cc3m_tamp_t5only_${STAMP}" "$CKPT_TAMP" "okvqa_eval_tamp_${STAMP}_seed${SEED}_fullval" "$((MASTER_PORT_BASE + model_i))"
      ;;
    *)
      echo "[FATAL] unknown MODELS entry: $model" >&2
      exit 1
      ;;
  esac
done

if [[ -f "$ATV_ROOT/qwen/activation_aware_pruner.py" ]]; then
  validate_args=()
  if [[ "$VALIDATE_STRICT" == "1" ]]; then
    validate_args+=(--strict)
  fi
  python scripts/blip2/validate_atv_migration.py \
    --original_atv_root "$ATV_ROOT" \
    --lavis_root "$REPO_ROOT" \
    --out_dir "$REPORT_DIR" \
    --preserve_existing \
    --metrics_jsonl "$LAVIS_METRICS_JSONL" \
    "${OKVQA_EVIDENCE_ARGS[@]}" \
    --eval_provenance_csv "$EVAL_PROVENANCE_CSV" \
    --eval_seed "$SEED" \
    "${validate_args[@]}"
else
  echo "[WARN] skip validate_atv_migration.py: ATV_ROOT not found: $ATV_ROOT"
fi

echo ""
echo "[OK] metrics jsonl: $LAVIS_METRICS_JSONL"
echo "[OK] validation report: $REPORT_DIR"
