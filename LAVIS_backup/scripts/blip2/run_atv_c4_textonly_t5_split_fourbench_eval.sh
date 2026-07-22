#!/usr/bin/env bash
# =============================================================================
# C4 text-only ATV-entry pruning, then four-benchmark evaluation.
#
# This script intentionally passes ONLY the C4 text calibration file into the
# pruning stage:
#   C4 text -> --prune_calib_mode t5_c4_text -> blipt5_atv_pruner
#
# Because C4 has no images, there are no BLIP2 visual/query tokens. In the
# current BLIP2-T5 port, text-only calibration sets temp_label to all False, so
# the ATV entry degenerates to text-token-only Wanda-style T5 pruning with
# uniform sparsity. ViT/Q-Former/t5_proj stay dense. This is therefore the
# C4 text side of a split-pruning baseline, not true multimodal ATV.
#
# Four evals:
#   MMBench / OKVQA / MMMU / MathVista
#
# Usage from LAVIS_backup:
#   cd /data/data2/mfs/2/LAVIS_backup
#   bash scripts/blip2/run_atv_c4_textonly_t5_split_fourbench_eval.sh
#
# Common overrides:
#   BASE=/data/data2/mfs
#   C4_JSON=/data/data2/mfs/c4_calib_128.json
#   NUM_DATA=128 BS=8 T5_SPEC=24-0.5-1.0-1.0 ATV_ALPHA=1.0
#   RUN_PRUNE=0 JOB_ID=<existing_job_id>
#   RUN_EVAL=0
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

BASE="${BASE:-/data/data2/mfs}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$BASE/model_cache}"
ECOFLAP_ENV="${ECOFLAP_ENV:-${MODEL_CACHE_ROOT}/ecoflap_model_env.sh}"
if [[ -f "$ECOFLAP_ENV" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "$ECOFLAP_ENV"
  set -u
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$MODEL_CACHE_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TORCH_HOME="${TORCH_HOME:-$MODEL_CACHE_ROOT/torch}"
mkdir -p "$HF_HOME/hub" "$HF_HOME/transformers" "$TORCH_HOME/hub/checkpoints" 2>/dev/null || true

_resolve_hub_snapshot_dir() {
  local repo_dir="$1"
  [[ -d "$repo_dir/snapshots" ]] || { echo ""; return 0; }
  find "$repo_dir/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | head -1
}

HUB_ROOT="${HUGGINGFACE_HUB_CACHE}"
if [[ -z "${BERT_BASE_UNCASED_SNAPSHOT:-}" ]]; then
  BERT_BASE_UNCASED_SNAPSHOT="$(_resolve_hub_snapshot_dir "$HUB_ROOT/models--bert-base-uncased")"
fi
if [[ -z "${FLAN_T5_XL_SNAPSHOT:-}" ]]; then
  FLAN_T5_XL_SNAPSHOT="$(_resolve_hub_snapshot_dir "$HUB_ROOT/models--google--flan-t5-xl")"
fi
export BERT_BASE_UNCASED_SNAPSHOT
export FLAN_T5_XL_SNAPSHOT

BLIP2_PRETRAINED="${BLIP2_PRETRAINED:-$TORCH_HOME/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
if [[ ! -f "$BLIP2_PRETRAINED" ]]; then
  BLIP2_PRETRAINED="$BASE/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth"
fi

CFG="${CFG:-$REPO_ROOT/lavis/projects/blip2/eval/t5_c4_text_prune_calib.yaml}"
C4_JSON="${C4_JSON:-$BASE/c4_calib_128.json}"

NUM_DATA="${NUM_DATA:-128}"
BS="${BS:-8}"
T5_SPEC="${T5_SPEC:-24-0.5-1.0-1.0}"
ATV_ALPHA="${ATV_ALPHA:-1.0}"
SEED="${SEED:-42}"
export LAVIS_DISTRIBUTED_SAMPLER_SEED="${LAVIS_DISTRIBUTED_SAMPLER_SEED:-$SEED}"

JOB_STAMP="${JOB_STAMP:-$(date +%Y%m%d_%H%M%S)}"
JOB_ID="${JOB_ID:-atv_c4_textonly_t5only_${JOB_STAMP}_seed${SEED}}"
CKPT="$REPO_ROOT/pruned_checkpoint/${JOB_ID}.pth"

RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_SPARSITY_CHECK="${RUN_SPARSITY_CHECK:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
MASTER_PORT_OKVQA="${MASTER_PORT_OKVQA:-29780}"
OKVQA_EVAL_NUM_WORKERS="${OKVQA_EVAL_NUM_WORKERS:-0}"

REPORT_DIR="${REPORT_DIR:-$BASE/atv_c4_textonly_t5_split_${JOB_STAMP}_seed${SEED}}"
PRUNE_LOG="$REPORT_DIR/${JOB_ID}_prune.log"
SPARSITY_CSV="$REPORT_DIR/${JOB_ID}_sparsity.csv"
METRICS_JSONL="${LAVIS_METRICS_JSONL:-$REPORT_DIR/${JOB_ID}_fourbench_metrics.jsonl}"
SUMMARY_MD="${SUMMARY_MD:-$REPORT_DIR/${JOB_ID}_fourbench_summary.md}"
SUMMARY_TSV="${SUMMARY_TSV:-$REPORT_DIR/${JOB_ID}_fourbench_summary.tsv}"
mkdir -p "$REPORT_DIR" "$REPO_ROOT/pruned_checkpoint" "$REPO_ROOT/lavis/output/BLIP2" "$REPO_ROOT/training_statistics"

ECOFLAP_BENCH_ROOT="${ECOFLAP_BENCH_ROOT:-$BASE}"
MMBENCH_ROOT="${MMBENCH_ROOT:-$ECOFLAP_BENCH_ROOT/MMBench_eval}"
MMMU_ROOT="${MMMU_ROOT:-$ECOFLAP_BENCH_ROOT/MMMU_single_image}"
MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
MMMU_SPLIT="${MMMU_SPLIT:-test}"
OKVQA_EVAL_CFG="${OKVQA_EVAL_CFG:-$REPO_ROOT/lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml}"
MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-$ECOFLAP_BENCH_ROOT/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
MATHVISTA_IMAGES_DIR="${MATHVISTA_IMAGES_DIR:-$ECOFLAP_BENCH_ROOT/MathVista_eval_testmini_mc/images}"

_require_file() {
  local path="$1"
  local label="$2"
  [[ -f "$path" ]] || {
    echo "[FATAL] missing $label: $path" >&2
    exit 1
  }
}

_require_dir() {
  local path="$1"
  local label="$2"
  [[ -d "$path" ]] || {
    echo "[FATAL] missing $label: $path" >&2
    exit 1
  }
}

preflight() {
  _require_file "$BLIP2_PRETRAINED" "BLIP2_PRETRAINED"
  _require_file "$CFG" "C4 text-only prune cfg"
  _require_file "$C4_JSON" "C4 calibration JSON"
  _require_dir "${BERT_BASE_UNCASED_SNAPSHOT:-}" "bert-base-uncased snapshot"
  _require_dir "${FLAN_T5_XL_SNAPSHOT:-}" "flan-t5-xl snapshot"

  if (( NUM_DATA % BS != 0 )); then
    echo "[FATAL] NUM_DATA ($NUM_DATA) must be divisible by BS ($BS)." >&2
    exit 1
  fi

  if [[ "$RUN_EVAL" == "1" ]]; then
    _require_dir "$MMBENCH_ROOT" "MMBench eval root"
    _require_dir "$MMMU_ROOT" "MMMU eval root"
    _require_file "$OKVQA_EVAL_CFG" "OKVQA eval cfg"
    _require_file "$MATHVISTA_EVAL_JSON" "MathVista eval JSON"
    _require_dir "$MATHVISTA_IMAGES_DIR" "MathVista images dir"
  fi
}

_t5_sparsity_target() {
  local depth sparsity rest
  IFS='-' read -r depth sparsity rest <<< "$T5_SPEC"
  echo "$sparsity"
}

prune_c4_text_only() {
  echo ""
  echo "========== C4 text-only ATV-entry pruning =========="
  echo "[INFO] REPO_ROOT=$REPO_ROOT"
  echo "[INFO] C4_JSON=$C4_JSON"
  echo "[INFO] JOB_ID=$JOB_ID"
  echo "[INFO] CKPT=$CKPT"
  echo "[INFO] T5_SPEC=$T5_SPEC ATV_ALPHA=$ATV_ALPHA NUM_DATA=$NUM_DATA BS=$BS"
  echo "[WARN] Calibration stage receives no images; ViT is not pruned."
  echo "[WARN] ATV visual/query-token selection degenerates because temp_label is all False."

  LAVIS_ATV_DIAGNOSTIC_DIR="${LAVIS_ATV_DIAGNOSTIC_DIR:-$REPORT_DIR}" \
  python evaluate_blip.py \
    --cfg-path "$CFG" \
    --options "model.pretrained=${BLIP2_PRETRAINED}" \
    --prune_calib_mode t5_c4_text \
    --c4_calib_json "$C4_JSON" \
    --pruning_method blipt5_atv_pruner \
    --atv_alpha "$ATV_ALPHA" \
    --no_prune_vit \
    --t5_prune_spec "$T5_SPEC" \
    --num_data "$NUM_DATA" \
    --num_data_first_stage "$NUM_DATA" \
    --prunining_dataset_batch_size "$BS" \
    --job_id "$JOB_ID" \
    --save_pruned_model 2>&1 | tee "$PRUNE_LOG"

  _require_file "$CKPT" "saved C4 text-only ATV checkpoint"
}

check_sparsity() {
  local target
  target="$(_t5_sparsity_target)"
  echo ""
  echo "========== sparsity check =========="
  python scripts/blip2/check_ckpt_sparsity.py \
    --ckpt "$CKPT" \
    --tag "$JOB_ID" \
    --expect_t5 "$target" \
    --vit_max 0.01 \
    --non_t5_max 0.01 \
    --tol 0.05 \
    --out_csv "$SPARSITY_CSV"
}

eval_four_bench() {
  local eval_tag okvqa_job
  eval_tag="atv_c4_textonly_t5only_${JOB_STAMP}_seed${SEED}"
  okvqa_job="okvqa_eval_${eval_tag}_fullval"

  _require_file "$CKPT" "checkpoint for eval"
  export LAVIS_METRICS_JSONL="$METRICS_JSONL"
  export LAVIS_EVAL_CALIB_TAG="$eval_tag"
  : > "$LAVIS_METRICS_JSONL"

  echo ""
  echo "========== four-benchmark eval =========="
  echo "[INFO] checkpoint=$CKPT"
  echo "[INFO] metrics=$LAVIS_METRICS_JSONL"

  echo ""
  echo ">>> [$eval_tag] MMBench"
  export LAVIS_METRICS_BENCHMARK="MMBench"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMBENCH_ROOT" \
    --split "$MMBENCH_SPLIT" \
    --ckpt "$CKPT" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --device cuda \
    --overall_only

  echo ""
  echo ">>> [$eval_tag] OKVQA full val"
  export LAVIS_METRICS_BENCHMARK="OKVQA"
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$MASTER_PORT_OKVQA" evaluate_blip.py \
    --cfg-path "$OKVQA_EVAL_CFG" \
    --options "model.pretrained=${BLIP2_PRETRAINED}" "run.num_workers=${OKVQA_EVAL_NUM_WORKERS}" \
    --t5_pruned_checkpoint "$CKPT" \
    --job_id "$okvqa_job"

  echo ""
  echo ">>> [$eval_tag] MMMU"
  export LAVIS_METRICS_BENCHMARK="MMMU"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMMU_ROOT" \
    --split "$MMMU_SPLIT" \
    --ckpt "$CKPT" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --device cuda \
    --overall_only

  echo ""
  echo ">>> [$eval_tag] MathVista MC"
  export LAVIS_METRICS_BENCHMARK="MathVista_MC"
  python scripts/blip2/mathvista_mc_eval.py \
    --eval_json "$MATHVISTA_EVAL_JSON" \
    --images_dir "$MATHVISTA_IMAGES_DIR" \
    --ckpt "$CKPT" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --device cuda

  if [[ -f "$SCRIPT_DIR/collect_lavisbackup_eval_summary.py" ]]; then
    echo ""
    echo "========== eval summary =========="
    python "$SCRIPT_DIR/collect_lavisbackup_eval_summary.py" \
      --repo-root "$REPO_ROOT" \
      --metrics-jsonl "$LAVIS_METRICS_JSONL" \
      --out-md "$SUMMARY_MD" \
      --out-tsv "$SUMMARY_TSV" \
      --suites "${eval_tag}:${okvqa_job}" || true
  fi
}

preflight

echo "========== run config =========="
echo "[INFO] BASE=$BASE"
echo "[INFO] HF_HOME=$HF_HOME"
echo "[INFO] BLIP2_PRETRAINED=$BLIP2_PRETRAINED"
echo "[INFO] C4_JSON=$C4_JSON"
echo "[INFO] RUN_PRUNE=$RUN_PRUNE RUN_EVAL=$RUN_EVAL RUN_SPARSITY_CHECK=$RUN_SPARSITY_CHECK"
echo "[INFO] REPORT_DIR=$REPORT_DIR"
echo "================================"

if [[ "$RUN_PRUNE" == "1" ]]; then
  prune_c4_text_only
else
  echo "[INFO] RUN_PRUNE=0, reuse checkpoint: $CKPT"
  _require_file "$CKPT" "existing checkpoint"
fi

if [[ "$RUN_SPARSITY_CHECK" == "1" ]]; then
  check_sparsity
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  eval_four_bench
else
  echo "[INFO] RUN_EVAL=0, skip four-benchmark eval."
fi

echo ""
echo "========== done =========="
echo "[INFO] checkpoint: $CKPT"
echo "[INFO] report_dir:  $REPORT_DIR"
echo "[INFO] metrics:     $METRICS_JSONL"
echo "[INFO] summary_md:  $SUMMARY_MD"
echo "[INFO] summary_tsv: $SUMMARY_TSV"
