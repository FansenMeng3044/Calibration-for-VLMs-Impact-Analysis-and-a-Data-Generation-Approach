#!/usr/bin/env bash
# =============================================================================
# CC3M multimodal calibration -> ATV-Pruning -> four-benchmark evaluation.
#
# This is the clean one-shot driver for the experiment:
#   1. Use CC3M image + caption calibration data.
#   2. Run BLIP2-T5 with blipt5_atv_pruner.
#   3. Prune only the LLM/T5 side. ViT is kept dense.
#   4. Evaluate the saved checkpoint on MMBench / OKVQA / MMMU / MathVista.
#
# Usage:
#   cd /data/data2/mfs/2/LAVIS_backup
#   bash scripts/blip2/run_atv_cc3m_joint_llmonly_prune_eval_fourbench.sh
#
# Common overrides:
#   BASE=/data/data2/mfs
#   CC3M_JSON=/data/data2/mfs/CC3M_calib_128/cc3m_calib_128.json
#   CC3M_IMAGES_DIR=/data/data2/mfs/CC3M_calib_128/images
#   T5_SPEC=24-0.5-1.0-1.0 ATV_ALPHA=1.0 NUM_DATA=128 BS=8
#   RUN_PRUNE=0 JOB_ID=<existing_job_id> bash scripts/blip2/run_atv_cc3m_joint_llmonly_prune_eval_fourbench.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

BASE="${BASE:-/data/data2/mfs}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$BASE/model_cache}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

ECOFLAP_ENV="${ECOFLAP_ENV:-${MODEL_CACHE_ROOT}/ecoflap_model_env.sh}"
if [[ -f "$ECOFLAP_ENV" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$ECOFLAP_ENV"
  set -u
fi

export TORCH_HOME="${TORCH_HOME:-${MODEL_CACHE_ROOT}/torch}"
export HF_HOME="${HF_HOME:-${MODEL_CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

resolve_hub_snapshot_dir() {
  local repo_dir="$1"
  [[ -d "$repo_dir/snapshots" ]] || { echo ""; return 0; }
  find "$repo_dir/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | head -1
}

if [[ -z "${BERT_BASE_UNCASED_SNAPSHOT:-}" ]]; then
  BERT_BASE_UNCASED_SNAPSHOT="$(resolve_hub_snapshot_dir "$HUGGINGFACE_HUB_CACHE/models--bert-base-uncased")"
fi
if [[ -z "${FLAN_T5_XL_SNAPSHOT:-}" ]]; then
  FLAN_T5_XL_SNAPSHOT="$(resolve_hub_snapshot_dir "$HUGGINGFACE_HUB_CACHE/models--google--flan-t5-xl")"
fi
export BERT_BASE_UNCASED_SNAPSHOT
export FLAN_T5_XL_SNAPSHOT

BLIP2_PRETRAINED="${BLIP2_PRETRAINED:-${MODEL_CACHE_ROOT}/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
CC3M_TEMPLATE_CFG="${CC3M_TEMPLATE_CFG:-$REPO_ROOT/lavis/projects/blip2/eval/cc_prefix_derivative_compute_cc3m_calib128.yaml}"
CC3M_JSON="${CC3M_JSON:-$BASE/CC3M_calib_128/cc3m_calib_128.json}"
CC3M_IMAGES_DIR="${CC3M_IMAGES_DIR:-$BASE/CC3M_calib_128/images}"

JOB_STAMP="${JOB_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
export LAVIS_DISTRIBUTED_SAMPLER_SEED="${LAVIS_DISTRIBUTED_SAMPLER_SEED:-$SEED}"

ATV_ALPHA="${ATV_ALPHA:-1.0}"
T5_SPEC="${T5_SPEC:-24-0.5-1.0-1.0}"
T5_SPARSITY_TARGET="${T5_SPARSITY_TARGET:-$(python - "$T5_SPEC" <<'PY'
import sys
parts = sys.argv[1].split("-")
try:
    keep_ratio = float(parts[1])
except (IndexError, ValueError) as exc:
    raise SystemExit("cannot parse T5_SPEC keep ratio from %r" % sys.argv[1]) from exc
print("%.6f" % (1.0 - keep_ratio))
PY
)}"
NUM_DATA="${NUM_DATA:-128}"
NUM_DATA_FIRST_STAGE="${NUM_DATA_FIRST_STAGE:-$NUM_DATA}"
BS="${BS:-8}"

RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_SPARSITY_CHECK="${RUN_SPARSITY_CHECK:-1}"

JOB_ID="${JOB_ID:-atv_cc3m_joint_llmonly_${JOB_STAMP}_seed${SEED}}"
CKPT="${CKPT:-$REPO_ROOT/pruned_checkpoint/${JOB_ID}.pth}"

REPORT_DIR="${REPORT_DIR:-$BASE/atv_cc3m_joint_llmonly_${JOB_STAMP}_seed${SEED}}"
CC3M_RUNTIME_CFG="${CC3M_RUNTIME_CFG:-$REPORT_DIR/cc3m_calib_runtime.yaml}"
PRUNE_LOG="${PRUNE_LOG:-$REPORT_DIR/${JOB_ID}_prune.log}"
SPARSITY_CSV="${SPARSITY_CSV:-$REPORT_DIR/${JOB_ID}_sparsity.csv}"
METRICS_JSONL="${METRICS_JSONL:-$REPORT_DIR/${JOB_ID}_fourbench_metrics.jsonl}"
SUMMARY_MD="${SUMMARY_MD:-$REPORT_DIR/${JOB_ID}_fourbench_summary.md}"
SUMMARY_TSV="${SUMMARY_TSV:-$REPORT_DIR/${JOB_ID}_fourbench_summary.tsv}"

MMBENCH_ROOT="${MMBENCH_ROOT:-$BASE/MMBench_eval}"
MMMU_ROOT="${MMMU_ROOT:-$BASE/MMMU_single_image}"
MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
MMMU_SPLIT="${MMMU_SPLIT:-test}"
OKVQA_EVAL_CFG="${OKVQA_EVAL_CFG:-$REPO_ROOT/lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml}"
OKVQA_EVAL_NUM_WORKERS="${OKVQA_EVAL_NUM_WORKERS:-0}"
MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-$BASE/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
MATHVISTA_IMAGES_DIR="${MATHVISTA_IMAGES_DIR:-$BASE/MathVista_eval_testmini_mc/images}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"

MASTER_PORT_PRUNE="${MASTER_PORT_PRUNE:-29720}"
MASTER_PORT_EVAL="${MASTER_PORT_EVAL:-29730}"

mkdir -p "$REPORT_DIR" "$REPO_ROOT/pruned_checkpoint"

next_eval_port() {
  local port="$MASTER_PORT_EVAL"
  MASTER_PORT_EVAL=$((MASTER_PORT_EVAL + 1))
  echo "$port"
}

preflight() {
  [[ -f "$BLIP2_PRETRAINED" ]] || { echo "[FATAL] BLIP2_PRETRAINED not found: $BLIP2_PRETRAINED" >&2; exit 1; }
  [[ -f "$CC3M_TEMPLATE_CFG" ]] || { echo "[FATAL] CC3M template cfg not found: $CC3M_TEMPLATE_CFG" >&2; exit 1; }
  [[ -f "$CC3M_JSON" ]] || { echo "[FATAL] CC3M_JSON not found: $CC3M_JSON" >&2; exit 1; }
  [[ -d "$CC3M_IMAGES_DIR" ]] || { echo "[FATAL] CC3M_IMAGES_DIR not found: $CC3M_IMAGES_DIR" >&2; exit 1; }
  [[ -d "${BERT_BASE_UNCASED_SNAPSHOT:-}" ]] || { echo "[FATAL] bert-base-uncased snapshot not found. Set HF_HOME or BERT_BASE_UNCASED_SNAPSHOT." >&2; exit 1; }
  [[ -d "${FLAN_T5_XL_SNAPSHOT:-}" ]] || { echo "[FATAL] flan-t5-xl snapshot not found. Set HF_HOME or FLAN_T5_XL_SNAPSHOT." >&2; exit 1; }

  if (( NUM_DATA % BS != 0 )); then
    echo "[FATAL] NUM_DATA ($NUM_DATA) must be divisible by BS ($BS)." >&2
    exit 1
  fi

  if [[ "$RUN_EVAL" == "1" ]]; then
    [[ -d "$MMBENCH_ROOT" ]] || { echo "[FATAL] MMBENCH_ROOT not found: $MMBENCH_ROOT" >&2; exit 1; }
    [[ -d "$MMMU_ROOT" ]] || { echo "[FATAL] MMMU_ROOT not found: $MMMU_ROOT" >&2; exit 1; }
    [[ -f "$OKVQA_EVAL_CFG" ]] || { echo "[FATAL] OKVQA_EVAL_CFG not found: $OKVQA_EVAL_CFG" >&2; exit 1; }
    [[ -f "$MATHVISTA_EVAL_JSON" || "${SKIP_MATHVISTA:-0}" == "1" ]] || { echo "[FATAL] MATHVISTA_EVAL_JSON not found: $MATHVISTA_EVAL_JSON" >&2; exit 1; }
    [[ -d "$MATHVISTA_IMAGES_DIR" || "${SKIP_MATHVISTA:-0}" == "1" ]] || { echo "[FATAL] MATHVISTA_IMAGES_DIR not found: $MATHVISTA_IMAGES_DIR" >&2; exit 1; }
  fi
}

materialize_cc3m_cfg() {
  python scripts/blip2/materialize_cc3m_calib_cfg.py \
    --src_cfg "$CC3M_TEMPLATE_CFG" \
    --out_cfg "$CC3M_RUNTIME_CFG" \
    --cc3m_json "$CC3M_JSON" \
    --cc3m_images_dir "$CC3M_IMAGES_DIR" \
    --pretrained "$BLIP2_PRETRAINED"
}

prune_atv_cc3m() {
  echo ""
  echo "========================================================================"
  echo "[PRUNE] CC3M multimodal calibration -> ATV, LLM-only"
  echo "[PRUNE] job_id=$JOB_ID"
  echo "[PRUNE] ckpt=$CKPT"
  echo "========================================================================"

  LAVIS_ATV_DIAGNOSTIC_DIR="${LAVIS_ATV_DIAGNOSTIC_DIR:-$REPORT_DIR}" \
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$MASTER_PORT_PRUNE" evaluate_blip.py \
    --cfg-path "$CC3M_RUNTIME_CFG" \
    --options "model.pretrained=${BLIP2_PRETRAINED}" \
    --prune_calib_mode multimodal \
    --importance_scope llm_only \
    --pruning_method blipt5_atv_pruner \
    --atv_alpha "$ATV_ALPHA" \
    --t5_prune_spec "$T5_SPEC" \
    --num_data "$NUM_DATA" \
    --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
    --prunining_dataset_batch_size "$BS" \
    --job_id "$JOB_ID" \
    --save_pruned_model 2>&1 | tee "$PRUNE_LOG"

  [[ -f "$CKPT" ]] || { echo "[FATAL] pruning finished but checkpoint is missing: $CKPT" >&2; exit 1; }
  echo "[OK] saved checkpoint: $CKPT"
}

eval_fourbench() {
  local eval_tag="atv_cc3m_joint_llmonly_${JOB_STAMP}_seed${SEED}"
  local okvqa_job="okvqa_eval_${eval_tag}_fullval"
  local -a suites=("${eval_tag}:${okvqa_job}")

  [[ -f "$CKPT" ]] || { echo "[FATAL] eval checkpoint not found: $CKPT" >&2; exit 1; }
  export LAVIS_EVAL_CALIB_TAG="$eval_tag"
  export LAVIS_METRICS_JSONL="$METRICS_JSONL"
  : > "$LAVIS_METRICS_JSONL"

  echo ""
  echo "========================================================================"
  echo "[EVAL] four benchmarks for $eval_tag"
  echo "========================================================================"

  if [[ "${SKIP_MMBENCH:-0}" != "1" ]]; then
    echo ">>> MMBench"
    export LAVIS_METRICS_BENCHMARK="MMBench"
    python scripts/blip2/mmmu_eval_by_discipline.py \
      --mmmu_root "$MMBENCH_ROOT" \
      --split "$MMBENCH_SPLIT" \
      --ckpt "$CKPT" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda \
      --overall_only
  fi

  if [[ "${SKIP_OKVQA:-0}" != "1" ]]; then
    echo ">>> OKVQA full val"
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$(next_eval_port)" evaluate_blip.py \
      --cfg-path "$OKVQA_EVAL_CFG" \
      --options "run.num_workers=${OKVQA_EVAL_NUM_WORKERS}" \
      --t5_pruned_checkpoint "$CKPT" \
      --vit_pruned_checkpoint "$CKPT" \
      --job_id "$okvqa_job"
  fi

  if [[ "${SKIP_MMMU:-0}" != "1" ]]; then
    echo ">>> MMMU"
    export LAVIS_METRICS_BENCHMARK="MMMU"
    python scripts/blip2/mmmu_eval_by_discipline.py \
      --mmmu_root "$MMMU_ROOT" \
      --split "$MMMU_SPLIT" \
      --ckpt "$CKPT" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda \
      --overall_only
  fi

  if [[ "${SKIP_MATHVISTA:-0}" != "1" ]]; then
    echo ">>> MathVista MC"
    export LAVIS_METRICS_BENCHMARK="MathVista_MC"
    python scripts/blip2/mathvista_mc_eval.py \
      --eval_json "$MATHVISTA_EVAL_JSON" \
      --images_dir "$MATHVISTA_IMAGES_DIR" \
      --ckpt "$CKPT" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda
  fi

  python scripts/blip2/collect_lavisbackup_eval_summary.py \
    --repo-root "$REPO_ROOT" \
    --metrics-jsonl "$LAVIS_METRICS_JSONL" \
    --out-md "$SUMMARY_MD" \
    --out-tsv "$SUMMARY_TSV" \
    --suites "${suites[@]}"
}

echo "========================================================================"
echo "ATV CC3M multimodal calibration, LLM-only pruning + four evals"
echo "========================================================================"
echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] BASE=$BASE"
echo "[INFO] CC3M_JSON=$CC3M_JSON"
echo "[INFO] CC3M_IMAGES_DIR=$CC3M_IMAGES_DIR"
echo "[INFO] BLIP2_PRETRAINED=$BLIP2_PRETRAINED"
echo "[INFO] T5_SPEC=$T5_SPEC T5_SPARSITY_TARGET=$T5_SPARSITY_TARGET ATV_ALPHA=$ATV_ALPHA NUM_DATA=$NUM_DATA BS=$BS"
echo "[INFO] RUN_PRUNE=$RUN_PRUNE RUN_EVAL=$RUN_EVAL"
echo "[INFO] REPORT_DIR=$REPORT_DIR"
echo "========================================================================"

preflight
materialize_cc3m_cfg

if [[ "$RUN_PRUNE" == "1" ]]; then
  prune_atv_cc3m
else
  echo "[INFO] RUN_PRUNE=0, reuse checkpoint: $CKPT"
  [[ -f "$CKPT" ]] || { echo "[FATAL] checkpoint missing for eval-only run: $CKPT" >&2; exit 1; }
fi

if [[ "$RUN_SPARSITY_CHECK" == "1" ]]; then
  python scripts/blip2/check_ckpt_sparsity.py \
    --ckpt "$CKPT" \
    --tag "$JOB_ID" \
    --expect_t5 "$T5_SPARSITY_TARGET" \
    --tol 0.05 \
    --out_csv "$SPARSITY_CSV"
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  eval_fourbench
else
  echo "[INFO] RUN_EVAL=0, skip evaluation."
fi

echo ""
echo "[OK] checkpoint: $CKPT"
echo "[OK] metrics:    $METRICS_JSONL"
echo "[OK] summary:    $SUMMARY_TSV"
echo "[OK] report dir: $REPORT_DIR"
