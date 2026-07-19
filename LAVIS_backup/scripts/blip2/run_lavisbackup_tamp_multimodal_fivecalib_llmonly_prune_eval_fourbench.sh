#!/usr/bin/env bash
# =============================================================================
# TAMP multimodal calibration with LLM-only pruning, then four evals.
#
# Calibration sets:
#   mmbench / mmmu / okvqa / mathvista / cc3m
#
# For each calibration set this script:
#   1. Runs BLIP2-T5 calibration with the original multimodal JSON + images.
#      Optional: HYBRID_C4_TEXT=1 pairs each calibration image with C4 text.
#   2. Uses --pruning_method blipt5_tamp_pruner in multimodal mode.
#   3. Prunes only the T5/LLM side with --importance_scope llm_only.
#   4. Saves one full-model checkpoint.
#   5. Evaluates the checkpoint on MMBench / OKVQA / MMMU / MathVista.
#
# Usage:
#   cd /data/data2/mfs/2/LAVIS_backup
#   bash scripts/blip2/run_lavisbackup_tamp_multimodal_fivecalib_llmonly_prune_eval_fourbench.sh
#
# Run only a subset:
#   CALIBS="mmbench cc3m" bash scripts/blip2/run_lavisbackup_tamp_multimodal_fivecalib_llmonly_prune_eval_fourbench.sh
#
# Prune only / eval only:
#   RUN_EVAL=0 bash scripts/blip2/run_lavisbackup_tamp_multimodal_fivecalib_llmonly_prune_eval_fourbench.sh
#   RUN_PRUNE=0 JOB_STAMP=20260720_000000 bash scripts/blip2/run_lavisbackup_tamp_multimodal_fivecalib_llmonly_prune_eval_fourbench.sh
#
# Common overrides:
#   BASE=/data/data2/mfs
#   NUM_DATA=128 BS=8 CALIB_SEEDS=42
#   T5_SPEC=24-0.5-1.0-1.0
#   HYBRID_C4_TEXT=1 C4_JSON=/data/data2/mfs/c4_calib_128.json
#   MMBENCH_ROOT=... MMMU_ROOT=... MATHVISTA_EVAL_JSON=... MATHVISTA_IMAGES_DIR=...
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
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
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

if [[ -z "${EVA_VIT_G_PTH:-}" ]]; then
  if [[ -f "${TORCH_HOME}/hub/checkpoints/eva_vit_g.pth" ]]; then
    export EVA_VIT_G_PTH="${TORCH_HOME}/hub/checkpoints/eva_vit_g.pth"
  elif [[ -f "${HOME}/.cache/torch/hub/checkpoints/eva_vit_g.pth" ]]; then
    export EVA_VIT_G_PTH="${HOME}/.cache/torch/hub/checkpoints/eva_vit_g.pth"
  fi
fi

JOB_STAMP="${JOB_STAMP:-$(date +%Y%m%d_%H%M%S)}"
CALIBS="${CALIBS:-mmbench mmmu okvqa mathvista cc3m}"
CALIB_SEEDS="${CALIB_SEEDS:-${SEEDS:-42}}"

RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

NUM_DATA="${NUM_DATA:-128}"
NUM_DATA_FIRST_STAGE="${NUM_DATA_FIRST_STAGE:-$NUM_DATA}"
BS="${BS:-8}"
T5_SPEC="${T5_SPEC:-24-0.5-1.0-1.0}"
PRUNE_METHOD="${PRUNE_METHOD:-blipt5_tamp_pruner}"
HYBRID_C4_TEXT="${HYBRID_C4_TEXT:-0}"
C4_JSON="${C4_JSON:-$BASE/c4_calib_128.json}"
HYBRID_DIR="${HYBRID_DIR:-$BASE/hybrid_c4_multimodal_calib_${NUM_DATA}}"
FORCE_HYBRID_BUILD="${FORCE_HYBRID_BUILD:-0}"
HYBRID_METADATA="${HYBRID_METADATA:-1}"

MMBENCH_ROOT="${MMBENCH_ROOT:-$BASE/MMBench_eval}"
MMMU_ROOT="${MMMU_ROOT:-$BASE/MMMU_single_image}"
MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
MMMU_SPLIT="${MMMU_SPLIT:-test}"
MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-$BASE/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
MATHVISTA_IMAGES_DIR="${MATHVISTA_IMAGES_DIR:-$BASE/MathVista_eval_testmini_mc/images}"
OKVQA_EVAL_CFG="${OKVQA_EVAL_CFG:-$REPO_ROOT/lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml}"
OKVQA_EVAL_NUM_WORKERS="${OKVQA_EVAL_NUM_WORKERS:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
MASTER_PORT="${MASTER_PORT:-29840}"

SUMMARY_DIR="${SUMMARY_DIR:-$REPO_ROOT/lavis/output/BLIP2}"
LOG_DIR="${LOG_DIR:-$SUMMARY_DIR/tamp_multimodal_llmonly_${JOB_STAMP}}"
mkdir -p "$REPO_ROOT/pruned_checkpoint" "$SUMMARY_DIR" "$LOG_DIR" "$HYBRID_DIR" "$LOG_DIR/hybrid_cfgs"

preflight() {
  [[ -f "$BLIP2_PRETRAINED" ]] || { echo "[FATAL] BLIP2_PRETRAINED not found: $BLIP2_PRETRAINED" >&2; exit 1; }
  [[ -d "${BERT_BASE_UNCASED_SNAPSHOT:-}" ]] || { echo "[FATAL] bert-base-uncased snapshot not found. Set HF_HOME or BERT_BASE_UNCASED_SNAPSHOT." >&2; exit 1; }
  [[ -d "${FLAN_T5_XL_SNAPSHOT:-}" ]] || { echo "[FATAL] flan-t5-xl snapshot not found. Set HF_HOME or FLAN_T5_XL_SNAPSHOT." >&2; exit 1; }
  [[ -n "${EVA_VIT_G_PTH:-}" && -f "$EVA_VIT_G_PTH" ]] || { echo "[FATAL] EVA_VIT_G_PTH not found. Set EVA_VIT_G_PTH." >&2; exit 1; }

  if (( NUM_DATA % BS != 0 )); then
    echo "[FATAL] NUM_DATA ($NUM_DATA) must be divisible by BS ($BS)." >&2
    exit 1
  fi

  for calib in $CALIBS; do
    local cfg
    cfg="$(calib_cfg "$calib")"
    [[ -f "$cfg" ]] || { echo "[FATAL] calibration cfg not found for $calib: $cfg" >&2; exit 1; }
    if [[ "$HYBRID_C4_TEXT" == "1" ]]; then
      local raw_json images_dir
      raw_json="$(calib_raw_json "$calib")"
      images_dir="$(calib_images_dir "$calib")"
      [[ -f "$raw_json" ]] || { echo "[FATAL] source image JSON not found for $calib: $raw_json" >&2; exit 1; }
      [[ -d "$images_dir" ]] || { echo "[FATAL] source images dir not found for $calib: $images_dir" >&2; exit 1; }
    fi
  done
  if [[ "$HYBRID_C4_TEXT" == "1" ]]; then
    [[ -f "$C4_JSON" ]] || { echo "[FATAL] C4_JSON not found: $C4_JSON" >&2; exit 1; }
  fi

  if [[ "$RUN_EVAL" == "1" ]]; then
    [[ -d "$MMBENCH_ROOT" ]] || { echo "[FATAL] MMBENCH_ROOT not found: $MMBENCH_ROOT" >&2; exit 1; }
    [[ -d "$MMMU_ROOT" ]] || { echo "[FATAL] MMMU_ROOT not found: $MMMU_ROOT" >&2; exit 1; }
    [[ -f "$OKVQA_EVAL_CFG" ]] || { echo "[FATAL] OKVQA_EVAL_CFG not found: $OKVQA_EVAL_CFG" >&2; exit 1; }
    [[ -f "$MATHVISTA_EVAL_JSON" || "${SKIP_MATHVISTA:-0}" == "1" ]] || { echo "[FATAL] MATHVISTA_EVAL_JSON not found: $MATHVISTA_EVAL_JSON" >&2; exit 1; }
    [[ -d "$MATHVISTA_IMAGES_DIR" || "${SKIP_MATHVISTA:-0}" == "1" ]] || { echo "[FATAL] MATHVISTA_IMAGES_DIR not found: $MATHVISTA_IMAGES_DIR" >&2; exit 1; }
  fi
}

calib_label() {
  case "$1" in
    mmbench) echo "MMBench" ;;
    mmmu) echo "MMMU" ;;
    okvqa) echo "OKVQA" ;;
    mathvista) echo "MathVista" ;;
    cc3m) echo "CC3M" ;;
    *) echo "[FATAL] unknown calibration set: $1" >&2; exit 1 ;;
  esac
}

calib_cfg() {
  case "$1" in
    mmbench) echo "$REPO_ROOT/lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_mmbench.yaml" ;;
    mmmu) echo "$REPO_ROOT/lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_mmmu_overall.yaml" ;;
    okvqa) echo "$REPO_ROOT/lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_okvqa_train_overall.yaml" ;;
    mathvista) echo "$REPO_ROOT/lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_mathvista.yaml" ;;
    cc3m) echo "$REPO_ROOT/lavis/projects/blip2/eval/cc_prefix_derivative_compute_cc3m_calib128.yaml" ;;
    *) echo "[FATAL] unknown calibration set: $1" >&2; exit 1 ;;
  esac
}

calib_raw_json() {
  local calib="$1"
  local var="RAW_${calib^^}"
  if [[ -n "${!var:-}" ]]; then
    echo "${!var}"
    return 0
  fi
  case "$calib" in
    mmbench) echo "$BASE/MMBench_calibration/mmbench_calibration_train.json" ;;
    mmmu) echo "$BASE/MMMU_calibration/mmmu_calibration_train.json" ;;
    okvqa) echo "$BASE/datasets/okvqa/annotations/okvqa_train.json" ;;
    mathvista) echo "$BASE/MathVista_calibration/mathvista_calibration_train.json" ;;
    cc3m) echo "$BASE/CC3M_calib_128/cc3m_calib_128.json" ;;
    *) echo "[FATAL] unknown calibration set: $calib" >&2; exit 1 ;;
  esac
}

calib_images_dir() {
  local calib="$1"
  local var="IMAGES_${calib^^}"
  if [[ -n "${!var:-}" ]]; then
    echo "${!var}"
    return 0
  fi
  case "$calib" in
    mmbench) echo "$BASE/MMBench_calibration/images" ;;
    mmmu) echo "$BASE/MMMU_calibration/images" ;;
    okvqa) echo "$BASE/datasets/okvqa" ;;
    mathvista) echo "$BASE/MathVista_calibration/images/images" ;;
    cc3m) echo "$BASE/CC3M_calib_128/images" ;;
    *) echo "[FATAL] unknown calibration set: $calib" >&2; exit 1 ;;
  esac
}

mode_prefix() {
  if [[ "$HYBRID_C4_TEXT" == "1" ]]; then
    echo "tamp_mm_c4txt_llmonly"
  else
    echo "tamp_mm_llmonly"
  fi
}

job_id_for() {
  local calib="$1"
  local seed="$2"
  local label
  label="$(calib_label "$calib")"
  echo "$(mode_prefix)_${label}_${JOB_STAMP}_seed${seed}"
}

ckpt_for() {
  local calib="$1"
  local seed="$2"
  echo "$REPO_ROOT/pruned_checkpoint/$(job_id_for "$calib" "$seed").pth"
}

next_port() {
  local port="$MASTER_PORT"
  MASTER_PORT=$((MASTER_PORT + 1))
  echo "$port"
}

prepare_hybrid_cfg() {
  local calib="$1"
  local seed="$2"
  local label raw_json images_dir src_cfg hybrid_json hybrid_cfg
  label="$(calib_label "$calib")"
  raw_json="$(calib_raw_json "$calib")"
  images_dir="$(calib_images_dir "$calib")"
  src_cfg="$(calib_cfg "$calib")"
  hybrid_json="$HYBRID_DIR/${calib}_images_c4_text_${NUM_DATA}_seed${seed}.json"
  hybrid_cfg="$LOG_DIR/hybrid_cfgs/${calib}_images_c4_text_${NUM_DATA}_seed${seed}.yaml"

  if [[ ! -f "$hybrid_json" || "$FORCE_HYBRID_BUILD" == "1" ]]; then
    local -a metadata_args=()
    if [[ "$HYBRID_METADATA" == "1" ]]; then
      metadata_args+=(--metadata)
    fi
    echo "[HYBRID] build $label image + C4 text JSON: $hybrid_json"
    python "$SCRIPT_DIR/build_hybrid_c4_multimodal_calib.py" \
      --image_json "$raw_json" \
      --c4_json "$C4_JSON" \
      --output "$hybrid_json" \
      --num "$NUM_DATA" \
      --seed "$seed" \
      "${metadata_args[@]}"
  else
    echo "[HYBRID] reuse JSON: $hybrid_json"
  fi

  if [[ ! -f "$hybrid_cfg" || "$FORCE_HYBRID_BUILD" == "1" ]]; then
    echo "[HYBRID] materialize cfg: $hybrid_cfg"
    python "$SCRIPT_DIR/materialize_multimodal_calib_cfg.py" \
      --src_cfg "$src_cfg" \
      --out_cfg "$hybrid_cfg" \
      --annotation_json "$hybrid_json" \
      --images_dir "$images_dir" \
      --pretrained "$BLIP2_PRETRAINED" \
      --run_seed "$seed"
  else
    echo "[HYBRID] reuse cfg: $hybrid_cfg"
  fi

  ACTIVE_PRUNE_CFG="$hybrid_cfg"
}

prune_one() {
  local calib="$1"
  local seed="$2"
  local cfg job_id ckpt port
  cfg="$(calib_cfg "$calib")"
  if [[ "$HYBRID_C4_TEXT" == "1" ]]; then
    prepare_hybrid_cfg "$calib" "$seed"
    cfg="$ACTIVE_PRUNE_CFG"
  fi
  job_id="$(job_id_for "$calib" "$seed")"
  ckpt="$(ckpt_for "$calib" "$seed")"
  port="$(next_port)"

  export LAVIS_DISTRIBUTED_SAMPLER_SEED="$seed"
  echo ""
  echo "========================================================================"
  echo "[PRUNE] calib=$calib seed=$seed"
  echo "[PRUNE] cfg=$cfg"
  if [[ "$HYBRID_C4_TEXT" == "1" ]]; then
    echo "[PRUNE] hybrid text=$C4_JSON"
  fi
  echo "[PRUNE] checkpoint=$ckpt"
  echo "========================================================================"

  python -m torch.distributed.run --nproc_per_node=1 --master_port="$port" evaluate_blip.py \
    --cfg-path "$cfg" \
    --options "model.pretrained=${BLIP2_PRETRAINED}" \
    --prune_calib_mode multimodal \
    --importance_scope llm_only \
    --pruning_method "$PRUNE_METHOD" \
    --t5_prune_spec "$T5_SPEC" \
    --num_data "$NUM_DATA" \
    --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
    --prunining_dataset_batch_size "$BS" \
    --job_id "$job_id" \
    --save_pruned_model

  [[ -f "$ckpt" ]] || { echo "[FATAL] pruning finished but checkpoint is missing: $ckpt" >&2; exit 1; }
  echo "[OK] saved checkpoint: $ckpt"
}

eval_fourbench_one() {
  local calib="$1"
  local seed="$2"
  local label ckpt eval_tag okvqa_job
  label="$(calib_label "$calib")"
  ckpt="$(ckpt_for "$calib" "$seed")"
  eval_tag="$(mode_prefix)_${label}_${JOB_STAMP}_seed${seed}"
  okvqa_job="okvqa_eval_${eval_tag}_fullval"

  [[ -f "$ckpt" ]] || { echo "[FATAL] eval checkpoint not found: $ckpt" >&2; exit 1; }

  export LAVIS_EVAL_CALIB_TAG="$eval_tag"
  echo ""
  echo "========================================================================"
  echo "[EVAL] calib=$calib seed=$seed tag=$eval_tag"
  echo "[EVAL] checkpoint=$ckpt"
  echo "========================================================================"

  if [[ "${SKIP_MMBENCH:-0}" != "1" ]]; then
    echo ""
    echo ">>> [$eval_tag] MMBench"
    export LAVIS_METRICS_BENCHMARK="MMBench"
    python scripts/blip2/mmmu_eval_by_discipline.py \
      --mmmu_root "$MMBENCH_ROOT" \
      --split "$MMBENCH_SPLIT" \
      --ckpt "$ckpt" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda \
      --overall_only
  fi

  if [[ "${SKIP_OKVQA:-0}" != "1" ]]; then
    echo ""
    echo ">>> [$eval_tag] OKVQA full val"
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$(next_port)" evaluate_blip.py \
      --cfg-path "$OKVQA_EVAL_CFG" \
      --options "run.num_workers=${OKVQA_EVAL_NUM_WORKERS}" \
      --t5_pruned_checkpoint "$ckpt" \
      --vit_pruned_checkpoint "$ckpt" \
      --job_id "$okvqa_job"
  fi

  if [[ "${SKIP_MMMU:-0}" != "1" ]]; then
    echo ""
    echo ">>> [$eval_tag] MMMU"
    export LAVIS_METRICS_BENCHMARK="MMMU"
    python scripts/blip2/mmmu_eval_by_discipline.py \
      --mmmu_root "$MMMU_ROOT" \
      --split "$MMMU_SPLIT" \
      --ckpt "$ckpt" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda \
      --overall_only
  fi

  if [[ "${SKIP_MATHVISTA:-0}" != "1" ]]; then
    echo ""
    echo ">>> [$eval_tag] MathVista MC"
    export LAVIS_METRICS_BENCHMARK="MathVista_MC"
    python scripts/blip2/mathvista_mc_eval.py \
      --eval_json "$MATHVISTA_EVAL_JSON" \
      --images_dir "$MATHVISTA_IMAGES_DIR" \
      --ckpt "$ckpt" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda
  fi
}

collect_summary() {
  local metrics_jsonl="$1"
  local summary_md="$2"
  local summary_tsv="$3"
  shift 3
  local -a suites=("$@")

  if [[ "$RUN_EVAL" != "1" || "${#suites[@]}" -eq 0 ]]; then
    return 0
  fi

  echo ""
  echo "========================================================================"
  echo "[SUMMARY] writing four-benchmark table"
  echo "========================================================================"
  python "$SCRIPT_DIR/collect_lavisbackup_eval_summary.py" \
    --repo-root "$REPO_ROOT" \
    --metrics-jsonl "$metrics_jsonl" \
    --out-md "$summary_md" \
    --out-tsv "$summary_tsv" \
    --suites "${suites[@]}"
  echo "[OK] summary md:  $summary_md"
  echo "[OK] summary tsv: $summary_tsv"
}

echo "========================================================================"
echo "TAMP multimodal calibration, LLM-only pruning + four-benchmark evaluation"
echo "========================================================================"
echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] BASE=$BASE"
echo "[INFO] BLIP2_PRETRAINED=$BLIP2_PRETRAINED"
echo "[INFO] EVA_VIT_G_PTH=${EVA_VIT_G_PTH:-}"
echo "[INFO] BERT_BASE_UNCASED_SNAPSHOT=${BERT_BASE_UNCASED_SNAPSHOT:-}"
echo "[INFO] FLAN_T5_XL_SNAPSHOT=${FLAN_T5_XL_SNAPSHOT:-}"
echo "[INFO] CALIBS=$CALIBS"
echo "[INFO] CALIB_SEEDS=$CALIB_SEEDS"
echo "[INFO] NUM_DATA=$NUM_DATA BS=$BS T5_SPEC=$T5_SPEC"
echo "[INFO] HYBRID_C4_TEXT=$HYBRID_C4_TEXT C4_JSON=$C4_JSON HYBRID_DIR=$HYBRID_DIR"
echo "[INFO] RUN_PRUNE=$RUN_PRUNE RUN_EVAL=$RUN_EVAL"
echo "[INFO] JOB_STAMP=$JOB_STAMP"
echo "========================================================================"

preflight

RUN_PREFIX="$(mode_prefix)"
METRICS_JSONL="$SUMMARY_DIR/${RUN_PREFIX}_fivecalib_fourbench_${JOB_STAMP}.jsonl"
SUMMARY_MD="$SUMMARY_DIR/${RUN_PREFIX}_fivecalib_fourbench_${JOB_STAMP}.md"
SUMMARY_TSV="$SUMMARY_DIR/${RUN_PREFIX}_fivecalib_fourbench_${JOB_STAMP}.tsv"
export LAVIS_METRICS_JSONL="$METRICS_JSONL"
: > "$LAVIS_METRICS_JSONL"

declare -a SUITE_SPECS=()

for seed in $CALIB_SEEDS; do
  export LAVIS_DISTRIBUTED_SAMPLER_SEED="$seed"
  for calib in $CALIBS; do
    label="$(calib_label "$calib")"
    eval_tag="$(mode_prefix)_${label}_${JOB_STAMP}_seed${seed}"
    okvqa_job="okvqa_eval_${eval_tag}_fullval"
    SUITE_SPECS+=("${eval_tag}:${okvqa_job}")

    if [[ "$RUN_PRUNE" == "1" ]]; then
      prune_one "$calib" "$seed"
    else
      ckpt="$(ckpt_for "$calib" "$seed")"
      echo "[INFO] RUN_PRUNE=0, reuse checkpoint: $ckpt"
      [[ -f "$ckpt" ]] || { echo "[FATAL] checkpoint missing for eval-only run: $ckpt" >&2; exit 1; }
    fi

    if [[ "$RUN_EVAL" == "1" ]]; then
      eval_fourbench_one "$calib" "$seed"
    fi
  done
done

collect_summary "$METRICS_JSONL" "$SUMMARY_MD" "$SUMMARY_TSV" "${SUITE_SPECS[@]}"

echo ""
echo "========================================================================"
echo "[DONE] TAMP multimodal LLM-only run completed."
echo "[DONE] checkpoints: $REPO_ROOT/pruned_checkpoint/${RUN_PREFIX}_*_${JOB_STAMP}_seed*.pth"
echo "[DONE] metrics:     $METRICS_JSONL"
echo "[DONE] summary:     $SUMMARY_TSV"
echo "========================================================================"
