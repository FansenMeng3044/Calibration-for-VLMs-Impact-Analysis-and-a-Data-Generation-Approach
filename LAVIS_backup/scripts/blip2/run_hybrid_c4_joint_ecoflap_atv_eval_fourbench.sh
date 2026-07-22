#!/usr/bin/env bash
# =============================================================================
# Hybrid C4 multimodal calibration -> joint ECoFLaP/ATV pruning -> four evals.
#
# Hybrid calibration means:
#   image side: MMBench / MMMU / OKVQA / MathVista / CC3M calibration images
#   text side : C4 text
#
# The prebuilt hybrid JSON files are expected under:
#   /data/data2/mfs/hybrid_c4_multimodal_calib_128/
#
# For each calibration set and each method, this script:
#   1. materializes a runtime multimodal YAML pointing to the hybrid JSON;
#   2. runs one joint multimodal pruning pass, pruning both T5 and ViT;
#   3. saves one full BLIP2-T5 checkpoint;
#   4. evaluates it on MMBench / OKVQA / MMMU / MathVista.
#
# Usage:
#   cd /data/data2/mfs/2/LAVIS_backup
#   bash scripts/blip2/run_hybrid_c4_joint_ecoflap_atv_eval_fourbench.sh
#
# Common overrides:
#   CALIBS="mmbench cc3m" METHODS="ecoflap atv"
#   RUN_EVAL=0
#   RUN_PRUNE=0 JOB_STAMP=20260722_000000
#   NUM_DATA=128 BS=8 T5_SPEC=24-0.5-1.0-1.0 VIT_SPEC=39-0.5-1.0-1.0
#   ECOFLAP_SCORE_METHOD=MEZO-GradOnly_sum ECOFLAP_SPARSITY_RATIO_GRANULARITY=block
#   ATV_ALPHA=1.0
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

if [[ -z "${EVA_VIT_G_PTH:-}" ]]; then
  if [[ -f "$TORCH_HOME/hub/checkpoints/eva_vit_g.pth" ]]; then
    export EVA_VIT_G_PTH="$TORCH_HOME/hub/checkpoints/eva_vit_g.pth"
  elif [[ -f "$MODEL_CACHE_ROOT/torch/hub/checkpoints/eva_vit_g.pth" ]]; then
    export EVA_VIT_G_PTH="$MODEL_CACHE_ROOT/torch/hub/checkpoints/eva_vit_g.pth"
  elif [[ -f "${HOME}/.cache/torch/hub/checkpoints/eva_vit_g.pth" ]]; then
    export EVA_VIT_G_PTH="${HOME}/.cache/torch/hub/checkpoints/eva_vit_g.pth"
  fi
fi

BLIP2_PRETRAINED="${BLIP2_PRETRAINED:-${MODEL_CACHE_ROOT}/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
HYBRID_DIR="${HYBRID_DIR:-$BASE/hybrid_c4_multimodal_calib_128}"
TEMPLATE_CFG="${TEMPLATE_CFG:-$REPO_ROOT/lavis/projects/blip2/eval/cc_prefix_derivative_compute_cc3m_calib128.yaml}"

CALIBS="${CALIBS:-mmbench mmmu okvqa mathvista cc3m}"
METHODS="${METHODS:-ecoflap atv}"
JOB_STAMP="${JOB_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
export LAVIS_DISTRIBUTED_SAMPLER_SEED="${LAVIS_DISTRIBUTED_SAMPLER_SEED:-$SEED}"

NUM_DATA="${NUM_DATA:-128}"
NUM_DATA_FIRST_STAGE="${NUM_DATA_FIRST_STAGE:-$NUM_DATA}"
BS="${BS:-8}"
T5_SPEC="${T5_SPEC:-24-0.5-1.0-1.0}"
VIT_SPEC="${VIT_SPEC:-39-0.5-1.0-1.0}"
ECOFLAP_SCORE_METHOD="${ECOFLAP_SCORE_METHOD:-MEZO-GradOnly_sum}"
ECOFLAP_SPARSITY_RATIO_GRANULARITY="${ECOFLAP_SPARSITY_RATIO_GRANULARITY:-block}"
ATV_ALPHA="${ATV_ALPHA:-1.0}"

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
VIT_SPARSITY_MAX="${VIT_SPARSITY_MAX:-$(python - "$VIT_SPEC" <<'PY'
import sys
parts = sys.argv[1].split("-")
try:
    keep_ratio = float(parts[1])
except (IndexError, ValueError) as exc:
    raise SystemExit("cannot parse VIT_SPEC keep ratio from %r" % sys.argv[1]) from exc
print("%.6f" % min(1.0, 1.0 - keep_ratio + 0.08))
PY
)}"
MAX_SPARSITY_PER_LAYER="${MAX_SPARSITY_PER_LAYER:-$(python - "$T5_SPEC" <<'PY'
import sys
parts = sys.argv[1].split("-")
try:
    keep_ratio = float(parts[1])
except (IndexError, ValueError) as exc:
    raise SystemExit("cannot parse T5_SPEC keep ratio from %r" % sys.argv[1]) from exc
print("%.6f" % min(1.0, 1.0 - keep_ratio + 0.1))
PY
)}"

RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_SPARSITY_CHECK="${RUN_SPARSITY_CHECK:-1}"

MMBENCH_ROOT="${MMBENCH_ROOT:-$BASE/MMBench_eval}"
MMMU_ROOT="${MMMU_ROOT:-$BASE/MMMU_single_image}"
MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
MMMU_SPLIT="${MMMU_SPLIT:-test}"
OKVQA_EVAL_CFG="${OKVQA_EVAL_CFG:-$REPO_ROOT/lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml}"
OKVQA_EVAL_NUM_WORKERS="${OKVQA_EVAL_NUM_WORKERS:-0}"
MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-$BASE/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
MATHVISTA_IMAGES_DIR="${MATHVISTA_IMAGES_DIR:-$BASE/MathVista_eval_testmini_mc/images}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"

MASTER_PORT="${MASTER_PORT:-29980}"
OUT_ROOT="${OUT_ROOT:-$BASE/hybrid_c4_joint_ecoflap_atv_${JOB_STAMP}_seed${SEED}}"
CFG_DIR="$OUT_ROOT/runtime_cfgs"
LOG_DIR="$OUT_ROOT/logs"
SUMMARY_DIR="$OUT_ROOT/summaries"
mkdir -p "$OUT_ROOT" "$CFG_DIR" "$LOG_DIR" "$SUMMARY_DIR" "$REPO_ROOT/pruned_checkpoint"

next_port() {
  local port="$MASTER_PORT"
  MASTER_PORT=$((MASTER_PORT + 1))
  echo "$port"
}

method_to_pruner() {
  case "$1" in
    ecoflap) echo "blipt5_wanda_pruner" ;;
    atv) echo "blipt5_atv_pruner" ;;
    *) echo "[FATAL] unknown method: $1" >&2; exit 1 ;;
  esac
}

method_label() {
  case "$1" in
    ecoflap) echo "ecoflap" ;;
    atv) echo "atv" ;;
    *) echo "[FATAL] unknown method: $1" >&2; exit 1 ;;
  esac
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

hybrid_json_for() {
  local calib="$1"
  local var="HYBRID_${calib^^}_JSON"
  if [[ -n "${!var:-}" ]]; then
    echo "${!var}"
    return 0
  fi

  local direct="$HYBRID_DIR/${calib}_images_c4_text_${NUM_DATA}_seed${SEED}.json"
  if [[ -f "$direct" ]]; then
    echo "$direct"
    return 0
  fi

  local first
  first="$(find "$HYBRID_DIR" -maxdepth 1 -type f \
    \( -name "${calib}_images_c4_text_${NUM_DATA}_seed*.json" -o -name "${calib}*c4*${NUM_DATA}*.json" \) \
    2>/dev/null | sort | head -1)"
  if [[ -n "$first" ]]; then
    echo "$first"
    return 0
  fi

  echo "$direct"
}

job_id_for() {
  local method="$1" calib="$2"
  echo "hybrid_c4_joint_$(method_label "$method")_${calib}_${JOB_STAMP}_seed${SEED}"
}

ckpt_for() {
  local method="$1" calib="$2"
  echo "$REPO_ROOT/pruned_checkpoint/$(job_id_for "$method" "$calib").pth"
}

runtime_cfg_for() {
  local method="$1" calib="$2"
  echo "$CFG_DIR/$(job_id_for "$method" "$calib").yaml"
}

preflight() {
  [[ -f "$BLIP2_PRETRAINED" ]] || { echo "[FATAL] BLIP2_PRETRAINED not found: $BLIP2_PRETRAINED" >&2; exit 1; }
  [[ -f "$TEMPLATE_CFG" ]] || { echo "[FATAL] TEMPLATE_CFG not found: $TEMPLATE_CFG" >&2; exit 1; }
  [[ -d "$HYBRID_DIR" ]] || { echo "[FATAL] HYBRID_DIR not found: $HYBRID_DIR" >&2; exit 1; }
  [[ -d "${BERT_BASE_UNCASED_SNAPSHOT:-}" ]] || { echo "[FATAL] bert-base-uncased snapshot not found. Set HF_HOME or BERT_BASE_UNCASED_SNAPSHOT." >&2; exit 1; }
  [[ -d "${FLAN_T5_XL_SNAPSHOT:-}" ]] || { echo "[FATAL] flan-t5-xl snapshot not found. Set HF_HOME or FLAN_T5_XL_SNAPSHOT." >&2; exit 1; }
  [[ -n "${EVA_VIT_G_PTH:-}" && -f "$EVA_VIT_G_PTH" ]] || { echo "[FATAL] EVA_VIT_G_PTH not found. Set EVA_VIT_G_PTH." >&2; exit 1; }

  if (( NUM_DATA % BS != 0 )); then
    echo "[FATAL] NUM_DATA ($NUM_DATA) must be divisible by BS ($BS)." >&2
    exit 1
  fi

  for method in $METHODS; do
    method_to_pruner "$method" >/dev/null
  done

  for calib in $CALIBS; do
    local json images
    json="$(hybrid_json_for "$calib")"
    images="$(calib_images_dir "$calib")"
    [[ -f "$json" ]] || { echo "[FATAL] hybrid JSON not found for $calib: $json" >&2; exit 1; }
    [[ -d "$images" ]] || { echo "[FATAL] images dir not found for $calib: $images" >&2; exit 1; }
  done

  if [[ "$RUN_EVAL" == "1" ]]; then
    [[ -d "$MMBENCH_ROOT" ]] || { echo "[FATAL] MMBENCH_ROOT not found: $MMBENCH_ROOT" >&2; exit 1; }
    [[ -d "$MMMU_ROOT" ]] || { echo "[FATAL] MMMU_ROOT not found: $MMMU_ROOT" >&2; exit 1; }
    [[ -f "$OKVQA_EVAL_CFG" ]] || { echo "[FATAL] OKVQA_EVAL_CFG not found: $OKVQA_EVAL_CFG" >&2; exit 1; }
    [[ -f "$MATHVISTA_EVAL_JSON" || "${SKIP_MATHVISTA:-0}" == "1" ]] || { echo "[FATAL] MATHVISTA_EVAL_JSON not found: $MATHVISTA_EVAL_JSON" >&2; exit 1; }
    [[ -d "$MATHVISTA_IMAGES_DIR" || "${SKIP_MATHVISTA:-0}" == "1" ]] || { echo "[FATAL] MATHVISTA_IMAGES_DIR not found: $MATHVISTA_IMAGES_DIR" >&2; exit 1; }
  fi
}

materialize_cfg() {
  local method="$1" calib="$2"
  local json images cfg
  json="$(hybrid_json_for "$calib")"
  images="$(calib_images_dir "$calib")"
  cfg="$(runtime_cfg_for "$method" "$calib")"

  python scripts/blip2/materialize_multimodal_calib_cfg.py \
    --src_cfg "$TEMPLATE_CFG" \
    --out_cfg "$cfg" \
    --annotation_json "$json" \
    --images_dir "$images" \
    --pretrained "$BLIP2_PRETRAINED" \
    --run_seed "$SEED" >/dev/null

  echo "$cfg"
}

prune_one() {
  local method="$1" calib="$2"
  local pruner job_id ckpt cfg port log
  local -a method_args=()
  pruner="$(method_to_pruner "$method")"
  job_id="$(job_id_for "$method" "$calib")"
  ckpt="$(ckpt_for "$method" "$calib")"
  cfg="$(materialize_cfg "$method" "$calib")"
  port="$(next_port)"
  log="$LOG_DIR/${job_id}_prune.log"

  case "$method" in
    ecoflap)
      method_args=(
        --score_method "$ECOFLAP_SCORE_METHOD"
        --sparsity_ratio_granularity "$ECOFLAP_SPARSITY_RATIO_GRANULARITY"
        --max_sparsity_per_layer "$MAX_SPARSITY_PER_LAYER"
      )
      ;;
    atv)
      method_args=(--atv_alpha "$ATV_ALPHA")
      ;;
  esac

  echo ""
  echo "========================================================================"
  echo "[PRUNE] method=$method calib=$calib"
  echo "[PRUNE] cfg=$cfg"
  echo "[PRUNE] ckpt=$ckpt"
  echo "========================================================================"

  LAVIS_ATV_DIAGNOSTIC_DIR="${LAVIS_ATV_DIAGNOSTIC_DIR:-$OUT_ROOT/atv_diagnostics}" \
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$port" evaluate_blip.py \
    --cfg-path "$cfg" \
    --options "model.pretrained=${BLIP2_PRETRAINED}" \
    --prune_calib_mode multimodal \
    --importance_scope joint \
    --pruning_method "$pruner" \
    --prune_vit \
    --t5_prune_spec "$T5_SPEC" \
    --vit_prune_spec "$VIT_SPEC" \
    --num_data "$NUM_DATA" \
    --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
    --prunining_dataset_batch_size "$BS" \
    "${method_args[@]}" \
    --job_id "$job_id" \
    --save_pruned_model 2>&1 | tee "$log"

  [[ -f "$ckpt" ]] || { echo "[FATAL] pruning finished but checkpoint missing: $ckpt" >&2; exit 1; }
  echo "[OK] saved checkpoint: $ckpt"
}

eval_one() {
  local method="$1" calib="$2" metrics_jsonl="$3"
  local job_id ckpt eval_tag okvqa_job
  job_id="$(job_id_for "$method" "$calib")"
  ckpt="$(ckpt_for "$method" "$calib")"
  eval_tag="$job_id"
  okvqa_job="okvqa_eval_${eval_tag}_fullval"

  [[ -f "$ckpt" ]] || { echo "[FATAL] eval checkpoint not found: $ckpt" >&2; exit 1; }
  export LAVIS_EVAL_CALIB_TAG="$eval_tag"
  export LAVIS_METRICS_JSONL="$metrics_jsonl"

  echo ""
  echo "========================================================================"
  echo "[EVAL] method=$method calib=$calib tag=$eval_tag"
  echo "[EVAL] ckpt=$ckpt"
  echo "========================================================================"

  if [[ "${SKIP_MMBENCH:-0}" != "1" ]]; then
    echo ">>> MMBench"
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
    echo ">>> OKVQA full val"
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$(next_port)" evaluate_blip.py \
      --cfg-path "$OKVQA_EVAL_CFG" \
      --options "run.num_workers=${OKVQA_EVAL_NUM_WORKERS}" \
      --t5_pruned_checkpoint "$ckpt" \
      --vit_pruned_checkpoint "$ckpt" \
      --job_id "$okvqa_job"
  fi

  if [[ "${SKIP_MMMU:-0}" != "1" ]]; then
    echo ">>> MMMU"
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
    echo ">>> MathVista MC"
    export LAVIS_METRICS_BENCHMARK="MathVista_MC"
    python scripts/blip2/mathvista_mc_eval.py \
      --eval_json "$MATHVISTA_EVAL_JSON" \
      --images_dir "$MATHVISTA_IMAGES_DIR" \
      --ckpt "$ckpt" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda
  fi
}

check_sparsity() {
  local method="$1" calib="$2"
  local job_id ckpt csv
  job_id="$(job_id_for "$method" "$calib")"
  ckpt="$(ckpt_for "$method" "$calib")"
  csv="$OUT_ROOT/${job_id}_sparsity.csv"

  python scripts/blip2/check_ckpt_sparsity.py \
    --ckpt "$ckpt" \
    --tag "$job_id" \
    --expect_t5 "$T5_SPARSITY_TARGET" \
    --vit_max "$VIT_SPARSITY_MAX" \
    --tol 0.08 \
    --out_csv "$csv"
}

echo "========================================================================"
echo "Hybrid C4 multimodal calibration, joint ECoFLaP/ATV pruning"
echo "========================================================================"
echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] BASE=$BASE"
echo "[INFO] HYBRID_DIR=$HYBRID_DIR"
echo "[INFO] CALIBS=$CALIBS"
echo "[INFO] METHODS=$METHODS"
echo "[INFO] T5_SPEC=$T5_SPEC VIT_SPEC=$VIT_SPEC NUM_DATA=$NUM_DATA BS=$BS"
echo "[INFO] ECOFLAP_SCORE_METHOD=$ECOFLAP_SCORE_METHOD ECOFLAP_SPARSITY_RATIO_GRANULARITY=$ECOFLAP_SPARSITY_RATIO_GRANULARITY"
echo "[INFO] MAX_SPARSITY_PER_LAYER=$MAX_SPARSITY_PER_LAYER ATV_ALPHA=$ATV_ALPHA"
echo "[INFO] T5_SPARSITY_TARGET=$T5_SPARSITY_TARGET VIT_SPARSITY_MAX=$VIT_SPARSITY_MAX"
echo "[INFO] RUN_PRUNE=$RUN_PRUNE RUN_EVAL=$RUN_EVAL"
echo "[INFO] OUT_ROOT=$OUT_ROOT"
echo "========================================================================"

preflight

declare -a SUITES=()
METRICS_JSONL="$OUT_ROOT/hybrid_c4_joint_ecoflap_atv_fourbench_${JOB_STAMP}_seed${SEED}.jsonl"
: > "$METRICS_JSONL"

for method in $METHODS; do
  for calib in $CALIBS; do
    job_id="$(job_id_for "$method" "$calib")"
    okvqa_job="okvqa_eval_${job_id}_fullval"
    SUITES+=("${job_id}:${okvqa_job}")

    if [[ "$RUN_PRUNE" == "1" ]]; then
      prune_one "$method" "$calib"
    else
      ckpt="$(ckpt_for "$method" "$calib")"
      echo "[INFO] RUN_PRUNE=0, reuse checkpoint: $ckpt"
      [[ -f "$ckpt" ]] || { echo "[FATAL] missing checkpoint for eval-only run: $ckpt" >&2; exit 1; }
    fi

    if [[ "$RUN_SPARSITY_CHECK" == "1" ]]; then
      check_sparsity "$method" "$calib"
    fi

    if [[ "$RUN_EVAL" == "1" ]]; then
      eval_one "$method" "$calib" "$METRICS_JSONL"
    fi
  done
done

if [[ "$RUN_EVAL" == "1" ]]; then
  SUMMARY_MD="$SUMMARY_DIR/hybrid_c4_joint_ecoflap_atv_fourbench_${JOB_STAMP}_seed${SEED}.md"
  SUMMARY_TSV="$SUMMARY_DIR/hybrid_c4_joint_ecoflap_atv_fourbench_${JOB_STAMP}_seed${SEED}.tsv"
  python scripts/blip2/collect_lavisbackup_eval_summary.py \
    --repo-root "$REPO_ROOT" \
    --metrics-jsonl "$METRICS_JSONL" \
    --out-md "$SUMMARY_MD" \
    --out-tsv "$SUMMARY_TSV" \
    --suites "${SUITES[@]}"
  echo "[OK] summary md:  $SUMMARY_MD"
  echo "[OK] summary tsv: $SUMMARY_TSV"
fi

echo ""
echo "[OK] metrics jsonl: $METRICS_JSONL"
echo "[OK] output root:   $OUT_ROOT"
