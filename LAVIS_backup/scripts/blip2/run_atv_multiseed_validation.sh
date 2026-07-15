#!/usr/bin/env bash
# =============================================================================
# Multi-seed ATV validation driver for BLIP2-T5.
#
# This script is the recommended top-level entry for paper-grade evidence:
#   1. Build mechanism/mask/token/importance evidence on MAIN_SEED.
#   2. Build matching ATV / alpha-ablation / naive-Wanda checkpoints for every seed.
#   3. Evaluate dense, ATV alpha={0,1,4}, naive Wanda, and TAMP/AMIA on four benchmarks.
#   4. Accumulate all eval rows into one REPORT_DIR and run the final strict gate.
#
# It intentionally keeps the expensive mechanism evidence anchored to one seed,
# then uses seeds 42/43/44 for performance stability (mean/std).
#
# Example:
#   cd /data/data2/mfs/2/LAVIS_backup
#   BASE=/data/data2/mfs \
#   ATV_ROOT=/data/data2/mfs/ATV-Pruning \
#   MASK_BASE_CKPT=/data/data2/mfs/model_cache/full_dense_blip2_t5_flant5xl_state_dict.pth \
#   CKPT_TAMP_SEED42=/path/to/tamp_or_amia_seed42.pth \
#   CKPT_TAMP_SEED43=/path/to/tamp_or_amia_seed43.pth \
#   CKPT_TAMP_SEED44=/path/to/tamp_or_amia_seed44.pth \
#   bash scripts/blip2/run_atv_multiseed_validation.sh
#
# Useful switches:
#   RUN_PRUNE=0          reuse existing checkpoints
#   RUN_EVAL=0           skip four-benchmark evaluation
#   RUN_PREFLIGHT=0      skip filesystem/input preflight checks
#   RUN_ENV_CAPTURE=0    skip Python/Torch/CUDA/git/source provenance capture
#   RUN_AUDIT=0          skip final strict manifest/traceability audit summary
#   RUN_SNAPSHOT=0       skip final artifact checksum/size snapshot
#   STRICT_FINAL=0       write final report without failing on incomplete evidence
#   SEEDS="42 43 44"     seed list for eval stability
#   MODELS="dense atv atv_alpha0 atv_alpha4 naive tamp"
#   RUN_DENSE_ONCE=0     evaluate dense once per seed instead of only first seed
#   CKPT_TAMP_TEMPLATE="/path/to/tamp_seed{seed}.pth"
#   ALLOW_SHARED_TAMP_CKPT=1  explicitly reuse one CKPT_TAMP for every seed
#   MASK_BASE_CKPT=...   full dense state_dict for dense-base-aware mask IoU
#   ALLOW_ZERO_ONLY_MASK_INFERENCE=1  debug-only mask IoU without dense base
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

BASE="${BASE:-/data/data2/mfs}"
LB_ROOT="${LB_ROOT:-$REPO_ROOT}"
ATV_ROOT="${ATV_ROOT:-$BASE/ATV-Pruning}"
SEEDS="${SEEDS:-42 43 44}"
MAIN_SEED="${MAIN_SEED:-42}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-$BASE/atv_validation_report_multiseed_${STAMP}}"
MODELS="${MODELS:-dense atv atv_alpha0 atv_alpha4 naive tamp}"
PRUNE_PROVENANCE_CSV="${PRUNE_PROVENANCE_CSV:-$REPORT_DIR/prune_provenance.csv}"
MASK_BASE_CKPT="${MASK_BASE_CKPT:-${DENSE_FULL_CKPT:-}}"
export PRUNE_PROVENANCE_CSV

RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_ENV_CAPTURE="${RUN_ENV_CAPTURE:-1}"
RUN_AUDIT="${RUN_AUDIT:-1}"
RUN_SNAPSHOT="${RUN_SNAPSHOT:-1}"
STRICT_FINAL="${STRICT_FINAL:-1}"
RUN_MAIN_REPRO_CHECK="${RUN_MAIN_REPRO_CHECK:-1}"
RUN_ALPHA_SWEEP="${RUN_ALPHA_SWEEP:-1}"
MAIN_ALPHA_GRID="${MAIN_ALPHA_GRID:-0 0.25 0.5 1 2 4}"
EVAL_SEED_ALPHA_GRID="${EVAL_SEED_ALPHA_GRID:-0 4}"
RUN_DENSE_ONCE="${RUN_DENSE_ONCE:-1}"
ALLOW_SHARED_TAMP_CKPT="${ALLOW_SHARED_TAMP_CKPT:-0}"
ALLOW_ZERO_ONLY_MASK_INFERENCE="${ALLOW_ZERO_ONLY_MASK_INFERENCE:-0}"
SNAPSHOT_HASH_MAX_BYTES="${SNAPSHOT_HASH_MAX_BYTES:-536870912}"
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  else
    echo "[FATAL] neither python nor python3 is available; set PYTHON=/path/to/python" >&2
    exit 1
  fi
fi

mkdir -p "$REPORT_DIR"
if [[ "$RUN_PRUNE" == "1" ]]; then
  rm -f "$PRUNE_PROVENANCE_CSV"
fi

models_include_tamp() {
  for model in $MODELS; do
    case "$model" in
      tamp|amia)
        return 0
        ;;
    esac
  done
  return 1
}

resolve_tamp_ckpt() {
  local seed="$1"
  local var="CKPT_TAMP_SEED${seed}"
  local ckpt="${!var:-}"
  if [[ -z "$ckpt" && -n "${CKPT_TAMP_TEMPLATE:-}" ]]; then
    ckpt="${CKPT_TAMP_TEMPLATE//\{seed\}/$seed}"
    ckpt="${ckpt//%SEED%/$seed}"
  fi
  if [[ -z "$ckpt" && -n "${CKPT_TAMP:-}" ]]; then
    if [[ "$ALLOW_SHARED_TAMP_CKPT" == "1" ]]; then
      ckpt="$CKPT_TAMP"
      echo "[WARN] reusing one CKPT_TAMP for seed=$seed because ALLOW_SHARED_TAMP_CKPT=1: $ckpt" >&2
    else
      echo "[FATAL] MODELS includes TAMP/AMIA but no seed-specific checkpoint was provided for seed=$seed." >&2
      echo "        Set CKPT_TAMP_SEED${seed}=/path/to/ckpt or CKPT_TAMP_TEMPLATE='/path/seed{seed}.pth'." >&2
      echo "        To intentionally reuse one baseline checkpoint, set ALLOW_SHARED_TAMP_CKPT=1." >&2
      exit 1
    fi
  fi
  if [[ -z "$ckpt" ]]; then
    echo "[FATAL] MODELS includes TAMP/AMIA but no checkpoint was provided for seed=$seed." >&2
    echo "        Set CKPT_TAMP_SEED${seed}=/path/to/ckpt or CKPT_TAMP_TEMPLATE='/path/seed{seed}.pth'." >&2
    echo "        To intentionally reuse one CKPT_TAMP for all seeds, set CKPT_TAMP and ALLOW_SHARED_TAMP_CKPT=1." >&2
    exit 1
  fi
  printf "%s" "$ckpt"
}

echo "########## ATV multi-seed validation ##########"
echo "  repo       : $LB_ROOT"
echo "  report     : $REPORT_DIR"
echo "  stamp      : $STAMP"
echo "  seeds      : $SEEDS"
echo "  main seed  : $MAIN_SEED"
echo "  models     : $MODELS"

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  echo ""
  echo "########## Preflight checks ##########"
  preflight_args=(
    --lavis_root "$LB_ROOT"
    --original_atv_root "$ATV_ROOT"
    --base "$BASE"
    --report_dir "$REPORT_DIR"
    --stamp "$STAMP"
    --seeds "$SEEDS"
    --models "$MODELS"
    --run_prune "$RUN_PRUNE"
    --run_eval "$RUN_EVAL"
  )
  [[ -n "${CC3M_JSON:-}" ]] && preflight_args+=(--cc3m_json "$CC3M_JSON")
  [[ -n "${CC3M_IMAGES_DIR:-}" ]] && preflight_args+=(--cc3m_images_dir "$CC3M_IMAGES_DIR")
  [[ -n "${EXPECTED_CALIB_SAMPLES:-}" ]] && preflight_args+=(--expected_calib_samples "$EXPECTED_CALIB_SAMPLES")
  [[ -n "${CALIB_IMAGE_FIELD:-}" ]] && preflight_args+=(--calib_image_field "$CALIB_IMAGE_FIELD")
  [[ -n "${CALIB_TEXT_FIELDS:-}" ]] && preflight_args+=(--calib_text_fields "$CALIB_TEXT_FIELDS")
  [[ -n "${IMAGE_PROBE_SAMPLES:-}" ]] && preflight_args+=(--image_probe_samples "$IMAGE_PROBE_SAMPLES")
  [[ -n "${DENSE_PRETRAIN_CKPT:-}" ]] && preflight_args+=(--dense_pretrain_ckpt "$DENSE_PRETRAIN_CKPT")
  [[ -n "$MASK_BASE_CKPT" ]] && preflight_args+=(--mask_base_ckpt "$MASK_BASE_CKPT")
  [[ "$ALLOW_ZERO_ONLY_MASK_INFERENCE" == "1" ]] && preflight_args+=(--allow_zero_only_mask_inference)
  [[ -n "${MMBENCH_ROOT:-}" ]] && preflight_args+=(--mmbench_root "$MMBENCH_ROOT")
  [[ -n "${MMMU_ROOT:-}" ]] && preflight_args+=(--mmmu_root "$MMMU_ROOT")
  [[ -n "${MATHVISTA_EVAL_JSON:-}" ]] && preflight_args+=(--mathvista_eval_json "$MATHVISTA_EVAL_JSON")
  [[ -n "${MATHVISTA_IMAGES_DIR:-}" ]] && preflight_args+=(--mathvista_images_dir "$MATHVISTA_IMAGES_DIR")
  [[ -n "${CKPT_TAMP:-}" ]] && preflight_args+=(--ckpt_tamp "$CKPT_TAMP")
  [[ -n "${CKPT_TAMP_TEMPLATE:-}" ]] && preflight_args+=(--ckpt_tamp_template "$CKPT_TAMP_TEMPLATE")
  for seed in $SEEDS; do
    seed_ckpt_var="CKPT_TAMP_SEED${seed}"
    if [[ -n "${!seed_ckpt_var:-}" ]]; then
      preflight_args+=(--ckpt_tamp_seed "$seed=${!seed_ckpt_var}")
    fi
  done
  if [[ "$ALLOW_SHARED_TAMP_CKPT" == "1" ]]; then
    preflight_args+=(--allow_shared_tamp_ckpt)
  fi
  "$PYTHON" scripts/blip2/preflight_atv_validation.py "${preflight_args[@]}"
else
  echo "[INFO] RUN_PREFLIGHT=0, skipping filesystem/input preflight checks."
fi

rm -f "$REPORT_DIR/runtime_environment.json" "$REPORT_DIR/runtime_environment.md"
if [[ "$RUN_ENV_CAPTURE" == "1" ]]; then
  echo ""
  echo "########## Runtime environment provenance ##########"
  "$PYTHON" scripts/blip2/capture_atv_runtime_env.py \
    --lavis_root "$LB_ROOT" \
    --original_atv_root "$ATV_ROOT" \
    --report_dir "$REPORT_DIR" \
    --base "$BASE" \
    --stamp "$STAMP" \
    --seeds "$SEEDS" \
    --models "$MODELS" \
    --run_prune "$RUN_PRUNE" \
    --run_eval "$RUN_EVAL"
else
  echo "[INFO] RUN_ENV_CAPTURE=0, skipping runtime environment provenance capture."
fi

if [[ "$RUN_PRUNE" == "1" ]]; then
  for seed in $SEEDS; do
    if [[ "$seed" == "$MAIN_SEED" ]]; then
      seed_report="$REPORT_DIR"
      run_smoke="${RUN_MAIN_SMOKE:-1}"
      run_repro="$RUN_MAIN_REPRO_CHECK"
      validate_report=1
      alpha_grid="$MAIN_ALPHA_GRID"
    else
      seed_report="$BASE/atv_validation_report_${STAMP}_seed${seed}_mechanism"
      run_smoke=0
      run_repro=0
      validate_report=0
      alpha_grid="$EVAL_SEED_ALPHA_GRID"
    fi
    ckpt_tamp_for_seed=""
    if models_include_tamp; then
      ckpt_tamp_for_seed="$(resolve_tamp_ckpt "$seed")"
    fi

    echo ""
    echo "########## Prune/evidence pass | seed=$seed report=$seed_report ##########"
    SEED="$seed" \
      STAMP="$STAMP" \
      REPORT_DIR="$seed_report" \
      SKIP_MMBENCH=1 \
      RUN_SMOKE="$run_smoke" \
      RUN_REPRO_CHECK="$run_repro" \
      RUN_ALPHA_SWEEP="$RUN_ALPHA_SWEEP" \
      ALPHA_GRID="$alpha_grid" \
      VALIDATE_REPORT="$validate_report" \
      MASK_BASE_CKPT="$MASK_BASE_CKPT" \
      CKPT_TAMP="$ckpt_tamp_for_seed" \
      ALLOW_ZERO_ONLY_MASK_INFERENCE="$ALLOW_ZERO_ONLY_MASK_INFERENCE" \
      bash scripts/blip2/run_atv_full_verify.sh
  done
else
  echo "[INFO] RUN_PRUNE=0, reusing existing checkpoints."
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  seed_i=0
  for seed in $SEEDS; do
    seed_i=$((seed_i + 1))
    ckpt_atv="$LB_ROOT/pruned_checkpoint/atv_cc3m_t5only_${STAMP}_seed${seed}.pth"
    ckpt_naive="$LB_ROOT/pruned_checkpoint/naive_wanda_cc3m_t5only_${STAMP}_seed${seed}.pth"
    ckpt_alpha0="$LB_ROOT/pruned_checkpoint/atv_cc3m_t5only_alpha0_${STAMP}_seed${seed}.pth"
    ckpt_alpha4="$LB_ROOT/pruned_checkpoint/atv_cc3m_t5only_alpha4_${STAMP}_seed${seed}.pth"
    ckpt_tamp=""
    if models_include_tamp; then
      ckpt_tamp="$(resolve_tamp_ckpt "$seed")"
    fi
    seed_models="$MODELS"
    if [[ "$RUN_DENSE_ONCE" == "1" && "$seed_i" -gt 1 ]]; then
      seed_models=""
      for model in $MODELS; do
        [[ "$model" == "dense" ]] && continue
        seed_models="${seed_models:+$seed_models }$model"
      done
    fi

    echo ""
    echo "########## Four-benchmark eval | seed=$seed models=$seed_models ##########"
    SEED="$seed" \
      STAMP="$STAMP" \
      REPORT_DIR="$REPORT_DIR" \
      MODELS="$seed_models" \
      CKPT_ATV="$ckpt_atv" \
      CKPT_NAIVE="$ckpt_naive" \
      CKPT_ATV_ALPHA0="$ckpt_alpha0" \
      CKPT_ATV_ALPHA4="$ckpt_alpha4" \
      CKPT_TAMP="$ckpt_tamp" \
      REQUIRE_MODEL_CKPTS=1 \
      VALIDATE_STRICT=0 \
      MASTER_PORT_BASE="$((29900 + 100 * seed_i))" \
      bash scripts/blip2/run_atv_eval_matrix_fourbench.sh
  done
else
  echo "[INFO] RUN_EVAL=0, skipping four-benchmark eval."
fi

echo ""
echo "########## Final validation report ##########"
final_args=()
if [[ "$STRICT_FINAL" == "1" ]]; then
  final_args+=(--strict)
fi
if [[ "$ALLOW_ZERO_ONLY_MASK_INFERENCE" == "1" ]]; then
  final_args+=(--allow_zero_only_mask_inference)
fi

set +e
"$PYTHON" scripts/blip2/validate_atv_migration.py \
  --original_atv_root "$ATV_ROOT" \
  --lavis_root "$LB_ROOT" \
  --out_dir "$REPORT_DIR" \
  --preserve_existing \
  --prune_provenance_csv "$PRUNE_PROVENANCE_CSV" \
  "${final_args[@]}"
validation_status=$?
set -e

audit_status=0
if [[ "$RUN_AUDIT" == "1" ]]; then
  echo ""
  echo "########## Strict audit summary ##########"
  audit_args=(--report_dir "$REPORT_DIR")
  if [[ "$STRICT_FINAL" != "1" ]]; then
    audit_args+=(--no_fail)
  fi
  set +e
  "$PYTHON" scripts/blip2/audit_atv_validation_report.py "${audit_args[@]}"
  audit_status=$?
  set -e
else
  echo "[INFO] RUN_AUDIT=0, skipping strict audit summary."
fi

if [[ "$RUN_SNAPSHOT" == "1" ]]; then
  echo ""
  echo "########## Artifact snapshot ##########"
  snapshot_args=(
    --report_dir "$REPORT_DIR"
    --hash_max_bytes "$SNAPSHOT_HASH_MAX_BYTES"
    --include "original_atv_pruner=$ATV_ROOT/qwen/activation_aware_pruner.py"
    --include "lavis_wanda_atv_pruner=$LB_ROOT/lavis/compression/pruners/wanda_pruner.py"
    --include "lavis_evaluate_blip=$LB_ROOT/evaluate_blip.py"
    --include "lavis_runtime_env_capture=$LB_ROOT/scripts/blip2/capture_atv_runtime_env.py"
    --include "lavis_strict_audit=$LB_ROOT/scripts/blip2/audit_atv_validation_report.py"
    --include "lavis_validator=$LB_ROOT/scripts/blip2/validate_atv_migration.py"
    --include "lavis_preflight=$LB_ROOT/scripts/blip2/preflight_atv_validation.py"
    --include "lavis_dense_export=$LB_ROOT/scripts/blip2/export_blip2_full_dense_state_dict.py"
    --include "lavis_materialize_cc3m_cfg=$LB_ROOT/scripts/blip2/materialize_cc3m_calib_cfg.py"
    --include "lavis_ckpt_sparsity_check=$LB_ROOT/scripts/blip2/check_ckpt_sparsity.py"
    --include "lavis_ckpt_compare=$LB_ROOT/scripts/blip2/compare_ckpts.py"
    --include "lavis_snapshot=$LB_ROOT/scripts/blip2/snapshot_atv_artifacts.py"
    --include "lavis_atv_prune_driver=$LB_ROOT/scripts/blip2/run_atv_cc3m_prune_then_eval.sh"
    --include "lavis_atv_eval_driver=$LB_ROOT/scripts/blip2/run_atv_eval_matrix_fourbench.sh"
    --include "lavis_atv_full_verify=$LB_ROOT/scripts/blip2/run_atv_full_verify.sh"
    --include "lavis_atv_multiseed_driver=$LB_ROOT/scripts/blip2/run_atv_multiseed_validation.sh"
  )
  for seed in $SEEDS; do
    snapshot_args+=(--include "ckpt_atv_alpha1_seed${seed}=$LB_ROOT/pruned_checkpoint/atv_cc3m_t5only_${STAMP}_seed${seed}.pth")
    snapshot_args+=(--include "ckpt_naive_wanda_seed${seed}=$LB_ROOT/pruned_checkpoint/naive_wanda_cc3m_t5only_${STAMP}_seed${seed}.pth")
    snapshot_args+=(--include "ckpt_atv_alpha0_seed${seed}=$LB_ROOT/pruned_checkpoint/atv_cc3m_t5only_alpha0_${STAMP}_seed${seed}.pth")
    snapshot_args+=(--include "ckpt_atv_alpha4_seed${seed}=$LB_ROOT/pruned_checkpoint/atv_cc3m_t5only_alpha4_${STAMP}_seed${seed}.pth")
    if [[ "$RUN_EVAL" == "1" ]] && models_include_tamp; then
      snapshot_args+=(--include "ckpt_tamp_seed${seed}=$(resolve_tamp_ckpt "$seed")")
    fi
  done
  "$PYTHON" scripts/blip2/snapshot_atv_artifacts.py "${snapshot_args[@]}"
else
  echo "[INFO] RUN_SNAPSHOT=0, skipping artifact snapshot."
fi

if [[ "$validation_status" -ne 0 ]]; then
  exit "$validation_status"
fi
if [[ "$audit_status" -ne 0 ]]; then
  exit "$audit_status"
fi

echo ""
echo "[OK] multi-seed validation report: $REPORT_DIR"
