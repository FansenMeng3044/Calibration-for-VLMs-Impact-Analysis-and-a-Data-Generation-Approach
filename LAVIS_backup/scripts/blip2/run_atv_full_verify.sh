#!/usr/bin/env bash
# =============================================================================
# End-to-end ATV migration verification for BLIP2-T5.
#
# This script is intentionally evidence-oriented. It does not only create an
# ATV checkpoint; it also collects the artifacts needed by
# validate_atv_migration.py:
#   1. smoke ATV prune + sparsity check
#   2. formal ATV alpha=1 prune + diagnostics
#   3. same-seed ATV rerun for reproducibility
#   4. naive Wanda checkpoint for ATV-vs-Wanda mask comparison
#   5. alpha sweep logs/checkpoints
#   6. final validation report folder
#
# It must be run on the GPU server with BLIP2-T5 weights and CC3M-128 data.
# Example:
#   bash scripts/blip2/run_atv_full_verify.sh
#
# Useful switches:
#   RUN_SMOKE=0          skip the 16-sample smoke prune
#   SKIP_MMBENCH=1       skip the ECoFLaP MMBench table step
#   RUN_REPRO_CHECK=0    skip the expensive same-seed rerun
#   RUN_ALPHA_SWEEP=0    skip alpha sweep
#   RUN_ENV_CAPTURE=0    skip Python/Torch/CUDA/git/source provenance capture
#   RUN_AUDIT=0          skip nonfatal strict audit summary for this partial report
#   MASK_BASE_CKPT=...   full dense state_dict for dense-base-aware mask IoU
#   CKPT_TAMP=...        TAMP/AMIA checkpoint for required ATV-vs-TAMP mask evidence
#   ALLOW_ZERO_ONLY_MASK_INFERENCE=1  debug-only mask IoU without dense base
# =============================================================================
set -euo pipefail

BASE="${BASE:-/data/data2/mfs}"
LB_ROOT="${LB_ROOT:-$BASE/2/LAVIS_backup}"
ECO_ROOT="${ECO_ROOT:-$BASE/2/ECoFLaP/LAVIS}"
ATV_ROOT="${ATV_ROOT:-$BASE/ATV-Pruning}"
SEED="${SEED:-42}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
T5_SPEC="${T5_SPEC:-24-0.5-1.0-1.0}"
NUM_DATA="${NUM_DATA:-128}"
CC3M_CFG="${CC3M_CFG:-$LB_ROOT/lavis/projects/blip2/eval/cc_prefix_derivative_compute_cc3m_calib128.yaml}"
CC3M_JSON_EFFECTIVE="${CC3M_JSON:-$BASE/CC3M_calib_128/cc3m_calib_128.json}"
CC3M_IMAGES_DIR_EFFECTIVE="${CC3M_IMAGES_DIR:-$BASE/CC3M_calib_128/images}"
BLIP2_PRETRAINED="${BLIP2_PRETRAINED:-$BASE/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
MASK_BASE_CKPT="${MASK_BASE_CKPT:-${DENSE_FULL_CKPT:-}}"
CKPT_TAMP="${CKPT_TAMP:-}"
ALLOW_ZERO_ONLY_MASK_INFERENCE="${ALLOW_ZERO_ONLY_MASK_INFERENCE:-0}"
MATERIALIZE_CC3M_CFG="${MATERIALIZE_CC3M_CFG:-1}"
SKIP_MMBENCH="${SKIP_MMBENCH:-0}"
VALIDATE_REPORT="${VALIDATE_REPORT:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_ALPHA_SWEEP="${RUN_ALPHA_SWEEP:-1}"
RUN_REPRO_CHECK="${RUN_REPRO_CHECK:-1}"
RUN_ENV_CAPTURE="${RUN_ENV_CAPTURE:-1}"
RUN_AUDIT="${RUN_AUDIT:-1}"
ALPHA_GRID="${ALPHA_GRID:-0 0.25 0.5 1 2 4}"
PYTHON="${PYTHON:-python}"
export SEED STAMP T5_SPEC NUM_DATA

ATV_JOB="atv_cc3m_t5only_${STAMP}_seed${SEED}"
ATV_REPRO_JOB="atv_cc3m_t5only_repro_${STAMP}_seed${SEED}"
NAIVE_JOB="naive_wanda_cc3m_t5only_${STAMP}_seed${SEED}"
ATV_CKPT="$LB_ROOT/pruned_checkpoint/${ATV_JOB}.pth"
ATV_REPRO_CKPT="$LB_ROOT/pruned_checkpoint/${ATV_REPRO_JOB}.pth"
NAIVE_CKPT="$LB_ROOT/pruned_checkpoint/${NAIVE_JOB}.pth"
REPORT_DIR="${REPORT_DIR:-$BASE/atv_validation_report_${STAMP}_seed${SEED}}"
ATV_SMOKE_LOG="$REPORT_DIR/atv_smoke.log"
ATV_LOG="$REPORT_DIR/atv_alpha1.log"
ATV_REPRO_LOG="$REPORT_DIR/atv_repro_same_seed.log"
NAIVE_LOG="$REPORT_DIR/naive_wanda.log"
SPARSITY_CSV="$REPORT_DIR/sparsity_summary.csv"
NAIVE_SPARSITY_CSV="$REPORT_DIR/${NAIVE_JOB}_sparsity.csv"
PRUNE_PROVENANCE_CSV="${PRUNE_PROVENANCE_CSV:-$REPORT_DIR/prune_provenance.csv}"
CC3M_CFG_RUNTIME="$CC3M_CFG"
ATV_LOG_ARGS=(--atv_log "alpha1=$ATV_LOG")
MASK_PAIR_ARGS=()
MASK_BASE_ARGS=()

for d in "$LB_ROOT" "$ECO_ROOT"; do
  [[ -d "$d" ]] || { echo "[FATAL] repository root not found: $d" >&2; exit 1; }
done
mkdir -p "$REPORT_DIR"
rm -f \
  "$REPORT_DIR/token_mask_integrity.csv" \
  "$REPORT_DIR/calibration_batch_trace.csv" \
  "$REPORT_DIR/selected_query_frequency.csv" \
  "$REPORT_DIR/importance_distribution.csv" \
  "$REPORT_DIR/alpha_sweep.csv" \
  "$REPORT_DIR/mask_iou_by_layer.csv" \
  "$REPORT_DIR/mask_iou_by_module.csv" \
  "$REPORT_DIR/sparsity_summary.csv" \
  "$REPORT_DIR/runtime_environment.json" \
  "$REPORT_DIR/runtime_environment.md" \
  "$REPORT_DIR/prune_provenance.csv" \
  "$REPORT_DIR/eval_results.csv" \
  "$REPORT_DIR/validation_manifest.csv" \
  "$REPORT_DIR/strict_audit_summary.csv" \
  "$REPORT_DIR/strict_audit_summary.md" \
  "$REPORT_DIR/static_mapping.md" \
  "$REPORT_DIR/unit_test_results.txt" \
  "$REPORT_DIR/final_validation_report.md" \
  "$REPORT_DIR/importance_distribution.png" \
  "$REPORT_DIR/scaler_row_distribution.png" \
  "$REPORT_DIR/selected_query_token_frequency.png"

if [[ -n "$MASK_BASE_CKPT" ]]; then
  [[ -f "$MASK_BASE_CKPT" ]] || { echo "[FATAL] MASK_BASE_CKPT not found: $MASK_BASE_CKPT" >&2; exit 1; }
  MASK_BASE_ARGS+=(--mask_base_ckpt "$MASK_BASE_CKPT")
elif [[ "$ALLOW_ZERO_ONLY_MASK_INFERENCE" == "1" ]]; then
  echo "[WARN] ALLOW_ZERO_ONLY_MASK_INFERENCE=1: mask IoU will use raw weight==0. This is debug-only, not strict evidence."
  MASK_BASE_ARGS+=(--allow_zero_only_mask_inference)
fi

if [[ "$MATERIALIZE_CC3M_CFG" == "1" ]]; then
  CC3M_CFG_RUNTIME="$REPORT_DIR/cc3m_calib_runtime_full_verify.yaml"
  ( cd "$LB_ROOT" && "$PYTHON" scripts/blip2/materialize_cc3m_calib_cfg.py \
      --src_cfg "$CC3M_CFG" \
      --out_cfg "$CC3M_CFG_RUNTIME" \
      --cc3m_json "$CC3M_JSON_EFFECTIVE" \
      --cc3m_images_dir "$CC3M_IMAGES_DIR_EFFECTIVE" \
      --pretrained "$BLIP2_PRETRAINED" )
fi

csv_cell() {
  local value="${1:-}"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

append_naive_prune_provenance() {
  if [[ ! -s "$PRUNE_PROVENANCE_CSV" ]]; then
    printf 'seed,method,role,alpha,job_id,ckpt,calib_cfg,calib_json,images_dir,num_data,t5_spec,t5_sparsity_target,vit_sparsity_target,pruning_scope,run_prune,prune_log,sparsity_csv\n' > "$PRUNE_PROVENANCE_CSV"
  fi
  {
    csv_cell "$SEED"; printf ','
    csv_cell "wanda"; printf ','
    csv_cell "main"; printf ','
    csv_cell ""; printf ','
    csv_cell "$NAIVE_JOB"; printf ','
    csv_cell "$NAIVE_CKPT"; printf ','
    csv_cell "$CC3M_CFG_RUNTIME"; printf ','
    csv_cell "$CC3M_JSON_EFFECTIVE"; printf ','
    csv_cell "$CC3M_IMAGES_DIR_EFFECTIVE"; printf ','
    csv_cell "$NUM_DATA"; printf ','
    csv_cell "$T5_SPEC"; printf ','
    csv_cell "0.5"; printf ','
    csv_cell "0.0"; printf ','
    csv_cell "t5_only"; printf ','
    csv_cell "1"; printf ','
    csv_cell "$NAIVE_LOG"; printf ','
    csv_cell "$NAIVE_SPARSITY_CSV"; printf '\n'
  } >> "$PRUNE_PROVENANCE_CSV"
}

if [[ "$RUN_ENV_CAPTURE" == "1" ]]; then
  echo "########## Runtime environment provenance ##########"
  ( cd "$LB_ROOT" && "$PYTHON" scripts/blip2/capture_atv_runtime_env.py \
      --lavis_root "$LB_ROOT" \
      --original_atv_root "$ATV_ROOT" \
      --report_dir "$REPORT_DIR" \
      --base "$BASE" \
      --stamp "$STAMP" \
      --seeds "$SEED" \
      --models "atv naive" \
      --run_prune 1 \
      --run_eval 0 )
else
  echo "[INFO] RUN_ENV_CAPTURE=0, skipping runtime environment provenance capture."
fi

if [[ "$RUN_SMOKE" == "1" ]]; then
  echo "########## 1) Smoke prune: ATV nsamples=16 ##########"
  ( cd "$LB_ROOT" && \
      NUM_DATA=16 \
      RUN_EVAL=0 \
      RUN_ENV_CAPTURE=0 \
      VALIDATE_REPORT=0 \
      REPORT_DIR="$REPORT_DIR/smoke" \
      PRUNE_PROVENANCE_CSV="$PRUNE_PROVENANCE_CSV" \
      PRUNE_PROVENANCE_ROLE="smoke" \
      CC3M_CFG="$CC3M_CFG_RUNTIME" \
      CC3M_JSON="$CC3M_JSON_EFFECTIVE" \
      CC3M_IMAGES_DIR="$CC3M_IMAGES_DIR_EFFECTIVE" \
      MATERIALIZE_CC3M_CFG=0 \
      LAVIS_ATV_DIAGNOSTIC_DIR="$REPORT_DIR/smoke" \
      JOB_STAMP="$STAMP" \
      JOB_ID="${ATV_JOB}_smoke" \
      bash scripts/blip2/run_atv_cc3m_prune_then_eval.sh ) 2>&1 | tee "$ATV_SMOKE_LOG"
  echo ">>> Smoke sparsity check: T5 near 0.5, non-T5 modules near dense"
  ( cd "$LB_ROOT" && python scripts/blip2/check_ckpt_sparsity.py \
      --ckpt "pruned_checkpoint/${ATV_JOB}_smoke.pth" --expect_t5 0.5 --tol 0.05 )
else
  echo "########## 1) Smoke prune skipped (RUN_SMOKE=0) ##########"
fi

echo ""
echo "########## 2) Formal ATV prune: CC3M-128, alpha=1 ##########"
( cd "$LB_ROOT" && \
  LAVIS_ATV_DIAGNOSTIC_DIR="$REPORT_DIR" \
  RUN_EVAL=0 \
  RUN_ENV_CAPTURE=0 \
  PRUNE_PROVENANCE_CSV="$PRUNE_PROVENANCE_CSV" \
  PRUNE_PROVENANCE_ROLE="main" \
  CC3M_CFG="$CC3M_CFG_RUNTIME" \
  CC3M_JSON="$CC3M_JSON_EFFECTIVE" \
  CC3M_IMAGES_DIR="$CC3M_IMAGES_DIR_EFFECTIVE" \
  MATERIALIZE_CC3M_CFG=0 \
  JOB_STAMP="$STAMP" \
  JOB_ID="$ATV_JOB" \
  bash scripts/blip2/run_atv_cc3m_prune_then_eval.sh ) 2>&1 | tee "$ATV_LOG"
echo ">>> Formal ATV sparsity check: T5 near 0.5, non-T5 modules near dense"
( cd "$LB_ROOT" && python scripts/blip2/check_ckpt_sparsity.py \
    --ckpt "$ATV_CKPT" --tag "atv_alpha1" --expect_t5 0.5 --tol 0.05 \
    --out_csv "$SPARSITY_CSV" )

if [[ "$RUN_REPRO_CHECK" == "1" ]]; then
  echo ""
  echo "########## 3) Same-seed ATV reproducibility rerun ##########"
  ( cd "$LB_ROOT" && \
    LAVIS_ATV_DIAGNOSTIC_DIR="$REPORT_DIR/repro" \
    RUN_EVAL=0 \
    RUN_ENV_CAPTURE=0 \
    VALIDATE_REPORT=0 \
    REPORT_DIR="$REPORT_DIR/repro" \
    PRUNE_PROVENANCE_CSV="$PRUNE_PROVENANCE_CSV" \
    PRUNE_PROVENANCE_ROLE="repro" \
    CC3M_CFG="$CC3M_CFG_RUNTIME" \
    CC3M_JSON="$CC3M_JSON_EFFECTIVE" \
    CC3M_IMAGES_DIR="$CC3M_IMAGES_DIR_EFFECTIVE" \
    MATERIALIZE_CC3M_CFG=0 \
    JOB_STAMP="$STAMP" \
    JOB_ID="$ATV_REPRO_JOB" \
    MASTER_PORT_PRUNE=29762 \
    bash scripts/blip2/run_atv_cc3m_prune_then_eval.sh ) 2>&1 | tee "$ATV_REPRO_LOG"
  [[ -f "$ATV_REPRO_CKPT" ]] || { echo "[FATAL] repro checkpoint missing: $ATV_REPRO_CKPT" >&2; exit 1; }
  MASK_PAIR_ARGS+=(--mask_pair atv_repro_same_seed "$ATV_CKPT" "$ATV_REPRO_CKPT")
fi

echo ""
echo "########## 4) Naive Wanda checkpoint with the same calibration path ##########"
( cd "$LB_ROOT" && python -m torch.distributed.run --nproc_per_node=1 --master_port 29761 evaluate_blip.py \
    --cfg-path "$CC3M_CFG_RUNTIME" \
    --pruning_method blipt5_wanda_pruner \
    --no_prune_vit \
    --save_pruned_model --prunining_dataset_batch_size 8 \
    --num_data "$NUM_DATA" --num_data_first_stage "$NUM_DATA" \
    --t5_prune_spec "$T5_SPEC" \
    --job_id "$NAIVE_JOB" ) 2>&1 | tee "$NAIVE_LOG"
[[ -f "$NAIVE_CKPT" ]] || { echo "[FATAL] naive Wanda checkpoint missing: $NAIVE_CKPT" >&2; exit 1; }
( cd "$LB_ROOT" && python scripts/blip2/check_ckpt_sparsity.py \
    --ckpt "$NAIVE_CKPT" --tag "naive_wanda" --expect_t5 0.5 --tol 0.05 \
    --out_csv "$NAIVE_SPARSITY_CSV" )
append_naive_prune_provenance

echo ""
echo "########## 5) ATV alpha=1 vs naive Wanda mask comparison ##########"
( cd "$LB_ROOT" && python scripts/blip2/compare_ckpts.py --a "$ATV_CKPT" --b "$NAIVE_CKPT" )
MASK_PAIR_ARGS+=(--mask_pair atv_alpha1_vs_wanda "$ATV_CKPT" "$NAIVE_CKPT")

if [[ "$RUN_ALPHA_SWEEP" == "1" ]]; then
  echo ""
  echo "########## 6) Alpha sweep: ATV token-selection behavior ##########"
  alpha_i=0
  for alpha in $ALPHA_GRID; do
    alpha_i=$((alpha_i + 1))
    if [[ "$alpha" == "1" || "$alpha" == "1.0" ]]; then
      continue
    fi
    alpha_safe="${alpha//./p}"
    alpha_safe="${alpha_safe//-/m}"
    sweep_job="atv_cc3m_t5only_alpha${alpha_safe}_${STAMP}_seed${SEED}"
    sweep_ckpt="$LB_ROOT/pruned_checkpoint/${sweep_job}.pth"
    sweep_log="$REPORT_DIR/atv_alpha${alpha_safe}.log"
    echo ">>> alpha=$alpha job=$sweep_job"
    ( cd "$LB_ROOT" && \
      ATV_ALPHA="$alpha" \
      RUN_EVAL=0 \
      RUN_ENV_CAPTURE=0 \
      VALIDATE_REPORT=0 \
      REPORT_DIR="$REPORT_DIR/alpha_${alpha_safe}" \
      PRUNE_PROVENANCE_CSV="$PRUNE_PROVENANCE_CSV" \
      PRUNE_PROVENANCE_ROLE="alpha_sweep" \
      CC3M_CFG="$CC3M_CFG_RUNTIME" \
      CC3M_JSON="$CC3M_JSON_EFFECTIVE" \
      CC3M_IMAGES_DIR="$CC3M_IMAGES_DIR_EFFECTIVE" \
      MATERIALIZE_CC3M_CFG=0 \
      JOB_STAMP="$STAMP" \
      JOB_ID="$sweep_job" \
      MASTER_PORT_PRUNE="$((29800 + alpha_i))" \
      bash scripts/blip2/run_atv_cc3m_prune_then_eval.sh ) 2>&1 | tee "$sweep_log"
    [[ -f "$sweep_ckpt" ]] || { echo "[FATAL] alpha=$alpha checkpoint missing: $sweep_ckpt" >&2; exit 1; }
    ATV_LOG_ARGS+=(--atv_log "alpha${alpha}=$sweep_log")
    MASK_PAIR_ARGS+=(--mask_pair "atv_alpha${alpha}_vs_wanda" "$sweep_ckpt" "$NAIVE_CKPT")
  done
fi

if [[ -n "$CKPT_TAMP" ]]; then
  [[ -f "$CKPT_TAMP" ]] || { echo "[FATAL] CKPT_TAMP not found: $CKPT_TAMP" >&2; exit 1; }
  MASK_PAIR_ARGS+=(--mask_pair atv_alpha1_vs_tamp "$ATV_CKPT" "$CKPT_TAMP")
else
  echo "[WARN] CKPT_TAMP is unset; strict validation will lack ATV-vs-TAMP/AMIA mask evidence."
fi

if [[ "$VALIDATE_REPORT" == "1" ]]; then
  echo ""
  echo "########## 7) ATV migration validation report ##########"
  if [[ "${#MASK_PAIR_ARGS[@]}" -gt 0 && -z "$MASK_BASE_CKPT" && "$ALLOW_ZERO_ONLY_MASK_INFERENCE" != "1" ]]; then
    echo "[FATAL] MASK_BASE_CKPT is required for strict dense-base-aware mask comparisons." >&2
    echo "        Use a full dense BLIP2-T5 state_dict, not blip2_pretrained_flant5xl.pth." >&2
    echo "        For local debugging only, set ALLOW_ZERO_ONLY_MASK_INFERENCE=1." >&2
    exit 1
  fi
  if [[ -f "$ATV_ROOT/qwen/activation_aware_pruner.py" ]]; then
    ( cd "$LB_ROOT" && python scripts/blip2/validate_atv_migration.py \
        --original_atv_root "$ATV_ROOT" \
        --lavis_root "$LB_ROOT" \
        --out_dir "$REPORT_DIR" \
        "${ATV_LOG_ARGS[@]}" \
        "${MASK_PAIR_ARGS[@]}" \
        "${MASK_BASE_ARGS[@]}" \
        --sparsity_csv "$SPARSITY_CSV" \
        --prune_provenance_csv "$PRUNE_PROVENANCE_CSV" \
        --token_mask_csv "$REPORT_DIR/token_mask_integrity.csv" \
        --calibration_batch_trace_csv "$REPORT_DIR/calibration_batch_trace.csv" \
        --importance_csv "$REPORT_DIR/importance_distribution.csv" \
        --selected_query_csv "$REPORT_DIR/selected_query_frequency.csv" )
  else
    echo "[WARN] skip validate_atv_migration.py: ATV_ROOT not found: $ATV_ROOT"
    echo "       set ATV_ROOT=/path/to/ATV-Pruning to enable static mapping evidence."
  fi

  if [[ "$RUN_AUDIT" == "1" ]]; then
    ( cd "$LB_ROOT" && python scripts/blip2/audit_atv_validation_report.py \
        --report_dir "$REPORT_DIR" \
        --no_fail )
  else
    echo "[INFO] RUN_AUDIT=0, skipping nonfatal strict audit summary."
  fi
fi

if [[ "$SKIP_MMBENCH" == "1" ]]; then
  echo "[INFO] SKIP_MMBENCH=1, skipping ECoFLaP MMBench table."
  exit 0
fi

echo ""
echo "########## 8) Optional ECoFLaP MMBench comparison table ##########"
eco_env=(
  "MODELS=${MODELS:-dense joint split atv}"
  "CKPT_ATV=$ATV_CKPT"
  "JOB_STAMP=$STAMP"
  "SEED=$SEED"
  "T5_SPEC=$T5_SPEC"
  "NUM_DATA=$NUM_DATA"
)
if [[ -n "${CKPT_TAMP:-}" ]]; then
  eco_env+=("CKPT_TAMP=$CKPT_TAMP")
fi
( cd "$ECO_ROOT" && env "${eco_env[@]}" \
    bash scripts/blip2/run_pure_wanda_cc3m_split_joint_dense_mmbench_full.sh )

echo ""
echo "########## Done ##########"
echo "  ATV ckpt       : $ATV_CKPT"
echo "  ATV repro ckpt : $ATV_REPRO_CKPT"
echo "  naive ckpt     : $NAIVE_CKPT"
echo "  report dir     : $REPORT_DIR"
echo "  comparison md  : $ECO_ROOT/lavis/output/BLIP2/mmbench_full_table_${STAMP}_seed${SEED}.md"
