#!/usr/bin/env bash
# =============================================================================
# Calibration Pairing -- internal-mechanism suite.
#
# Question: can independently calibrated modules draw from UNRELATED sources?
# The two accuracy tables collapse toward the mean under cross-sourcing. This
# suite explains WHY from internal quantities, never from the accuracy table.
#
# The decisive fact this suite is built around:
#   Under factorized calibration  S_L = Psi_L(L, D_txt)  and  S_V = Psi_V(V, D_img).
#   In the CROSS-source setting D_txt = C4 for every row, so S_L is FROZEN across
#   rows and only S_V varies.  In the SAME-source setting both S_L and S_V vary.
#   Comparing the two settings isolates how much of everything the text side drove.
#
# Four experiments, each reusing a validated tool:
#   Exp1  mask space      -- analyze_calibration_mask_mechanism.py  (L vs V overlap)
#   Exp2  Wanda statistic -- extract_wanda_statistics.py + analyze_calibration_statistics.py
#   Exp3  activation drift-- analyze_pruned_drift_by_token_group.py (vis/text/answer/KL)
#   Exp4  LLM input space -- extract_llm_input_embeddings.py + analyze_llm_embeddings.py
#
# ------------------------------------------------------------------ inputs ---
# Pruned checkpoints you already produced for the two tables, comma-separated
# "label=path". Use the SAME labels in both groups so outputs line up:
#   SAME_CKPTS   factorized, visual=text=source   (the Table-2-style diagonal set)
#   CROSS_CKPTS  visual=source, text=C4           (the C4-text tables)
# Sources for the GPU extractions (Exp2/Exp4), one line "label|calib_json|images_dir":
#   DATASETS     the 5 image sources AND C4 (C4 has no images -> leave images_dir empty)
# Held-out multimodal eval set for the drift pass (Exp3), NOT a calibration set:
#   EVAL_JSON, EVAL_IMAGES
# Optional:
#   DENSE_CKPT   dense blip2 pth (default: pretrained blip2_t5)
#   ACC_CSV      accuracy matrix csv, to correlate mechanism with outcome
#   METHOD       tag for the output folder (e.g. wanda / sparsegpt), default "wanda"
#   MAX_SAMPLES  default 128   BATCH_SIZE default 8   DRIFT_SAMPLES default 64
#   RUN_EXP1..RUN_EXP4  set to 0 to skip a stage (default 1). Exp1 is CPU-only.
#   OUT_ROOT     default $BASE/pairing_mechanism/$METHOD
#
# IMPORTANT: for S_L to be *exactly* frozen across the cross-source rows, the C4
# calibration must use the identical sample (same seed/segment) in every row.
# Confirm your table checkpoints were built that way, or the "overlap = 1.0"
# prediction will be diluted by C4 sampling noise rather than by the mechanism.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

BASE="${BASE:-${AUTODL_TMP:-/data/data2/mfs}}"
METHOD="${METHOD:-wanda}"
OUT_ROOT="${OUT_ROOT:-$BASE/pairing_mechanism/$METHOD}"
STATS_ROOT="$OUT_ROOT/stats"
EMB_ROOT="$OUT_ROOT/embeddings"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DRIFT_SAMPLES="${DRIFT_SAMPLES:-64}"
DENSE_CKPT="${DENSE_CKPT:-}"
ACC_CSV="${ACC_CSV:-}"
RUN_EXP1="${RUN_EXP1:-1}"
RUN_EXP2="${RUN_EXP2:-1}"
RUN_EXP3="${RUN_EXP3:-1}"
RUN_EXP4="${RUN_EXP4:-1}"

: "${SAME_CKPTS:?set SAME_CKPTS='MMBench=/p/a.pth,MMMU=/p/b.pth,...' (factorized same-source)}"
: "${CROSS_CKPTS:?set CROSS_CKPTS='MMBench=/p/a.pth,...' (visual=source, text=C4)}"
mkdir -p "$OUT_ROOT"

acc_arg=(); [[ -n "$ACC_CSV" ]] && acc_arg=(--accuracy_csv "$ACC_CSV")
dense_stat_arg=(); [[ -n "$DENSE_CKPT" ]] && dense_stat_arg=(--ckpt "$DENSE_CKPT")

# label=path,label=path  ->  ("--ckpt" "label=path" ...)
to_ckpt_args() {
  local _out=(); IFS=',' read -ra _pairs <<< "$1"
  for kv in "${_pairs[@]}"; do [[ -n "${kv// }" ]] && _out+=(--ckpt "$kv"); done
  printf '%s\n' "${_out[@]}"
}
mapfile -t SAME_ARGS  < <(to_ckpt_args "$SAME_CKPTS")
mapfile -t CROSS_ARGS < <(to_ckpt_args "$CROSS_CKPTS")

# =============================================================================
# Exp1 -- mask space (CPU).  Is the "choice" of source removed on the L side?
#   For each module component (t5 = language, vit = vision) and each provenance
#   setting, dump the pairwise KEEP-mask overlap across the 5 source rows.
#   Prediction:  cross-source L overlap ~ 1.0 (S_L frozen), while V overlap < 1;
#                same-source L overlap < 1 (S_L varies with the text side).
# =============================================================================
if [[ "$RUN_EXP1" == "1" ]]; then
  echo "########## Exp1: mask-space overlap (CPU) ##########"
  for comp in t5 vit; do
    python scripts/blip2/analyze_calibration_mask_mechanism.py \
      "${SAME_ARGS[@]}" --centrality_component "$comp" \
      --out_dir "$OUT_ROOT/exp1_mask/same_${comp}" "${acc_arg[@]}"
    python scripts/blip2/analyze_calibration_mask_mechanism.py \
      "${CROSS_ARGS[@]}" --centrality_component "$comp" \
      --out_dir "$OUT_ROOT/exp1_mask/cross_${comp}" "${acc_arg[@]}"
  done
  echo ">>> read: compare exp1_mask/{same,cross}_t5 vs _vit overlap matrices."
fi

# =============================================================================
# Exp2 -- Wanda statistic (GPU extract, then CPU analysis).
#   Per source, extract the per-channel Wanda scaler; then test how much of the
#   mask is fixed by |W| alone vs moved by calibration, and how the L-side
#   statistic responds to swapping the source.
#   NOTE: extract_wanda_statistics runs a MULTIMODAL forward and splits by token
#   group; its 'text' group is a diagnostic, not the exact text-only S_L. The
#   faithful L statistic lives in the pruned masks of Exp1 -- treat Exp2 as the
#   |W|-dominance / sensitivity texture behind that result.
# =============================================================================
if [[ "$RUN_EXP2" == "1" ]]; then
  echo "########## Exp2: Wanda statistic extraction (GPU) ##########"
  : "${DATASETS:?Exp2 needs DATASETS='label|calib_json|images_dir' lines (5 sources + C4)}"
  mkdir -p "$STATS_ROOT"
  STATS_ARGS=()
  while IFS='|' read -r label cjson cimg; do
    [[ -z "${label// }" ]] && continue
    if [[ -z "${cimg// }" ]]; then
      # extract_wanda_statistics.py has only a MULTIMODAL forward (ViT ->
      # Q-Former -> visual prefix + text) and hard-requires --images_dir, so a
      # text-only source (C4) cannot go through it. The text-only L statistic
      # for such a source is covered by Exp4 (--input_mode text_only) and by the
      # real factorized masks in Exp1 -- skip it here rather than crash.
      echo ">>> skip stat [$label]: text-only source, no images -> not applicable to Exp2."
      continue
    fi
    echo ">>> extract stat [$label]"
    python scripts/blip2/extract_wanda_statistics.py \
      --label "$label" --calib_json "$cjson" --images_dir "$cimg" \
      --out_dir "$STATS_ROOT/$label" \
      --max_samples "$MAX_SAMPLES" --batch_size "$BATCH_SIZE" \
      "${dense_stat_arg[@]}"
    STATS_ARGS+=(--stats "$label=$STATS_ROOT/$label")
  done <<< "$DATASETS"

  echo "########## Exp2: statistic analysis (CPU) ##########"
  for grp in all text visual; do
    python scripts/blip2/analyze_calibration_statistics.py \
      "${STATS_ARGS[@]}" --component both --group "$grp" \
      --out_dir "$OUT_ROOT/exp2_stat/group_${grp}" "${acc_arg[@]}"
  done
fi

# =============================================================================
# Exp3 -- activation drift by token group (GPU).
#   Run every config + dense on the SAME held-out multimodal eval rows and split
#   the deviation into visual-prefix / text / answer positions + logit KL.
#   Prediction: across the cross-source configs the TEXT/answer drift barely
#   varies (S_L frozen), while any visual-source signal is confined to the
#   visual prefix and decays before the answer positions.
# =============================================================================
if [[ "$RUN_EXP3" == "1" ]]; then
  echo "########## Exp3: pruned drift by token group (GPU) ##########"
  : "${EVAL_JSON:?Exp3 needs EVAL_JSON (held-out multimodal eval rows)}"
  : "${EVAL_IMAGES:?Exp3 needs EVAL_IMAGES}"
  dense_arg=(); [[ -n "$DENSE_CKPT" ]] && dense_arg=(--dense_ckpt "$DENSE_CKPT")
  # tag each label so same/cross are distinguishable in one drift plot
  DRIFT_ARGS=()
  IFS=',' read -ra _p <<< "$SAME_CKPTS";  for kv in "${_p[@]}"; do [[ -n "${kv// }" ]] && DRIFT_ARGS+=(--ckpt "same_${kv}"); done
  IFS=',' read -ra _p <<< "$CROSS_CKPTS"; for kv in "${_p[@]}"; do [[ -n "${kv// }" ]] && DRIFT_ARGS+=(--ckpt "cross_${kv}"); done
  python scripts/blip2/analyze_pruned_drift_by_token_group.py \
    --eval_json "$EVAL_JSON" --images_dir "$EVAL_IMAGES" \
    "${dense_arg[@]}" "${DRIFT_ARGS[@]}" \
    --out_dir "$OUT_ROOT/exp3_drift" \
    --max_samples "$DRIFT_SAMPLES" --batch_size 2
fi

# =============================================================================
# Exp4 -- LLM input-embedding space (GPU extract, then CPU analysis).
#   Per source capture the pooled visual-prefix and text embeddings that enter
#   L. text_only reproduces the factorized L input; multimodal gives the visual
#   side. This closes the loop: cross-source fixes the TEXT input to C4 for every
#   row, so the L input distribution (hence S_L) is identical -> Exp1's overlap=1.
# =============================================================================
if [[ "$RUN_EXP4" == "1" ]]; then
  echo "########## Exp4: LLM input embeddings (GPU) ##########"
  : "${DATASETS:?Exp4 needs DATASETS (same as Exp2)}"
  mkdir -p "$EMB_ROOT"
  EMB_ARGS=()
  while IFS='|' read -r label cjson cimg; do
    [[ -z "${label// }" ]] && continue
    if [[ -n "${cimg// }" ]]; then
      mode=multimodal; img_arg=(--images_dir "$cimg")
    else
      mode=text_only;  img_arg=()          # C4 / text-only sources
    fi
    echo ">>> embed [$label] mode=$mode"
    python scripts/blip2/extract_llm_input_embeddings.py \
      --label "$label" --calib_json "$cjson" "${img_arg[@]}" \
      --input_mode "$mode" --out_dir "$EMB_ROOT/$label" \
      --max_samples "$MAX_SAMPLES" --batch_size "$BATCH_SIZE" --fp32 \
      "${dense_stat_arg[@]}"
    EMB_ARGS+=(--emb "$label=$EMB_ROOT/$label")
  done <<< "$DATASETS"

  python scripts/blip2/analyze_llm_embeddings.py --mode semantic \
    "${EMB_ARGS[@]}" --part both \
    --out_dir "$OUT_ROOT/exp4_embeddings" "${acc_arg[@]}"
fi

echo "########## DONE ##########"
echo "outputs under: $OUT_ROOT"
echo
echo "How to read the mechanism, end to end:"
echo "  Exp4  cross-source fixes the L text input to C4  -> L input distribution identical across rows"
echo "  Exp2  -> L per-channel Wanda scaler identical; |W| already fixes most of the mask anyway"
echo "  Exp1  -> L keep-mask overlap ~ 1.0 across cross rows (frozen), V overlap < 1 (varies)"
echo "  Exp3  -> the visual-source signal lives in the visual prefix and decays before the answer"
echo "  => accuracy is insensitive to the visual-source choice; the column collapses to its mean."
