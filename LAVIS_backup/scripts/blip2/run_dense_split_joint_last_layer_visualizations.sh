#!/usr/bin/env bash
# =============================================================================
# Record and visualize last-layer output activation similarity:
#   dense full BLIP2-T5 vs pure-Wanda split-merge vs pure-Wanda joint multimodal.
#
# You must provide the forward dataset used for comparison.  Use the SAME
# forward rows for all three model states so differences are attributable to
# pruning, not to input changes.  JSON+images and MMMU/MMBench-style parquet are
# both supported.
#
# Example:
#   FWD_JSON=/data/data2/mfs/MMBench_eval/mmbench_dev.json \
#   FWD_IMAGES=/data/data2/mfs/MMBench_eval/images \
#   FWD_NAME=mmbench_dev \
#   bash scripts/blip2/run_dense_split_joint_last_layer_visualizations.sh
#
# Parquet example:
#   FWD_PARQUET_DIR=/data/data2/mfs/MMBench_eval \
#   FWD_SUBJECT=Other \
#   FWD_SPLIT=dev \
#   FWD_NAME=mmbench_dev \
#   bash scripts/blip2/run_dense_split_joint_last_layer_visualizations.sh
#
# Common overrides:
#   MAX_SAMPLES=128 BATCH_SIZE=2 DEVICE=cuda
#   NO_DECODER=1              # default; set 0 to include T5 decoder
#   RUN_RECORD=0              # reuse existing activation dirs
#   OUT_ROOT=/data/data2/mfs/dense_split_joint_last_layer_vis/mmbench_dev
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

BASE="${BASE:-${AUTODL_TMP:-/data/data2/mfs}}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$BASE/model_cache}"
ECOFLAP_ENV="${ECOFLAP_ENV:-${MODEL_CACHE_ROOT}/ecoflap_model_env.sh}"
if [[ -f "$ECOFLAP_ENV" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "$ECOFLAP_ENV"
  set -u
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$MODEL_CACHE_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_HOME="${TORCH_HOME:-$MODEL_CACHE_ROOT/torch}"

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

DENSE_CKPT="${DENSE_CKPT:-$TORCH_HOME/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
SPLIT_CKPT="${SPLIT_CKPT:-/data/data2/mfs/2/ECoFLaP/LAVIS/pruned_checkpoint/merged_pure_wanda_cc3m_caption_t5_pure_wanda_cc3m_caption_t5only_20260621_025350_seed42__cc3m_image_vit_pure_wanda_cc3m_caption_cc3m_image_vitonly_20260621_025350_seed42.pth}"
JOINT_CKPT="${JOINT_CKPT:-/data/data2/mfs/2/ECoFLaP/LAVIS/pruned_checkpoint/pure_wanda_joint_cc3m128_multimodal_20260621_044626_seed42.pth}"

FWD_JSON="${FWD_JSON:-}"
FWD_IMAGES="${FWD_IMAGES:-}"
FWD_PARQUET="${FWD_PARQUET:-}"
FWD_PARQUET_DIR="${FWD_PARQUET_DIR:-}"
FWD_SPLIT="${FWD_SPLIT:-dev}"
FWD_SUBJECT="${FWD_SUBJECT:-}"

if [[ -n "$FWD_JSON" && "$FWD_JSON" == *.parquet && -z "$FWD_PARQUET" ]]; then
  FWD_PARQUET="$FWD_JSON"
  FWD_JSON=""
fi

if [[ -z "$FWD_JSON" && -z "$FWD_PARQUET" && -z "$FWD_PARQUET_DIR" ]]; then
  echo "[FATAL] set FWD_JSON/FWD_IMAGES or FWD_PARQUET or FWD_PARQUET_DIR." >&2
  exit 1
fi

if [[ -z "${FWD_NAME:-}" ]]; then
  if [[ -n "$FWD_JSON" ]]; then
    FWD_NAME="$(basename "${FWD_JSON%.*}")"
  elif [[ -n "$FWD_PARQUET" ]]; then
    FWD_NAME="$(basename "${FWD_PARQUET%.*}")"
  else
    FWD_NAME="$(basename "$FWD_PARQUET_DIR")_${FWD_SPLIT}"
  fi
fi
MAX_SAMPLES="${MAX_SAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-2}"
DEVICE="${DEVICE:-cuda}"
NO_DECODER="${NO_DECODER:-1}"
RUN_RECORD="${RUN_RECORD:-1}"
JOB_STAMP="${JOB_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$BASE/dense_split_joint_last_layer_vis/${FWD_NAME}_${JOB_STAMP}}"

DENSE_OUT="${DENSE_OUT:-$OUT_ROOT/dense}"
SPLIT_OUT="${SPLIT_OUT:-$OUT_ROOT/split_merge}"
JOINT_OUT="${JOINT_OUT:-$OUT_ROOT/joint_multimodal}"
COMPARE_OUT="${COMPARE_OUT:-$OUT_ROOT/compare}"

COMPONENTS="${COMPONENTS:-vit:all qformer:all t5_encoder:visual t5_encoder:text}"
if [[ "$NO_DECODER" == "0" ]]; then
  COMPONENTS="${COMPONENTS} t5_decoder:decoder"
fi

mkdir -p "$OUT_ROOT" "$DENSE_OUT" "$SPLIT_OUT" "$JOINT_OUT" "$COMPARE_OUT"

if [[ -n "$FWD_PARQUET" || -n "$FWD_PARQUET_DIR" ]]; then
  CONVERT_DIR="$OUT_ROOT/forward_from_parquet"
  CONVERT_JSON="$CONVERT_DIR/forward.json"
  CONVERT_IMAGES="$CONVERT_DIR/images"
  CONVERT_ARGS=(
    --out_json "$CONVERT_JSON"
    --out_images_dir "$CONVERT_IMAGES"
    --max_samples "$MAX_SAMPLES"
    --split "$FWD_SPLIT"
  )
  if [[ -n "$FWD_PARQUET" ]]; then
    CONVERT_ARGS+=(--parquet "$FWD_PARQUET")
  else
    CONVERT_ARGS+=(--parquet_dir "$FWD_PARQUET_DIR")
  fi
  if [[ -n "$FWD_SUBJECT" ]]; then
    CONVERT_ARGS+=(--subject "$FWD_SUBJECT")
  fi

  echo ">>> [convert] parquet forward data"
  python "$SCRIPT_DIR/parquet_forward_to_json_images.py" "${CONVERT_ARGS[@]}"
  FWD_JSON="$CONVERT_JSON"
  FWD_IMAGES="$CONVERT_IMAGES"
else
  : "${FWD_JSON:?set FWD_JSON=/path/to/forward_dataset.json}"
  : "${FWD_IMAGES:?set FWD_IMAGES=/path/to/forward_images_dir}"
fi

echo "========== dense/split/joint last-layer visualization =========="
echo "[INFO] forward json:   $FWD_JSON"
echo "[INFO] forward images: $FWD_IMAGES"
echo "[INFO] dense ckpt:     $DENSE_CKPT"
echo "[INFO] split ckpt:     $SPLIT_CKPT"
echo "[INFO] joint ckpt:     $JOINT_CKPT"
echo "[INFO] out root:       $OUT_ROOT"
echo "[INFO] components:     $COMPONENTS"
echo "==============================================================="

record_one() {
  local label="$1"
  local ckpt="$2"
  local out_dir="$3"
  local extra=()
  if [[ "$NO_DECODER" == "1" ]]; then
    extra+=(--no_decoder)
  fi
  echo ""
  echo ">>> [record] $label"
  python "$SCRIPT_DIR/record_multimodal_per_sample_ffn_activations.py" \
    --calib_json "$FWD_JSON" \
    --images_dir "$FWD_IMAGES" \
    --ckpt "$ckpt" \
    --out_dir "$out_dir" \
    --max_samples "$MAX_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    "${extra[@]}"
}

if [[ "$RUN_RECORD" == "1" ]]; then
  record_one "dense" "$DENSE_CKPT" "$DENSE_OUT"
  record_one "split_merge" "$SPLIT_CKPT" "$SPLIT_OUT"
  record_one "joint_multimodal" "$JOINT_CKPT" "$JOINT_OUT"
else
  echo "[INFO] RUN_RECORD=0, reuse existing activation directories."
fi

echo ""
echo ">>> [visualize] dense reference paired similarity"
# shellcheck disable=SC2206
COMPONENT_ARRAY=($COMPONENTS)
python "$SCRIPT_DIR/compare_dense_split_joint_last_layer_outputs.py" \
  --dense "$DENSE_OUT" \
  --split "$SPLIT_OUT" \
  --joint "$JOINT_OUT" \
  --out_dir "$COMPARE_OUT" \
  --components "${COMPONENT_ARRAY[@]}"

echo ""
echo "[OK] wrote outputs:"
echo "  dense activations: $DENSE_OUT"
echo "  split activations: $SPLIT_OUT"
echo "  joint activations: $JOINT_OUT"
echo "  visualizations:    $COMPARE_OUT"
