#!/usr/bin/env bash
# =============================================================================
# SparseGPT unimodal split pruning for five calibration datasets.
#
# For each calibration dataset:
#   cc3m / mmbench / mmmu / okvqa / mathvista
#
# this script does NOT run joint multimodal pruning.  Instead it:
#   1. extracts text/question/caption fields into a text-only JSON;
#   2. keeps the corresponding images in an image-only JSON;
#   3. prunes T5 only with SparseGPT on the text-only JSON;
#   4. prunes ViT only with SparseGPT on the images;
#   5. merges the T5-only and ViT-only checkpoints;
#   6. runs four evals: MMBench / MMMU / OKVQA / MathVista.
#
# Usage from ECoFLaP/LAVIS:
#   bash scripts/blip2/run_sparsegpt_fivecalib_unimodal_split_then_fourbench_eval.sh
#
# Run a subset:
#   CALIBS="cc3m mmbench" bash scripts/blip2/run_sparsegpt_fivecalib_unimodal_split_then_fourbench_eval.sh
#
# Common overrides:
#   BASE=/data/data2/mfs
#   NUM_DATA=128 PRUNING_CALIB_BATCH=8 RUN_EVAL=0
#   CC3M_CALIB_JSON=... CC3M_CALIB_IMAGES=...
#   MMBENCH_CALIB_JSON=... MMBENCH_CALIB_IMAGES=...
#   MMMU_CALIB_JSON=... MMMU_CALIB_IMAGES=...
#   OKVQA_CALIB_JSON=... OKVQA_CALIB_IMAGES=...
#   MATHVISTA_CALIB_JSON=... MATHVISTA_CALIB_IMAGES=...
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ "${SPARSEGPT_FIVECALIB_SPLIT_INNER:-0}" != "1" ]]; then
  if [[ "${CALIB:-}" == "all" ]]; then
    CALIB="cc3m mmbench mmmu okvqa mathvista"
  fi
  CALIBS="${CALIBS:-${CALIB:-cc3m mmbench mmmu okvqa mathvista}}"
  JOB_STAMP="${JOB_STAMP:-$(date +%Y%m%d_%H%M%S)}"
  export JOB_STAMP
  read -r -a _calibs <<< "$CALIBS"
  if [[ "${#_calibs[@]}" -eq 0 ]]; then
    echo "[FATAL] CALIBS is empty" >&2
    exit 1
  fi
  if [[ "${#_calibs[@]}" -gt 1 ]]; then
    for _calib in "${_calibs[@]}"; do
      echo ""
      echo "========================================================================"
      echo "== SparseGPT unimodal split calibration: ${_calib}"
      echo "========================================================================"
      CALIB="$_calib" SPARSEGPT_FIVECALIB_SPLIT_INNER=1 bash "$0" "$@"
    done
    echo ""
    echo "========================================================================"
    echo "== all SparseGPT unimodal split calibrations finished: ${CALIBS}"
    echo "========================================================================"
    exit 0
  fi
  export CALIB="${_calibs[0]}"
  export SPARSEGPT_FIVECALIB_SPLIT_INNER=1
fi

BASE="${BASE:-${AUTODL_TMP:-/data/data2/mfs}}"
AUTODL_TMP="$BASE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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
mkdir -p "${HF_HOME}/hub" "${HF_HOME}/transformers" "${TORCH_HOME}/hub/checkpoints" 2>/dev/null || true

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
export BLIP2_PRETRAINED

_resolve_eva_vit_g() {
  local candidates=(
    "${EVA_VIT_G_PTH:-}"
    "${TORCH_HOME}/hub/checkpoints/eva_vit_g.pth"
    "${MODEL_CACHE_ROOT}/torch/hub/checkpoints/eva_vit_g.pth"
    "${HOME}/.cache/torch/hub/checkpoints/eva_vit_g.pth"
    /root/autodl-tmp/cache_moved/torch/hub/checkpoints/eva_vit_g.pth
  )
  local p
  for p in "${candidates[@]}"; do
    [[ -n "$p" && -f "$p" ]] || continue
    echo "$p"
    return 0
  done
  return 1
}
if EVA_VIT_G_PTH="$(_resolve_eva_vit_g || true)" && [[ -n "$EVA_VIT_G_PTH" ]]; then
  export EVA_VIT_G_PTH
fi

CALIB="${CALIB:-cc3m}"
case "$CALIB" in
  cc3m) CALIB_TAG="cc3m" ;;
  mmbench) CALIB_TAG="mmbench" ;;
  mmmu) CALIB_TAG="mmmu" ;;
  okvqa) CALIB_TAG="okvqa" ;;
  mathvista) CALIB_TAG="mathvista" ;;
  *)
    echo "[FATAL] CALIB must be one of: cc3m mmbench mmmu okvqa mathvista. Got: $CALIB" >&2
    exit 1
    ;;
esac

case "$CALIB_TAG" in
  cc3m)
    RAW_JSON="${CC3M_CALIB_JSON:-$BASE/CC3M_calib_128/cc3m_calib_128.json}"
    RAW_IMAGES="${CC3M_CALIB_IMAGES:-$BASE/CC3M_calib_128/images}"
    ;;
  mmbench)
    RAW_JSON="${MMBENCH_CALIB_JSON:-$BASE/MMBench_calibration/mmbench_calibration_train.json}"
    RAW_IMAGES="${MMBENCH_CALIB_IMAGES:-$BASE/MMBench_calibration/images}"
    ;;
  mmmu)
    RAW_JSON="${MMMU_CALIB_JSON:-$BASE/MMMU_calibration/mmmu_calibration_train.json}"
    RAW_IMAGES="${MMMU_CALIB_IMAGES:-$BASE/MMMU_calibration/images}"
    ;;
  okvqa)
    RAW_JSON="${OKVQA_CALIB_JSON:-$BASE/datasets/okvqa/annotations/okvqa_train.json}"
    RAW_IMAGES="${OKVQA_CALIB_IMAGES:-$BASE/datasets/okvqa_official/images}"
    ;;
  mathvista)
    RAW_JSON="${MATHVISTA_CALIB_JSON:-$BASE/MathVista_calibration/mathvista_calibration_train.json}"
    RAW_IMAGES="${MATHVISTA_CALIB_IMAGES:-$BASE/MathVista_calibration/images}"
    ;;
esac

CFG_T5_BASE="${CFG_T5_BASE:-$REPO_ROOT/lavis/projects/blip2/eval/t5_c4_text_prune_calib.yaml}"
CFG_VIT_BASE="${CFG_VIT_BASE:-$REPO_ROOT/lavis/projects/blip2/eval/cc_prefix_derivative_compute_cc3m_calib128.yaml}"

T5_SPEC="${T5_SPEC:-${T5_PRUNE_SPEC:-24-0.5-1.0-1.0}}"
VIT_SPEC="${VIT_SPEC:-${VIT_PRUNE_SPEC:-39-0.5-1.0-1.0}}"
NUM_DATA="${NUM_DATA:-128}"
PRUNING_CALIB_BATCH="${PRUNING_CALIB_BATCH:-${BS:-8}}"
NUM_DATA_FIRST_STAGE="${NUM_DATA_FIRST_STAGE:-$NUM_DATA}"
SCORE_METHOD="${SCORE_METHOD:-MEZO-GradOnly_sum}"
MAX_SPARSITY_PER_LAYER="${MAX_SPARSITY_PER_LAYER:-0.6}"
MASTER_PORT_VIT="${MASTER_PORT_VIT:-29517}"
MASTER_PORT_EVAL="${MASTER_PORT_EVAL:-29600}"

SEED="${SEED:-${LAVIS_DISTRIBUTED_SAMPLER_SEED:-42}}"
export LAVIS_DISTRIBUTED_SAMPLER_SEED="$SEED"
JOB_STAMP="${JOB_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PRUNE_T5="${RUN_PRUNE_T5:-$RUN_PRUNE}"
RUN_PRUNE_VIT="${RUN_PRUNE_VIT:-$RUN_PRUNE}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"

if (( NUM_DATA % PRUNING_CALIB_BATCH != 0 )); then
  echo "[FATAL] NUM_DATA ($NUM_DATA) must be divisible by PRUNING_CALIB_BATCH ($PRUNING_CALIB_BATCH)." >&2
  exit 1
fi

export ECOFLAP_BENCH_ROOT="${ECOFLAP_BENCH_ROOT:-$BASE}"
export MMBENCH_ROOT="${MMBENCH_ROOT:-$BASE/MMBench_eval}"
export MMMU_ROOT="${MMMU_ROOT:-$BASE/MMMU_single_image}"
export MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
export MMMU_SPLIT="${MMMU_SPLIT:-test}"
export MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-$BASE/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
export MATHVISTA_IMAGES_DIR="${MATHVISTA_IMAGES_DIR:-$BASE/MathVista_eval_testmini_mc/images}"

JID_T5="sparsegpt_split_${CALIB_TAG}_text_t5only_${JOB_STAMP}_seed${SEED}"
JID_VIT="sparsegpt_split_${CALIB_TAG}_image_vitonly_${JOB_STAMP}_seed${SEED}"
CKPT_T5="$REPO_ROOT/pruned_checkpoint/${JID_T5}.pth"
CKPT_VIT="$REPO_ROOT/pruned_checkpoint/${JID_VIT}.pth"
MERGED_CKPT="${MERGED_CKPT:-$REPO_ROOT/pruned_checkpoint/merged_sparsegpt_split_${CALIB_TAG}_text_t5_${JID_T5}__image_vit_${JID_VIT}.pth}"

METRICS_JSONL="${LAVIS_METRICS_JSONL:-$REPO_ROOT/lavis/output/BLIP2/sparsegpt_split_${CALIB_TAG}_${JOB_STAMP}_seed${SEED}.jsonl}"
SUMMARY_MD="${SUMMARY_MD:-$REPO_ROOT/lavis/output/BLIP2/sparsegpt_split_${CALIB_TAG}_${JOB_STAMP}_seed${SEED}.md}"
SUMMARY_TSV="${SUMMARY_TSV:-$REPO_ROOT/lavis/output/BLIP2/sparsegpt_split_${CALIB_TAG}_${JOB_STAMP}_seed${SEED}.tsv}"
EVAL_TAG="sparsegpt_split_${CALIB_TAG}_${JOB_STAMP}_seed${SEED}"
EVAL_JOB_OKVQA="${EVAL_JOB_OKVQA:-okvqa_eval_${EVAL_TAG}_fullval}"

CFG_WORK_DIR="${CFG_WORK_DIR:-$REPO_ROOT/lavis/output/BLIP2/runtime_cfgs}"
SPLIT_DATA_DIR="${SPLIT_DATA_DIR:-$REPO_ROOT/lavis/output/BLIP2/runtime_calib_splits}"
RUNTIME_T5_CFG="$CFG_WORK_DIR/sparsegpt_split_${CALIB_TAG}_t5_${JOB_STAMP}_seed${SEED}.yaml"
RUNTIME_VIT_CFG="$CFG_WORK_DIR/sparsegpt_split_${CALIB_TAG}_vit_${JOB_STAMP}_seed${SEED}.yaml"
TEXT_JSON="$SPLIT_DATA_DIR/${CALIB_TAG}_${JOB_STAMP}_seed${SEED}_text_only.json"
IMAGE_JSON="$SPLIT_DATA_DIR/${CALIB_TAG}_${JOB_STAMP}_seed${SEED}_image_only.json"
MANIFEST_JSON="$SPLIT_DATA_DIR/${CALIB_TAG}_${JOB_STAMP}_seed${SEED}_manifest.json"

mkdir -p "$REPO_ROOT/pruned_checkpoint" "$REPO_ROOT/lavis/output/BLIP2" "$REPO_ROOT/training_statistics" "$CFG_WORK_DIR" "$SPLIT_DATA_DIR"

_write_runtime_cfg() {
  local src="$1"
  local dst="$2"
  local ann="$3"
  local images="$4"
  python - "$src" "$dst" "$BLIP2_PRETRAINED" "$ann" "$images" <<'PY'
import os
import sys
import yaml

src, dst, pretrained, ann, images = sys.argv[1:6]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("model", {})["pretrained"] = pretrained
datasets = cfg.setdefault("datasets", {})
if not datasets:
    datasets["prefix_conceptual_caption_3m"] = {}
for ds in datasets.values():
    build_info = ds.setdefault("build_info", {})
    ann_info = build_info.setdefault("annotations", {}).setdefault("train", {})
    ann_info["url"] = [ann]
    ann_info["storage"] = [ann]
    build_info.setdefault("images", {})["storage"] = images
os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print(dst)
PY
}

_prepare_unimodal_jsons() {
  python - "$RAW_JSON" "$RAW_IMAGES" "$TEXT_JSON" "$IMAGE_JSON" "$MANIFEST_JSON" "$NUM_DATA" "$CALIB_TAG" <<'PY'
import json
import os
import sys

raw_json, images_dir, text_json, image_json, manifest_json, num_data, tag = sys.argv[1:8]
num_data = int(num_data)

def value_to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(value_to_text(v) for v in value if value_to_text(v)).strip()
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            txt = value_to_text(val)
            if txt:
                parts.append(f"{key}. {txt}")
        return "\n".join(parts).strip()
    return str(value).strip()

def build_text(row):
    fields = ("text", "caption", "text_input", "question", "prompt", "output")
    selected = ""
    selected_field = ""
    for field in fields:
        if field in row:
            selected = value_to_text(row.get(field))
            if selected:
                selected_field = field
                break
    if not selected:
        raise ValueError("missing text/caption/question field")

    parts = [selected]
    hint = value_to_text(row.get("hint"))
    if hint and "hint:" not in selected.lower():
        parts.append("Hint: " + hint)
    options = row.get("options")
    if options:
        opt_text = value_to_text(options)
        if opt_text and opt_text not in selected:
            parts.append(opt_text)
    for letter in "ABCDEFG":
        opt = value_to_text(row.get(letter))
        if opt and f"{letter}." not in selected:
            parts.append(f"{letter}. {opt}")
    return "\n".join(parts).strip(), selected_field

if not os.path.isfile(raw_json):
    raise SystemExit(f"[FATAL] raw calibration JSON not found: {raw_json}")
if not os.path.isdir(images_dir):
    raise SystemExit(f"[FATAL] images dir not found: {images_dir}")
with open(raw_json, "r", encoding="utf-8") as f:
    rows = json.load(f)
if not isinstance(rows, list) or not rows:
    raise SystemExit(f"[FATAL] {raw_json} must be a non-empty JSON list")
if len(rows) < num_data:
    raise SystemExit(f"[FATAL] {tag} has only {len(rows)} rows, need NUM_DATA={num_data}")

text_rows = []
image_rows = []
field_counts = {}
for i, row in enumerate(rows[:num_data]):
    if not isinstance(row, dict):
        raise SystemExit(f"[FATAL] row {i} is not a JSON object")
    image = value_to_text(row.get("image"))
    if not image:
        raise SystemExit(f"[FATAL] row {i} missing image")
    full = image if os.path.isabs(image) else os.path.join(images_dir, image)
    if not os.path.isfile(full):
        raise SystemExit(f"[FATAL] row {i} image file not found: {full}")
    text, field = build_text(row)
    field_counts[field] = field_counts.get(field, 0) + 1
    text_rows.append({"text": text})
    out = dict(row)
    out["image"] = image
    out["caption"] = text
    out["text_input"] = text
    out.setdefault("answer", value_to_text(row.get("answer")))
    image_rows.append(out)

os.makedirs(os.path.dirname(os.path.abspath(text_json)), exist_ok=True)
with open(text_json, "w", encoding="utf-8") as f:
    json.dump(text_rows, f, ensure_ascii=False, indent=2)
    f.write("\n")
with open(image_json, "w", encoding="utf-8") as f:
    json.dump(image_rows, f, ensure_ascii=False, indent=2)
    f.write("\n")
manifest = {
    "tag": tag,
    "raw_json": raw_json,
    "images_dir": images_dir,
    "num_data": num_data,
    "text_json": text_json,
    "image_json": image_json,
    "text_field_counts": field_counts,
}
with open(manifest_json, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(f"[OK] wrote text_only={text_json}")
print(f"[OK] wrote image_only={image_json}")
print(f"[OK] text fields={field_counts}")
PY
}

_preflight() {
  if [[ ! -f "$BLIP2_PRETRAINED" ]]; then
    echo "[FATAL] BLIP2_PRETRAINED not found: $BLIP2_PRETRAINED" >&2
    exit 1
  fi
  if [[ ! -d "${BERT_BASE_UNCASED_SNAPSHOT:-}" ]]; then
    echo "[FATAL] bert-base-uncased snapshot not found. Set BERT_BASE_UNCASED_SNAPSHOT or HF_HOME." >&2
    exit 1
  fi
  if [[ ! -d "${FLAN_T5_XL_SNAPSHOT:-}" ]]; then
    echo "[FATAL] flan-t5-xl snapshot not found. Set FLAN_T5_XL_SNAPSHOT or HF_HOME." >&2
    exit 1
  fi
  if [[ -z "${EVA_VIT_G_PTH:-}" || ! -f "$EVA_VIT_G_PTH" ]]; then
    echo "[FATAL] eva_vit_g.pth not found. Set EVA_VIT_G_PTH or TORCH_HOME." >&2
    exit 1
  fi
  if [[ ! -f "$CFG_T5_BASE" ]]; then
    echo "[FATAL] CFG_T5_BASE not found: $CFG_T5_BASE" >&2
    exit 1
  fi
  if [[ ! -f "$CFG_VIT_BASE" ]]; then
    echo "[FATAL] CFG_VIT_BASE not found: $CFG_VIT_BASE" >&2
    exit 1
  fi
  _prepare_unimodal_jsons
  _write_runtime_cfg "$CFG_T5_BASE" "$RUNTIME_T5_CFG" "$IMAGE_JSON" "$RAW_IMAGES" >/dev/null
  _write_runtime_cfg "$CFG_VIT_BASE" "$RUNTIME_VIT_CFG" "$IMAGE_JSON" "$RAW_IMAGES" >/dev/null
}

run_prune_t5_text_only() {
  echo ""
  echo ">>> [prune 1/2] SparseGPT | ${CALIB_TAG} text only | prune T5"
  local extra=()
  if [[ "${T5_ENCODER_ONLY:-0}" == "1" ]]; then
    extra+=(--t5_c4_encoder_only)
  fi
  python evaluate_blip.py \
    --cfg-path "$RUNTIME_T5_CFG" \
    --options "model.pretrained=${BLIP2_PRETRAINED}" \
    --prune_calib_mode t5_c4_text \
    --importance_scope llm_only \
    --c4_calib_json "$TEXT_JSON" \
    --no_prune_vit \
    --pruning_method blipt5_sparsegpt_pruner \
    --t5_prune_spec "$T5_SPEC" \
    --score_method "$SCORE_METHOD" \
    --max_sparsity_per_layer "$MAX_SPARSITY_PER_LAYER" \
    --num_data "$NUM_DATA" \
    --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
    --prunining_dataset_batch_size "$PRUNING_CALIB_BATCH" \
    --job_id "$JID_T5" \
    --save_pruned_model \
    "${extra[@]}"
  [[ -f "$CKPT_T5" ]] || { echo "[FATAL] missing T5 ckpt after pruning: $CKPT_T5" >&2; exit 1; }
}

run_prune_vit_image_only() {
  echo ""
  echo ">>> [prune 2/2] SparseGPT | ${CALIB_TAG} images only | prune ViT"
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$MASTER_PORT_VIT" evaluate_blip.py \
    --cfg-path "$RUNTIME_VIT_CFG" \
    --options "model.pretrained=${BLIP2_PRETRAINED}" \
    --prune_calib_mode vit_image_only \
    --importance_scope vit_only_encode \
    --no_prune_t5 \
    --pruning_method blipt5_sparsegpt_pruner \
    --vit_prune_spec "$VIT_SPEC" \
    --score_method "$SCORE_METHOD" \
    --max_sparsity_per_layer "$MAX_SPARSITY_PER_LAYER" \
    --num_data "$NUM_DATA" \
    --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
    --prunining_dataset_batch_size "$PRUNING_CALIB_BATCH" \
    --job_id "$JID_VIT" \
    --save_pruned_model
  [[ -f "$CKPT_VIT" ]] || { echo "[FATAL] missing ViT ckpt after pruning: $CKPT_VIT" >&2; exit 1; }
}

echo "========== SparseGPT unimodal split: ${CALIB_TAG} text -> T5 + images -> ViT =========="
echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] RAW_JSON=$RAW_JSON"
echo "[INFO] RAW_IMAGES=$RAW_IMAGES"
echo "[INFO] TEXT_JSON=$TEXT_JSON"
echo "[INFO] IMAGE_JSON=$IMAGE_JSON"
echo "[INFO] T5 ckpt: $CKPT_T5"
echo "[INFO] ViT ckpt: $CKPT_VIT"
echo "[INFO] Merged:   $MERGED_CKPT"
echo "[INFO] RUN_PRUNE=$RUN_PRUNE (T5=$RUN_PRUNE_T5 ViT=$RUN_PRUNE_VIT) RUN_EVAL=$RUN_EVAL"
echo "====================================================================================="

_preflight

if [[ "$RUN_PRUNE_T5" == "1" ]]; then
  run_prune_t5_text_only
else
  echo "[INFO] RUN_PRUNE_T5=0, using existing T5 ckpt: $CKPT_T5"
  [[ -f "$CKPT_T5" ]] || { echo "[FATAL] missing T5 ckpt: $CKPT_T5" >&2; exit 1; }
fi

if [[ "$RUN_PRUNE_VIT" == "1" ]]; then
  run_prune_vit_image_only
else
  echo "[INFO] RUN_PRUNE_VIT=0, using existing ViT ckpt: $CKPT_VIT"
  [[ -f "$CKPT_VIT" ]] || { echo "[FATAL] missing ViT ckpt: $CKPT_VIT" >&2; exit 1; }
fi

if [[ "$RUN_EVAL" != "1" ]]; then
  echo "[INFO] RUN_EVAL=0, skip merge and benchmark evaluation"
  echo "[INFO] T5: $CKPT_T5"
  echo "[INFO] ViT: $CKPT_VIT"
  exit 0
fi

: > "$METRICS_JSONL"

export CKPT_T5_ONLY="$CKPT_T5"
export CKPT_VIT_ONLY="$CKPT_VIT"
export MERGED_CKPT
export RUN_MERGE=1
export RUN_EVAL=1
export EVAL_BATCH_SIZE
export MASTER_PORT="$MASTER_PORT_EVAL"
export EVAL_JOB_OKVQA
export LAVIS_METRICS_JSONL="$METRICS_JSONL"
export LAVIS_EVAL_CALIB_TAG="$EVAL_TAG"

echo ""
echo ">>> [merge + four benchmark eval] run_ecoflap_split_merge_eval_fourbench.sh"
bash "$SCRIPT_DIR/run_ecoflap_split_merge_eval_fourbench.sh"

if [[ -f "$REPO_ROOT/scripts/blip2/collect_ecoflap_eval_summary.py" ]]; then
  echo ""
  echo ">>> [summary] collect_ecoflap_eval_summary.py"
  python "$REPO_ROOT/scripts/blip2/collect_ecoflap_eval_summary.py" \
    --repo-root "$REPO_ROOT" \
    --metrics-jsonl "$METRICS_JSONL" \
    --out-md "$SUMMARY_MD" \
    --out-tsv "$SUMMARY_TSV" \
    --suites "${EVAL_TAG}:${EVAL_JOB_OKVQA}" || true
fi

echo ""
echo "========== ALL DONE =========="
echo "  Calibration:   $CALIB_TAG"
echo "  T5 text-only:  $CKPT_T5"
echo "  ViT image-only: $CKPT_VIT"
echo "  Merged:        $MERGED_CKPT"
echo "  Metrics:       $METRICS_JSONL"
echo "  Summary:       $SUMMARY_MD"
