#!/usr/bin/env bash
# =============================================================================
# Pure Wanda pruning with the PPO-selected MMBench calibration set, then run
# four evaluations: MMBench / MMMU / OKVQA / MathVista.
#
# Intended calibration JSON:
#   mmbench_calib_ppo_selected128_from_full4329.json
#
# Usage from ECoFLaP/LAVIS:
#   bash scripts/blip2/run_pure_wanda_mmbench_ppo_selected128_then_fourbench_eval.sh
#
# Common overrides:
#   BASE=/data/data2/mfs
#   MMBENCH_PPO_CALIB_JSON=/path/to/mmbench_calib_ppo_selected128_from_full4329.json
#   MMBENCH_PPO_CALIB_IMAGES=/path/to/MMBench_calibration/images
#   NUM_DATA=128 BS=8 RUN_EVAL=0 PRUNE_VIT=1
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
mkdir -p "${HF_HOME}/hub" "${HF_HOME}/transformers" "${TORCH_HOME}/hub/checkpoints" \
  "$REPO_ROOT/pruned_checkpoint" "$REPO_ROOT/training_statistics" "$REPO_ROOT/lavis/output/BLIP2/runtime_cfgs" \
  2>/dev/null || true

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

if [[ ! -d "$BERT_BASE_UNCASED_SNAPSHOT" ]]; then
  echo "[FATAL] Missing bert-base-uncased snapshot. Set HF_HOME or BERT_BASE_UNCASED_SNAPSHOT." >&2
  exit 1
fi
if [[ ! -d "$FLAN_T5_XL_SNAPSHOT" ]]; then
  echo "[FATAL] Missing google/flan-t5-xl snapshot. Set HF_HOME or FLAN_T5_XL_SNAPSHOT." >&2
  exit 1
fi

BLIP2_PRETRAINED="${BLIP2_PRETRAINED:-$TORCH_HOME/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
if [[ ! -f "$BLIP2_PRETRAINED" ]]; then
  BLIP2_PRETRAINED="$BASE/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth"
fi
if [[ ! -f "$BLIP2_PRETRAINED" ]]; then
  echo "[FATAL] Missing BLIP2_PRETRAINED: $BLIP2_PRETRAINED" >&2
  exit 1
fi

_pick_existing_file() {
  local f
  for f in "$@"; do
    [[ -n "$f" && -f "$f" ]] || continue
    echo "$f"
    return 0
  done
  return 1
}

MMBENCH_PPO_CALIB_JSON="${MMBENCH_PPO_CALIB_JSON:-}"
if [[ -z "$MMBENCH_PPO_CALIB_JSON" ]]; then
  MMBENCH_PPO_CALIB_JSON="$(_pick_existing_file \
    "$BASE/MMBench_calibration/mmbench_calib_ppo_selected128_from_full4329.json" \
    "$BASE/mmbench_calib_ppo_selected128_from_full4329.json" \
    "$REPO_ROOT/lavis/output/BLIP2/mmbench_calib_ppo_selected128_from_full4329.json" \
    "$REPO_ROOT/lavis/output/BLIP2/ppo_selection/mmbench_calib_ppo_selected128_from_full4329.json" \
    "$(pwd)/mmbench_calib_ppo_selected128_from_full4329.json" \
    || true)"
fi
if [[ -z "$MMBENCH_PPO_CALIB_JSON" || ! -f "$MMBENCH_PPO_CALIB_JSON" ]]; then
  echo "[FATAL] Could not find mmbench_calib_ppo_selected128_from_full4329.json." >&2
  echo "        Set MMBENCH_PPO_CALIB_JSON=/path/to/mmbench_calib_ppo_selected128_from_full4329.json" >&2
  exit 1
fi

MMBENCH_PPO_CALIB_IMAGES="${MMBENCH_PPO_CALIB_IMAGES:-${MMBENCH_CALIB_IMAGES:-$BASE/MMBench_calibration/images}}"
if [[ ! -d "$MMBENCH_PPO_CALIB_IMAGES" ]]; then
  echo "[FATAL] Missing MMBench calibration image dir: $MMBENCH_PPO_CALIB_IMAGES" >&2
  echo "        Set MMBENCH_PPO_CALIB_IMAGES=/path/to/images" >&2
  exit 1
fi

BASE_CFG="${BASE_CFG:-$REPO_ROOT/lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_mmbench.yaml}"
if [[ ! -f "$BASE_CFG" ]]; then
  echo "[FATAL] Missing base cfg: $BASE_CFG" >&2
  exit 1
fi

SEED="${SEED:-${LAVIS_DISTRIBUTED_SAMPLER_SEED:-42}}"
export LAVIS_DISTRIBUTED_SAMPLER_SEED="$SEED"
JOB_STAMP="${JOB_STAMP:-$(date +%Y%m%d_%H%M%S)}"
JOB_ID="${JOB_ID:-pure_wanda_mmbench_ppo_selected128_from_full4329_${JOB_STAMP}_seed${SEED}}"
RUNTIME_CFG="${RUNTIME_CFG:-$REPO_ROOT/lavis/output/BLIP2/runtime_cfgs/${JOB_ID}.yaml}"
JOINT_CKPT="$REPO_ROOT/pruned_checkpoint/${JOB_ID}.pth"

NUM_DATA="${NUM_DATA:-128}"
BS="${BS:-${PRUNING_CALIB_BATCH:-8}}"
NUM_DATA_FIRST_STAGE="${NUM_DATA_FIRST_STAGE:-32}"
T5_SPEC="${T5_SPEC:-${T5_PRUNE_SPEC:-24-0.5-1.0-1.0}}"
VIT_SPEC="${VIT_SPEC:-${VIT_PRUNE_SPEC:-39-0.5-1.0-1.0}}"
PRUNE_VIT="${PRUNE_VIT:-0}"
RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
MASTER_PORT_PRUNE="${MASTER_PORT_PRUNE:-29530}"
MASTER_PORT_EVAL="${MASTER_PORT_EVAL:-29600}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"

if (( NUM_DATA % BS != 0 )); then
  echo "[FATAL] NUM_DATA ($NUM_DATA) must be divisible by BS ($BS)." >&2
  exit 1
fi

python - "$BASE_CFG" "$RUNTIME_CFG" "$MMBENCH_PPO_CALIB_JSON" "$MMBENCH_PPO_CALIB_IMAGES" "$JOB_ID" <<'PY'
import copy
import os
import sys

base_cfg, out_cfg, calib_json, images_dir, job_id = sys.argv[1:6]

try:
    import yaml
except Exception:
    yaml = None

if yaml is None:
    with open(base_cfg, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.replace(
        "/root/autodl-tmp/MMBench_calibration/mmbench_calibration_train.json",
        calib_json,
    )
    text = text.replace("/root/autodl-tmp/MMBench_calibration/images", images_dir)
    text = text.replace('output_dir: "output/BLIP2/OKVQA_calibration"', f'output_dir: "output/BLIP2/{job_id}"')
else:
    with open(base_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ds = cfg["datasets"]["prefix_okvqa_calibration"]["build_info"]
    ds["annotations"]["train"]["storage"] = [calib_json]
    ds["images"]["storage"] = images_dir
    cfg.setdefault("run", {})["output_dir"] = "output/BLIP2/%s" % job_id
    text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)

os.makedirs(os.path.dirname(out_cfg), exist_ok=True)
with open(out_cfg, "w", encoding="utf-8") as f:
    f.write(text)
PY

echo "[INFO] calibration json: $MMBENCH_PPO_CALIB_JSON"
echo "[INFO] calibration images: $MMBENCH_PPO_CALIB_IMAGES"
echo "[INFO] runtime cfg: $RUNTIME_CFG"
echo "[INFO] job id: $JOB_ID"
echo "[INFO] output ckpt: $JOINT_CKPT"

export ECOFLAP_BENCH_ROOT="${ECOFLAP_BENCH_ROOT:-$BASE}"
export MMBENCH_ROOT="${MMBENCH_ROOT:-$BASE/MMBench_eval}"
export MMMU_ROOT="${MMMU_ROOT:-$BASE/MMMU_single_image}"
export MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
export MMMU_SPLIT="${MMMU_SPLIT:-test}"
export MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-$BASE/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
export MATHVISTA_IMAGES_DIR="${MATHVISTA_IMAGES_DIR:-$BASE/MathVista_eval_testmini_mc/images}"
export LAVIS_METRICS_JSONL="${LAVIS_METRICS_JSONL:-$REPO_ROOT/training_statistics/${JOB_ID}_fourbench_metrics.jsonl}"

PRUNE_EXTRA=()
if [[ "$PRUNE_VIT" == "1" ]]; then
  PRUNE_EXTRA+=(--prune_vit)
fi

if [[ "$RUN_PRUNE" == "1" ]]; then
  python -m torch.distributed.run --nproc_per_node=1 --master_port "$MASTER_PORT_PRUNE" evaluate_blip.py \
    --cfg-path "$RUNTIME_CFG" \
    --options model.pretrained="${BLIP2_PRETRAINED}" \
    --pruning_method blipt5_wanda_pruner \
    --save_pruned_model \
    --prunining_dataset_batch_size "$BS" \
    --num_data "$NUM_DATA" \
    --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
    --t5_prune_spec "$T5_SPEC" \
    --vit_prune_spec "$VIT_SPEC" \
    "${PRUNE_EXTRA[@]}" \
    --job_id "$JOB_ID" \
    "$@"
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  if [[ ! -f "$JOINT_CKPT" ]]; then
    echo "[FATAL] Missing pruned checkpoint: $JOINT_CKPT" >&2
    exit 1
  fi
  export JOINT_SINGLE_CKPT="$JOINT_CKPT"
  export MASTER_PORT="$MASTER_PORT_EVAL"
  export EVAL_BATCH_SIZE
  export EVAL_JOB_OKVQA="${EVAL_JOB_OKVQA:-okvqa_eval_${JOB_ID}}"
  bash "$SCRIPT_DIR/run_ecoflap_split_merge_eval_fourbench.sh"
fi

echo "[OK] finished pure Wanda MMBench PPO-selected calibration run."
echo "[OK] ckpt: $JOINT_CKPT"
echo "[OK] metrics: $LAVIS_METRICS_JSONL"
