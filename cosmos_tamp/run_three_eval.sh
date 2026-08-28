#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/private/workspace/hycui/mfs/cosmos_tamp}"
PROJECT_DIR="${PROJECT_DIR:-/private/workspace/hycui/project/Tamp}"
PYTHON_BIN="${PYTHON_BIN:-/private/workspace/hycui/envs/cosmos3-edge/bin/python}"
MODEL_PATH="${1:-/private/workspace/hycui/model/Cosmos3-Edge}"
TASK_SELECTION="${2:-all}"
OUTPUT_DIR="${3:-/private/workspace/hycui/Results/mfs/cosmos_tamp_eval/$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-0}"

case "${TASK_SELECTION}" in
  all) TASKS="mmbench_en_dev_local,mmmu_val_local,okvqa_val2014_local"; BENCHMARKS="mmbench mmmu okvqa" ;;
  mmbench) TASKS="mmbench_en_dev_local"; BENCHMARKS="mmbench" ;;
  mmmu) TASKS="mmmu_val_local"; BENCHMARKS="mmmu" ;;
  okvqa) TASKS="okvqa_val2014_local"; BENCHMARKS="okvqa" ;;
  *)
    echo "TASK_SELECTION must be one of: all, mmbench, mmmu, okvqa" >&2
    exit 2
    ;;
esac

for required in \
  "${MODEL_PATH}/config.json" \
  "${PROJECT_DIR}/lmms_tasks/mmbench_en_dev_local.yaml" \
  "${PROJECT_DIR}/lmms_tasks/mmmu_local/mmmu_val_local.yaml" \
  "${PROJECT_DIR}/lmms_tasks/okvqa_local/okvqa_val2014_local.yaml"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LMMS_EVAL_PLUGINS="cosmos_lmms_plugin"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hycui_cosmos_lmms_cache}"
# lmms-eval 0.4.0 constructs an unused OpenAI judge while importing the MMMU
# parser. New OpenAI clients require a non-empty key even though this local task
# never calls the judge. Network access remains disabled by the offline flags.
export OPENAI_API_KEY="${OPENAI_API_KEY:-offline-local-parser-only}"

MODEL_ARGS="pretrained=${MODEL_PATH},device=cuda:0,dtype=${DTYPE:-bfloat16},attn_implementation=${ATTN_IMPLEMENTATION:-eager},max_length=${MAX_LENGTH:-4096},min_image_pixels=${MIN_IMAGE_PIXELS:-65536},max_image_pixels=${MAX_IMAGE_PIXELS:-1048576},enable_thinking=${ENABLE_THINKING:-false}"

ARGS=(
  --model cosmos3_edge
  --model_args "${MODEL_ARGS}"
  --tasks "${TASKS}"
  --include_path "${PROJECT_DIR}/lmms_tasks"
  --batch_size 1
  --log_samples
  --output_path "${OUTPUT_DIR}"
  --verbosity INFO
)
if [[ -n "${TAMP_EVAL_LIMIT:-}" ]]; then
  ARGS+=(--limit "${TAMP_EVAL_LIMIT}")
fi

printf 'GPU physical id: %s\n' "${GPU_ID}"
printf 'Model: %s\n' "${MODEL_PATH}"
printf 'Tasks: %s\n' "${TASKS}"
printf 'Output: %s\n' "${OUTPUT_DIR}"
"${PYTHON_BIN}" -m lmms_eval "${ARGS[@]}"

for benchmark in ${BENCHMARKS}; do
  VALIDATE_ARGS=(
    --output-dir "${OUTPUT_DIR}"
    --benchmark "${benchmark}"
  )
  if [[ -n "${TAMP_EVAL_LIMIT:-}" && "${TAMP_EVAL_LIMIT}" =~ ^[0-9]+$ ]]; then
    VALIDATE_ARGS+=(--expected-count "${TAMP_EVAL_LIMIT}")
  fi
  "${PYTHON_BIN}" "${ROOT}/validate_eval_output.py" "${VALIDATE_ARGS[@]}"
done
