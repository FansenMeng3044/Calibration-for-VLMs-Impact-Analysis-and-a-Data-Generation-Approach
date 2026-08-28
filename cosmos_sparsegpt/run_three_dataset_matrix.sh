#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
  echo "usage: $0 {joint|separate} PHYSICAL_GPU RESULTS_ROOT [NSAMPLES]" >&2
  exit 2
fi

protocol="$1"
physical_gpu="$2"
results_root="$3"
nsamples="${4:-128}"
datasets="${COSMOS_SPARSEGPT_DATASETS:-mmbench mmmu okvqa}"
code_root="/private/workspace/hycui/mfs/cosmos_sparsegpt"
python_bin="/private/workspace/hycui/envs/cosmos3-edge/bin/python"

if [[ "${protocol}" != "joint" && "${protocol}" != "separate" ]]; then
  echo "protocol must be joint or separate" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONUNBUFFERED=1

# Validate every selected source before loading model weights.
for dataset in ${datasets}; do
  "${python_bin}" "${code_root}/cosmos_sparsegpt_prune.py" \
    --protocol "${protocol}" \
    --calibration-preset "${dataset}" \
    --nsamples "${nsamples}" \
    --preflight-only >/dev/null
done

for dataset in ${datasets}; do
  output_dir="${results_root}/${dataset}"
  log_path="${results_root}/${dataset}.log"
  if [[ -s "${output_dir}/state.json" ]] && \
     [[ "$(jq -r '.phase // empty' "${output_dir}/state.json")" == "complete" ]]; then
    echo "skip complete: ${output_dir}"
    continue
  fi
  if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "refusing non-empty incomplete output: ${output_dir}" >&2
    exit 3
  fi
  mkdir -p "${results_root}"
  "${python_bin}" "${code_root}/cosmos_sparsegpt_prune.py" \
    --protocol "${protocol}" \
    --calibration-preset "${dataset}" \
    --model-path /private/workspace/hycui/model/Cosmos3-Edge \
    --nsamples "${nsamples}" \
    --vision-sparsity 0.5 \
    --ar-sparsity 0.5 \
    --blocksize 128 \
    --percdamp 0.01 \
    --budget-mode exact_k_budget \
    --max-cholesky-retries 8 \
    --device cuda:0 \
    --dtype bfloat16 \
    --attn-implementation eager \
    --save-model \
    --output-dir "${output_dir}" 2>&1 | tee "${log_path}"
done
