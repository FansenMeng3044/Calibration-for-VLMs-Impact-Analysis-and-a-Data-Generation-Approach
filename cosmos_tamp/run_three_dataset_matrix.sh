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
datasets="${COSMOS_TAMP_DATASETS:-mmbench mmmu okvqa}"
code_root="/private/workspace/hycui/mfs/cosmos_tamp"
python_bin="/private/workspace/hycui/envs/cosmos3-edge/bin/python"

if [[ "${protocol}" != "joint" && "${protocol}" != "separate" ]]; then
  echo "protocol must be joint or separate" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONUNBUFFERED=1

# Validate every selected source before loading model weights.
for dataset in ${datasets}; do
  "${python_bin}" "${code_root}/cosmos_tamp_prune.py" \
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
    if [[ -s "${output_dir}/.checkpoint_validated" ]]; then
      echo "skip complete and validated: ${output_dir}"
    else
      echo "validate existing complete checkpoint: ${output_dir}"
      "${python_bin}" "${code_root}/validate_cosmos_checkpoint.py" \
        --run-dir "${output_dir}" \
        --protocol "${protocol}" \
        --preset "${dataset}" \
        --device cuda:0 \
        --expected-nsamples "${nsamples}" \
        --expected-seed 42 \
        --expected-ar-sparsity 0.5 \
        --expected-max-sparsity-per-linear 0.6 \
        --expected-dtype bfloat16 \
        --expected-attention-implementation eager \
        --expected-min-image-pixels 65536 \
        --expected-max-image-pixels 1048576 2>&1 | tee -a "${log_path}"
    fi
    continue
  fi
  if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "refusing non-empty incomplete output: ${output_dir}" >&2
    exit 3
  fi
  mkdir -p "${results_root}"
  "${python_bin}" "${code_root}/cosmos_tamp_prune.py" \
    --protocol "${protocol}" \
    --calibration-preset "${dataset}" \
    --model-path /private/workspace/hycui/model/Cosmos3-Edge \
    --nsamples "${nsamples}" \
    --vision-sparsity 0 \
    --ar-sparsity 0.5 \
    --max-sparsity-per-linear 0.6 \
    --device cuda:0 \
    --dtype bfloat16 \
    --attn-implementation eager \
    --save-model \
    --output-dir "${output_dir}" 2>&1 | tee "${log_path}"
  "${python_bin}" "${code_root}/validate_cosmos_checkpoint.py" \
    --run-dir "${output_dir}" \
    --protocol "${protocol}" \
    --preset "${dataset}" \
    --device cuda:0 \
    --expected-nsamples "${nsamples}" \
    --expected-seed 42 \
    --expected-ar-sparsity 0.5 \
    --expected-max-sparsity-per-linear 0.6 \
    --expected-dtype bfloat16 \
    --expected-attention-implementation eager \
    --expected-min-image-pixels 65536 \
    --expected-max-image-pixels 1048576 2>&1 | tee -a "${log_path}"
done
