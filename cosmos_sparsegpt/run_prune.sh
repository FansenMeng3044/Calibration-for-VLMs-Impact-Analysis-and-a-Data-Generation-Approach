#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 || "$#" -gt 5 ]]; then
  echo "usage: $0 {joint|separate} PHYSICAL_GPU OUTPUT_DIR {mmbench|mmmu|okvqa} [NSAMPLES]" >&2
  exit 2
fi

protocol="$1"
physical_gpu="$2"
output_dir="$3"
dataset="$4"
nsamples="${5:-128}"

if [[ "${protocol}" != "joint" && "${protocol}" != "separate" ]]; then
  echo "protocol must be joint or separate" >&2
  exit 2
fi
if [[ "${dataset}" != "mmbench" && "${dataset}" != "mmmu" && "${dataset}" != "okvqa" ]]; then
  echo "dataset must be mmbench, mmmu, or okvqa" >&2
  exit 2
fi

code_root="/private/workspace/hycui/mfs/cosmos_sparsegpt"
python_bin="/private/workspace/hycui/envs/cosmos3-edge/bin/python"
log_path="${output_dir}.log"

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONUNBUFFERED=1

"${python_bin}" "${code_root}/cosmos_sparsegpt_prune.py" \
  --protocol "${protocol}" \
  --model-path /private/workspace/hycui/model/Cosmos3-Edge \
  --calibration-preset "${dataset}" \
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

echo "output=${output_dir}"
echo "log=${log_path}"
