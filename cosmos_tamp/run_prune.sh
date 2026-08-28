#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 || "$#" -gt 7 ]]; then
  echo "usage: $0 {joint|separate} PHYSICAL_GPU OUTPUT_DIR {mmbench|mmmu|okvqa} [NSAMPLES] [AR_SPARSITY] [MAX_SPARSITY_PER_LINEAR]" >&2
  exit 2
fi

protocol="$1"
physical_gpu="$2"
output_dir="$3"
dataset="$4"
nsamples="${5:-128}"
ar_sparsity="${6:-0.5}"
max_sparsity_per_linear="${7:-0.6}"

if [[ "${protocol}" != "joint" && "${protocol}" != "separate" ]]; then
  echo "protocol must be joint or separate" >&2
  exit 2
fi
if [[ "${dataset}" != "mmbench" && "${dataset}" != "mmmu" && "${dataset}" != "okvqa" ]]; then
  echo "dataset must be mmbench, mmmu, or okvqa" >&2
  exit 2
fi

code_root="/private/workspace/hycui/mfs/cosmos_tamp"
python_bin="/private/workspace/hycui/envs/cosmos3-edge/bin/python"
log_path="${output_dir}.log"

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONUNBUFFERED=1

"${python_bin}" "${code_root}/cosmos_tamp_prune.py" \
  --protocol "${protocol}" \
  --model-path /private/workspace/hycui/model/Cosmos3-Edge \
  --calibration-preset "${dataset}" \
  --nsamples "${nsamples}" \
  --vision-sparsity 0 \
  --ar-sparsity "${ar_sparsity}" \
  --max-sparsity-per-linear "${max_sparsity_per_linear}" \
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
  --expected-ar-sparsity "${ar_sparsity}" \
  --expected-max-sparsity-per-linear "${max_sparsity_per_linear}" \
  --expected-dtype bfloat16 \
  --expected-attention-implementation eager \
  --expected-min-image-pixels 65536 \
  --expected-max-image-pixels 1048576 2>&1 | tee -a "${log_path}"

echo "output=${output_dir}"
echo "log=${log_path}"
