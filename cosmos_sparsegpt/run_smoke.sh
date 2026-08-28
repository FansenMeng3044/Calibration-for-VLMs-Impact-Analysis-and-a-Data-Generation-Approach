#!/usr/bin/env bash
set -euo pipefail

protocol="${1:-separate}"
physical_gpu="${2:-0}"
dataset="${3:-okvqa}"

if [[ "${protocol}" != "joint" && "${protocol}" != "separate" ]]; then
  echo "usage: $0 {joint|separate} [physical_gpu] [mmbench|mmmu|okvqa]" >&2
  exit 2
fi
if [[ "${dataset}" != "mmbench" && "${dataset}" != "mmmu" && "${dataset}" != "okvqa" ]]; then
  echo "dataset must be mmbench, mmmu, or okvqa" >&2
  exit 2
fi

code_root="/private/workspace/hycui/mfs/cosmos_sparsegpt"
python_bin="/private/workspace/hycui/envs/cosmos3-edge/bin/python"
stamp="$(date +%Y%m%d_%H%M%S)"
output_dir="/private/workspace/hycui/Results/mfs/cosmos_sparsegpt_smoke_${protocol}_${dataset}_${stamp}"
log_path="${output_dir}.log"

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONUNBUFFERED=1

"${python_bin}" "${code_root}/cosmos_sparsegpt_prune.py" \
  --protocol "${protocol}" \
  --model-path /private/workspace/hycui/model/Cosmos3-Edge \
  --calibration-preset "${dataset}" \
  --nsamples 1 \
  --vision-sparsity 0.5 \
  --ar-sparsity 0.5 \
  --blocksize 128 \
  --percdamp 0.01 \
  --budget-mode exact_k_budget \
  --max-cholesky-retries 8 \
  --max-vision-layers 1 \
  --max-ar-layers 1 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation eager \
  --no-save-model \
  --output-dir "${output_dir}" 2>&1 | tee "${log_path}"

echo "output=${output_dir}"
echo "log=${log_path}"
