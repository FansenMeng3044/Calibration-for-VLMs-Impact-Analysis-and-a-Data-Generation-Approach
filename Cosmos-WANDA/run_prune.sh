#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 6 ]]; then
  echo "usage: $0 {joint|separate} PHYSICAL_GPU OUTPUT_DIR [CALIBRATION_JSON] [IMAGE_ROOT] [NSAMPLES]" >&2
  exit 2
fi

protocol="$1"
physical_gpu="$2"
output_dir="$3"
calibration_json="${4:-/private/workspace/hycui/mfs/okvqa_20260626_181458_seed42_paired.json}"
image_root="${5:-/private/workspace/hycui/mfs/okvqa}"
nsamples="${6:-128}"

if [[ "${protocol}" != "joint" && "${protocol}" != "separate" ]]; then
  echo "protocol must be joint or separate" >&2
  exit 2
fi

code_root="/private/workspace/hycui/mfs/cosmos_wanda"
python_bin="/private/workspace/hycui/envs/cosmos3-edge/bin/python"
log_path="${output_dir}.log"

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONUNBUFFERED=1

"${python_bin}" "${code_root}/cosmos_wanda_prune.py" \
  --protocol "${protocol}" \
  --model-path /private/workspace/hycui/model/Cosmos3-Edge \
  --calibration-json "${calibration_json}" \
  --image-root "${image_root}" \
  --nsamples "${nsamples}" \
  --vision-sparsity 0.5 \
  --ar-sparsity 0.5 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation eager \
  --save-model \
  --output-dir "${output_dir}" 2>&1 | tee "${log_path}"

echo "output=${output_dir}"
echo "log=${log_path}"
