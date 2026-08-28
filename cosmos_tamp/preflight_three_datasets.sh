#!/usr/bin/env bash
set -euo pipefail

protocol="${1:-joint}"
nsamples="${2:-128}"
code_root="/private/workspace/hycui/mfs/cosmos_tamp"
python_bin="/private/workspace/hycui/envs/cosmos3-edge/bin/python"
datasets="${COSMOS_TAMP_DATASETS:-mmbench mmmu okvqa}"

if [[ "${protocol}" != "joint" && "${protocol}" != "separate" ]]; then
  echo "usage: $0 {joint|separate} [nsamples]" >&2
  exit 2
fi

for dataset in ${datasets}; do
  echo "===== ${protocol} calibration preflight: ${dataset} ====="
  "${python_bin}" "${code_root}/cosmos_tamp_prune.py" \
    --protocol "${protocol}" \
    --calibration-preset "${dataset}" \
    --nsamples "${nsamples}" \
    --preflight-only
done
