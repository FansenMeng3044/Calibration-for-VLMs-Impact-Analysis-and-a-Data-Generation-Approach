#!/usr/bin/env bash
set -euo pipefail

# Wrapper: 顺序跑 LAVIS_backup + ECoFLaP，使用 MathVista calibration，sampler seed=0

export DATE_TAG="${DATE_TAG:-$(date +%m%d)}"
export LAVIS_DISTRIBUTED_SAMPLER_SEED="${LAVIS_DISTRIBUTED_SAMPLER_SEED:-0}"

echo "[WRAPPER] DATE_TAG=$DATE_TAG sampler_seed=$LAVIS_DISTRIBUTED_SAMPLER_SEED"

(cd /root/autodl-tmp && bash run_mathvista_prune_eval_lavisbackup_seed0.sh)
(cd /root/autodl-tmp && bash run_mathvista_prune_eval_ecoflap_seed0.sh)

echo "[WRAPPER] DONE"
