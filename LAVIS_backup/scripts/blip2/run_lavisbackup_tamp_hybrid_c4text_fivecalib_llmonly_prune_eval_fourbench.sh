#!/usr/bin/env bash
# Convenience entry for:
#   MMBench/MMMU/OKVQA/MathVista/CC3M images + C4 text
#   -> multimodal BLIP2-T5 calibration
#   -> TAMP AMIA/DAS
#   -> prune T5/LLM only
#   -> four-benchmark eval.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HYBRID_C4_TEXT=1

exec bash "$SCRIPT_DIR/run_lavisbackup_tamp_multimodal_fivecalib_llmonly_prune_eval_fourbench.sh" "$@"
