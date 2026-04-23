#!/usr/bin/env bash
# 薄封装：默认 LAVIS_backup；逻辑见 run_mathvista_mc_eval.sh
set -euo pipefail
export LAVIS_REPO_ROOT="${LAVIS_REPO_ROOT:-/root/autodl-tmp/LAVIS_backup}"
exec bash /root/autodl-tmp/run_mathvista_mc_eval.sh
