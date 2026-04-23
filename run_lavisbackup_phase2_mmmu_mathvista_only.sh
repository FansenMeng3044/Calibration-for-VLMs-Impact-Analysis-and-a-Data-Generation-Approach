#!/usr/bin/env bash
# 仅 Phase2：不剪枝，用已有 MathVista 标定权重跑 MMMU overall + MathVista MC
# 默认权重：okvqa_cf_0.5_mathvista_overall_0327_s0.pth
# 覆盖： LB_PHASE2_CKPT=/path/to.pth bash ...
set -euo pipefail
export RUN_PHASE1=0
export RUN_PHASE3=0
exec bash /root/autodl-tmp/run_mathvista_prune_eval_suite.sh
