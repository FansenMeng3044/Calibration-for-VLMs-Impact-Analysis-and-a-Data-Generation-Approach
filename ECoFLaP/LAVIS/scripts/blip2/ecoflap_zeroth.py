# =============================================================================
# 当前入口脚本：使用 OK-VQA 做 calibration 的剪枝流程
# 原 CC3M calibration 版本已保留在: ecoflap_zeroth_cc3m_calibration.py
# 对应配置: lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa.yaml
# 用法: python ecoflap_zeroth.py <GPU> <port>
# =============================================================================
import subprocess
import random

import sys

GPU = sys.argv[1]
port = sys.argv[2]


method = "blipt5_wanda_pruner"
sparsity_ratio_granularity = "block"

score_method = "MEZO-GradOnly_sum"

ratio = 0.5
ratios = f"{ratio}-1.0-1.0"

max_sparsity_per_layer = f"{round(1.0 - ratio + 0.1, 1)}"
prunining_dataset_batch_size = 8

# -----------------------------------------------------------------------------
# 【改】calibration 来源与 job_id：当前用 GHLC（Geography_History_Language_and_Culture）
# 之前 CF（Cooking_and_Food）已注释，见下方。
# -----------------------------------------------------------------------------
# 使用 GHLC，job_id 带 ghlc 前缀，保存为新 .pth（如 okvqa_ghlc-blipt5_wanda_pruner_...pth）
job_id = f"okvqa_ghlc-{method}_{ratios}_{score_method}{max_sparsity_per_layer}_{sparsity_ratio_granularity}_bs{prunining_dataset_batch_size}"

# GHLC calibration 剪枝（calibration 数据在 lavis/configs/datasets/okvqa/calibration.yaml）
program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run"
f" --nproc_per_node=1 --master_port {port} evaluate_blip.py"
f" --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa.yaml"
f" --pruning_method '{method}' --save_pruned_model"
f" --score_method {score_method}"
f" --sparsity_ratio_granularity {sparsity_ratio_granularity}"
f" --max_sparsity_per_layer {max_sparsity_per_layer}"
f" --prunining_dataset_batch_size {prunining_dataset_batch_size}"
f" --t5_prune_spec 24-{ratios} --vit_prune_spec 39-{ratios} --job_id '{job_id}'")

print(program)
subprocess.call(program, shell=True)

# 仅做剪枝：执行完上面的剪枝命令后直接退出，不再跑后续评测
sys.exit(0)

# ========== 之前 CF（Cooking_and_Food）calibration 剪枝（已注释，保留）==========
# job_id = f"okvqa-{method}_{ratios}_{score_method}{max_sparsity_per_layer}_{sparsity_ratio_granularity}_bs{prunining_dataset_batch_size}"
# program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run"
# f" --nproc_per_node=1 --master_port {port} evaluate_blip.py"
# f" --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa.yaml"
# f" --pruning_method '{method}' --save_pruned_model"
# f" --score_method {score_method}"
# f" --sparsity_ratio_granularity {sparsity_ratio_granularity}"
# f" --max_sparsity_per_layer {max_sparsity_per_layer}"
# f" --prunining_dataset_batch_size {prunining_dataset_batch_size}"
# f" --t5_prune_spec 24-{ratios} --vit_prune_spec 39-{ratios} --job_id '{job_id}'")
# print(program)
# subprocess.call(program, shell=True)
# ========== 原 CF calibration 代码结束 ==========

method = "blipt5_wanda_pruner"

# ========== 原先这里会用剪枝后的 checkpoint 跑 VQAv2 / GQA / OK-VQA / NoCaps / Flickr 等评测 ==========
