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
# 【改】calibration 来源与 job_id 前缀：原 CC3M → 现 OK-VQA（见下方注释掉的原代码）
# -----------------------------------------------------------------------------
# job_id 前缀改为 okvqa，便于与 cc3m 剪枝结果区分
job_id = f"okvqa-{method}_{ratios}_{score_method}{max_sparsity_per_layer}_{sparsity_ratio_granularity}_bs{prunining_dataset_batch_size}"

# 【改】cfg-path：使用 OK-VQA calibration 配置（原为 cc_prefix_derivative_compute.yaml）
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

# ========== 以下为原 CC3M calibration 的剪枝代码（已注释，保留备查）==========
# job_id = f"cc3m-{method}_{ratios}_{score_method}{max_sparsity_per_layer}_{sparsity_ratio_granularity}_bs{prunining_dataset_batch_size}"
# program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run"
# f" --nproc_per_node=1 --master_port {port} evaluate_blip.py"
# f" --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute.yaml"
# f" --pruning_method '{method}' --save_pruned_model"
# f" --score_method {score_method}"
# f" --sparsity_ratio_granularity {sparsity_ratio_granularity}"
# f" --max_sparsity_per_layer {max_sparsity_per_layer}"
# f" --prunining_dataset_batch_size {prunining_dataset_batch_size}"
# f" --t5_prune_spec 24-{ratios} --vit_prune_spec 39-{ratios} --job_id '{job_id}'")
# print(program)
# subprocess.call(program, shell=True)
# ========== 原 CC3M calibration 代码结束 ==========

method = "blipt5_wanda_pruner"

for task in ["vqav2_zeroshot_flant5xl_eval", "gqa_zeroshot_flant5xl_eval", "okvqa_zeroshot_flant5xl_eval", "nocaps_flant5xl_eval", "ret_flickr_eval"]:

    ratios = f"{ratio}-1.0-1.0"
    
    # 【改】与上面剪枝阶段的 job_id 一致，使用 okvqa 前缀
    job_id = f"okvqa-{method}_{ratios}_{score_method}{max_sparsity_per_layer}_{sparsity_ratio_granularity}_bs{prunining_dataset_batch_size}"

    vit_pruned_checkpoint = f"pruned_checkpoint/{job_id}.pth"
    t5_pruned_checkpoint = f"pruned_checkpoint/{job_id}.pth"

    program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run"
    f" --nproc_per_node=1 --master_port {port} evaluate_blip.py"
    f" --cfg-path lavis/projects/blip2/eval/{task}.yaml"
    f" --pruning_method '{method}'"
    f" --t5_pruned_checkpoint {t5_pruned_checkpoint}"
    f" --vit_pruned_checkpoint {vit_pruned_checkpoint}"
    f" --job_id '{job_id}'")

    print(program)
    subprocess.call(program, shell=True)

# ========== 原 CC3M 版本中此处 job_id 为 cc3m-...，已改为上面 okvqa-... ==========
