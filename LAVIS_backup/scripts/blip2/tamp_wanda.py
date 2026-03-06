"""
Run BLIP-2 pruning with TAMP config (Wanda + DAS + AMIA).
Usage: python scripts/blip2/tamp_wanda.py <GPU_ID> <MASTER_PORT>
Example: python scripts/blip2/tamp_wanda.py 0 29500
"""
import os
import subprocess
import sys

# 国内网络：使用 HuggingFace 镜像，避免 bert-base-uncased 等下载超时
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

GPU = sys.argv[1] if len(sys.argv) > 1 else "0"
port = sys.argv[2] if len(sys.argv) > 2 else "29500"

method = "blipt5_tamp_pruner"
ratio = 0.5
ratios = f"{ratio}-1.0-1.0"
job_id = f"okvqa_cf_{ratios}"

program = (
    f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run"
    f" --nproc_per_node=1 --master_port {port} evaluate_blip.py"
    f" --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa.yaml"
    f" --pruning_method {method} --save_pruned_model"
    f" --t5_prune_spec 24-{ratios} --vit_prune_spec 39-{ratios} --job_id '{job_id}'"
)
print(program)
subprocess.call(program, shell=True)
