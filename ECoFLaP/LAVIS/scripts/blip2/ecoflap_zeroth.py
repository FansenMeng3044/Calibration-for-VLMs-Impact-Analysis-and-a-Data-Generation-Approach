# =============================================================================
# 当前入口脚本：使用 OK-VQA 做 calibration 的剪枝流程
# 原 CC3M calibration 版本已保留在: ecoflap_zeroth_cc3m_calibration.py
# 对应配置: lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa.yaml
#
# 原用法: python ecoflap_zeroth.py <GPU> <port>
# 现在支持: python ecoflap_zeroth.py <GPU> <port> --calib_source mmbench
# =============================================================================
import argparse
import os
import shutil
import subprocess
import sys


def write_mmbench_calibration_yaml(calib_yaml_path: str, train_json: str, images_root: str) -> None:
    """
    Overwrite ECoFLaP okvqa calibration.yaml so that OKVQACalibrationDataset reads:
      ann["image"] under images_root
      ann["answer"] as letter "A"/"B"/"C"/"D"
    """

    os.makedirs(os.path.dirname(calib_yaml_path), exist_ok=True)
    content = f"""datasets:
  prefix_okvqa_calibration:
    data_type: images
    build_info:
      annotations:
        train:
          url:
              - {train_json}
          storage:
              - {train_json}
      images:
          storage: {images_root}
"""
    with open(calib_yaml_path, "w", encoding="utf-8") as f:
        f.write(content)


def write_mme_calibration_yaml(calib_yaml_path: str, train_json: str, images_root: str) -> None:
    """
    Overwrite ECoFLaP okvqa calibration.yaml for Classic MME (Yes/No).

    OKVQACalibrationDataset只把 ann["answer"] 当作 text_output（不做 A/B/C/D 限制），
    因此这里直接把我们统一成的小写 "yes"/"no" 写进 JSON 即可。
    """

    os.makedirs(os.path.dirname(calib_yaml_path), exist_ok=True)
    content = f"""datasets:
  prefix_okvqa_calibration:
    data_type: images
    build_info:
      annotations:
        train:
          url:
              - {train_json}
          storage:
              - {train_json}
      images:
          storage: {images_root}
"""
    with open(calib_yaml_path, "w", encoding="utf-8") as f:
        f.write(content)


method = "blipt5_wanda_pruner"
sparsity_ratio_granularity = "block"

score_method = "MEZO-GradOnly_sum"

ratio = 0.5
ratios = f"{ratio}-1.0-1.0"

max_sparsity_per_layer = f"{round(1.0 - ratio + 0.1, 1)}"
prunining_dataset_batch_size = 8

# -----------------------------------------------------------------------------
# calibration 来源
# 默认保持当前 repo 里的 calibration.yaml（通常是你之前配置好的 category）。
# 当 --calib_source mmbench 时，覆盖 calibration.yaml -> MMBench calibration。
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("gpu", help="GPU id, e.g. 0")
parser.add_argument("port", help="Master port, e.g. 29501")
parser.add_argument(
    "--calib_source",
    default="current",
    choices=["current", "mmbench", "mme"],
    help="current: keep existing calibration.yaml; mmbench/mme: overwrite calibration.yaml to /root/autodl-tmp/*_calibration",
)
args = parser.parse_args()

GPU = args.gpu
port = args.port

calib_yaml_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../lavis/configs/datasets/okvqa/calibration.yaml")
)
mmbench_train_json = "/root/autodl-tmp/MMBench_calibration/mmbench_calibration_train.json"
mmbench_images_root = "/root/autodl-tmp/MMBench_calibration/images"

mme_train_json = "/root/autodl-tmp/MME_calibration/mmE_calibration_train.json"
mme_images_root = "/root/autodl-tmp/MME_calibration/images"

_calib_backup_path = calib_yaml_path + ".backup_before_mmbench"
_calib_backup_path_mme = calib_yaml_path + ".backup_before_mme"
use_mmbench = args.calib_source == "mmbench"
use_mme = args.calib_source == "mme"

if use_mmbench:
    if os.path.isfile(calib_yaml_path) and not os.path.isfile(_calib_backup_path):
        shutil.copy2(calib_yaml_path, _calib_backup_path)
    write_mmbench_calibration_yaml(calib_yaml_path, mmbench_train_json, mmbench_images_root)
elif use_mme:
    if os.path.isfile(calib_yaml_path) and not os.path.isfile(_calib_backup_path_mme):
        shutil.copy2(calib_yaml_path, _calib_backup_path_mme)
    write_mme_calibration_yaml(calib_yaml_path, mme_train_json, mme_images_root)

job_prefix = "okvqa_mmbench" if use_mmbench else ("okvqa_mme" if use_mme else "okvqa_ghlc")
job_id = f"{job_prefix}-{method}_{ratios}_{score_method}{max_sparsity_per_layer}_{sparsity_ratio_granularity}_bs{prunining_dataset_batch_size}"

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
