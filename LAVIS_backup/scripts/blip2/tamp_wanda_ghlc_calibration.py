# Copyright (c) 2022, salesforce.com, inc.
# SPDX-License-Identifier: BSD-3-Clause
"""
默认用 Geography_History_Language_and_Culture 类做 calibration 再跑 TAMP 剪枝，存成新 pth；
也支持用 MMBench calibration（--calib_source mmbench）或 Classic MME calibration（--calib_source mme），再跑 11 类 eval（与原先 eval_okvqa_by_category.py 一致）。

用法（须在 LAVIS_backup 根目录）:
  python scripts/blip2/tamp_wanda_ghlc_calibration.py <GPU_ID> <MASTER_PORT> [--data_root DIR] [--calib_source {ghlc,mmbench,mme}]

示例:
  python scripts/blip2/tamp_wanda_ghlc_calibration.py 0 29500
  python scripts/blip2/tamp_wanda_ghlc_calibration.py 0 29500 --data_root /root/autodl-tmp/datasets
  python scripts/blip2/tamp_wanda_ghlc_calibration.py 0 29500 --calib_source mmbench

剪枝完成后会输出 pruned_checkpoint/okvqa_cf_0.5_Geography_History_Language_and_Culture.pth，
并提示你如何用该 ckpt 跑 11 类 eval。
"""
import argparse
import os
import subprocess
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# -----------------------------------------------------------------------------
# 原先的 TAMP 剪枝（用 calibration.yaml 里配的类别，如 Cooking_and_Food）— 已注释
# -----------------------------------------------------------------------------
# GPU = sys.argv[1] if len(sys.argv) > 1 else "0"
# port = sys.argv[2] if len(sys.argv) > 2 else "29500"
# method = "blipt5_tamp_pruner"
# ratio = 0.5
# ratios = f"{ratio}-1.0-1.0"
# job_id = f"okvqa_cf_{ratios}"
# program = (
#     f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run"
#     f" --nproc_per_node=1 --master_port {port} evaluate_blip.py"
#     f" --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa.yaml"
#     f" --pruning_method {method} --save_pruned_model"
#     f" --t5_prune_spec 24-{ratios} --vit_prune_spec 39-{ratios} --job_id '{job_id}'"
# )
# print(program)
# subprocess.call(program, shell=True)

# -----------------------------------------------------------------------------
# 本次：支持 ghcl / mmbench / mme 三种 calibration
# -----------------------------------------------------------------------------
DEFAULT_DATA_ROOT = "/root/autodl-tmp/datasets"
CALIBRATION_CATEGORY = "Geography_History_Language_and_Culture"
GHLC_CFG = "lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_ghlc.yaml"
MMBENCH_CFG = "lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_mmbench.yaml"
MME_CFG = "lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_mme.yaml"
RATIO = 0.5
RATIOS = f"{RATIO}-1.0-1.0"
JOB_ID_GHLC = f"okvqa_cf_0.5_{CALIBRATION_CATEGORY}"
CKPT_GHLC = f"pruned_checkpoint/{JOB_ID_GHLC}.pth"
JOB_ID_MMBENCH = "okvqa_cf_0.5_MMBench"
CKPT_MMBENCH = f"pruned_checkpoint/{JOB_ID_MMBENCH}.pth"
JOB_ID_MME = "okvqa_cf_0.5_MME"
CKPT_MME = f"pruned_checkpoint/{JOB_ID_MME}.pth"


def main():
    parser = argparse.ArgumentParser(
        description="TAMP pruning with ghlc/mmbench/mme as calibration, save new .pth"
    )
    parser.add_argument("gpu", nargs="?", default="0", help="GPU id")
    parser.add_argument("port", nargs="?", default="29500", help="Master port")
    parser.add_argument(
        "--data_root",
        default=DEFAULT_DATA_ROOT,
        help="Data root (used only if you need to override paths in the yaml)",
    )
    parser.add_argument(
        "--calib_source",
        default="ghlc",
        choices=["ghlc", "mmbench", "mme"],
        help="Calibration source",
    )
    args = parser.parse_args()

    # 当前 yaml 内路径已写死为 /root/autodl-tmp/datasets；若 data_root 不同，可在这里改 yaml 或加 env
    data_root = os.path.abspath(args.data_root)
    if data_root != DEFAULT_DATA_ROOT:
        print(f"[INFO] data_root={data_root}; 若 yaml 内路径与之一致，请先改 cc_prefix_derivative_compute_okvqa_ghlc.yaml 中路径")

    method = "blipt5_tamp_pruner"

    if args.calib_source == "mmbench":
        cfg_path = MMBENCH_CFG
        job_id = JOB_ID_MMBENCH
        ckpt_path = CKPT_MMBENCH
        job_id_prefix = "okvqa_cf_0.5_MMBench"
    elif args.calib_source == "mme":
        cfg_path = MME_CFG
        job_id = JOB_ID_MME
        ckpt_path = CKPT_MME
        job_id_prefix = "okvqa_cf_0.5_MME"
    else:
        cfg_path = GHLC_CFG
        job_id = JOB_ID_GHLC
        ckpt_path = CKPT_GHLC
        job_id_prefix = "okvqa_cf_0.5_GHLC"

    program = (
        f"CUDA_VISIBLE_DEVICES={args.gpu} python -m torch.distributed.run"
        f" --nproc_per_node=1 --master_port {args.port} evaluate_blip.py"
        f" --cfg-path {cfg_path}"
        f" --pruning_method {method} --save_pruned_model"
        f" --t5_prune_spec 24-{RATIOS} --vit_prune_spec 39-{RATIOS} --job_id '{job_id}'"
    )
    print(program)
    ret = subprocess.call(program, shell=True)
    if ret != 0:
        print(f"[WARN] Pruning exited with {ret}")
        sys.exit(ret)

    ckpt_abs = os.path.abspath(ckpt_path)
    if os.path.isfile(ckpt_abs):
        print("")
        print("Calibration + pruning 已完成，新 ckpt:")
        print(f"  {ckpt_abs}")
        print("")
        print("用该 ckpt 跑 11 类 eval（与之前一样）:")
        print(
            f"  python scripts/blip2/eval_okvqa_by_category.py {args.gpu} 29501"
            f" --ckpt {ckpt_path} --job_id_prefix {job_id_prefix}"
        )
        print("")
        print("汇总结果:")
        print(f"  python scripts/blip2/summarize_okvqa_by_category.py --job_id_prefix {job_id_prefix}")
    else:
        print(f"[WARN] Expected checkpoint not found: {ckpt_abs}")


if __name__ == "__main__":
    main()
