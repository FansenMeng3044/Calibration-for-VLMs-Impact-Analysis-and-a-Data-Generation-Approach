# Copyright (c) 2022, salesforce.com, inc.
# SPDX-License-Identifier: BSD-3-Clause
"""
双层循环：外层用「剩余 9 类」依次做 calibration + TAMP 剪枝（各存一个新 pth），
内层用该 pth 跑 11 类 eval（与 eval_okvqa_by_category.py 一致）。
使用 HuggingFace 镜像，并在 conda 环境 ecoflap 下执行。

用法（须在 LAVIS_backup 根目录）:
  conda activate ecoflap
  python scripts/blip2/run_all_calibration_then_eval.py <GPU_ID> [--data_root DIR] [--port_calib PORT] [--port_eval PORT]

示例:
  conda activate ecoflap
  python scripts/blip2/run_all_calibration_then_eval.py 0
  python scripts/blip2/run_all_calibration_then_eval.py 0 --data_root /root/autodl-tmp/datasets
"""
import argparse
import os
import subprocess
import sys

# HuggingFace 镜像（国内）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

CONDA_ENV = "ecoflap"
DEFAULT_DATA_ROOT = "/root/autodl-tmp/datasets"
RATIO = 0.5
RATIOS = f"{RATIO}-1.0-1.0"
EVAL_DIR = "lavis/projects/blip2/eval"
CC_PREFIX_TEMPLATE = "cc_prefix_derivative_compute_okvqa"

# 11 类（eval 用）
OK_VQA_CATEGORIES = [
    "Brands_Companies_and_Products",
    "Cooking_and_Food",
    "Geography_History_Language_and_Culture",
    "Objects_Material_and_Clothing",
    "Other",
    "People_and_Everyday_life",
    "Plants_and_Animals",
    "Science_and_Technology",
    "Sports_and_Recreation",
    "Vehicles_and_Transportation",
    "Weather_and_Climate",
]

# 剩余 9 类做 calibration（已做过 Cooking_and_Food 默认、Geography_History_Language_and_Culture 单独脚本）
CALIBRATION_CATEGORIES = [
    "Brands_Companies_and_Products",
    "Objects_Material_and_Clothing",
    "Other",
    "People_and_Everyday_life",
    "Plants_and_Animals",
    "Science_and_Technology",
    "Sports_and_Recreation",
    "Vehicles_and_Transportation",
    "Weather_and_Climate",
]

# -----------------------------------------------------------------------------
# 原先单类 GHLC 的流程（已注释）
# -----------------------------------------------------------------------------
# 见 tamp_wanda_ghlc_calibration.py：仅用 Geography_History_Language_and_Culture 做 calibration，
# 存 pruned_checkpoint/okvqa_cf_0.5_Geography_History_Language_and_Culture.pth，再手动跑 11 类 eval。


def ensure_conda_env():
    """若当前不在 ecoflap 环境，则用 conda run -n ecoflap 重新执行本脚本。"""
    if os.environ.get("CONDA_DEFAULT_ENV") == CONDA_ENV:
        return
    conda_exe = os.environ.get("CONDA_EXE") or "conda"
    cmd = [conda_exe, "run", "-n", CONDA_ENV, "--no-capture-output", sys.executable] + sys.argv
    print(f"[INFO] 激活 conda 环境 {CONDA_ENV} 并重新执行: {' '.join(cmd)}")
    ret = subprocess.call(cmd)
    sys.exit(ret)


def write_calibration_yaml(data_root: str, category: str, out_path: str) -> None:
    """为指定 category 写 calibration 用的 cc_prefix_derivative_compute_okvqa_<Category>.yaml"""
    train_json = os.path.join(data_root, "okvqa_by_category", category, "okvqa_train.json")
    images_storage = os.path.join(data_root, "okvqa_official", "images")
    yaml_content = f"""# Auto-generated: calibration 仅用 {category}
model:
  arch: blip2_t5
  model_type: pretrain_flant5xl
  use_grad_checkpoint: False

datasets:
  prefix_okvqa_calibration:
    vis_processor:
        train:
          name: "blip2_image_train"
          image_size: 224
    text_processor:
        train:
          name: "blip_question"
    build_info:
      annotations:
        train:
          storage:
              - {train_json}
      images:
          storage: {images_storage}

run:
  task: image_text_pretrain
  lr_sched: "linear_warmup_cosine_lr"
  init_lr: 1e-4
  min_lr: 1e-5
  warmup_lr: 1e-6
  weight_decay: 0.05
  max_epoch: 1
  batch_size_train: 40
  batch_size_eval: 40
  num_workers: 4
  warmup_steps: 1000
  seed: 42
  output_dir: "output/BLIP2/OKVQA_calibration"
  amp: True
  resume_ckpt_path: null
  evaluate: False
  train_splits: ["train"]
  test_splits: ["train"]
  device: "cuda"
  world_size: 1
  dist_url: "env://"
  distributed: True
"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(yaml_content)


def main():
    parser = argparse.ArgumentParser(
        description="Outer: 9 calibration categories -> pth each; Inner: 11-class eval per pth. Uses conda ecoflap and HF mirror."
    )
    parser.add_argument("gpu", nargs="?", default="0", help="GPU id")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT, help="Data root for okvqa_by_category, okvqa_official")
    parser.add_argument("--port_calib", default="29500", help="Master port for calibration/pruning runs")
    parser.add_argument("--port_eval", default="29501", help="Master port for eval runs")
    parser.add_argument(
        "--calibration_categories",
        nargs="+",
        default=None,
        help=f"Override calibration categories (default: {len(CALIBRATION_CATEGORIES)} classes)",
    )
    parser.add_argument("--skip_calibration", action="store_true", help="Skip pruning step, only run eval for existing pths")
    parser.add_argument("--skip_eval", action="store_true", help="Only run calibration/pruning, skip 11-class eval")
    args = parser.parse_args()

    ensure_conda_env()

    data_root = os.path.abspath(args.data_root)
    calib_categories = args.calibration_categories if args.calibration_categories else CALIBRATION_CATEGORIES

    for idx, calib_cat in enumerate(calib_categories):
        job_id_calib = f"okvqa_cf_0.5_{calib_cat}"
        ckpt_path = f"pruned_checkpoint/{job_id_calib}.pth"

        # ----- 外层：Calibration + 剪枝，存新 pth -----
        if not args.skip_calibration:
            cfg_path = os.path.join(EVAL_DIR, f"{CC_PREFIX_TEMPLATE}_{calib_cat}.yaml")
            train_json = os.path.join(data_root, "okvqa_by_category", calib_cat, "okvqa_train.json")
            if not os.path.isfile(train_json):
                print(f"[SKIP] calibration category {calib_cat}: not found {train_json}")
                continue
            write_calibration_yaml(data_root, calib_cat, cfg_path)
            program = (
                f"CUDA_VISIBLE_DEVICES={args.gpu} python -m torch.distributed.run"
                f" --nproc_per_node=1 --master_port {args.port_calib} evaluate_blip.py"
                f" --cfg-path {cfg_path}"
                f" --pruning_method blipt5_tamp_pruner --save_pruned_model"
                f" --t5_prune_spec 24-{RATIOS} --vit_prune_spec 39-{RATIOS} --job_id '{job_id_calib}'"
            )
            print(f"\n[{idx+1}/{len(calib_categories)}] Calibration + 剪枝: {calib_cat} -> {ckpt_path}")
            print(program)
            ret = subprocess.call(program, shell=True)
            if ret != 0:
                print(f"[WARN] Calibration {calib_cat} exited with {ret}")
                continue
            if not os.path.isfile(ckpt_path):
                print(f"[WARN] Checkpoint not found: {ckpt_path}, skip eval for this calibration.")
                continue
        else:
            if not os.path.isfile(ckpt_path):
                print(f"[SKIP] --skip_calibration and ckpt not found: {ckpt_path}")
                continue

        # ----- 内层：用当前 pth 跑 11 类 eval -----
        if args.skip_eval:
            continue
        print(f"\n[{idx+1}/{len(calib_categories)}] 11 类 eval with ckpt: {ckpt_path}")
        eval_cmd = [
            sys.executable,
            "scripts/blip2/eval_okvqa_by_category.py",
            args.gpu,
            args.port_eval,
            "--data_root", data_root,
            "--ckpt", ckpt_path,
            "--job_id_prefix", job_id_calib,
        ]
        ret = subprocess.call(eval_cmd)
        if ret != 0:
            print(f"[WARN] Eval for calibration {calib_cat} exited with {ret}")

    print("\nDone. 汇总某次 calibration 的 11 类结果:")
    print("  python scripts/blip2/summarize_okvqa_by_category.py --job_id_prefix okvqa_cf_0.5_<CalibCategory>")


if __name__ == "__main__":
    main()
