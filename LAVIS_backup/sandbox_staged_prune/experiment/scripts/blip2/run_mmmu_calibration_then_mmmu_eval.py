# Copyright (c) 2022, salesforce.com, inc.
# SPDX-License-Identifier: BSD-3-Clause
"""
完整流程：6 种 MMMU calibration（每领域 90 条）→ 6 个 checkpoint → 每个 ckpt 跑一次 MMMU eval。
数据须先由 mmmu_to_calibration_format.py --by_discipline --max_per_discipline 90 生成。

用法（须在 LAVIS_backup 根目录）:
  conda activate ecoflap
  python scripts/blip2/run_mmmu_calibration_then_mmmu_eval.py <GPU_ID> [--mmmu_calibration_dir DIR] [--mmmu_root DIR] [--port_calib PORT] [--skip_calibration] [--skip_mmmu_eval]

示例（GPU 1）:
  python scripts/blip2/run_mmmu_calibration_then_mmmu_eval.py 1
  python scripts/blip2/run_mmmu_calibration_then_mmmu_eval.py 1 --port_calib 29600
"""
import argparse
import os
import subprocess
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

CONDA_ENV = "ecoflap"
DEFAULT_MMMU_CALIBRATION = "/root/autodl-tmp/MMMU_calibration"
DEFAULT_MMMU_ROOT = "/root/autodl-tmp/MMMU_single_image"
RATIO = 0.5
RATIOS = f"{RATIO}-1.0-1.0"
EVAL_DIR = "lavis/projects/blip2/eval"
CC_PREFIX_TEMPLATE = "cc_prefix_derivative_compute_okvqa"

MMMU_DISCIPLINES = [
    "Art & Design",
    "Business",
    "Science",
    "Health & Medicine",
    "Humanities & Social Science",
    "Tech & Engineering",
]


def discipline_to_filename_key(discipline: str) -> str:
    return discipline.replace(" & ", "_").replace(" ", "_")


def ensure_conda_env():
    if os.environ.get("CONDA_DEFAULT_ENV") == CONDA_ENV:
        return
    conda_exe = os.environ.get("CONDA_EXE") or "conda"
    cmd = [conda_exe, "run", "-n", CONDA_ENV, "--no-capture-output", sys.executable] + sys.argv
    print(f"[INFO] 激活 conda 环境 {CONDA_ENV} 并重新执行: {' '.join(cmd)}")
    ret = subprocess.call(cmd)
    sys.exit(ret)


def write_mmmu_calibration_yaml(mmmu_calibration_dir: str, discipline_key: str, train_json: str, out_path: str) -> None:
    images_storage = os.path.join(mmmu_calibration_dir, "images")
    yaml_content = f"""# Auto-generated: MMMU calibration 仅用 {discipline_key}（90 条 dev+validation 单图多选题）
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
        description="LAVIS_backup: 6 MMMU calibrations -> 6 pths -> 6 MMMU evals (one run per checkpoint, report by discipline)."
    )
    parser.add_argument("gpu", nargs="?", default="0", help="GPU id (e.g. 1 for GPU 1)")
    parser.add_argument("--mmmu_calibration_dir", default=DEFAULT_MMMU_CALIBRATION, help="MMMU_calibration root")
    parser.add_argument("--mmmu_root", default=DEFAULT_MMMU_ROOT, help="MMMU_single_image root for eval")
    parser.add_argument("--port_calib", type=int, default=50100, help="Base master port for calibration (each run uses port_calib+idx). Default 50100 to avoid conflict; if busy use 50200 etc.")
    parser.add_argument("--skip_calibration", action="store_true", help="Skip pruning, only run MMMU eval for existing pths")
    parser.add_argument("--skip_mmmu_eval", action="store_true", help="Only run calibration/pruning, skip MMMU eval")
    parser.add_argument("--mmmu_split", default="test", choices=["dev", "validation", "test"], help="MMMU split to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for MMMU eval")
    args = parser.parse_args()

    ensure_conda_env()

    lavis_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(lavis_root)
    mmmu_calibration_dir = os.path.abspath(args.mmmu_calibration_dir)
    mmmu_root = os.path.abspath(args.mmmu_root)

    for idx, disc in enumerate(MMMU_DISCIPLINES):
        key = discipline_to_filename_key(disc)
        train_json = os.path.join(mmmu_calibration_dir, f"mmmu_calibration_train_{key}.json")
        cfg_path = os.path.join(lavis_root, EVAL_DIR, f"{CC_PREFIX_TEMPLATE}_mmmu_{key}.yaml")
        job_id_calib = f"okvqa_mmmu_{key}_0.5"
        ckpt_path = os.path.join(lavis_root, "pruned_checkpoint", f"{job_id_calib}.pth")

        # ----- 1) Calibration + 剪枝 -----
        if not args.skip_calibration:
            if not os.path.isfile(train_json):
                print(f"[SKIP] MMMU calibration {disc}: not found {train_json}")
                continue
            write_mmmu_calibration_yaml(mmmu_calibration_dir, key, train_json, cfg_path)
            port = args.port_calib + idx
            program = (
                f"CUDA_VISIBLE_DEVICES={args.gpu} python -m torch.distributed.run"
                f" --nproc_per_node=1 --master_port {port} evaluate_blip.py"
                f" --cfg-path {cfg_path}"
                f" --pruning_method blipt5_tamp_pruner --save_pruned_model"
                f" --t5_prune_spec 24-{RATIOS} --vit_prune_spec 39-{RATIOS} --job_id '{job_id_calib}'"
            )
            print(f"\n[{idx+1}/6] MMMU calibration + 剪枝: {disc} -> {ckpt_path}")
            print(program)
            ret = subprocess.call(program, shell=True)
            if ret != 0:
                print(f"[WARN] Calibration {disc} exited with {ret}")
                continue
            if not os.path.isfile(ckpt_path):
                print(f"[WARN] Checkpoint not found: {ckpt_path}, skip MMMU eval.")
                continue
        else:
            if not os.path.isfile(ckpt_path):
                print(f"[SKIP] --skip_calibration and ckpt not found: {ckpt_path}")
                continue

        # ----- 2) MMMU eval（该 checkpoint 对应一个 calibration 类别，跑一次 MMMU test，输出 overall + 6 领域）-----
        if args.skip_mmmu_eval:
            continue
        print(f"\n[{idx+1}/6] MMMU eval (ckpt calibrated on {disc}): {ckpt_path}")
        mmmu_eval_cmd = (
            f"CUDA_VISIBLE_DEVICES={args.gpu} python scripts/blip2/mmmu_eval_by_discipline.py"
            f" --mmmu_root {mmmu_root}"
            f" --split {args.mmmu_split}"
            f" --ckpt {ckpt_path}"
            f" --batch_size {args.batch_size}"
            f" --device cuda"
        )
        ret = subprocess.call(mmmu_eval_cmd, shell=True)
        if ret != 0:
            print(f"[WARN] MMMU eval for {disc} exited with {ret}")

    print("\nDone. 6 个 checkpoint 各跑了一次 MMMU eval（Overall + 6 大领域 breakdown）。")


if __name__ == "__main__":
    main()
