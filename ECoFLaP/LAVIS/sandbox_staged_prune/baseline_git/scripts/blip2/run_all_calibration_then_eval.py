# =============================================================================
# ECoFLaP 双层循环：外层用「剩余 9 类」依次做 calibration + 剪枝（各存一个新 pth），
# 内层用该 pth 跑 11 类 eval。使用 HuggingFace 镜像，并在 conda 环境 ecoflap 下执行。
# 工作目录为 ECoFLaP/LAVIS（本脚本所在目录的上一级为 LAVIS）。
# =============================================================================
# 用法（在 ECoFLaP/LAVIS 根目录）:
#   conda activate ecoflap
#   python scripts/blip2/run_all_calibration_then_eval.py <GPU_ID> [--data_root DIR] [--port_calib PORT] [--port_eval PORT]
#
# 示例:
#   cd /root/autodl-tmp/ECoFLaP/LAVIS && conda activate ecoflap
#   python scripts/blip2/run_all_calibration_then_eval.py 0
# =============================================================================
import argparse
import os
import shutil
import subprocess
import sys

# HuggingFace 镜像（国内）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

CONDA_ENV = "ecoflap"
DEFAULT_DATA_ROOT = "/root/autodl-tmp/datasets"

# ECoFLaP 剪枝参数（与 ecoflap_zeroth.py 一致）
METHOD = "blipt5_wanda_pruner"
SCORE_METHOD = "MEZO-GradOnly_sum"
SPARSITY_RATIO_GRANULARITY = "block"
RATIO = 0.5
RATIOS = f"{RATIO}-1.0-1.0"
MAX_SPARSITY_PER_LAYER = f"{round(1.0 - RATIO + 0.1, 1)}"
PRUNING_BS = 8

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

# 剩余 9 类做 calibration（已做过 Cooking_and_Food、Geography_History_Language_and_Culture）
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

CALIBRATION_YAML = "lavis/configs/datasets/okvqa/calibration.yaml"
CC_PREFIX_OKVQA = "lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa.yaml"
EVAL_CONFIG_BASE = "lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval.yaml"
PER_CATEGORY_DIR = "lavis/projects/blip2/eval/okvqa_per_category"
TEST_FILES = [
    "vqa_val_eval.json",
    "answer_list.json",
    "OpenEnded_mscoco_val2014_questions.json",
    "mscoco_val2014_annotations.json",
]

# -----------------------------------------------------------------------------
# 原先单类 GHLC 的流程（ecoflap_zeroth.py，已注释保留）
# -----------------------------------------------------------------------------
# job_id = "okvqa_ghlc-blipt5_wanda_pruner_0.5-1.0-1.0_MEZO-GradOnly_sum0.6_block_bs8"
# program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run"
#   ... --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa.yaml
#   ... --job_id '{job_id}'")
# 剪枝后 pth: pruned_checkpoint/okvqa_ghlc-blipt5_wanda_pruner_....pth


def ensure_conda_env():
    if os.environ.get("CONDA_DEFAULT_ENV") == CONDA_ENV:
        return
    conda_exe = os.environ.get("CONDA_EXE") or "conda"
    cmd = [conda_exe, "run", "-n", CONDA_ENV, "--no-capture-output", sys.executable] + sys.argv
    print(f"[INFO] 激活 conda 环境 {CONDA_ENV} 并重新执行: {' '.join(cmd)}")
    ret = subprocess.call(cmd)
    sys.exit(ret)


def write_calibration_yaml(data_root: str, category: str, yaml_path: str) -> None:
    train_json = os.path.join(data_root, "okvqa_by_category", category, "okvqa_train.json")
    images_storage = os.path.join(data_root, "okvqa_official", "images")
    content = f"""# OK-VQA for BLIP2 calibration. Auto-generated for category: {category}
datasets:
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
          storage: {images_storage}
"""
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(content)


def write_eval_yaml(data_root: str, eval_category: str, base_yaml_path: str, out_path: str) -> None:
    cat_dir = os.path.join(data_root, "okvqa_by_category", eval_category)
    images_path = os.path.join(data_root, "okvqa_official", "images")
    test_storage_paths = [os.path.join(cat_dir, f) for f in TEST_FILES]
    test_storage_yaml = "\n            - ".join(test_storage_paths)
    build_info_block = f"""    build_info:
      annotations:
        test:
          storage:
            - {test_storage_yaml}
      images:
        storage: {images_path}
"""
    with open(base_yaml_path, "r") as f:
        base_yaml = f.read()
    marker = 'name: "blip_question"'
    if marker in base_yaml and "build_info:" not in base_yaml.split("run:")[0]:
        base_yaml = base_yaml.replace(marker, marker + "\n" + build_info_block, 1)
    else:
        base_yaml = base_yaml.replace(marker, marker + "\n" + build_info_block, 1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(base_yaml)


def main():
    parser = argparse.ArgumentParser(
        description="ECoFLaP: Outer=9 calibration categories -> pth each; Inner=11-class eval per pth. Uses conda ecoflap and HF mirror."
    )
    parser.add_argument("gpu", nargs="?", default="0", help="GPU id")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT, help="Data root for okvqa_by_category, okvqa_official")
    parser.add_argument("--port_calib", default="29500", help="Master port for calibration/pruning")
    parser.add_argument("--port_eval", default="29501", help="Master port for eval")
    parser.add_argument("--calibration_categories", nargs="+", default=None, help="Override calibration categories")
    parser.add_argument("--skip_calibration", action="store_true", help="Skip pruning, only run eval for existing pths")
    parser.add_argument("--skip_eval", action="store_true", help="Only run calibration/pruning, skip eval")
    args = parser.parse_args()

    ensure_conda_env()

    # 确保在 ECoFLaP/LAVIS 根目录
    lavis_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(lavis_root)
    data_root = os.path.abspath(args.data_root)
    calib_categories = args.calibration_categories if args.calibration_categories else CALIBRATION_CATEGORIES

    calib_yaml_path = os.path.join(lavis_root, CALIBRATION_YAML)
    backup_path = calib_yaml_path + ".backup_run_all"
    if os.path.isfile(calib_yaml_path) and not os.path.isfile(backup_path):
        shutil.copy2(calib_yaml_path, backup_path)
        print(f"[INFO] Backed up {CALIBRATION_YAML} -> {backup_path}")

    try:
        for idx, calib_cat in enumerate(calib_categories):
            job_id_prune = f"okvqa_{calib_cat}-{METHOD}_{RATIOS}_{SCORE_METHOD}{MAX_SPARSITY_PER_LAYER}_{SPARSITY_RATIO_GRANULARITY}_bs{PRUNING_BS}"
            ckpt_path = os.path.join(lavis_root, "pruned_checkpoint", job_id_prune + ".pth")

            # ----- 外层：Calibration + 剪枝 -----
            if not args.skip_calibration:
                train_json = os.path.join(data_root, "okvqa_by_category", calib_cat, "okvqa_train.json")
                if not os.path.isfile(train_json):
                    print(f"[SKIP] calibration {calib_cat}: not found {train_json}")
                    continue
                write_calibration_yaml(data_root, calib_cat, calib_yaml_path)
                program = (
                    f"CUDA_VISIBLE_DEVICES={args.gpu} python -m torch.distributed.run"
                    f" --nproc_per_node=1 --master_port {args.port_calib} evaluate_blip.py"
                    f" --cfg-path {CC_PREFIX_OKVQA}"
                    f" --pruning_method '{METHOD}' --save_pruned_model"
                    f" --score_method {SCORE_METHOD}"
                    f" --sparsity_ratio_granularity {SPARSITY_RATIO_GRANULARITY}"
                    f" --max_sparsity_per_layer {MAX_SPARSITY_PER_LAYER}"
                    f" --prunining_dataset_batch_size {PRUNING_BS}"
                    f" --t5_prune_spec 24-{RATIOS} --vit_prune_spec 39-{RATIOS} --job_id '{job_id_prune}'"
                )
                print(f"\n[{idx+1}/{len(calib_categories)}] Calibration + 剪枝: {calib_cat} -> {ckpt_path}")
                print(program)
                ret = subprocess.call(program, shell=True)
                if ret != 0:
                    print(f"[WARN] Calibration {calib_cat} exited with {ret}")
                    continue
                if not os.path.isfile(ckpt_path):
                    print(f"[WARN] Checkpoint not found: {ckpt_path}, skip eval.")
                    continue
            else:
                if not os.path.isfile(ckpt_path):
                    print(f"[SKIP] --skip_calibration and ckpt not found: {ckpt_path}")
                    continue

            # ----- 内层：11 类 eval -----
            if args.skip_eval:
                continue
            base_yaml_path = os.path.join(lavis_root, EVAL_CONFIG_BASE)
            per_cat_dir = os.path.join(lavis_root, PER_CATEGORY_DIR)
            os.makedirs(per_cat_dir, exist_ok=True)
            print(f"\n[{idx+1}/{len(calib_categories)}] 11 类 eval with ckpt: {ckpt_path}")
            for j, eval_cat in enumerate(OK_VQA_CATEGORIES):
                val_path = os.path.join(data_root, "okvqa_by_category", eval_cat, "vqa_val_eval.json")
                if not os.path.isfile(val_path):
                    print(f"  [SKIP] eval category {eval_cat}: not found {val_path}")
                    continue
                cfg_path = os.path.join(per_cat_dir, f"okvqa_zeroshot_flant5xl_eval_{eval_cat}.yaml")
                write_eval_yaml(data_root, eval_cat, base_yaml_path, cfg_path)
                job_id_eval = f"okvqa_{calib_cat}_eval_{eval_cat}"
                program = (
                    f"CUDA_VISIBLE_DEVICES={args.gpu} python -m torch.distributed.run"
                    f" --nproc_per_node=1 --master_port {args.port_eval} evaluate_blip.py"
                    f" --cfg-path {cfg_path}"
                    f" --t5_pruned_checkpoint {ckpt_path}"
                    f" --vit_pruned_checkpoint {ckpt_path}"
                    f" --job_id '{job_id_eval}'"
                )
                print(f"  [{j+1}/11] Eval: {eval_cat}")
                ret = subprocess.call(program, shell=True)
                if ret != 0:
                    print(f"  [WARN] Eval {eval_cat} exited with {ret}")
    finally:
        if os.path.isfile(backup_path):
            shutil.copy2(backup_path, calib_yaml_path)
            print(f"\n[INFO] Restored {CALIBRATION_YAML} from backup")

    print("\nDone. 汇总某次 calibration 的 11 类结果（在 ECoFLaP/LAVIS 下）:")
    print("  python scripts/blip2/summarize_okvqa_by_category.py --job_id_prefix okvqa_<CalibCategory>_eval")


if __name__ == "__main__":
    main()
