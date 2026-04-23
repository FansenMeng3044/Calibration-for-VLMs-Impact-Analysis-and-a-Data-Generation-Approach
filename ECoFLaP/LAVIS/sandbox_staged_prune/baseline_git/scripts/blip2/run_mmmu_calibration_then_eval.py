# =============================================================================
# ECoFLaP：以 MMMU 单图多选题（6 大领域）为 calibration，每领域 90 条（dev+validation），
# 共 6 次剪枝，每次一个 pth；可选再跑 11 类 OK-VQA eval。
# 数据须先由 mmmu_to_calibration_format.py --by_discipline --max_per_discipline 90 生成。
# =============================================================================
# 用法（在 ECoFLaP/LAVIS 根目录）:
#   conda activate ecoflap
#   python scripts/blip2/run_mmmu_calibration_then_eval.py <GPU_ID> [--mmmu_calibration_dir DIR] [--skip_eval]
#
# 示例:
#   cd /root/autodl-tmp/ECoFLaP/LAVIS && conda activate ecoflap
#   python scripts/blip2/run_mmmu_calibration_then_eval.py 0
# =============================================================================
import argparse
import os
import shutil
import subprocess
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

CONDA_ENV = "ecoflap"
DEFAULT_MMMU_CALIBRATION = "/root/autodl-tmp/MMMU_calibration"
DEFAULT_DATA_ROOT = "/root/autodl-tmp/datasets"

# 6 大领域（与 mmmu_to_calibration_format.py / mmmu_eval_by_discipline 一致）
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


# ECoFLaP 剪枝参数（与 ecoflap_zeroth / run_all_calibration_then_eval 一致）
METHOD = "blipt5_wanda_pruner"
SCORE_METHOD = "MEZO-GradOnly_sum"
SPARSITY_RATIO_GRANULARITY = "block"
RATIO = 0.5
RATIOS = f"{RATIO}-1.0-1.0"
MAX_SPARSITY_PER_LAYER = f"{round(1.0 - RATIO + 0.1, 1)}"
PRUNING_BS = 8

CALIBRATION_YAML = "lavis/configs/datasets/okvqa/calibration.yaml"
CC_PREFIX_OKVQA = "lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa.yaml"
EVAL_CONFIG_BASE = "lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval.yaml"
PER_CATEGORY_DIR = "lavis/projects/blip2/eval/okvqa_per_category"
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
TEST_FILES = [
    "vqa_val_eval.json",
    "answer_list.json",
    "OpenEnded_mscoco_val2014_questions.json",
    "mscoco_val2014_annotations.json",
]


def ensure_conda_env():
    if os.environ.get("CONDA_DEFAULT_ENV") == CONDA_ENV:
        return
    conda_exe = os.environ.get("CONDA_EXE") or "conda"
    cmd = [conda_exe, "run", "-n", CONDA_ENV, "--no-capture-output", sys.executable] + sys.argv
    print(f"[INFO] 激活 conda 环境 {CONDA_ENV} 并重新执行: {' '.join(cmd)}")
    ret = subprocess.call(cmd)
    sys.exit(ret)


def write_mmmu_calibration_yaml(mmmu_calibration_dir: str, discipline_key: str, train_json: str, yaml_path: str) -> None:
    images_storage = os.path.join(mmmu_calibration_dir, "images")
    content = f"""# MMMU calibration: 6 大领域之一 ({discipline_key}), 90 条 dev+validation 单图多选题
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
        description="ECoFLaP: 6 MMMU disciplines (90 samples each, dev+validation) -> 6 pths; optional 11-class OK-VQA eval."
    )
    parser.add_argument("gpu", nargs="?", default="0", help="GPU id")
    parser.add_argument("--mmmu_calibration_dir", default=DEFAULT_MMMU_CALIBRATION, help="MMMU_calibration root (images/ and mmmu_calibration_train_*.json)")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT, help="Data root for OK-VQA eval (okvqa_by_category, okvqa_official)")
    parser.add_argument("--port_calib", type=int, default=50100, help="Base master port for calibration; each run uses port_calib+idx. If busy use --port_calib 50200")
    parser.add_argument("--port_eval", type=int, default=50110, help="Base master port for eval")
    parser.add_argument("--skip_calibration", action="store_true", help="Skip pruning, only run eval for existing pths")
    parser.add_argument("--skip_eval", action="store_true", help="Only run calibration/pruning, skip 11-class eval")
    args = parser.parse_args()

    ensure_conda_env()

    lavis_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(lavis_root)
    mmmu_calibration_dir = os.path.abspath(args.mmmu_calibration_dir)
    data_root = os.path.abspath(args.data_root)
    calib_yaml_path = os.path.join(lavis_root, CALIBRATION_YAML)
    backup_path = calib_yaml_path + ".backup_run_mmmu"
    if os.path.isfile(calib_yaml_path) and not os.path.isfile(backup_path):
        shutil.copy2(calib_yaml_path, backup_path)
        print(f"[INFO] Backed up {CALIBRATION_YAML} -> {backup_path}")

    try:
        for idx, disc in enumerate(MMMU_DISCIPLINES):
            key = discipline_to_filename_key(disc)
            train_json = os.path.join(mmmu_calibration_dir, f"mmmu_calibration_train_{key}.json")
            job_id_prune = f"okvqa_mmmu_{key}-{METHOD}_{RATIOS}_{SCORE_METHOD}{MAX_SPARSITY_PER_LAYER}_{SPARSITY_RATIO_GRANULARITY}_bs{PRUNING_BS}"
            ckpt_path = os.path.join(lavis_root, "pruned_checkpoint", job_id_prune + ".pth")

            if not args.skip_calibration:
                if not os.path.isfile(train_json):
                    print(f"[SKIP] MMMU calibration {disc}: not found {train_json}")
                    continue
                write_mmmu_calibration_yaml(mmmu_calibration_dir, key, train_json, calib_yaml_path)
                port = args.port_calib + idx
                program = (
                    f"CUDA_VISIBLE_DEVICES={args.gpu} python -m torch.distributed.run"
                    f" --nproc_per_node=1 --master_port {port} evaluate_blip.py"
                    f" --cfg-path {CC_PREFIX_OKVQA}"
                    f" --pruning_method '{METHOD}' --save_pruned_model"
                    f" --score_method {SCORE_METHOD}"
                    f" --sparsity_ratio_granularity {SPARSITY_RATIO_GRANULARITY}"
                    f" --max_sparsity_per_layer {MAX_SPARSITY_PER_LAYER}"
                    f" --prunining_dataset_batch_size {PRUNING_BS}"
                    f" --t5_prune_spec 24-{RATIOS} --vit_prune_spec 39-{RATIOS} --job_id '{job_id_prune}'"
                )
                print(f"\n[{idx+1}/6] MMMU calibration + 剪枝: {disc} -> {ckpt_path}")
                print(program)
                ret = subprocess.call(program, shell=True)
                if ret != 0:
                    print(f"[WARN] Calibration {disc} exited with {ret}")
                    continue
                if not os.path.isfile(ckpt_path):
                    print(f"[WARN] Checkpoint not found: {ckpt_path}, skip eval.")
                    continue
            else:
                if not os.path.isfile(ckpt_path):
                    print(f"[SKIP] --skip_calibration and ckpt not found: {ckpt_path}")
                    continue

            if args.skip_eval:
                continue
            base_yaml_path = os.path.join(lavis_root, EVAL_CONFIG_BASE)
            per_cat_dir = os.path.join(lavis_root, PER_CATEGORY_DIR)
            os.makedirs(per_cat_dir, exist_ok=True)
            print(f"\n[{idx+1}/6] 11 类 eval with ckpt: {ckpt_path}")
            for j, eval_cat in enumerate(OK_VQA_CATEGORIES):
                val_path = os.path.join(data_root, "okvqa_by_category", eval_cat, "vqa_val_eval.json")
                if not os.path.isfile(val_path):
                    print(f"  [SKIP] eval category {eval_cat}: not found {val_path}")
                    continue
                cfg_path = os.path.join(per_cat_dir, f"okvqa_zeroshot_flant5xl_eval_{eval_cat}.yaml")
                write_eval_yaml(data_root, eval_cat, base_yaml_path, cfg_path)
                job_id_eval = f"okvqa_mmmu_{key}_eval_{eval_cat}"
                eval_port = args.port_eval + idx
                program = (
                    f"CUDA_VISIBLE_DEVICES={args.gpu} python -m torch.distributed.run"
                    f" --nproc_per_node=1 --master_port {eval_port} evaluate_blip.py"
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

    print("\nDone. 汇总某次 MMMU calibration 的 11 类结果:")
    print("  python scripts/blip2/summarize_okvqa_by_category.py --job_id_prefix okvqa_mmmu_<DisciplineKey>_eval")


if __name__ == "__main__":
    main()
