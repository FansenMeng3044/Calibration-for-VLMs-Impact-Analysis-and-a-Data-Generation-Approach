# Copyright (c) 2022, salesforce.com, inc.
# SPDX-License-Identifier: BSD-3-Clause
# OK-VQA eval：按 11 类分别跑 eval（路径约定与 ECoFLaP 一致），用本仓库 TAMP/CF-calibration 剪枝好的模型。
# 用法（须在 LAVIS_backup 根目录）: python scripts/blip2/eval_okvqa_by_category.py <GPU_ID> <MASTER_PORT> [--data_root DIR] [--ckpt PATH] ...
# 示例: python scripts/blip2/eval_okvqa_by_category.py 0 29501
#       python scripts/blip2/eval_okvqa_by_category.py 0 29501 --data_root /root/autodl-tmp/datasets --ckpt pruned_checkpoint/okvqa_cf_0.5-1.0-1.0.pth

import argparse
import os
import subprocess
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# OK-VQA 类别与 ECoFLaP/数据集一致：优先从 data_root/okvqa_by_category/configs/okvqa_*.yaml 发现，否则用下列 11 类
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

DEFAULT_DATA_ROOT = "/root/autodl-tmp/datasets"
DEFAULT_CKPT = "pruned_checkpoint/okvqa_cf_0.5-1.0-1.0.pth"
EVAL_CONFIG_BASE = "lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval.yaml"
PER_CATEGORY_DIR = "lavis/projects/blip2/eval/okvqa_per_category"


def main():
    parser = argparse.ArgumentParser(description="OK-VQA eval per category (TAMP pruned model), path convention same as ECoFLaP")
    parser.add_argument("gpu", nargs="?", default="0", help="GPU id")
    parser.add_argument("port", nargs="?", default="29501", help="Master port")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT, help="Root for okvqa_by_category, okvqa/annotations, okvqa_official/images (ECoFLaP 约定)")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="TAMP 剪枝后的 .pth (CF-calibration)")
    parser.add_argument("--categories", nargs="+", default=None, help="Override category list (default: 11 categories)")
    parser.add_argument("--job_id_prefix", default="okvqa_cf_0.5", help="job_id 前缀，每类为 {prefix}_{category}")
    parser.add_argument("--val_filename", default="vqa_val_eval.json", help="每类目录下 val 标注文件名（与 ECoFLaP 一致）")
    args = parser.parse_args()

    os.makedirs(PER_CATEGORY_DIR, exist_ok=True)
    data_root = os.path.abspath(args.data_root)
    # 与 ECoFLaP 一致：若有 okvqa_by_category/configs/okvqa_*.yaml 则从中发现类别列表，否则用默认 11 类
    configs_dir = os.path.join(data_root, "okvqa_by_category", "configs")
    if args.categories is not None:
        categories = args.categories
    elif os.path.isdir(configs_dir):
        categories = []
        for f in sorted(os.listdir(configs_dir)):
            if f.startswith("okvqa_") and f.endswith(".yaml"):
                cat = f[:-5].replace("okvqa_", "", 1)  # okvqa_Cooking_and_Food.yaml -> Cooking_and_Food
                categories.append(cat)
        if not categories:
            categories = OK_VQA_CATEGORIES
    else:
        categories = OK_VQA_CATEGORIES
    ckpt = os.path.abspath(args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt
    if not os.path.isfile(ckpt):
        print(f"[WARN] Checkpoint not found: {ckpt}")
        print("       Run tamp_wanda.py first and set --ckpt to the saved .pth")

    # 与 ECoFLaP 一致：每类目录下 test 为 4 个文件（同目录）
    TEST_FILES = ["vqa_val_eval.json", "answer_list.json", "OpenEnded_mscoco_val2014_questions.json", "mscoco_val2014_annotations.json"]
    images_path = os.path.join(data_root, "okvqa_official", "images")

    for i, cat in enumerate(categories):
        cat_dir = os.path.join(data_root, "okvqa_by_category", cat)
        val_eval_path = os.path.join(cat_dir, args.val_filename)
        if not os.path.isfile(val_eval_path):
            print(f"[SKIP] category {cat}: not found {val_eval_path}")
            continue

        # 与 ECoFLaP 一致：test.storage 为 4 个文件；build_info 须与 vis_processor/text_processor 同级，即 ok_vqa 下 4 空格
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
        cfg_path = os.path.join(PER_CATEGORY_DIR, f"okvqa_zeroshot_flant5xl_eval_{cat}.yaml")
        with open(EVAL_CONFIG_BASE, "r") as f:
            base_yaml = f.read()
        marker = 'name: "blip_question"'
        if marker in base_yaml and "build_info:" not in base_yaml.split("run:")[0]:
            base_yaml = base_yaml.replace(marker, marker + "\n" + build_info_block, 1)
        elif "#     build_info:" in base_yaml:
            base_yaml = base_yaml.replace(
                "#     build_info:\n#         images:\n#             storage: '/export/share/datasets/vision/coco/images/'",
                build_info_block.rstrip(),
            )
        else:
            base_yaml = base_yaml.replace(marker, marker + "\n" + build_info_block, 1)
        with open(cfg_path, "w") as f:
            f.write(base_yaml)

        job_id = f"{args.job_id_prefix}_{cat}"
        program = (
            f"CUDA_VISIBLE_DEVICES={args.gpu} python -m torch.distributed.run"
            f" --nproc_per_node=1 --master_port {args.port} evaluate_blip.py"
            f" --cfg-path {cfg_path}"
            f" --t5_pruned_checkpoint {ckpt}"
            f" --vit_pruned_checkpoint {ckpt}"
            f" --job_id '{job_id}'"
        )
        print(f"[{i+1}/{len(categories)}] Eval category: {cat}")
        print(program)
        ret = subprocess.call(program, shell=True)
        if ret != 0:
            print(f"[WARN] category {cat} eval exited with {ret}")

    print("Done. Check output/BLIP2/OKVQA or logs for per-category accuracy.")


if __name__ == "__main__":
    main()
