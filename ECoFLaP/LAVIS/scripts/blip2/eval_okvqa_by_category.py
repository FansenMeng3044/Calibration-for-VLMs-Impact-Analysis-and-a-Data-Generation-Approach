# Copyright (c) 2022, salesforce.com, inc.
# SPDX-License-Identifier: BSD-3-Clause
# OK-VQA：按 11 类分别 evaluate_blip（ECoFLaP / Wanda 剪枝 .pth）
# 用法（在 ECoFLaP/LAVIS 根目录）:
#   python scripts/blip2/eval_okvqa_by_category.py <GPU_FOR_CUDA_VISIBLE> <MASTER_PORT> [--ckpt PATH] [--job_id_prefix PREFIX] ...

import argparse
import os
import subprocess
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

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
DEFAULT_CKPT = "pruned_checkpoint/okvqa_mme-blipt5_wanda_pruner_0.5-1.0-1.0_MEZO-GradOnly_sum0.6_block_bs8.pth"
EVAL_CONFIG_BASE = "lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval.yaml"
PER_CATEGORY_DIR = "lavis/projects/blip2/eval/okvqa_per_category"


def main():
    parser = argparse.ArgumentParser(description="OK-VQA eval per category (ECoFLaP wanda pruned .pth)")
    parser.add_argument("gpu", nargs="?", default="0", help="传给子进程的 CUDA_VISIBLE_DEVICES（单卡暴露时填 0）")
    parser.add_argument("port", nargs="?", default="29501", help="torch.distributed master_port")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT, help="okvqa_by_category / okvqa_official 根目录")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="剪枝后的 .pth")
    parser.add_argument("--categories", nargs="+", default=None, help="覆盖类别列表")
    parser.add_argument("--job_id_prefix", default="okvqa_eval_ecoflap", help="每类 job_id = {prefix}_{category}")
    parser.add_argument("--val_filename", default="vqa_val_eval.json", help="每类目录下 val 标注文件名")
    args = parser.parse_args()

    os.makedirs(PER_CATEGORY_DIR, exist_ok=True)
    data_root = os.path.abspath(args.data_root)
    configs_dir = os.path.join(data_root, "okvqa_by_category", "configs")
    if args.categories is not None:
        categories = args.categories
    elif os.path.isdir(configs_dir):
        categories = []
        for f in sorted(os.listdir(configs_dir)):
            if f.startswith("okvqa_") and f.endswith(".yaml"):
                cat = f[:-5].replace("okvqa_", "", 1)
                categories.append(cat)
        if not categories:
            categories = OK_VQA_CATEGORIES
    else:
        categories = OK_VQA_CATEGORIES

    ckpt = os.path.abspath(args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt
    if not os.path.isfile(ckpt):
        print(f"[WARN] Checkpoint not found: {ckpt}")

    TEST_FILES = [
        "vqa_val_eval.json",
        "answer_list.json",
        "OpenEnded_mscoco_val2014_questions.json",
        "mscoco_val2014_annotations.json",
    ]
    images_path = os.path.join(data_root, "okvqa_official", "images")

    for i, cat in enumerate(categories):
        cat_dir = os.path.join(data_root, "okvqa_by_category", cat)
        val_eval_path = os.path.join(cat_dir, args.val_filename)
        if not os.path.isfile(val_eval_path):
            print(f"[SKIP] category {cat}: not found {val_eval_path}")
            continue

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
        with open(EVAL_CONFIG_BASE, "r", encoding="utf-8") as f:
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
        with open(cfg_path, "w", encoding="utf-8") as f:
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

    print("Done. 结果目录: lavis/output/BLIP2/OKVQA/<job_id>/")


if __name__ == "__main__":
    main()
