#!/usr/bin/env python3
"""
从已有的 test_vqa_result.json + 标注 重新计算 11 类 OK-VQA 准确率并打印汇总。
用法（在 UKMP-main/LAVIS 下）:
  python scripts/structured_blip2/summarize_okvqa_results.py
"""
import os
import sys

# 只把 vqa_tools 的父目录加入 path，避免导入整个 lavis
LAVIS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.basename(LAVIS_ROOT) != "LAVIS":
    LAVIS_ROOT = os.path.join(LAVIS_ROOT, "LAVIS")
sys.path.insert(0, os.path.join(LAVIS_ROOT, "lavis", "common"))
from vqa_tools.vqa import VQA
from vqa_tools.vqa_eval import VQAEval

# (eval_label, 类别目录名)
CATEGORIES = [
    ("VT", "Vehicles_and_Transportation"),
    ("BCP", "Brands_Companies_and_Products"),
    ("OMC", "Objects_Material_and_Clothing"),
    ("SR", "Sports_and_Recreation"),
    ("CF", "Cooking_and_Food"),
    ("GHLC", "Geography_History_Language_and_Culture"),
    ("PEL", "People_and_Everyday_life"),
    ("PA", "Plants_and_Animals"),
    ("ST", "Science_and_Technology"),
    ("WC", "Weather_and_Climate"),
    ("Other", "Other"),
]
DATA_ROOT = "/root/autodl-tmp/datasets/okvqa_by_category"
OUTPUT_ROOT = os.path.join(LAVIS_ROOT, "lavis", "output", "BLIP2", "OKVQA")


def main():
    results = []
    for label, cat_dir in CATEGORIES:
        anno = os.path.join(DATA_ROOT, cat_dir, "mscoco_val2014_annotations.json")
        ques = os.path.join(DATA_ROOT, cat_dir, "OpenEnded_mscoco_val2014_questions.json")
        res_file = os.path.join(OUTPUT_ROOT, f"okvqa_eval_{label}", "result", "test_vqa_result.json")
        if not os.path.isfile(anno) or not os.path.isfile(ques) or not os.path.isfile(res_file):
            results.append((label, cat_dir, None))
            continue
        try:
            vqa = VQA(anno, ques)
            vqa_result = vqa.loadRes(res_file, ques)
            scorer = VQAEval(vqa, vqa_result, n=2)
            scorer.evaluate()
            acc = scorer.accuracy["overall"]
            results.append((label, cat_dir, acc))
        except Exception as e:
            results.append((label, cat_dir, f"err: {e}"))

    # 打印表格
    print("\n" + "=" * 70)
    print("OK-VQA 11 类 Overall Accuracy 汇总（当前 result 来自最近一次 eval 运行）")
    print("=" * 70)
    print(f"{'类别':<6} {'准确率 (%)':<12} {'类别名'}")
    print("-" * 70)
    for label, cat_dir, acc in results:
        if acc is None:
            print(f"{label:<6} {'(缺失文件)':<12} {cat_dir}")
        elif isinstance(acc, str):
            print(f"{label:<6} {acc:<12} {cat_dir}")
        else:
            print(f"{label:<6} {acc:.2f}%        {cat_dir}")
    print("-" * 70)
    valid = [r[2] for r in results if isinstance(r[2], (int, float))]
    if valid:
        import numpy as np
        print(f"平均 (共 {len(valid)} 类): {np.mean(valid):.2f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
