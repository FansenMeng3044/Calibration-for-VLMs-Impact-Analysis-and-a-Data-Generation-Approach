#!/usr/bin/env python3
"""
汇总所有 OK-VQA 结果：自动发现 output/BLIP2/OKVQA 下所有 prefix（含 9 类 calibration 的 okvqa_eval_calib*），
对每个 prefix 计算 11 类准确率，输出总表（行=eval 类别，列=calibration/模型）。
用法（在 UKMP-main/LAVIS 下）:
  python scripts/structured_blip2/summarize_all_okvqa_results.py
  python scripts/structured_blip2/summarize_all_okvqa_results.py --csv summary.csv  # 同时保存 CSV
"""
import argparse
import os
import sys

LAVIS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.basename(LAVIS_ROOT) != "LAVIS":
    LAVIS_ROOT = os.path.join(LAVIS_ROOT, "LAVIS")
sys.path.insert(0, os.path.join(LAVIS_ROOT, "lavis", "common"))
from vqa_tools.vqa import VQA
from vqa_tools.vqa_eval import VQAEval

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
LABELS = [c[0] for c in CATEGORIES]
DATA_ROOT = "/root/autodl-tmp/datasets/okvqa_by_category"
OUTPUT_ROOT = os.path.join(LAVIS_ROOT, "lavis", "output", "BLIP2", "OKVQA")


def discover_prefixes():
    """发现所有包含完整 11 类结果的 prefix（目录名 prefix_<Label> 且均有 result/test_vqa_result.json）。"""
    if not os.path.isdir(OUTPUT_ROOT):
        return []
    found = {}
    for d in os.listdir(OUTPUT_ROOT):
        full = os.path.join(OUTPUT_ROOT, d)
        if not os.path.isdir(full):
            continue
        for label in LABELS:
            if not d.endswith("_" + label):
                continue
            prefix = d[: -(len(label) + 1)].rstrip("_")
            if not prefix:
                continue
            res_file = os.path.join(OUTPUT_ROOT, d, "result", "test_vqa_result.json")
            if os.path.isfile(res_file):
                found.setdefault(prefix, set()).add(label)
            break
    # 只保留 11 类都有的 prefix
    prefixes = [p for p, labels in found.items() if labels >= set(LABELS)]
    # 排序：fullprecision 优先，然后 okvqa_eval（无 calib），再按 calib 名
    def key(p):
        if "fullprecision" in p:
            return (0, p)
        if "calib" not in p:
            return (1, p)
        return (2, p)
    prefixes.sort(key=key)
    return prefixes


def run_one(prefix):
    """对单个 prefix 计算 11 类准确率，返回 list of (label, acc float or None or str)。"""
    results = []
    for label, cat_dir in CATEGORIES:
        anno = os.path.join(DATA_ROOT, cat_dir, "mscoco_val2014_annotations.json")
        ques = os.path.join(DATA_ROOT, cat_dir, "OpenEnded_mscoco_val2014_questions.json")
        res_file = os.path.join(OUTPUT_ROOT, f"{prefix}_{label}", "result", "test_vqa_result.json")
        if not os.path.isfile(anno) or not os.path.isfile(ques) or not os.path.isfile(res_file):
            results.append((label, None))
            continue
        try:
            vqa = VQA(anno, ques)
            vqa_result = vqa.loadRes(res_file, ques)
            scorer = VQAEval(vqa, vqa_result, n=2)
            scorer.evaluate()
            acc = scorer.accuracy["overall"]
            results.append((label, acc))
        except Exception as e:
            results.append((label, f"err: {e}"))
    return results


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description="汇总所有 OK-VQA 结果")
    parser.add_argument("--csv", type=str, default="", help="若指定则写入该 CSV 文件")
    parser.add_argument("--prefix", type=str, default="", help="只汇总该 prefix（不指定则自动发现全部）")
    args = parser.parse_args()

    if args.prefix:
        prefixes = [p.strip() for p in args.prefix.split(",") if p.strip()]
    else:
        prefixes = discover_prefixes()

    if not prefixes:
        print("[WARN] 未发现任何完整 11 类结果。请确认 lavis/output/BLIP2/OKVQA 下存在 okvqa_eval_* 或 okvqa_eval_calib*_* 目录。")
        return

    print(f"[INFO] 发现 {len(prefixes)} 个 prefix: {prefixes}\n")

    # 列名：简短显示（去掉 okvqa_eval_ 前缀）
    def short_name(p):
        if p == "okvqa_fullprecision_eval":
            return "fullprecision"
        if p.startswith("okvqa_eval_"):
            return p.replace("okvqa_eval_", "", 1)
        return p

    # 计算每个 prefix 的 11 类准确率
    table = {}  # prefix -> list of acc (indexed by LABELS)
    for prefix in prefixes:
        rows = run_one(prefix)
        accs = []
        for label in LABELS:
            acc = next((r[1] for r in rows if r[0] == label), None)
            accs.append(acc)
        table[prefix] = accs

    # 打印总表：行=类别，列=prefix
    col_names = [short_name(p) for p in prefixes]
    col_width = max(10, max(len(c) for c in col_names))
    head = f"{'类别':<8}" + "".join(f"{c:>{col_width}}" for c in col_names) + "  (平均)"
    print("=" * len(head))
    print("OK-VQA 汇总（行=eval 类别，列=calibration/模型）")
    print("=" * len(head))
    print(head)
    print("-" * len(head))

    for i, label in enumerate(LABELS):
        row_vals = []
        for prefix in prefixes:
            v = table[prefix][i]
            if v is None:
                s = "-"
            elif isinstance(v, str):
                s = "err"
            else:
                s = f"{v:.2f}"
            row_vals.append(s)
        row_str = f"{label:<8}" + "".join(f"{s:>{col_width}}" for s in row_vals)
        # 行平均（仅数值）
        nums = []
        for prefix in prefixes:
            v = table[prefix][i]
            if isinstance(v, (int, float)):
                nums.append(v)
        row_str += f"  {np.mean(nums):.2f}" if nums else "  -"
        print(row_str)

    print("-" * len(head))
    # 列平均
    avg_row = "平均    "
    for prefix in prefixes:
        vals = [v for v in table[prefix] if isinstance(v, (int, float))]
        avg_row += f"{np.mean(vals):>{col_width}.2f}" if vals else f"{'-':>{col_width}}"
    all_vals = [v for accs in table.values() for v in accs if isinstance(v, (int, float))]
    avg_row += f"  {np.mean(all_vals):.2f}" if all_vals else "  -"
    print(avg_row)
    print("=" * len(head))

    # 可选：写 CSV
    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write("category," + ",".join(short_name(p) for p in prefixes) + ",mean\n")
            for i, label in enumerate(LABELS):
                row_vals = []
                for prefix in prefixes:
                    v = table[prefix][i]
                    if v is None:
                        row_vals.append("")
                    elif isinstance(v, str):
                        row_vals.append("err")
                    else:
                        row_vals.append(f"{v:.2f}")
                nums = [table[p][i] for p in prefixes if isinstance(table[p][i], (int, float))]
                row_vals.append(f"{np.mean(nums):.2f}" if nums else "")
                f.write(f"{label}," + ",".join(row_vals) + "\n")
            # 平均行
            avg_vals = []
            for prefix in prefixes:
                vals = [v for v in table[prefix] if isinstance(v, (int, float))]
                avg_vals.append(f"{np.mean(vals):.2f}" if vals else "")
            all_vals = [v for accs in table.values() for v in accs if isinstance(v, (int, float))]
            avg_vals.append(f"{np.mean(all_vals):.2f}" if all_vals else "")
            f.write("mean," + ",".join(avg_vals) + "\n")
        print(f"\n[INFO] 已写入 CSV: {args.csv}")


if __name__ == "__main__":
    main()
