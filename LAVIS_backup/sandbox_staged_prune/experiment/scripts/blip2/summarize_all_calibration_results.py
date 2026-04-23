# Copyright (c) 2022, salesforce.com, inc.
# SPDX-License-Identifier: BSD-3-Clause
# 汇总「所有 calibration」的 11 类 eval 结果：自动发现所有 prefix，打总表（行=eval 类，列=calibration run）并每列平均。
# 用法（在 LAVIS_backup 根目录）: python scripts/blip2/summarize_all_calibration_results.py [--output_base DIR] [--csv FILE]

import argparse
import csv
import json
import os

DEFAULT_OUTPUT_BASE = "lavis/output/BLIP2/OKVQA"

# 11 个 eval 类（用于从目录名反推 prefix）
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


def get_acc_from_evaluate_txt(path: str):
    acc = None
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                acc = data.get("agg_metrics")
            except json.JSONDecodeError:
                continue
    return acc


def main():
    parser = argparse.ArgumentParser(description="Summarize all calibration runs: one table, columns=calibration, rows=eval category")
    parser.add_argument("--output_base", default=DEFAULT_OUTPUT_BASE, help="Base dir for OKVQA results")
    parser.add_argument("--csv", default=None, help="Also write results to CSV file (e.g. results.csv)")
    args = parser.parse_args()

    base = os.path.abspath(args.output_base)
    if not os.path.isdir(base):
        print(f"[WARN] Not a directory: {base}")
        return

    # 收集 (prefix, category, acc)；目录名为 prefix_category，且以 11 类之一结尾
    prefix_to_cat_acc = {}  # prefix -> { category -> acc }
    for name in sorted(os.listdir(base)):
        if not os.path.isdir(os.path.join(base, name)):
            continue
        eval_file = os.path.join(base, name, "evaluate.txt")
        if not os.path.isfile(eval_file):
            continue
        for cat in OK_VQA_CATEGORIES:
            suffix = "_" + cat
            if name.endswith(suffix) and len(name) > len(suffix):
                prefix = name[: -len(suffix)]
                acc = get_acc_from_evaluate_txt(eval_file)
                if prefix not in prefix_to_cat_acc:
                    prefix_to_cat_acc[prefix] = {}
                prefix_to_cat_acc[prefix][cat] = acc
                break

    if not prefix_to_cat_acc:
        print("No result dirs (with evaluate.txt) found under", base)
        return

    # 列顺序：按 prefix 字符串排序
    prefixes = sorted(prefix_to_cat_acc.keys())
    cat_width = 43
    col_width = 8
    header = "Eval Category" + " " * (cat_width - 11)
    for p in prefixes:
        short = (p.replace("okvqa_cf_0.5_", "")[: col_width - 1]) if len(p) > col_width else p
        header += " | " + short[: col_width - 1].ljust(col_width - 1)
    header += " | Avg"
    print(header)
    print("-" * len(header))

    for cat in OK_VQA_CATEGORIES:
        row = cat[:cat_width].ljust(cat_width)
        vals = []
        for p in prefixes:
            acc = prefix_to_cat_acc[p].get(cat)
            if acc is not None:
                row += f" | {acc:5.2f}%"
                vals.append(acc)
            else:
                row += " |   --  "
        if vals:
            row += f" | {sum(vals)/len(vals):5.2f}%"
        else:
            row += " |   --  "
        print(row)

    print("-" * len(header))
    row_avg = "Average (11 cats)".ljust(cat_width)
    total_avgs = []
    for p in prefixes:
        d = prefix_to_cat_acc[p]
        v = [d[c] for c in OK_VQA_CATEGORIES if d.get(c) is not None]
        avg = sum(v) / len(v) if v else None
        if avg is not None:
            row_avg += f" | {avg:5.2f}%"
            total_avgs.append(avg)
        else:
            row_avg += " |   --  "
    if total_avgs:
        row_avg += f" | {sum(total_avgs)/len(total_avgs):5.2f}%"
    print(row_avg)
    print("")
    print("Prefixes (calibration runs):", prefixes)

    # 输出 CSV
    if args.csv:
        csv_path = os.path.abspath(args.csv)
        d = os.path.dirname(csv_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Eval Category"] + prefixes + ["Avg"])
            for cat in OK_VQA_CATEGORIES:
                vals = []
                row = [cat]
                for p in prefixes:
                    acc = prefix_to_cat_acc[p].get(cat)
                    row.append(round(acc, 2) if acc is not None else "")
                    if acc is not None:
                        vals.append(acc)
                row.append(round(sum(vals) / len(vals), 2) if vals else "")
                w.writerow(row)
            avg_row = ["Average (11 cats)"]
            for p in prefixes:
                d = prefix_to_cat_acc[p]
                v = [d[c] for c in OK_VQA_CATEGORIES if d.get(c) is not None]
                avg_row.append(round(sum(v) / len(v), 2) if v else "")
            avg_row.append(round(sum(total_avgs) / len(total_avgs), 2) if total_avgs else "")
            w.writerow(avg_row)
        print(f"CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
