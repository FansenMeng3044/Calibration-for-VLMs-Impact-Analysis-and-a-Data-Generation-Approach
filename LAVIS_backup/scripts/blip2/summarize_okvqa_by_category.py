# Copyright (c) 2022, salesforce.com, inc.
# SPDX-License-Identifier: BSD-3-Clause
# 汇总 OK-VQA 按类 eval 结果：读取 output/BLIP2/OKVQA/<job_id_prefix>_<Category>/evaluate.txt 中的 agg_metrics，打表并算平均。
# 用法（在 LAVIS_backup 根目录）: python scripts/blip2/summarize_okvqa_by_category.py [--output_base DIR] [--job_id_prefix PREFIX]

import argparse
import json
import os

DEFAULT_OUTPUT_BASE = "lavis/output/BLIP2/OKVQA"
DEFAULT_JOB_PREFIX = "okvqa_cf_0.5"


def main():
    parser = argparse.ArgumentParser(description="Summarize per-category OK-VQA eval results")
    parser.add_argument("--output_base", default=DEFAULT_OUTPUT_BASE, help="Base dir for OKVQA results (e.g. lavis/output/BLIP2/OKVQA)")
    parser.add_argument("--job_id_prefix", default=DEFAULT_JOB_PREFIX, help="Job id prefix, e.g. okvqa_cf_0.5 -> okvqa_cf_0.5_Cooking_and_Food")
    args = parser.parse_args()

    base = os.path.abspath(args.output_base)
    if not os.path.isdir(base):
        print(f"[WARN] Not a directory: {base}")
        return

    prefix = args.job_id_prefix + "_"
    results = []
    for name in sorted(os.listdir(base)):
        if not name.startswith(prefix) or not os.path.isdir(os.path.join(base, name)):
            continue
        category = name[len(prefix):]
        eval_file = os.path.join(base, name, "evaluate.txt")
        if not os.path.isfile(eval_file):
            results.append((category, None))
            continue
        acc = None
        with open(eval_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    acc = data.get("agg_metrics")
                    if acc is not None:
                        pass  # 取最后一次（最新一次运行）
                except json.JSONDecodeError:
                    continue
        results.append((category, acc))

    if not results:
        print("No per-category result dirs found under", base, "with prefix", prefix)
        return

    print("Category                                    | Accuracy")
    print("-" * 55)
    valid = []
    for cat, acc in results:
        if acc is not None:
            print(f"{cat:43} | {acc:.2f}%")
            valid.append(acc)
        else:
            print(f"{cat:43} | (no agg_metrics)")
    print("-" * 55)
    if valid:
        avg = sum(valid) / len(valid)
        print(f"{'Average (' + str(len(valid)) + ' categories)':43} | {avg:.2f}%")


if __name__ == "__main__":
    main()
