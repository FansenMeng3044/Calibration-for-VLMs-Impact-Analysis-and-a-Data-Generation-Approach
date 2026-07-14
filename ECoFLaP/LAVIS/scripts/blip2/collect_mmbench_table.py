#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 run_pure_wanda_cc3m_split_joint_dense_mmbench_full.sh 的 SUMMARY_LOG
解析成「方法 × (Overall + 各学科)」的一张对照表（Markdown + TSV）。

日志约定（由 mmbench_full() 与 mmmu_eval_by_discipline.py 产出）：
  ========== MMBench 全量 | 模型=<tag> | split=dev ==========
  ...
  Overall accuracy: 61.34%
  By discipline:
    Science: 58.10% (123)
    ...

用法:
  python scripts/blip2/collect_mmbench_table.py --log <SUMMARY_LOG> \
      --out-md <table.md> --out-tsv <table.tsv>
"""
import argparse
import re

MODEL_RE = re.compile(r"模型=([^|]+?)\s*\|")
OVERALL_RE = re.compile(r"Overall accuracy:\s*([\d.]+)%")
DISC_RE = re.compile(r"^\s+(.+?):\s*([\d.]+)%\s*\((\d+)\)\s*$")


def parse(log_path):
    models = []            # 保序: [(tag, {"Overall": f, disc: (acc, n)})]
    cur = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = MODEL_RE.search(line)
            if m:
                cur = {"__tag__": m.group(1).strip()}
                models.append(cur)
                continue
            if cur is None:
                continue
            mo = OVERALL_RE.search(line)
            if mo and "Overall" not in cur:
                cur["Overall"] = float(mo.group(1))
                continue
            md = DISC_RE.match(line)
            if md:
                name = md.group(1).strip()
                if name.lower().startswith("overall"):
                    continue
                cur[name] = (float(md.group(2)), int(md.group(3)))
    return models


def build_table(models):
    # 学科列 = 各模型出现过的学科并集，按首次出现顺序
    discs = []
    for mdl in models:
        for k in mdl:
            if k in ("__tag__", "Overall") or k in discs:
                continue
            discs.append(k)
    cols = ["Overall"] + discs
    rows = []
    for mdl in models:
        row = [mdl["__tag__"]]
        for c in cols:
            if c == "Overall":
                row.append(f"{mdl.get('Overall'):.2f}" if "Overall" in mdl else "-")
            else:
                v = mdl.get(c)
                row.append(f"{v[0]:.2f}" if v else "-")
        rows.append(row)
    return cols, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-tsv", default=None)
    args = ap.parse_args()

    models = parse(args.log)
    if not models:
        raise SystemExit(f"[FATAL] 未从日志解析到任何 '模型=' 段: {args.log}")
    cols, rows = build_table(models)
    header = ["method"] + cols

    # Markdown
    md = []
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    md_str = "\n".join(md)

    # TSV
    tsv = ["\t".join(header)] + ["\t".join(r) for r in rows]
    tsv_str = "\n".join(tsv)

    print(md_str)
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(md_str + "\n")
        print(f"\n[OK] Markdown 表 -> {args.out_md}")
    if args.out_tsv:
        with open(args.out_tsv, "w", encoding="utf-8") as f:
            f.write(tsv_str + "\n")
        print(f"[OK] TSV 表 -> {args.out_tsv}")


if __name__ == "__main__":
    main()
