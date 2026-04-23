#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
汇总「纯 Wanda + 四 calibration × 四基准」跑出来的结果。

数据来源（与 run_pure_wanda_fourcalib_prune_then_fourbench_each.sh 一致）:
  - MMBench / MMMU / MathVista：若评测时设置了 LAVIS_METRICS_JSONL（脚本已默认写入
    training_statistics/pure_wanda_fourbench_metrics.jsonl），从 jsonl 读 overall_accuracy_percent。
  - OKVQA：从 lavis/output/BLIP2/OKVQA/okvqa_eval_<job_id>/evaluate.txt 解析最后一行含 agg_metrics 的 JSON。

联合剪枝单文件时：LAVIS_EVAL_CALIB_TAG = joint_<stem>，stem = pruned_checkpoint 文件名去 .pth；
OKVQA 的 --job_id = okvqa_eval_<stem>。

用法（在 ECoFLaP/LAVIS 根目录）:
  python scripts/blip2/collect_pure_wanda_fourbench_summary.py
  python scripts/blip2/collect_pure_wanda_fourbench_summary.py \\
    --job-prefix pure_wanda_calib \\
    --metrics-jsonl training_statistics/pure_wanda_fourbench_metrics.jsonl \\
    --out-md training_statistics/pure_wanda_fourbench_summary.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _last_agg_metrics(evaluate_txt: Path) -> Optional[float]:
    if not evaluate_txt.is_file():
        return None
    last = None
    with open(evaluate_txt, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "agg_metrics" in data:
                v = data["agg_metrics"]
                if v is not None:
                    last = float(v)
    return last


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _index_jsonl(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """(calib_tag, benchmark) -> record"""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        ct = r.get("calib_tag")
        bm = r.get("benchmark")
        if not ct or not bm:
            continue
        out[(str(ct), str(bm))] = r
    return out


def _fmt(v: Optional[float]) -> str:
    return f"{v:.2f}" if v is not None else "—"


def build_table(
    repo_root: Path,
    job_prefix: str,
    calibs: List[str],
    metrics_jsonl: Path,
) -> Tuple[str, str]:
    lib_root = repo_root / "lavis"
    okvqa_root = lib_root / "output" / "BLIP2" / "OKVQA"
    rows = _load_jsonl(metrics_jsonl)
    idx = _index_jsonl(rows)

    md: List[str] = [
        "# 纯 Wanda 四 calibration × 四基准 汇总",
        "",
        f"- repo: `{repo_root}`",
        f"- job 前缀: `{job_prefix}_<calib>`（与 pruned_checkpoint 文件名一致）",
        f"- metrics jsonl: `{metrics_jsonl}`（MMBench / MMMU / MathVista）",
        f"- OKVQA: `{okvqa_root}/okvqa_eval_<job_id>/evaluate.txt`",
        "",
        "| calibration 来源 | stem (ckpt) | MMBench % | MMMU % | OKVQA agg | MathVista % |",
        "|---|---|--:|--:|--:|--:|",
    ]
    tsv: List[str] = [
        "calib\tstem\tmmbench_pct\tmmmu_pct\tokvqa_agg\tmathvista_pct",
    ]

    for calib in calibs:
        stem = f"{job_prefix}_{calib}"
        tag = f"joint_{stem}"
        okvqa_job = f"okvqa_eval_{stem}"
        mb = idx.get((tag, "MMBench"), {})
        mm = idx.get((tag, "MMMU"), {})
        mv = idx.get((tag, "MathVista_MC"), {})
        ok = _last_agg_metrics(okvqa_root / okvqa_job / "evaluate.txt")

        def gv(d: Dict[str, Any]) -> Optional[float]:
            v = d.get("value")
            return float(v) if v is not None else None

        md.append(
            f"| {calib} | `{stem}` | {_fmt(gv(mb))} | {_fmt(gv(mm))} | {_fmt(ok)} | {_fmt(gv(mv))} |"
        )
        tsv.append(
            "\t".join(
                [
                    calib,
                    stem,
                    _fmt(gv(mb)).replace("—", ""),
                    _fmt(gv(mm)).replace("—", ""),
                    _fmt(ok).replace("—", ""),
                    _fmt(gv(mv)).replace("—", ""),
                ]
            )
        )

    md.extend(
        [
            "",
            "## 说明",
            "",
            "- 若 MMBench/MMMU/MathVista 为 「—」，检查是否在本次评测前启用了 `LAVIS_METRICS_JSONL`（"
            "`run_pure_wanda_fourcalib_prune_then_fourbench_each.sh` 会默认追加到 "
            "`training_statistics/pure_wanda_fourbench_metrics.jsonl`）。旧跑次可重跑四 eval 或手动补记。",
            "- OKVQA 为 `evaluate_blip` 全量 val 的 `agg_metrics`。",
            "- 权重路径：`pruned_checkpoint/<stem>.pth`。",
            "",
        ]
    )
    return "\n".join(md) + "\n", "\n".join(tsv) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
        help="ECoFLaP/LAVIS 根目录",
    )
    ap.add_argument("--job-prefix", default="pure_wanda_calib", help="与剪枝时 JOB_PREFIX 一致")
    ap.add_argument(
        "--calibs",
        default="mmbench,mmmu,okvqa,mathvista",
        help="逗号分隔，与跑实验时 CALIBS 一致",
    )
    ap.add_argument(
        "--metrics-jsonl",
        type=Path,
        default=None,
        help="默认: <repo>/training_statistics/pure_wanda_fourbench_metrics.jsonl",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="默认: <repo>/training_statistics/pure_wanda_fourbench_summary.md",
    )
    ap.add_argument("--out-tsv", type=Path, default=None)
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    mj = args.metrics_jsonl or (repo / "training_statistics" / "pure_wanda_fourbench_metrics.jsonl")
    mj = mj.resolve()

    calibs = [c.strip() for c in args.calibs.split(",") if c.strip()]

    md_s, tsv_s = build_table(repo, args.job_prefix, calibs, mj)

    out_md = args.out_md or (repo / "training_statistics" / "pure_wanda_fourbench_summary.md")
    out_md = out_md.resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md_s, encoding="utf-8")
    print(md_s)
    print(f"[INFO] 已写入: {out_md}")

    if args.out_tsv:
        p = args.out_tsv.resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(tsv_s, encoding="utf-8")
        print(f"[INFO] TSV: {p}")


if __name__ == "__main__":
    main()
