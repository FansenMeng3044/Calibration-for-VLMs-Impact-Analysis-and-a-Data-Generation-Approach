#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
汇总 UKMP 剪枝权重在多 benchmark 上的评测（仅 overall）。
与 run_ukmp_eval_mme_okvqa_mmmu_calib_ckpts.sh 中的 job_id / calib_tag 一致。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_SUITES: List[Tuple[str, str]] = [
    ("OKVQA_train_overall_calib", "okvqa_eval_ukmp_calibOKVQAoverall_fullval"),
    # ("MME_calib", "okvqa_eval_ukmp_calibMME_fullval"),
    # ("MMMU_overall_calib", "okvqa_eval_ukmp_calibMMMUoverall_fullval"),
]


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


def _pick_metrics(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        ct = r.get("calib_tag")
        bm = r.get("benchmark")
        if not ct or not bm:
            continue
        out[(str(ct), str(bm))] = r
    return out


def build_table(
    repo_root: Path,
    metrics_jsonl: Path,
    suites: List[Tuple[str, str]],
) -> Tuple[str, str]:
    okvqa_root = repo_root / "lavis" / "output" / "BLIP2" / "OKVQA"
    picked = _pick_metrics(_load_jsonl(metrics_jsonl))

    md_lines = [
        "# UKMP 剪枝评测汇总（仅 overall）",
        "",
        f"- metrics jsonl: `{metrics_jsonl}`",
        f"- OKVQA: `{okvqa_root}`",
        "",
        "## 总表（准确率 %）",
        "",
        "| Calibration | MMBench | MME yes/no | OKVQA overall (full val) | MMMU |",
        "|---|---:|---:|---:|---:|",
    ]
    tsv_lines = [
        "calibration\tmmbench_pct\tmme_yesno_pct\tokvqa_overall_pct\tmmmu_pct"
    ]

    for calib_tag, fullval_job in suites:
        mb = picked.get((calib_tag, "MMBench"), {})
        mme = picked.get((calib_tag, "MME_yesno"), {})
        mmmu = picked.get((calib_tag, "MMMU"), {})
        okvqa = _last_agg_metrics(okvqa_root / fullval_job / "evaluate.txt")

        def fmt(x: Optional[float]) -> str:
            return f"{x:.2f}" if x is not None else "—"

        md_lines.append(
            f"| {calib_tag} | {fmt(mb.get('value'))} | {fmt(mme.get('value'))} | "
            f"{fmt(okvqa)} | {fmt(mmmu.get('value'))} |"
        )
        tsv_lines.append(
            "\t".join(
                [
                    calib_tag,
                    fmt(mb.get("value")).replace("—", ""),
                    fmt(mme.get("value")).replace("—", ""),
                    fmt(okvqa).replace("—", ""),
                    fmt(mmmu.get("value")).replace("—", ""),
                ]
            )
        )

    md_lines.extend(
        [
            "",
            "## 说明",
            "",
            "- **MMBench / MMMU**：`scripts/blip2/mmmu_eval_by_discipline.py --overall_only`。",
            "- **OKVQA**：`evaluate_blip2_pruned.py` 全量 val，`evaluate.txt` 中 `agg_metrics`。",
            "",
        ]
    )

    return "\n".join(md_lines) + "\n", "\n".join(tsv_lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--metrics-jsonl", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-tsv", default="", help="可选，写入 TSV")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    mj = Path(args.metrics_jsonl).resolve()
    md_s, tsv_s = build_table(repo, mj, DEFAULT_SUITES)

    out_md = Path(args.out_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md_s, encoding="utf-8")
    print(md_s)
    if args.out_tsv:
        Path(args.out_tsv).write_text(tsv_s, encoding="utf-8")


if __name__ == "__main__":
    main()
