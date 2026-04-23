#!/usr/bin/env python3
"""
Classic MME calibration -> MMMU_single_image eval-parquet (for mmu_eval_by_discipline.py)

We reuse:
  /root/autodl-tmp/MME_calibration/mmE_calibration_train.json
which already stores:
  - question: original MME question text
  - answer: lowercase "yes" / "no"
  - image: filename under /root/autodl-tmp/MME_calibration/images

Then we map:
  - options: ['yes', 'no']
  - answer letter:
      yes -> 'A'
      no  -> 'B'

Parquet columns required by mmmu_eval_by_discipline.py:
  - id
  - question
  - options (string repr of python list works best)
  - answer (letter 'A'/'B')
  - image_1..image_7 with exactly one image dict containing {"bytes": ...}
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List

import pandas as pd


def clean_mme_question(q: str) -> str:
    if q is None:
        return ""
    s = str(q)
    # MME questions usually end with: "Please answer yes or no."
    s = re.sub(r"(?i)\s*please answer yes or no\.?\s*$", "", s).strip()
    return s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calib_json",
        default="/root/autodl-tmp/MME_calibration/mmE_calibration_train.json",
        help="mmE_calibration_train.json (list of {image, question, answer})",
    )
    parser.add_argument(
        "--images_root",
        default="/root/autodl-tmp/MME_calibration/images",
        help="Directory containing exported MME_calibration images",
    )
    parser.add_argument("--out_root", default="/root/autodl-tmp/MME_eval", help="Output root")
    parser.add_argument("--subject", default="Other", help="Parquet folder name (mmmu eval discipline)")
    parser.add_argument("--file_prefix", default="test", help="Split prefix used by mmu eval script")
    parser.add_argument("--chunk_size", type=int, default=500, help="Rows per parquet file")
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    with open(args.calib_json, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    if not data:
        raise RuntimeError(f"Empty calibration json: {args.calib_json}")

    if args.max_samples and args.max_samples > 0:
        data = data[: args.max_samples]

    os.makedirs(os.path.join(args.out_root, args.subject), exist_ok=True)
    out_subj_dir = os.path.join(args.out_root, args.subject)

    options_repr = str(["yes", "no"])

    image_bytes_cache: Dict[str, bytes] = {}

    records: List[Dict[str, Any]] = []
    file_idx = 0

    def flush() -> None:
        nonlocal records, file_idx
        if not records:
            return
        df = pd.DataFrame.from_records(records)
        col_order = [
            "id",
            "question",
            "options",
            "answer",
            "image_1",
            "image_2",
            "image_3",
            "image_4",
            "image_5",
            "image_6",
            "image_7",
        ]
        existing = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]
        df = df[existing]
        out_path = os.path.join(out_subj_dir, f"{args.file_prefix}-{file_idx:05d}.parquet")
        df.to_parquet(out_path, index=False)
        records = []
        file_idx += 1

    for i, row in enumerate(data):
        img_name = str(row.get("image", "")).strip()
        if not img_name:
            continue
        ans = str(row.get("answer", "")).strip().lower()
        if ans not in ("yes", "no"):
            raise ValueError(f"Unexpected answer in calib json at idx={i}: {ans!r}")
        answer_letter = "A" if ans == "yes" else "B"

        q = clean_mme_question(str(row.get("question", "")).strip())

        img_path = os.path.join(args.images_root, img_name)
        if img_path not in image_bytes_cache:
            with open(img_path, "rb") as f:
                image_bytes_cache[img_path] = f.read()

        image_1 = {"bytes": image_bytes_cache[img_path], "path": ""}

        records.append(
            {
                "id": str(i),
                "question": q,
                "options": options_repr,
                "answer": answer_letter,
                "image_1": image_1,
                "image_2": None,
                "image_3": None,
                "image_4": None,
                "image_5": None,
                "image_6": None,
                "image_7": None,
            }
        )

        if len(records) >= args.chunk_size:
            flush()

    flush()
    print(f"[DONE] Wrote {file_idx} parquet file(s) under {out_subj_dir}")


if __name__ == "__main__":
    main()

