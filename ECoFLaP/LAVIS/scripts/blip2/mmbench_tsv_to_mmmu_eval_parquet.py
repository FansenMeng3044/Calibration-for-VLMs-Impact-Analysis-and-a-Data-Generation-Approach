#!/usr/bin/env python3
"""
MMBench TEST/DEV TSV (VLMEvalKit format) -> MMMU_single_image-style parquet

Goal: reuse /root/autodl-tmp/ECoFLaP/LAVIS/scripts/blip2/mmmu_eval_by_discipline.py
to directly compute accuracy with its custom multi-choice judging logic.

The eval script expects parquet rows to look like:
  id: str
  question: str
  options: str (a python-list string is safest for ast.literal_eval)
  answer: str (A/B/C/D)
  image_1..image_7: only image_1 has {"bytes": <bytes>, ...}

We decode TSV base64 to raw encoded image bytes and store them inside image_1.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import os
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd


def _fix_base64_padding(b64: str) -> str:
    b64 = (b64 or "").strip()
    # Base64 length should be multiple of 4
    padding = (-len(b64)) % 4
    if padding:
        b64 = b64 + ("=" * padding)
    return b64


def decode_image_bytes(b64_str: str) -> bytes:
    """
    Decode image bytes from base64 field in VLMEvalKit TSV.
    Output bytes are expected to be directly consumable by PIL.Image.open(BytesIO(...)).
    """
    b64 = "".join((b64_str or "").split())
    b64 = _fix_base64_padding(b64)

    try:
        return base64.b64decode(b64, validate=False)
    except Exception:
        # fallback for urlsafe alphabet
        return base64.urlsafe_b64decode(b64)


def iter_tsv_rows(tsv_path: str) -> Iterator[Dict[str, str]]:
    with open(tsv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quotechar='"')
        for row in reader:
            yield row


def build_question(question: str, hint: str) -> str:
    q = (question or "").strip()
    h = (hint or "").strip()
    if h:
        q = q + "\nHint: " + h
    return q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", required=True, help="MMBench_TEST_EN.tsv or MMBench_DEV_EN.tsv")
    parser.add_argument("--out_root", required=True, help="Output root, will create <out_root>/Other/ ...")
    parser.add_argument("--subject", default="Other", help="MMMU eval subject folder name")
    parser.add_argument("--file_prefix", default="test", help="Filename prefix: test/dev/validation")
    parser.add_argument("--chunk_size", type=int, default=100, help="Rows per parquet file")
    parser.add_argument("--max_samples", type=int, default=0, help="For debugging; 0 = all")
    parser.add_argument("--skip_bad_images", action="store_true", help="Skip rows with decode failures")
    parser.add_argument("--cache_image_min_len", type=int, default=1000, help=">= this length => base64 image")
    args = parser.parse_args()

    out_subj_dir = os.path.join(args.out_root, args.subject)
    os.makedirs(out_subj_dir, exist_ok=True)

    # Pass 1: cache base64 strings keyed by image_id (= row['index'] for long base64 rows).
    base64_by_image_id: Dict[str, str] = {}
    n_pass1 = 0
    for row in iter_tsv_rows(args.tsv):
        idx = str(row.get("index", "") or "").strip()
        img_field = str(row.get("image", "") or "").strip()
        if not idx or not img_field:
            continue
        if len(img_field) >= args.cache_image_min_len:
            base64_by_image_id[idx] = img_field
        n_pass1 += 1
    if not base64_by_image_id:
        raise RuntimeError("No long base64 images found; TSV format unexpected or cache_image_min_len too high.")

    # Pass 2: decode (dedup by image_id) and write parquet chunks.
    decoded_bytes_cache: Dict[str, bytes] = {}
    records: List[Dict[str, object]] = []
    written_files = 0
    total_written_rows = 0

    def flush_chunk(chunk_idx: int) -> None:
        nonlocal records, written_files, total_written_rows
        if not records:
            return
        df = pd.DataFrame.from_records(records)
        # Ensure deterministic column order for easier sanity checks
        col_order = ["id", "question", "options", "answer", "image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"]
        existing = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]
        df = df[existing]
        out_path = os.path.join(out_subj_dir, f"{args.file_prefix}-{chunk_idx:05d}.parquet")
        df.to_parquet(out_path, index=False)
        written_files += 1
        total_written_rows += len(df)
        records = []

    max_samples = int(args.max_samples or 0)
    seen = 0

    for row in iter_tsv_rows(args.tsv):
        idx = str(row.get("index", "") or "").strip()
        if not idx:
            continue

        question = build_question(row.get("question", ""), row.get("hint", ""))
        # Options: eval script expects ast.literal_eval(parse_options()) to work,
        # so store python-list repr as string (like existing MMMU parquet).
        options_list = [str(row.get(k, "") or "") for k in ["A", "B", "C", "D"]]
        options_repr = str(options_list)
        answer = str(row.get("answer", "") or "").strip()

        img_field = str(row.get("image", "") or "").strip()
        if len(img_field) >= args.cache_image_min_len:
            image_id = idx
            img_b64 = img_field
        else:
            image_id = img_field
            img_b64 = base64_by_image_id.get(image_id)
            if not img_b64:
                if args.skip_bad_images:
                    continue
                raise RuntimeError(f"Missing base64 for image_id={image_id} (row index={idx})")

        if image_id not in decoded_bytes_cache:
            try:
                decoded_bytes_cache[image_id] = decode_image_bytes(img_b64)
            except Exception:
                if args.skip_bad_images:
                    continue
                raise

        img_bytes = decoded_bytes_cache[image_id]
        image_1 = {"bytes": img_bytes, "path": ""}

        records.append(
            {
                "id": idx,
                "question": question,
                "options": options_repr,
                "answer": answer,
                "image_1": image_1,
                "image_2": None,
                "image_3": None,
                "image_4": None,
                "image_5": None,
                "image_6": None,
                "image_7": None,
            }
        )

        seen += 1
        if len(records) >= args.chunk_size:
            flush_chunk(chunk_idx=written_files)

        if max_samples > 0 and seen >= max_samples:
            break

    # flush tail
    flush_chunk(chunk_idx=written_files)

    print(f"[DONE] TSV -> parquet: files={written_files}, rows={total_written_rows} (subject={args.subject})")


if __name__ == "__main__":
    main()

