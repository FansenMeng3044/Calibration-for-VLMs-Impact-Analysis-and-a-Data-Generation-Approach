#!/usr/bin/env python3
"""
MMBench TSV (VLMEvalKit) -> ECoFLaP OK-VQA calibration (image, question, answer)

Input:
  - MMBench_DEV_EN.tsv / MMBench_TEST_EN.tsv (columns include: index, question, hint,
    A/B/C/D, answer, image(base64 jpeg), split, ...)

Output (for ecoflap / OKVQACalibrationDataset):
  - out_dir/images/ : decoded images as jpg
  - out_dir/mmbench_calibration_train.json : list of {image, question, answer}

Answer mode:
  - "letter": answer is "A"/"B"/"C"/"D"

Question construction:
  - question = original question + optional hint + options A/B/C/D lines.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
from typing import Dict, List

from PIL import Image


def norm_letter(s: str) -> str:
    return str(s).strip().upper()[:1]


def build_question(row: Dict[str, str]) -> str:
    q = (row.get("question") or "").strip()
    hint = (row.get("hint") or "").strip()
    if hint:
        q = q + f"\nHint: {hint}"

    # options in fixed order to keep evaluation consistent
    for letter in ["A", "B", "C", "D"]:
        opt = (row.get(letter) or "")
        q = q + f"\n{letter}. {opt}"
    return q


def decode_image_to_rgb(b64_str: str) -> Image.Image:
    # b64 is usually a raw base64 string (not a data URL)
    import binascii

    b64_str = (b64_str or "").strip()
    # remove potential whitespace/newlines inside the field
    b64_str = "".join(b64_str.split())
    # fix missing padding: base64 length should be a multiple of 4
    padding = (-len(b64_str)) % 4
    if padding:
        b64_str = b64_str + ("=" * padding)

    try:
        img_bytes = base64.b64decode(b64_str, validate=False)
    except binascii.Error:
        # fallback: urlsafe variant (just in case '-' '_' appear)
        img_bytes = base64.urlsafe_b64decode(b64_str)
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img
    except Exception as e:
        # Provide useful debug context so we can locate broken TSV rows.
        raise RuntimeError(
            f"Cannot identify image from base64. "
            f"b64_len={len(b64_str)} decoded_bytes_len={len(img_bytes)} err={type(e).__name__}: {e}"
        ) from e


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", required=True, help="Path to MMBench_DEV_EN.tsv or MMBench_TEST_EN.tsv")
    parser.add_argument("--out_dir", required=True, help="Output dir root (creates images/ and json)")
    parser.add_argument("--prefix", default="dev", help="Filename prefix: e.g. dev/test")
    parser.add_argument("--json_name", default="mmbench_calibration_train.json")
    parser.add_argument("--skip_bad_images", action="store_true", help="Skip rows whose images cannot be decoded")
    parser.add_argument(
        "--dedup_by_image_id",
        action="store_true",
        help="Save one image per unique image_id (image short values are treated as image_id references).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Pass 1: collect base64 images keyed by image_id (= the `index` of rows whose `image` is long base64)
    base64_by_image_id: Dict[str, str] = {}
    with open(args.tsv, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quotechar='"')
        for row in reader:
            idx = str(row.get("index", "")).strip()
            img_field = str(row.get("image", "") or "").strip()
            if not idx or not img_field:
                continue
            if len(img_field) >= 1000:
                base64_by_image_id[idx] = img_field

    if not base64_by_image_id:
        raise RuntimeError("No long base64 images found in TSV; unexpected MMBench TSV format.")

    # Pass 2: build calibration json + save images
    records: List[Dict[str, str]] = []
    bad_indices: List[str] = []
    saved_image_ids: set[str] = set()

    with open(args.tsv, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quotechar='"')
        required_cols = ["index", "question", "hint", "A", "B", "C", "D", "answer", "image"]
        for c in required_cols:
            if c not in reader.fieldnames:
                raise ValueError(f"Missing column '{c}'. Fields: {reader.fieldnames}")

        for row in reader:
            idx = str(row.get("index", "")).strip()
            if not idx:
                continue

            img_field = str(row.get("image", "") or "").strip()
            # In VLMEvalKit TSV: long base64 means actual image; short values are image_id references.
            if len(img_field) >= 1000:
                image_id = idx
                img_b64 = img_field
            else:
                image_id = img_field
                img_b64 = base64_by_image_id.get(image_id)
                if not img_b64:
                    msg = f"[BAD] Missing base64 for image_id={image_id} (row idx={idx})"
                    if args.skip_bad_images:
                        print(msg)
                        bad_indices.append(idx)
                        continue
                    raise RuntimeError(msg)

            img_filename = f"{args.prefix}_{image_id}.jpg" if args.dedup_by_image_id else f"{args.prefix}_{idx}.jpg"
            img_path = os.path.join(img_dir, img_filename)

            # Save image (dedup by image_id if enabled)
            try:
                if not args.dedup_by_image_id or (image_id not in saved_image_ids) or (not os.path.exists(img_path)):
                    img = decode_image_to_rgb(img_b64)
                    img.save(img_path, format="JPEG")
                    saved_image_ids.add(image_id)
            except Exception as e:
                ans = str(row.get("answer", "") or "")
                preview = (img_b64 or "")[:24]
                msg = f"[BAD] MMBench image decode failed. idx={idx} answer={ans} image_id={image_id} preview={preview!r} err={e}"
                if args.skip_bad_images:
                    print(msg)
                    bad_indices.append(idx)
                    continue
                raise RuntimeError(msg) from e

            question = build_question(row)
            answer = norm_letter(row.get("answer", ""))
            if answer not in {"A", "B", "C", "D"}:
                msg = f"[BAD] Unexpected answer letter '{answer}' for index={idx}"
                if args.skip_bad_images:
                    print(msg)
                    bad_indices.append(idx)
                    continue
                raise ValueError(msg)

            records.append({"image": img_filename, "question": question, "answer": answer})

    json_path = os.path.join(args.out_dir, args.json_name)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(records, jf, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} records to {json_path}")
    print(f"Images in {img_dir}: {len(os.listdir(img_dir))}")
    if bad_indices:
        print(f"Skipped bad rows: {len(bad_indices)}")
        print("Bad indices (first 30):", bad_indices[:30])


if __name__ == "__main__":
    main()

