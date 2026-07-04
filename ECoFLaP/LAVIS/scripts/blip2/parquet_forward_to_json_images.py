#!/usr/bin/env python3
"""Convert MMMU/MMBench-style parquet rows to recorder JSON plus image files.

The activation recorder expects rows like:
  {"image": "...jpg", "question": "...", "answer": "..."}

The four-benchmark eval parquet files usually store image bytes in image_1 and
keep options in a separate column.  This script extracts those image bytes,
writes real image files, and builds a JSON file whose question includes the
formatted options.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import shutil
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_COLUMNS = ("image", "image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7")
AUTO_TEXT_FIELDS = ("question", "caption", "text_input", "text", "prompt")
AUTO_OUTPUT_FIELDS = ("answer", "text_output", "caption", "text", "question")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert parquet benchmark rows into JSON/images for activation recording.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--parquet", nargs="*", default=[], help="One or more parquet files.")
    parser.add_argument("--parquet_dir", default=None, help="Root containing split parquet files.")
    parser.add_argument("--split", default="dev", help="File prefix to select under parquet_dir, e.g. dev/test/validation.")
    parser.add_argument("--subject", default=None, help="Optional subdirectory under parquet_dir, e.g. Other.")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_images_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--text_field", default="auto")
    parser.add_argument("--output_field", default="auto")
    parser.add_argument("--image_format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--skip_bad_rows", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (dict, list, tuple, bytes, bytearray, memoryview)):
        return False
    try:
        return bool(value != value)
    except Exception:
        return False
    return False


def value_to_text(value: Any) -> str:
    if is_missing(value):
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(value_to_text(v) for v in value if not is_missing(v)).strip()
    if isinstance(value, dict):
        return " ".join(value_to_text(v) for v in value.values() if not is_missing(v)).strip()
    return str(value).strip()


def parse_options(options_val: Any) -> List[str]:
    if is_missing(options_val):
        return []
    if isinstance(options_val, (list, tuple)):
        return [str(x).strip() for x in options_val if value_to_text(x)]
    s = str(options_val).strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if value_to_text(x)]
    except (ValueError, SyntaxError):
        pass
    return []


def format_options(options_val: Any) -> str:
    opts = parse_options(options_val)
    if not opts:
        return ""
    return "Options: " + " ".join("%s. %s" % (chr(ord("A") + i), opt) for i, opt in enumerate(opts))


def select_text(row: Any, field: str, auto_fields: Sequence[str]) -> Tuple[str, str]:
    fields = [field] if field != "auto" else list(auto_fields)
    for name in fields:
        if name in row and value_to_text(row.get(name)):
            text = value_to_text(row.get(name))
            text = re.sub(r"\s*<image\s*\d*>\s*", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"\s+", " ", text).strip()
            return text, name
    raise KeyError("missing non-empty text field among: %s" % ", ".join(fields))


def bytes_from_value(value: Any) -> Optional[bytes]:
    if is_missing(value):
        return None
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw is not None:
            return bytes_from_value(raw)
        path = value.get("path")
        if value_to_text(path) and os.path.exists(str(path)):
            with open(str(path), "rb") as handle:
                return handle.read()
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if hasattr(value, "tobytes") and not isinstance(value, str):
        try:
            return value.tobytes()
        except Exception:
            return None
    if isinstance(value, str) and os.path.exists(value):
        with open(value, "rb") as handle:
            return handle.read()
    return None


def extract_image_bytes(row: Any) -> bytes:
    for col in IMAGE_COLUMNS:
        if col not in row:
            continue
        img_bytes = bytes_from_value(row.get(col))
        if img_bytes:
            return img_bytes
    raise KeyError("missing image bytes in columns: %s" % ", ".join(IMAGE_COLUMNS))


def write_image(img_bytes: bytes, out_path: str, image_format: str) -> None:
    from PIL import Image

    ensure_dir(os.path.dirname(out_path))
    with Image.open(io.BytesIO(img_bytes)) as img:
        if image_format == "jpg":
            img.convert("RGB").save(out_path, format="JPEG", quality=95)
        else:
            img.save(out_path, format="PNG")


def find_parquet_files(parquet_files: Sequence[str], parquet_dir: Optional[str], split: str, subject: Optional[str]) -> List[str]:
    files: List[str] = [os.path.abspath(p) for p in parquet_files if p]
    if parquet_dir:
        root = os.path.abspath(os.path.join(parquet_dir, subject)) if subject else os.path.abspath(parquet_dir)
        if not os.path.isdir(root):
            raise FileNotFoundError("parquet root does not exist: %s" % root)
        prefix = (split or "").lower()
        for current_root, _, names in os.walk(root):
            for name in names:
                if not name.endswith(".parquet"):
                    continue
                file_split = name.split("-")[0].split(".")[0].lower()
                if prefix and file_split != prefix:
                    continue
                files.append(os.path.join(current_root, name))
    files = sorted(dict.fromkeys(files))
    if not files:
        raise FileNotFoundError("no parquet files found")
    return files


def build_output_row(
    row: Any,
    image_name: str,
    text_field: str,
    output_field: str,
    source_path: str,
    source_row_index: int,
) -> Dict[str, Any]:
    question, selected_text_field = select_text(row, text_field, AUTO_TEXT_FIELDS)
    options_text = format_options(row.get("options"))
    if options_text:
        question = question + "\n" + options_text

    answer, selected_output_field = select_text(row, output_field, AUTO_OUTPUT_FIELDS)
    sample_id = value_to_text(row.get("id")) or "%s:%d" % (os.path.basename(source_path), source_row_index)

    return {
        "image": image_name,
        "question": question,
        "text_input": question,
        "answer": answer,
        "text_output": answer,
        "source_id": sample_id,
        "source_parquet": source_path,
        "source_row_index": int(source_row_index),
        "source_text_field": selected_text_field,
        "source_output_field": selected_output_field,
    }


def main() -> None:
    args = parse_args()
    import pandas as pd

    ensure_dir(os.path.dirname(os.path.abspath(args.out_json)))
    if os.path.isdir(args.out_images_dir):
        shutil.rmtree(args.out_images_dir)
    ensure_dir(args.out_images_dir)

    parquet_files = find_parquet_files(args.parquet, args.parquet_dir, args.split, args.subject)
    rows_out: List[Dict[str, Any]] = []
    skipped = 0
    ext = args.image_format

    for path in parquet_files:
        df = pd.read_parquet(path)
        for row_index, row in df.iterrows():
            if args.max_samples is not None and len(rows_out) >= args.max_samples:
                break
            try:
                img_bytes = extract_image_bytes(row)
                image_name = "%06d.%s" % (len(rows_out), ext)
                write_image(img_bytes, os.path.join(args.out_images_dir, image_name), args.image_format)
                rows_out.append(
                    build_output_row(
                        row=row,
                        image_name=image_name,
                        text_field=args.text_field,
                        output_field=args.output_field,
                        source_path=path,
                        source_row_index=int(row_index),
                    )
                )
            except Exception as exc:
                if not args.skip_bad_rows:
                    raise RuntimeError("%s row %s failed: %s" % (path, row_index, exc)) from exc
                skipped += 1
            if args.log_every and len(rows_out) and len(rows_out) % args.log_every == 0:
                print("[INFO] converted %d rows" % len(rows_out), flush=True)
        if args.max_samples is not None and len(rows_out) >= args.max_samples:
            break

    if not rows_out:
        raise RuntimeError("No usable rows converted from parquet.")

    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(rows_out, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    manifest_path = os.path.join(os.path.dirname(os.path.abspath(args.out_json)), "forward_from_parquet_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "parquet_files": parquet_files,
                "out_json": os.path.abspath(args.out_json),
                "out_images_dir": os.path.abspath(args.out_images_dir),
                "converted_rows": len(rows_out),
                "skipped_rows": skipped,
                "split": args.split,
                "subject": args.subject,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    print("[OK] converted parquet rows: %d (skipped=%d)" % (len(rows_out), skipped))
    print("[OK] wrote json: %s" % args.out_json)
    print("[OK] wrote images: %s" % args.out_images_dir)


if __name__ == "__main__":
    main()
