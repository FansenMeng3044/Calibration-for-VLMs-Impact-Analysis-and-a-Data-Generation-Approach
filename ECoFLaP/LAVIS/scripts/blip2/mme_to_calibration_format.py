#!/usr/bin/env python3
"""
Classic MME benchmark -> ecoflap/ukmp/lavisbackup OK-VQA calibration format

Input (extracted):
  /root/autodl-tmp/datasets/MME/extracted/MME_Benchmark_release_version/MME_Benchmark/

Output:
  /root/autodl-tmp/MME_calibration/
    images/  (JPEGs)
    mmE_calibration_train.json  (list of {image, question, answer})

Parsing:
  Each line in MME *.txt is:
    question<TAB>Yes|No

Answer normalization:
  "Yes" -> "yes", "No" -> "no" (lowercase)

Image mapping:
  For each txt file:
    - If its path is .../questions_answers_YN/*.txt
        corresponding images are in .../images/<stem>.jpg/.png
    - else images are in the same directory as the txt:
        <stem>.jpg/<stem>.png

Image export:
  For robustness and downstream compatibility, we save every used source image
  to output as JPEG:
    mme_<task>_<stem>.jpg

Dedup:
  If multiple questions reference the same source image, we save it once.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


VALID_TASKS = [
    "OCR",
    "artwork",
    "celebrity",
    "code_reasoning",
    "color",
    "commonsense_reasoning",
    "count",
    "existence",
    "landmark",
    "numerical_calculation",
    "position",
    "posters",
    "scene",
    "text_translation",
]


def norm_answer(s: str) -> str:
    v = str(s).strip().lower()
    if v == "yes":
        return "yes"
    if v == "no":
        return "no"
    raise ValueError(f"Unexpected answer: {s!r}")


def parse_txt_lines(txt_path: Path) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" not in line:
                # Some MME files are 2-column separated by tab; if not, skip with warning.
                continue
            q, a = line.split("\t", 1)
            out.append((q.strip(), a.strip()))
    return out


def find_image_for_txt(txt_path: Path) -> Optional[Path]:
    stem = txt_path.stem  # without .txt
    parts = txt_path.parts

    # Pattern 1: .../questions_answers_YN/<file>.txt -> .../images/<file>.jpg
    if "questions_answers_YN" in parts:
        idx = parts.index("questions_answers_YN")
        image_dir = Path(*parts[:idx]) / "images"
    else:
        image_dir = txt_path.parent

    for ext in [".jpg", ".jpeg", ".png"]:
        cand = image_dir / f"{stem}{ext}"
        if cand.is_file():
            return cand
    return None


def save_as_jpg(src_img: Path, dst_img: Path) -> None:
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_img) as im:
        im = im.convert("RGB")
        im.save(dst_img, format="JPEG", quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mme_root", required=True, help="MME_Benchmark directory")
    parser.add_argument("--out_dir", required=True, help="Output root (will create images/ and json)")
    parser.add_argument("--tasks", nargs="*", default=VALID_TASKS, help="Tasks to include (default all 14)")
    args = parser.parse_args()

    mme_root = Path(args.mme_root)
    out_dir = Path(args.out_dir)
    img_out_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir.mkdir(parents=True, exist_ok=True)

    # source image dedup: src_path -> dst_filename
    image_cache: Dict[str, str] = {}

    records: List[Dict[str, str]] = []
    total_lines = 0

    tasks_set = set(args.tasks)
    # We scan all txt under each task directory (handles nested scene/questions_answers_YN).
    for task in args.tasks:
        task_dir = mme_root / task
        if not task_dir.exists():
            print(f"[WARN] Task dir not found: {task_dir}")
            continue

        for txt_path in sorted(task_dir.rglob("*.txt")):
            # Determine task name for output naming:
            # Use provided task (top folder name) for stable mapping.
            task_name = task
            stem = txt_path.stem

            qa_pairs = parse_txt_lines(txt_path)
            if not qa_pairs:
                continue

            for q, a in qa_pairs:
                total_lines += 1
                ans = norm_answer(a)

                src_img = find_image_for_txt(txt_path)
                if src_img is None:
                    raise RuntimeError(f"Cannot find image for txt={txt_path}")

                src_key = str(src_img.resolve())
                if src_key not in image_cache:
                    # Export once
                    dst_filename = f"mme_{task_name}_{stem}.jpg"
                    dst_path = img_out_dir / dst_filename
                    save_as_jpg(src_img, dst_path)
                    image_cache[src_key] = dst_filename

                records.append({"image": image_cache[src_key], "question": q, "answer": ans})

    json_path = out_dir / "mmE_calibration_train.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Wrote {len(records)} records -> {json_path}")
    print(f"[INFO] Unique images: {len(set(r['image'] for r in records))} (saved to {img_out_dir})")
    print(f"[INFO] Total qa lines parsed: {total_lines}")


if __name__ == "__main__":
    main()

