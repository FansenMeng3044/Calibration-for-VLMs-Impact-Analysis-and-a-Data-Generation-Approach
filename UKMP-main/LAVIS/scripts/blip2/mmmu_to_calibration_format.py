#!/usr/bin/env python3
"""
MMMU 单图题 → ECoFLaP calibration 格式转换。

从 MMMU_single_image 读取 parquet，导出图片到目录，生成 okvqa 同构的 JSON，
供 prefix_okvqa_calibration / OKVQACalibrationDataset 使用（不改 LAVIS 代码）。

用法:
  python scripts/blip2/mmmu_to_calibration_format.py [--splits dev validation] [--out_dir DIR]

默认: --out_dir /root/autodl-tmp/MMMU_calibration
      --splits dev validation
      生成 MMMU_calibration/images/ 与 MMMU_calibration/mmmu_calibration_train.json

  --by_discipline: 额外按 6 大领域各生成一个 JSON（如 mmmu_calibration_train_Art_Design.json），
      图片仍共用一个 images/ 目录。

  --max_per_discipline N: 与 --by_discipline 一起用时，每个领域最多取 N 条（从 dev+validation 中先到先得），
      用于「每类别一个 calibration、每类 90 条」：--by_discipline --max_per_discipline 90
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re

import pandas as pd
from PIL import Image


def has_bytes(val):
    return isinstance(val, dict) and val.get("bytes") is not None


def count_images(row):
    n = 0
    for col in [f"image_{i}" for i in range(1, 8)]:
        if has_bytes(row.get(col)):
            n += 1
    return n


def get_first_image_bytes(row):
    for col in [f"image_{i}" for i in range(1, 8)]:
        val = row.get(col)
        if has_bytes(val):
            return val["bytes"]
    return None


def parse_options(options_val):
    """Parse MMMU options (string or list) to list of strings."""
    if options_val is None:
        return []
    if isinstance(options_val, list):
        return [str(x) for x in options_val]
    s = str(options_val).strip()
    if not s:
        return []
    try:
        out = ast.literal_eval(s)
        return [str(x) for x in out] if isinstance(out, list) else []
    except (ValueError, SyntaxError):
        return []


def answer_letter_to_text(answer_letter, options_list):
    """Map A/B/C/D + options list to single answer text."""
    letter = str(answer_letter).strip().upper()
    if not letter or not options_list:
        return str(answer_letter).strip()
    idx = ord(letter[0]) - ord("A")
    if 0 <= idx < len(options_list):
        return options_list[idx]
    return str(answer_letter).strip()


def safe_filename(subject: str, sample_id: str) -> str:
    """Unique filename: subject_id, no path chars."""
    raw = f"{subject}_{sample_id}"
    raw = re.sub(r"[/\\]", "_", raw)
    raw = re.sub(r"\s+", "_", raw)
    return raw + ".png"


# 30 学科 -> 6 大领域（与前面分析一致）
DISCIPLINES = {
    "Art & Design": ["Art", "Art_Theory", "Design"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Agriculture", "Biology", "Chemistry", "Geography", "Math", "Materials", "Physics"],
    "Health & Medicine": ["Basic_Medical_Science", "Clinical_Medicine", "Diagnostics_and_Laboratory_Medicine", "Pharmacy", "Public_Health"],
    "Humanities & Social Science": ["History", "Literature", "Music", "Psychology", "Sociology"],
    "Tech & Engineering": ["Architecture_and_Engineering", "Computer_Science", "Electronics", "Energy_and_Power", "Mechanical_Engineering"],
}


def subject_to_discipline(subject: str) -> str:
    """Return discipline name for a subject (e.g. Physics -> Science)."""
    for disc, subs in DISCIPLINES.items():
        if subject in subs:
            return disc
    return "Other"


def discipline_to_filename_key(discipline: str) -> str:
    """e.g. 'Art & Design' -> 'Art_Design' for JSON filename."""
    return discipline.replace(" & ", "_").replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(description="MMMU single-image → calibration JSON + images")
    parser.add_argument("--mmmu_root", default="/root/autodl-tmp/MMMU_single_image", help="MMMU_single_image root")
    parser.add_argument("--out_dir", default="/root/autodl-tmp/MMMU_calibration", help="Output root: images/ and JSON here")
    parser.add_argument("--splits", nargs="+", default=["dev", "validation"], help="Splits to include (dev, validation, test)")
    parser.add_argument("--json_name", default="mmmu_calibration_train.json", help="JSON filename under out_dir")
    parser.add_argument("--by_discipline", action="store_true", help="Also write one JSON per 6 disciplines (mmmu_calibration_train_Art_Design.json etc.)")
    parser.add_argument("--max_per_discipline", type=int, default=None, help="When using --by_discipline, cap each discipline to this many samples (e.g. 90)")
    args = parser.parse_args()

    image_dir = os.path.join(args.out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, args.json_name)

    annotations = []
    by_disc: dict[str, list[dict]] = {disc: [] for disc in DISCIPLINES}
    by_disc["Other"] = []
    seen = set()
    subjects = sorted(
        d
        for d in os.listdir(args.mmmu_root)
        if os.path.isdir(os.path.join(args.mmmu_root, d)) and not d.startswith(".")
    )

    for subject in subjects:
        subj_dir = os.path.join(args.mmmu_root, subject)
        for fname in os.listdir(subj_dir):
            if not fname.endswith(".parquet"):
                continue
            split = fname.split("-")[0].lower()
            if split not in args.splits:
                continue
            path = os.path.join(subj_dir, fname)
            df = pd.read_parquet(path)
            for _, row in df.iterrows():
                if count_images(row) != 1:
                    continue
                img_bytes = get_first_image_bytes(row)
                if img_bytes is None:
                    continue
                sample_id = str(row.get("id", "")).strip()
                key = (subject, sample_id)
                if key in seen:
                    continue
                seen.add(key)
                filename = safe_filename(subject, sample_id)
                img_path = os.path.join(image_dir, filename)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img.save(img_path)

                options_list = parse_options(row.get("options"))
                answer_text = answer_letter_to_text(row.get("answer"), options_list)
                question = str(row.get("question", "")).strip()
                question = re.sub(r"\s*<image\s*\d*>\s*", " ", question, flags=re.IGNORECASE).strip()

                ann = {
                    "image": filename,
                    "question": question,
                    "answer": answer_text,
                }
                annotations.append(ann)
                disc = subject_to_discipline(subject)
                if disc not in by_disc:
                    by_disc[disc] = []
                by_disc[disc].append(ann)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    print("Wrote %d samples to %s" % (len(annotations), json_path))
    print("Images in %s" % image_dir)

    if args.by_discipline:
        base_name = os.path.splitext(args.json_name)[0]
        for disc, ann_list in by_disc.items():
            if disc == "Other" or not ann_list:
                continue
            if args.max_per_discipline is not None:
                ann_list = ann_list[: args.max_per_discipline]
            key = discipline_to_filename_key(disc)
            disc_json = os.path.join(args.out_dir, f"{base_name}_{key}.json")
            with open(disc_json, "w", encoding="utf-8") as f:
                json.dump(ann_list, f, ensure_ascii=False, indent=2)
            print("  by_discipline: %s -> %s (%d samples)" % (disc, disc_json, len(ann_list)))

    print("Calibration YAML: set annotations.train.storage to %s and images.storage to %s" % (json_path, image_dir))


if __name__ == "__main__":
    main()
