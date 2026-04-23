#!/usr/bin/env python3
"""
MME (yes/no) evaluation on parquet.

Input parquet format (from `mme_calibration_to_mmmu_eval_parquet.py`):
  - id
  - question
  - answer: 'A'/'B' (A -> yes, B -> no)
  - image_1..image_7: only image_1 contains {"bytes": ...}
  - options (optional, ignored here)

This script forces the model prompt to output ONLY `yes` or `no`,
and judges by extracting a yes/no token from the generated text.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
from typing import Dict, Iterator, Tuple

import pandas as pd
import torch
from PIL import Image

# Ensure we can import `lavis` when executed from the repo root.
_LAVIS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _LAVIS_ROOT not in sys.path:
    sys.path.insert(0, _LAVIS_ROOT)

from lavis.models import load_model
from lavis.processors import load_processor


def has_bytes(val) -> bool:
    return isinstance(val, dict) and val.get("bytes") is not None


def count_images(row) -> int:
    n = 0
    for i in range(1, 8):
        if has_bytes(row.get(f"image_{i}")):
            n += 1
    return n


def get_first_image_bytes(row):
    for i in range(1, 8):
        val = row.get(f"image_{i}")
        if has_bytes(val):
            return val["bytes"]
    return None


def gt_to_yes_no(gt_raw) -> str:
    """
    Convert parquet `answer` to 'yes'/'no'.
    Supported inputs:
      - 'A'/'B' (from mme_calibration_to_mmmu_eval_parquet)
      - 'yes'/'no' (in case you generate other parquet formats)
    """
    if gt_raw is None:
        return ""
    s = str(gt_raw).strip()
    if not s:
        return ""
    su = s.upper()
    if su == "A":
        return "yes"
    if su == "B":
        return "no"
    sl = s.lower()
    if sl == "yes":
        return "yes"
    if sl == "no":
        return "no"
    return ""


def extract_yes_no(pred_text) -> str:
    """
    Extract 'yes'/'no' from model output (case-insensitive).
    We use word-boundary regex to avoid matching inside other words.
    """
    if pred_text is None:
        return ""
    s = str(pred_text).strip().lower()
    if not s:
        return ""
    if re.search(r"\byes\b", s):
        return "yes"
    if re.search(r"\bno\b", s):
        return "no"
    return ""


def load_mme_parquet(mme_root: str, split: str) -> Iterator[Tuple[str, bytes, str, str]]:
    """
    Yield tuples: (sample_id, image_bytes, question, gt_yes_no)
    """
    subjects = sorted(
        d for d in os.listdir(mme_root)
        if os.path.isdir(os.path.join(mme_root, d)) and not d.startswith(".")
    )

    for subject in subjects:
        subj_dir = os.path.join(mme_root, subject)
        for fname in os.listdir(subj_dir):
            if not fname.endswith(".parquet"):
                continue
            file_split = fname.split("-")[0].lower()
            if file_split != split.lower():
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
                question = str(row.get("question", "")).strip()
                gt_raw = row.get("answer", "")
                gt = gt_to_yes_no(gt_raw)
                if not gt:
                    continue
                yield sample_id, img_bytes, question, gt


def main() -> None:
    parser = argparse.ArgumentParser(description="MME yes/no eval on parquet (token-level yes/no parsing)")
    parser.add_argument("--mme_root", default="/root/autodl-tmp/MME_eval", help="MME_eval parquet root")
    parser.add_argument("--split", default="test", choices=["dev", "validation", "test"], help="Split to evaluate")
    parser.add_argument("--ckpt", default=None, help="Optional: pruned model checkpoint")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap number of samples (debug)")
    parser.add_argument("--max_new_tokens", type=int, default=5, help="Max new tokens for yes/no generation")
    parser.add_argument("--num_beams", type=int, default=3, help="Beam search width")
    args = parser.parse_args()

    samples = list(load_mme_parquet(args.mme_root, args.split))
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        print(f"No samples found under {args.mme_root} split={args.split}")
        return

    print(f"Loaded {len(samples)} MME yes/no samples (split={args.split})")

    if args.ckpt is None:
        model = load_model("blip2_t5", "pretrain_flant5xl", is_eval=True, device=args.device)
    else:
        model = load_model(
            "blip2_t5",
            "pretrain_flant5xl",
            is_eval=True,
            device=args.device,
            checkpoint=args.ckpt,
        )

    vis_processor = load_processor("blip_image_eval").build(image_size=224)
    text_processor = load_processor("blip_question").build(max_words=99999, remove_punctuation=False)

    correct = 0
    total = 0
    pred_dist: Dict[str, int] = {"yes": 0, "no": 0, "": 0}

    # Batch inference
    for i in range(0, len(samples), args.batch_size):
        batch = samples[i : i + args.batch_size]

        images = []
        questions = []
        gt_list = []
        for _, img_bytes, question, gt in batch:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            images.append(vis_processor(img))
            # Force exact yes/no. Even if casing varies, we parse with case-insensitive regex.
            prompt = f"Question: {question}\nAnswer with exactly one word: yes or no."
            questions.append(text_processor(prompt))
            gt_list.append(gt)

        image_tensor = torch.stack(images).to(args.device)

        with torch.no_grad():
            preds = model.predict_answers(
                {"image": image_tensor, "text_input": questions},
                num_beams=args.num_beams,
                max_len=args.max_new_tokens,  # generation wrapper uses this as max_new_tokens
                min_len=1,
                inference_method="generate",
            )

        for gt, pred in zip(gt_list, preds):
            pred_ans = extract_yes_no(pred)
            pred_dist[pred_ans] = pred_dist.get(pred_ans, 0) + 1
            total += 1
            if pred_ans == gt:
                correct += 1

    acc = 100.0 * correct / total if total else 0.0
    print(f"MME yes/no accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"Pred distribution (yes/no/unknown): {pred_dist}")

    _mp = os.environ.get("LAVIS_METRICS_JSONL")
    if _mp:
        import json

        _calib = os.environ.get("LAVIS_EVAL_CALIB_TAG", "")
        rec = {
            "calib_tag": _calib,
            "benchmark": "MME_yesno",
            "split": args.split,
            "metric": "overall_accuracy_percent",
            "value": round(acc, 4),
            "n": int(total),
        }
        with open(_mp, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

