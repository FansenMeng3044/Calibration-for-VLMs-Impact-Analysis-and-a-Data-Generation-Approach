#!/usr/bin/env python3
"""
MathVista multi-choice eval from JSON produced by convert_mathvista_mc_to_eval.py.

GT ``answer`` is already a single letter from choice matching at conversion time.
Pred parsing matches ``judge_multi_choice_tamp_style`` in mmmu_eval_by_discipline.py:
  - If ``:`` in processed text: split by ``:``, take the first segment of length 1 that is
    in ``a``–``g`` (this split uses option letters only, not the first char of the string).
  - Else: first match of ``[a-g]`` in the processed text.

Run from LAVIS repo root (ECoFLaP/LAVIS or LAVIS_backup; same as mmmu_eval_by_discipline.py).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys

import torch
from PIL import Image

_LAVIS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _LAVIS_ROOT not in sys.path:
    sys.path.insert(0, _LAVIS_ROOT)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from lavis.models import load_model
from lavis.processors import load_processor
from load_blip2_t5_split_ckpts import load_blip2_t5_for_eval

# Reuse TAMP-style normalization from MMMU eval (scripts/ is not a Python package)
_mmu_path = os.path.join(_LAVIS_ROOT, "scripts", "blip2", "mmmu_eval_by_discipline.py")
_spec = importlib.util.spec_from_file_location("mmmu_eval_by_discipline", _mmu_path)
_mmu = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mmu)
process_answer = _mmu.process_answer

# Match MMMU TAMP multi-choice letter set; current MathVista testmini MC uses up to G.
_MC_LETTERS = ["a", "b", "c", "d", "e", "f", "g"]
_MC_CLASS = "[a-g]"


def normalize_gt_letter(gt_letter: str) -> str:
    s = str(gt_letter or "").strip()
    if not s:
        return ""
    first = s[0].upper()
    if first in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return first.lower()
    return process_answer(gt_letter)


def judge_mathvista_mc(gt_letter: str, pred_raw: str) -> int:
    """Same structure as ``judge_multi_choice_tamp_style`` (MMMU), with ``a``–``g``."""
    gt_ans = normalize_gt_letter(gt_letter)
    if not gt_ans:
        return -1

    pred_ans = process_answer(pred_raw)
    if not pred_ans:
        return 0

    if ":" in pred_ans:
        parts = [a.strip() for a in pred_ans.split(":")]
        for a in parts:
            if len(a) == 1 and a in _MC_LETTERS:
                pred_ans = a
                break
    else:
        m = re.search(_MC_CLASS, pred_ans)
        if m:
            pred_ans = m.group(0)

    return 1 if pred_ans == gt_ans else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval_json",
        default="/root/autodl-tmp/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json",
        help="JSON list: pid, image, question, answer (letter), ...",
    )
    ap.add_argument(
        "--images_dir",
        default=None,
        help="Override image directory (default: sibling images/ next to eval_json)",
    )
    ap.add_argument(
        "--ckpt",
        default=None,
        help="Full pruned .pth. Omit it to evaluate the dense LAVIS BLIP2-T5 model.",
    )
    ap.add_argument(
        "--vit_ckpt",
        default=None,
        help="ViT-only prune .pth; use with --t5_ckpt for combined ViT+T5 split eval",
    )
    ap.add_argument(
        "--t5_ckpt",
        default=None,
        help="T5-only prune .pth; use with --vit_ckpt",
    )
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    if args.ckpt and (args.vit_ckpt or args.t5_ckpt):
        ap.error("Use either --ckpt or (--vit_ckpt and --t5_ckpt), not both")
    if (args.vit_ckpt or args.t5_ckpt) and not (args.vit_ckpt and args.t5_ckpt):
        ap.error("Both --vit_ckpt and --t5_ckpt are required together")

    json_path = os.path.abspath(args.eval_json)
    base = os.path.dirname(json_path)
    images_dir = os.path.abspath(args.images_dir) if args.images_dir else os.path.join(base, "images")

    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    n = len(rows)
    if n == 0:
        print("No samples in", json_path)
        return

    print("Loaded %d MathVista MC samples from %s" % (n, json_path))
    print("images_dir:", images_dir)

    if args.vit_ckpt and args.t5_ckpt:
        model = load_blip2_t5_for_eval(args.device, vit_ckpt=args.vit_ckpt, t5_ckpt=args.t5_ckpt)
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
    max_len, min_len, num_beams = 10, 1, 5

    correct = 0
    for i in range(0, n, args.batch_size):
        batch = rows[i : i + args.batch_size]
        images = []
        questions = []
        for r in batch:
            img_path = os.path.join(images_dir, r["image"])
            img = Image.open(img_path).convert("RGB")
            images.append(vis_processor(img))
            questions.append(text_processor(str(r["question"])))
        image_tensor = torch.stack(images).to(args.device)
        with torch.no_grad():
            preds = model.predict_answers(
                {"image": image_tensor, "text_input": questions},
                num_beams=num_beams,
                max_len=max_len,
                min_len=min_len,
                inference_method="generate",
            )
        for r, pred in zip(batch, preds):
            gt = r.get("answer", "")
            sc = judge_mathvista_mc(gt, pred)
            if sc >= 0:
                correct += sc

    acc = 100.0 * correct / n if n else 0.0
    print("\n===== MathVista MC (n=%d) =====" % n)
    print("Overall accuracy (letter match): %.2f%%" % acc)

    _mp = os.environ.get("LAVIS_METRICS_JSONL")
    if _mp:
        import json as json_mod

        _bench = os.environ.get("LAVIS_METRICS_BENCHMARK", "MathVista_MC")
        _calib = os.environ.get("LAVIS_EVAL_CALIB_TAG", "")
        rec = {
            "calib_tag": _calib,
            "benchmark": _bench,
            "split": "eval_json",
            "metric": "overall_accuracy_percent",
            "value": round(acc, 4),
            "n": int(n),
            "eval_json": json_path,
        }
        with open(_mp, "a", encoding="utf-8") as f:
            f.write(json_mod.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
