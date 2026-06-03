#!/usr/bin/env python3
"""Generate question-focused hints for calibration JSON files.

Run from a LAVIS root, for example:

  python scripts/blip2/generate_calibration_hints.py \
    --calib_json /data/data2/mfs/MMMU_calibration/mmmu_calibration_train.json \
    --images_dir /data/data2/mfs/MMMU_calibration/images \
    --ckpt /data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth \
    --out_hints_jsonl /data/data2/mfs/MMMU_calibration/mmmu_generated_hints.jsonl \
    --out_calib_json /data/data2/mfs/MMMU_calibration/mmmu_calibration_train_with_hint.json \
    --batch_size 4 \
    --max_samples 128
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_LAVIS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _LAVIS_ROOT not in sys.path:
    sys.path.insert(0, _LAVIS_ROOT)


PROMPT_NO_OPTIONS = """Generate a short question-focused hint for this visual question.

First, carefully interpret what the question is asking.
Then mention the kind of visual evidence that should be checked in the image.
The hint should clarify the question's intent, not solve it.

Rules:
- Do not answer the question.
- Do not mention any option letter such as A, B, C, or D.
- Do not mention the correct choice.
- Do not copy an answer option as the hint.
- Do not describe the whole image.
- Do not provide step-by-step reasoning.
- Keep it under {max_words} words.
- Output only one hint sentence.

Question:
{question}

Hint:"""


PROMPT_WITH_OPTIONS = """Generate a short question-focused hint for this visual question.

First, carefully interpret what the question is asking.
Then mention the kind of visual evidence that should be checked in the image.
The hint should clarify the question's intent, not solve it.

Rules:
- Do not answer the question.
- Do not mention any option letter such as A, B, C, or D.
- Do not mention the correct choice.
- Do not copy an answer option as the hint.
- Do not describe the whole image.
- Do not provide step-by-step reasoning.
- Keep it under {max_words} words.
- Output only one hint sentence.

Question:
{question}

Options:
{options}

Hint:"""


BAD_SUBSTRINGS = [
    "answer is",
    "correct",
    "choose",
    "option",
    "the answer",
    "best answer",
    "right answer",
    "selected",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate question-focused hints for calibration data.")
    ap.add_argument("--calib_json", required=True, help="Input calibration JSON list.")
    ap.add_argument(
        "--images_dir",
        default=None,
        help="Image directory. Defaults to sibling images/ next to --calib_json.",
    )
    ap.add_argument(
        "--ckpt",
        required=True,
        help="Original unpruned BLIP2 checkpoint, e.g. blip2_pretrained_flant5xl.pth.",
    )
    ap.add_argument("--out_hints_jsonl", required=True, help="Audit JSONL output.")
    ap.add_argument("--out_calib_json", required=True, help="Calibration JSON with hints appended.")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument(
        "--device",
        default=None,
        help="Device for generation. Defaults to cuda when available, otherwise cpu.",
    )
    ap.add_argument("--num_beams", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=24, help="Generation max_new_tokens.")
    ap.add_argument("--min_len", type=int, default=3)
    ap.add_argument("--hint_max_words", type=int, default=18)
    ap.add_argument("--question_field", default="question")
    ap.add_argument("--image_field", default="image")
    ap.add_argument("--answer_field", default="answer")
    ap.add_argument(
        "--bad_hint_policy",
        choices=["drop", "append"],
        default="drop",
        help="drop keeps the original question when a hint is filtered.",
    )
    ap.add_argument(
        "--append_template",
        default="{question}\nHint: {hint}",
        help="Template used for accepted hints. Available fields: question, hint.",
    )
    ap.add_argument(
        "--add_hint_metadata",
        action="store_true",
        help="Also store generated_hint/status/reasons in the output calibration JSON.",
    )
    ap.add_argument("--log_every", type=int, default=20)
    return ap.parse_args()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_rows(path: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise TypeError("Input calibration JSON must be a list of objects.")
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError("Row %d is not a JSON object." % i)
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def resolve_image_path(images_dir: str, image_value: Any) -> str:
    image_name = str(image_value)
    if os.path.isabs(image_name):
        return image_name
    return os.path.join(images_dir, image_name)


def build_options_text(row: Dict[str, Any]) -> str:
    choices = row.get("choices")
    if isinstance(choices, list) and choices:
        lines = []
        for i, choice in enumerate(choices):
            letter = chr(ord("A") + i)
            lines.append("%s. %s" % (letter, str(choice)))
        return "\n".join(lines)

    lines = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if letter in row and str(row.get(letter, "")).strip():
            lines.append("%s. %s" % (letter, str(row[letter]).strip()))
    return "\n".join(lines)


def build_prompt(question: str, row: Dict[str, Any], max_words: int) -> str:
    options = build_options_text(row)
    if options:
        return PROMPT_WITH_OPTIONS.format(question=question, options=options, max_words=max_words)
    return PROMPT_NO_OPTIONS.format(question=question, max_words=max_words)


def strip_wrapping_quotes(text: str) -> str:
    s = text.strip()
    changed = True
    while changed and len(s) >= 2:
        changed = False
        pairs = [('"', '"'), ("'", "'"), ("`", "`"), ("\u201c", "\u201d")]
        for left, right in pairs:
            if s.startswith(left) and s.endswith(right):
                s = s[1:-1].strip()
                changed = True
    return s


def first_sentence(text: str) -> str:
    m = re.match(r"(.+?[.!?])(?:\s|$)", text)
    if m:
        return m.group(1).strip()
    return text.strip()


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;:")


def clean_hint(raw_hint: Any, max_words: int) -> str:
    s = str(raw_hint or "").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = strip_wrapping_quotes(s)
    s = re.sub(r"^(?:hint|visual hint|question-focused hint)\s*:\s*", "", s, flags=re.IGNORECASE).strip()
    s = strip_wrapping_quotes(s)
    s = first_sentence(s)
    s = truncate_words(s, max_words)
    return s.strip()


def normalize_text(text: Any) -> str:
    s = str(text or "").casefold()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def iter_answer_values(answer: Any) -> Iterable[str]:
    if isinstance(answer, list):
        for item in answer:
            yield str(item)
    elif answer is not None:
        yield str(answer)


def contains_answer_leak(hint: str, answer: Any) -> bool:
    hint_norm = normalize_text(hint)
    if not hint_norm:
        return False

    for value in iter_answer_values(answer):
        ans = str(value).strip()
        if not ans:
            continue
        if re.fullmatch(r"[A-Za-z]", ans):
            continue
        ans_norm = normalize_text(ans)
        if not ans_norm:
            continue
        if ans_norm in {"yes", "no"}:
            if re.search(r"(^| )%s( |$)" % re.escape(ans_norm), hint_norm):
                return True
            continue
        if len(ans_norm) < 3:
            continue
        if re.search(r"(^| )%s( |$)" % re.escape(ans_norm), hint_norm):
            return True
    return False


def has_option_letter_leak(text: str) -> bool:
    if re.fullmatch(r"\s*[A-D]\s*", text):
        return True
    if re.search(r"(?<![A-Za-z0-9])\(?[A-D]\)?[\).:]", text):
        return True
    return False


def filter_hint(raw_hint: Any, clean: str, answer: Any) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    combined = ("%s %s" % (str(raw_hint or ""), clean)).strip()
    combined_lower = combined.casefold()

    if not clean:
        reasons.append("empty")

    for bad in BAD_SUBSTRINGS:
        if bad in combined_lower:
            reasons.append("banned:%s" % bad)

    if has_option_letter_leak(combined):
        reasons.append("option_letter")

    if contains_answer_leak(combined, answer):
        reasons.append("answer_leak")

    status = "ok" if not reasons else "filtered"
    return status, reasons


def make_output_row(
    row: Dict[str, Any],
    question_field: str,
    hint: str,
    status: str,
    reasons: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    out = copy.deepcopy(row)
    original_question = str(row.get(question_field, ""))
    if status == "ok" or args.bad_hint_policy == "append":
        out[question_field] = args.append_template.format(question=original_question, hint=hint)
    else:
        out[question_field] = original_question

    if args.add_hint_metadata:
        out["generated_hint"] = hint
        out["generated_hint_status"] = status
        out["generated_hint_reasons"] = list(reasons)

    return out


def write_json(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()

    try:
        import torch
        from PIL import Image
        from lavis.models import load_model
        from lavis.processors import load_processor
    except ImportError as e:
        raise SystemExit(
            "Missing runtime dependency: %s. Run this script in the LAVIS environment with torch/PIL installed."
            % e
        ) from e

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    calib_json = os.path.abspath(args.calib_json)
    images_dir = (
        os.path.abspath(args.images_dir)
        if args.images_dir
        else os.path.join(os.path.dirname(calib_json), "images")
    )

    rows = load_rows(calib_json, args.max_samples)
    if not rows:
        raise RuntimeError("No rows found in %s" % calib_json)

    print("Loaded %d calibration rows from %s" % (len(rows), calib_json))
    print("images_dir:", images_dir)
    print("device:", args.device)

    model = load_model(
        "blip2_t5",
        "pretrain_flant5xl",
        is_eval=True,
        device=args.device,
        checkpoint=args.ckpt,
    )
    model.eval()
    vis_processor = load_processor("blip_image_eval").build(image_size=224)

    ensure_parent_dir(args.out_hints_jsonl)
    output_rows: List[Dict[str, Any]] = []
    ok_count = 0
    filtered_count = 0

    with open(args.out_hints_jsonl, "w", encoding="utf-8") as audit_f:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            images = []
            prompts = []

            for row_idx, row in enumerate(batch, start=start):
                if args.question_field not in row:
                    raise KeyError("Row %d missing question field %r" % (row_idx, args.question_field))
                if args.image_field not in row:
                    raise KeyError("Row %d missing image field %r" % (row_idx, args.image_field))

                question = str(row.get(args.question_field, ""))
                img_path = resolve_image_path(images_dir, row[args.image_field])
                if not os.path.isfile(img_path):
                    raise FileNotFoundError("Image not found for row %d: %s" % (row_idx, img_path))

                img = Image.open(img_path).convert("RGB")
                images.append(vis_processor(img))
                prompts.append(build_prompt(question, row, args.hint_max_words))

            image_tensor = torch.stack(images).to(args.device)
            with torch.no_grad():
                hints_raw = model.predict_answers(
                    {"image": image_tensor, "text_input": prompts},
                    num_beams=args.num_beams,
                    max_len=args.max_len,
                    min_len=args.min_len,
                    inference_method="generate",
                )

            for row, prompt, raw_hint in zip(batch, prompts, hints_raw):
                clean = clean_hint(raw_hint, args.hint_max_words)
                status, reasons = filter_hint(raw_hint, clean, row.get(args.answer_field))
                if status == "ok":
                    ok_count += 1
                else:
                    filtered_count += 1

                output_rows.append(
                    make_output_row(
                        row=row,
                        question_field=args.question_field,
                        hint=clean,
                        status=status,
                        reasons=reasons,
                        args=args,
                    )
                )

                audit = {
                    "image": row.get(args.image_field),
                    "question": row.get(args.question_field),
                    "answer": row.get(args.answer_field),
                    "hint_prompt": prompt,
                    "hint_raw": raw_hint,
                    "hint_clean": clean,
                    "status": status,
                    "reasons": list(reasons),
                }
                audit_f.write(json.dumps(audit, ensure_ascii=False) + "\n")
            audit_f.flush()

            done = min(start + len(batch), len(rows))
            if args.log_every > 0 and (done == len(rows) or done % args.log_every == 0):
                print("Processed %d/%d rows (ok=%d filtered=%d)" % (done, len(rows), ok_count, filtered_count))

    write_json(args.out_calib_json, output_rows)
    print("[OK] wrote audit hints:", os.path.abspath(args.out_hints_jsonl))
    print("[OK] wrote calibration JSON:", os.path.abspath(args.out_calib_json))
    print("[OK] accepted hints: %d | filtered hints: %d" % (ok_count, filtered_count))


if __name__ == "__main__":
    main()
