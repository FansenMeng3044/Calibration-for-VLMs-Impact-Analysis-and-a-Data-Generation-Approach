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


PROMPT_NO_OPTIONS = """Convert the visual question into a calibration hint.

Write new text that tells the reader what to inspect before answering.
Use an imperative checklist style. Start with Check, Compare, Inspect, Use, Read, or Locate.
Choose the evidence types implied by the question, such as labels, values, units, table positions, visual regions, object relationships, chart markings, or blank cells.

Avoid these failure modes:
- Do not answer or guess the answer.
- Do not restate the question or turn it into another question.
- Do not copy a phrase from the question longer than a few words.
- Do not include exact numbers, formulas, dollar amounts, percentages, or final values.
- Do not name a final class, disease, artist, object identity, or conclusion.
- Do not write a general caption such as "the image shows".
- Do not output generic wording such as "evidence categories".

Output only the hint text.

Question: {question}
Hint:"""


PROMPT_WITH_OPTIONS = """Convert the visual multiple-choice question into a calibration hint.

Use the question and options only to infer the task and answer type.
Write new text that tells the reader what to inspect before answering.
Use an imperative checklist style. Start with Check, Compare, Inspect, Use, Read, or Locate.
Choose the evidence types implied by the question, such as labels, values, units, table positions, visual regions, object relationships, chart markings, or blank cells.

Avoid these failure modes:
- Do not answer or guess the answer.
- Do not mention option letters.
- Do not copy an option.
- Do not restate the question or turn it into another question.
- Do not copy a phrase from the question longer than a few words.
- Do not include exact numbers, formulas, dollar amounts, percentages, or final values.
- Do not name a final class, disease, artist, object identity, or conclusion.
- Do not write a general caption such as "the image shows".
- Do not output generic wording such as "evidence categories".

Output only the hint text.

Question: {question}
Options: {options}
Hint:"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate question-focused hints for calibration data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    ap.add_argument(
        "--hint_generation_mode",
        choices=["t5_text_only", "blip2"],
        default="t5_text_only",
        help="Use Flan-T5-XL text-only generation or the original BLIP2 image+prompt generation path.",
    )
    ap.add_argument("--num_beams", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=40, help="Generation max_new_tokens.")
    ap.add_argument("--min_len", type=int, default=8)
    ap.add_argument(
        "--hint_min_words",
        type=int,
        default=8,
        help="Hard minimum accepted hint length. Shorter hints are retried.",
    )
    ap.add_argument(
        "--hint_target_min_words",
        type=int,
        default=12,
        help="Soft quality target. Shorter accepted hints are marked weak in the audit file.",
    )
    ap.add_argument(
        "--hint_max_words",
        type=int,
        default=None,
        help="Optional post-cleaning word cap. Defaults to no word cap.",
    )
    ap.add_argument(
        "--max_hint_attempts",
        type=int,
        default=3,
        help="Regenerate a bad hint up to this many attempts, then keep the original question unchanged.",
    )
    ap.add_argument("--question_field", default="question")
    ap.add_argument("--image_field", default="image")
    ap.add_argument("--answer_field", default="answer")
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


def build_prompt(question: str, row: Dict[str, Any]) -> str:
    options = build_options_text(row)
    if options:
        return PROMPT_WITH_OPTIONS.format(
            question=question,
            options=options,
        )
    return PROMPT_NO_OPTIONS.format(
        question=question,
    )


def build_retry_prompt(
    base_prompt: str,
    rejected_hint: Any,
    reasons: Sequence[str],
) -> str:
    reason_text = ", ".join(reasons) if reasons else "invalid hint"
    return (
        base_prompt
        + "\n\nThe previous hint was rejected and must not be repeated.\n"
        + "Rejected hint: %s\n" % str(rejected_hint or "").strip()
        + "Rejected because: %s\n" % reason_text
        + "Write a different imperative checklist-style hint.\n"
        + "Start with Check, Compare, Inspect, Use, Read, or Locate.\n"
        + "Do not answer, copy an option, copy question phrases, include exact numbers, or reuse the rejected hint.\n"
        + "Hint:"
    )


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


def truncate_words(text: str, max_words: Optional[int]) -> str:
    if max_words is None:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;:")


def clean_hint(raw_hint: Any, max_words: Optional[int]) -> str:
    s = str(raw_hint or "").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = strip_wrapping_quotes(s)
    s = re.sub(r"^(?:hint|visual hint|question-focused hint)\s*:\s*", "", s, flags=re.IGNORECASE).strip()
    s = strip_wrapping_quotes(s)
    s = truncate_words(s, max_words)
    return s.strip()


def normalize_text(text: Any) -> str:
    s = str(text or "").casefold()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def word_tokens(text: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").casefold())


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


def filter_hint(
    raw_hint: Any,
    clean: str,
    answer: Any,
    min_words: int,
    target_min_words: int,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    hard_reasons: List[str] = []
    combined = ("%s %s" % (str(raw_hint or ""), clean)).strip()

    clean_word_count = len(word_tokens(clean))
    if not clean:
        reasons.append("empty")
        hard_reasons.append("empty")
    elif clean_word_count < min_words:
        reasons.append("too_short")
        hard_reasons.append("too_short")
    elif clean_word_count < target_min_words:
        reasons.append("weak:shorter_than_target")

    if has_option_letter_leak(combined):
        reasons.append("option_letter")
        hard_reasons.append("option_letter")

    if contains_answer_leak(combined, answer):
        reasons.append("answer_leak")
        hard_reasons.append("answer_leak")

    status = "ok" if not hard_reasons else "filtered"
    return status, reasons


def make_output_row(
    row: Dict[str, Any],
    question_field: str,
    hint: str,
    reasons: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    out = copy.deepcopy(row)
    original_question = str(row.get(question_field, ""))
    out[question_field] = args.append_template.format(question=original_question, hint=hint)

    if args.add_hint_metadata:
        out["generated_hint"] = hint
        out["generated_hint_status"] = "ok"
        out["generated_hint_reasons"] = list(reasons)

    return out


def make_original_output_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(row)


def write_json(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def generate_t5_text_only(
    model: Any,
    prompts: Sequence[str],
    args: argparse.Namespace,
    torch_module: Any,
) -> List[str]:
    input_tokens = model.t5_tokenizer(
        list(prompts),
        padding="longest",
        return_tensors="pt",
    ).to(args.device)

    with model.maybe_autocast(dtype=torch_module.bfloat16):
        outputs = model.t5_model.generate(
            input_ids=input_tokens.input_ids,
            attention_mask=input_tokens.attention_mask,
            num_beams=args.num_beams,
            max_new_tokens=args.max_len,
            min_length=args.min_len,
        )
        return model.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)


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
    if args.max_hint_attempts < 1:
        raise ValueError("--max_hint_attempts must be >= 1")
    if args.hint_min_words < 1:
        raise ValueError("--hint_min_words must be >= 1")
    if args.hint_target_min_words < args.hint_min_words:
        raise ValueError("--hint_target_min_words must be >= --hint_min_words")
    if args.hint_max_words is not None:
        if args.hint_max_words < 1:
            raise ValueError("--hint_max_words must be >= 1")
        if args.hint_target_min_words > args.hint_max_words:
            raise ValueError("--hint_target_min_words must be <= --hint_max_words")
        if args.hint_min_words > args.hint_max_words:
            raise ValueError("--hint_min_words must be <= --hint_max_words")

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
    print("hint_generation_mode:", args.hint_generation_mode)
    model = load_model(
        "blip2_t5",
        "pretrain_flant5xl",
        is_eval=True,
        device=args.device,
        checkpoint=args.ckpt,
    )
    model.eval()
    vis_processor = None
    if args.hint_generation_mode == "blip2":
        vis_processor = load_processor("blip_image_eval").build(image_size=224)

    ensure_parent_dir(args.out_hints_jsonl)
    output_rows: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    ok_count = 0
    retry_count = 0
    keep_original_count = 0

    with open(args.out_hints_jsonl, "w", encoding="utf-8") as audit_f:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            image_tensors = []
            base_prompts = []

            for row_idx, row in enumerate(batch, start=start):
                if args.question_field not in row:
                    raise KeyError("Row %d missing question field %r" % (row_idx, args.question_field))
                if args.image_field not in row:
                    raise KeyError("Row %d missing image field %r" % (row_idx, args.image_field))

                question = str(row.get(args.question_field, ""))
                if args.hint_generation_mode == "blip2":
                    img_path = resolve_image_path(images_dir, row[args.image_field])
                    if not os.path.isfile(img_path):
                        raise FileNotFoundError("Image not found for row %d: %s" % (row_idx, img_path))

                    img = Image.open(img_path).convert("RGB")
                    image_tensors.append(vis_processor(img))
                base_prompts.append(build_prompt(question, row))

            pending = list(range(len(batch)))
            last_raw: Dict[int, Any] = {}
            last_reasons: Dict[int, List[str]] = {}

            for attempt in range(1, args.max_hint_attempts + 1):
                if not pending:
                    break

                prompts = []
                images = []
                for local_idx in pending:
                    if attempt == 1:
                        prompt = base_prompts[local_idx]
                    else:
                        prompt = build_retry_prompt(
                            base_prompts[local_idx],
                            last_raw.get(local_idx, ""),
                            last_reasons.get(local_idx, []),
                        )
                    prompts.append(prompt)
                    if args.hint_generation_mode == "blip2":
                        images.append(image_tensors[local_idx])

                with torch.no_grad():
                    if args.hint_generation_mode == "t5_text_only":
                        hints_raw = generate_t5_text_only(model, prompts, args, torch)
                    else:
                        image_tensor = torch.stack(images).to(args.device)
                        hints_raw = model.generate(
                            {"image": image_tensor, "prompt": prompts},
                            num_beams=args.num_beams,
                            max_length=args.max_len,
                            min_length=args.min_len,
                        )

                still_pending = []
                for local_idx, prompt, raw_hint in zip(pending, prompts, hints_raw):
                    row = batch[local_idx]
                    clean = clean_hint(raw_hint, args.hint_max_words)
                    status, reasons = filter_hint(
                        raw_hint,
                        clean,
                        row.get(args.answer_field),
                        args.hint_min_words,
                        args.hint_target_min_words,
                    )
                    audit_status = "ok" if status == "ok" else "retry"

                    audit = {
                        "image": row.get(args.image_field),
                        "question": row.get(args.question_field),
                        "answer": row.get(args.answer_field),
                        "hint_prompt": prompt,
                        "hint_raw": raw_hint,
                        "hint_clean": clean,
                        "status": audit_status,
                        "attempt": attempt,
                        "reasons": list(reasons),
                    }

                    if status == "ok":
                        ok_count += 1
                        output_rows[start + local_idx] = make_output_row(
                            row=row,
                            question_field=args.question_field,
                            hint=clean,
                            reasons=reasons,
                            args=args,
                        )
                    else:
                        retry_count += 1
                        last_raw[local_idx] = raw_hint
                        last_reasons[local_idx] = list(reasons)
                        if attempt >= args.max_hint_attempts:
                            keep_original_count += 1
                            audit["status"] = "failed_keep_original"
                            output_rows[start + local_idx] = make_original_output_row(row)
                        else:
                            still_pending.append(local_idx)

                    audit_f.write(json.dumps(audit, ensure_ascii=False) + "\n")

                pending = still_pending
            audit_f.flush()

            done = min(start + len(batch), len(rows))
            if args.log_every > 0 and (done == len(rows) or done % args.log_every == 0):
                print(
                    "Processed %d/%d rows (ok=%d retries=%d keep_original=%d)"
                    % (done, len(rows), ok_count, retry_count, keep_original_count)
                )

    final_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(output_rows):
        if row is None:
            raise RuntimeError("Internal error: row %d has no accepted hint." % idx)
        final_rows.append(row)

    write_json(args.out_calib_json, final_rows)
    print("[OK] wrote audit hints:", os.path.abspath(args.out_hints_jsonl))
    print("[OK] wrote calibration JSON:", os.path.abspath(args.out_calib_json))
    print(
        "[OK] accepted hints: %d | retry attempts: %d | kept original: %d"
        % (ok_count, retry_count, keep_original_count)
    )


if __name__ == "__main__":
    main()
