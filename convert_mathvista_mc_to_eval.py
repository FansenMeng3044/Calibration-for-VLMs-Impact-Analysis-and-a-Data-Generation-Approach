#!/usr/bin/env python3
"""Export MathVista multi-choice split to a local eval pack (images + JSON).

Ground truth:
  - ``answer`` is the single correct option letter (A–Z), aligned with ``choices`` order
    (same mapping as ``convert_mathvista_to_ecoflap_okvqa_calibration.py``).
  - ``answer_raw`` is the dataset's original ``answer`` string (the option text).

Intended eval rule: normalize model output to one letter and check equality with ``answer``.

Input: HuggingFace ``datasets.load_from_disk`` folder containing only multi_choice rows
  (e.g. /root/autodl-tmp/MathVista_testmini_multi_choice).

This script does not modify LAVIS/ECoFLaP code.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_from_disk

DEFAULT_SUFFIX = (
    "\n\nOnly output the single correct option letter (A-Z). "
    "Do not output any other words, explanation, or punctuation."
)


def norm_text(x: Any) -> str:
    s = str(x).replace("\xa0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


LETTER_BY_IDX = [chr(ord("A") + i) for i in range(26)]


def save_image(decoded_image: Any, out_path: Path) -> None:
    img = decoded_image
    try:
        img = img.convert("RGB")
    except Exception:
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="JPEG", quality=95)


def multi_choice_to_letter(
    choices: List[Any],
    answer: Any,
    pid: str,
    fail_on_mismatch: bool,
    debug: bool,
) -> str:
    if not isinstance(choices, list) or len(choices) == 0:
        raise ValueError(f"multi_choice has empty/non-list choices: pid={pid}")

    answer_norm = norm_text(answer)
    choice_norms = [norm_text(c) for c in choices]

    if answer_norm in choice_norms:
        i = choice_norms.index(answer_norm)
        if i >= len(LETTER_BY_IDX):
            raise ValueError(f"pid={pid} choices_len={len(choices)} cannot map index {i} to A-Z")
        return LETTER_BY_IDX[i]

    ans_upper = str(answer).strip().upper()
    if len(ans_upper) == 1 and ans_upper in LETTER_BY_IDX:
        return ans_upper

    if debug:
        print("[MISMATCH] pid=", pid)
        print("  answer:", answer)
        print("  choices:", choices)

    if fail_on_mismatch:
        raise ValueError(f"multi_choice mapping mismatch for pid={pid}")

    return str(answer).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--multi_dir",
        required=True,
        help="load_from_disk path (multi_choice rows only), e.g. MathVista_testmini_multi_choice",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="output root: writes images/ and JSON list",
    )
    ap.add_argument(
        "--json_name",
        default="mathvista_multi_choice_eval.json",
        help="output JSON filename under out_dir",
    )
    ap.add_argument("--fail_on_mismatch", type=int, default=1)
    ap.add_argument("--debug_mismatch", action="store_true")
    ap.add_argument(
        "--append_letter_only_suffix",
        type=int,
        default=1,
        help="append short instruction so prompts match calibration-style MC (default: 1)",
    )
    ap.add_argument(
        "--suffix_text",
        default=DEFAULT_SUFFIX,
        help="suffix when --append_letter_only_suffix is 1",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    images_dir = out_dir / "images"
    json_path = out_dir / args.json_name
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(args.multi_dir)
    rows: List[Dict[str, Any]] = []

    for ex in ds:
        pid = str(ex.get("pid", ""))
        if not pid:
            raise ValueError(f"Empty pid in sample: {ex}")

        decoded_image = ex.get("decoded_image")
        if decoded_image is None:
            raise ValueError(f"Missing decoded_image for pid={pid}")

        choices = ex.get("choices")
        answer_raw = ex.get("answer")
        letter = multi_choice_to_letter(
            choices=choices if isinstance(choices, list) else [],
            answer=answer_raw,
            pid=pid,
            fail_on_mismatch=bool(args.fail_on_mismatch),
            debug=bool(args.debug_mismatch),
        )

        question_text = ex.get("query")
        if question_text is None:
            question_text = ex.get("question")
        if question_text is None:
            raise ValueError(f"Missing question/query for pid={pid}")
        question_text = str(question_text)
        if args.append_letter_only_suffix and args.suffix_text:
            if args.suffix_text.strip() not in question_text:
                question_text = question_text.rstrip() + args.suffix_text

        img_name = f"{pid}.jpg"
        save_image(decoded_image, images_dir / img_name)

        row: Dict[str, Any] = {
            "pid": pid,
            "image": img_name,
            "question": question_text,
            "choices": [str(c) for c in choices] if isinstance(choices, list) else [],
            "answer": letter,
            "answer_raw": str(answer_raw) if answer_raw is not None else "",
            "question_type": str(ex.get("question_type", "multi_choice")),
        }
        rows.append(row)

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {len(rows)} multi-choice eval samples")
    print("json:", json_path)
    print("images_dir:", images_dir)


if __name__ == "__main__":
    main()
