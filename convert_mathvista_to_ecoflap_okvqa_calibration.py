#!/usr/bin/env python3
"""Convert MathVista HF dataset splits into ECoFLaP/LAVIS OKVQA calibration format.

Output:
  out_dir/images/<pid>.jpg
  out_dir/mathvista_calibration_train.json : JSON list of {"image","question","answer"}

Answer mapping for multi_choice:
  - normalize answer and each choice (strip/collapse spaces + casefold)
  - if answer matches a choice at index i -> letter A/B/C/D by i
  - else if answer itself is a letter A-D -> use it
  - else print mismatch info and fail (default)

This script does not modify any LAVIS/ECoFLaP code.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_from_disk


def norm_text(x: Any) -> str:
    s = str(x).replace(" ", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


LETTER_BY_IDX = [chr(ord("A") + i) for i in range(26)]  # A-Z


def save_image(decoded_image: Any, out_path: Path) -> None:
    img = decoded_image
    try:
        img = img.convert("RGB")
    except Exception:
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="JPEG", quality=95)


def multi_choice_to_letter(choices: List[Any], answer: Any, pid: str, fail_on_mismatch: bool, debug: bool) -> str:
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
    if ans_upper in LETTER_BY_IDX:
        return ans_upper

    if debug:
        print("[MISMATCH] pid=", pid)
        print("  answer:", answer)
        print("  answer_norm:", answer_norm)
        print("  choices:", choices)
        print("  choice_norms:", choice_norms)

    if fail_on_mismatch:
        raise ValueError(f"multi_choice mapping mismatch for pid={pid}")

    return str(answer).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi_dir", required=True)
    ap.add_argument("--free_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--json_name", default="mathvista_calibration_train.json")
    ap.add_argument("--fail_on_mismatch", type=int, default=1)
    ap.add_argument("--debug_mismatch", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    images_dir = out_dir / "images"
    json_path = out_dir / args.json_name
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    multi_ds = load_from_disk(args.multi_dir)
    free_ds = load_from_disk(args.free_dir)

    annotations: List[Dict[str, Any]] = []

    def handle_split(ds, qtype: str) -> None:
        for ex in ds:
            pid = str(ex.get("pid", ""))
            if not pid:
                raise ValueError(f"Empty pid in {qtype} sample: {ex}")

            decoded_image = ex.get("decoded_image")
            if decoded_image is None:
                raise ValueError(f"Missing decoded_image for pid={pid}")

            img_name = f"{pid}.jpg"
            save_image(decoded_image, images_dir / img_name)

            question_text = ex.get("query")
            if question_text is None:
                question_text = ex.get("question")
            if question_text is None:
                raise ValueError(f"Missing question/query for pid={pid}")

            if qtype == "free_form":
                answer_text = str(ex.get("answer"))
            elif qtype == "multi_choice":
                answer_text = multi_choice_to_letter(
                    choices=ex.get("choices"),
                    answer=ex.get("answer"),
                    pid=pid,
                    fail_on_mismatch=bool(args.fail_on_mismatch),
                    debug=bool(args.debug_mismatch),
                )
            else:
                raise ValueError(f"Unknown qtype {qtype}")

            annotations.append({"image": img_name, "question": str(question_text), "answer": answer_text})

    handle_split(free_ds, "free_form")
    handle_split(multi_ds, "multi_choice")

    if len(annotations) != (len(free_ds) + len(multi_ds)):
        raise RuntimeError("Annotation count mismatch")

    json_path.write_text(json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {len(annotations)} samples")
    print("json:", json_path)
    print("images_dir:", images_dir)


if __name__ == "__main__":
    main()
