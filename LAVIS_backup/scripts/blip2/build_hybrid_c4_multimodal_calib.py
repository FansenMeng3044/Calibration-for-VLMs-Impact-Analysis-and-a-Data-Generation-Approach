#!/usr/bin/env python3
"""Build a multimodal calibration JSON with original images and C4 text.

The output keeps the image-side fields from the source calibration rows, but
replaces all common text fields with a C4 text string. This lets the existing
prefix_conceptual_caption_3m loader run a full BLIP-2 multimodal forward while
the language tokens come from C4 instead of the image dataset.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from typing import Any, Dict, Iterable, List, Sequence


TEXT_KEYS = (
    "caption",
    "text",
    "text_input",
    "text_output",
    "output",
    "question",
    "prompt",
    "sent",
    "query",
    "answer",
)

C4_TEXT_KEYS = (
    "text",
    "caption",
    "text_input",
    "text_output",
    "output",
    "question",
    "prompt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair images from one calibration set with C4 text strings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image_json", required=True, help="Source multimodal calibration JSON.")
    parser.add_argument("--c4_json", required=True, help="C4 text JSON, list[str] or list[dict].")
    parser.add_argument("--output", required=True, help="Output hybrid calibration JSON.")
    parser.add_argument("--num", type=int, default=128, help="Number of hybrid rows to write.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    parser.add_argument(
        "--shuffle_images",
        action="store_true",
        help="Shuffle image rows before taking --num. Default preserves source order.",
    )
    parser.add_argument(
        "--shuffle_c4",
        action="store_true",
        default=True,
        help="Shuffle C4 texts before pairing.",
    )
    parser.add_argument(
        "--no_shuffle_c4",
        dest="shuffle_c4",
        action="store_false",
        help="Use C4 texts in file order.",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Store original text fields under _hybrid_original_text for auditing.",
    )
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rows_from_json(data: Any, path: str) -> List[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("annotations", "data", "items", "samples", "questions"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return list(data.values())
    raise ValueError("%s has unsupported JSON top-level type: %s" % (path, type(data).__name__))


def extract_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in C4_TEXT_KEYS:
            value = item.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    return text
    text = str(item).strip()
    return text


def load_c4_texts(path: str) -> List[str]:
    rows = rows_from_json(load_json(path), path)
    texts = [extract_text(row) for row in rows]
    texts = [text for text in texts if text]
    if not texts:
        raise ValueError("No usable C4 text found in %s" % path)
    return texts


def repeat_to_length(items: Sequence[str], length: int) -> List[str]:
    if not items:
        raise ValueError("Cannot repeat an empty sequence.")
    out: List[str] = []
    while len(out) < length:
        out.extend(items)
    return out[:length]


def original_text_snapshot(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: copy.deepcopy(row[key]) for key in TEXT_KEYS if key in row}


def apply_c4_text(row: Dict[str, Any], c4_text: str, keep_metadata: bool) -> Dict[str, Any]:
    out = copy.deepcopy(row)
    if keep_metadata:
        out["_hybrid_original_text"] = original_text_snapshot(out)
    out["_hybrid_c4_text"] = c4_text

    # Different local dataset wrappers may read different names. Keep them all
    # synchronized so the forward path cannot silently use the old question or answer.
    out["caption"] = c4_text
    out["text"] = c4_text
    out["text_input"] = c4_text
    out["text_output"] = c4_text
    out["output"] = c4_text
    out["question"] = c4_text
    out["answer"] = c4_text
    return out


def main() -> int:
    args = parse_args()
    if args.num <= 0:
        raise ValueError("--num must be positive.")
    if not os.path.isfile(args.image_json):
        raise FileNotFoundError(args.image_json)
    if not os.path.isfile(args.c4_json):
        raise FileNotFoundError(args.c4_json)

    rng = random.Random(args.seed)
    image_rows = rows_from_json(load_json(args.image_json), args.image_json)
    if not image_rows:
        raise ValueError("No rows found in %s" % args.image_json)
    if args.shuffle_images:
        image_rows = list(image_rows)
        rng.shuffle(image_rows)
    image_rows = image_rows[: args.num]
    if len(image_rows) < args.num:
        raise ValueError(
            "%s has only %d rows, fewer than requested --num %d"
            % (args.image_json, len(image_rows), args.num)
        )

    c4_texts = load_c4_texts(args.c4_json)
    c4_texts = list(c4_texts)
    if args.shuffle_c4:
        rng.shuffle(c4_texts)
    c4_texts = repeat_to_length(c4_texts, len(image_rows))

    out_rows: List[Dict[str, Any]] = []
    for idx, (row, c4_text) in enumerate(zip(image_rows, c4_texts)):
        if not isinstance(row, dict):
            raise ValueError("Image row %d is not a JSON object." % idx)
        out_rows.append(apply_c4_text(row, c4_text, args.metadata))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)
        f.write("\n")

    preview = out_rows[0].get("image", "<missing image>") if out_rows else "<empty>"
    print("[OK] wrote hybrid calibration JSON: %s" % args.output)
    print("[OK] rows=%d first_image=%s" % (len(out_rows), preview))
    print("[OK] text source: %s" % args.c4_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
