#!/usr/bin/env python3
"""Materialise the OKVQA text-only calibration EXACTLY as the split pipeline does.

The five-calib unimodal-split scripts derive each source's T5 text through
``_prepare_unimodal_jsons`` -> ``build_text`` over ``rows[:num_data]`` (the FIRST
num_data rows, no shuffle, no dedup), writing a JSON list of ``{"text": ...}``.
To make a HYBRID run whose OKVQA text is byte-identical to OKVQA-as-same-source,
this reproduces that extraction verbatim (value_to_text / build_text copied from
run_sparsegpt_fivecalib_unimodal_split_then_fourbench_eval.sh).

Point --okvqa_json at the SAME OKVQA raw json the split runs use
(default $BASE/datasets/okvqa/annotations/okvqa_train.json) with the SAME --num.

Usage:
  python scripts/blip2/build_okvqa_text_calib.py \
      --okvqa_json /data/data2/mfs/datasets/okvqa/annotations/okvqa_train.json \
      --out /data/data2/mfs/okvqa_text_128.json --num 128
"""

from __future__ import annotations

import argparse
import json


# --- copied verbatim from _prepare_unimodal_jsons (keep in sync) -------------
def value_to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(value_to_text(v) for v in value if value_to_text(v)).strip()
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            txt = value_to_text(val)
            if txt:
                parts.append(f"{key}. {txt}")
        return "\n".join(parts).strip()
    return str(value).strip()


def build_text(row):
    fields = ("text", "caption", "text_input", "question", "prompt", "output")
    selected = ""
    selected_field = ""
    for field in fields:
        if field in row:
            selected = value_to_text(row.get(field))
            if selected:
                selected_field = field
                break
    if not selected:
        raise ValueError("missing text/caption/question field")

    parts = [selected]
    hint = value_to_text(row.get("hint"))
    if hint and "hint:" not in selected.lower():
        parts.append("Hint: " + hint)
    options = row.get("options")
    if options:
        opt_text = value_to_text(options)
        if opt_text and opt_text not in selected:
            parts.append(opt_text)
    for letter in "ABCDEFG":
        opt = value_to_text(row.get(letter))
        if opt and f"{letter}." not in selected:
            parts.append(f"{letter}. {opt}")
    return "\n".join(parts).strip(), selected_field
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build OKVQA text calib identical to same-source split.")
    p.add_argument("--okvqa_json", required=True, help="OKVQA raw calibration json (list of dicts).")
    p.add_argument("--out", required=True, help="Output JSON list of {'text': ...}.")
    p.add_argument("--num", type=int, default=128, help="Take the FIRST --num rows (matches rows[:num_data]).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.okvqa_json, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        raise SystemExit("[FATAL] %s is not a non-empty JSON list" % args.okvqa_json)
    if len(rows) < args.num:
        raise SystemExit("[FATAL] only %d rows, need num=%d" % (len(rows), args.num))

    text_rows, field_counts = [], {}
    for i, row in enumerate(rows[: args.num]):
        if not isinstance(row, dict):
            raise SystemExit("[FATAL] row %d is not a JSON object" % i)
        text, field = build_text(row)
        field_counts[field] = field_counts.get(field, 0) + 1
        text_rows.append({"text": text})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(text_rows, f, ensure_ascii=False, indent=0)
    print("[OK] wrote %d rows -> %s" % (len(text_rows), args.out))
    print("     text field used:", field_counts)
    for r in text_rows[:3]:
        print("     e.g. %r" % r["text"])


if __name__ == "__main__":
    main()
