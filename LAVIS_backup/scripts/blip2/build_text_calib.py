#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从某个数据集的标注 JSON 里抽取「纯文本」标定集，产出 C4 式的字符串列表 JSON，
供 TAMP / Wanda 的 `--prune_calib_mode t5_c4_text --c4_calib_json <out>` 使用。

支持的输入形态：
  - list[str]                          -> 直接用
  - list[dict]                         -> 从常见文本字段里取（question/caption/text/...）
  - {"annotations"|"data"|"questions": [...]}  -> 取该列表
  - {id: item, ...}                    -> 取 values()

文本字段自动探测顺序：text_input, text, caption, question, sent, query, prompt, output。
可选把选项(choices/options)与答案拼到文本后面，让标定文本更接近该数据集的真实 prompt。

例：
  python scripts/blip2/build_text_calib.py \
    --input  /data/data2/mfs/MMBench_calibration/mmbench_calibration_train.json \
    --output /data/data2/mfs/text_calib_128/mmbench_text_calib_128.json \
    --num 128 --seed 42 --shuffle --include-choices
"""
import argparse
import json
import os
import random

TEXT_KEYS = ["text_input", "text", "caption", "question", "sent", "query", "prompt", "output"]
CHOICE_KEYS = ["choices", "options", "option"]
ANSWER_KEYS = ["answer", "text_output", "answers", "gt_answer"]


def _first_str(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and value:
        return str(value[0]).strip()
    return ""


def extract_text(item, include_choices=False, include_answer=False):
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return ""

    base = ""
    for k in TEXT_KEYS:
        if k in item:
            base = _first_str(item[k])
            if base:
                break
    if not base:
        return ""

    parts = [base]
    if include_choices:
        for ck in CHOICE_KEYS:
            if ck in item and item[ck]:
                ch = item[ck]
                if isinstance(ch, dict):
                    ch = list(ch.values())
                if isinstance(ch, (list, tuple)):
                    parts.append(" ".join(str(c).strip() for c in ch if str(c).strip()))
                else:
                    parts.append(str(ch).strip())
                break
    if include_answer:
        for ak in ANSWER_KEYS:
            if ak in item and item[ak]:
                parts.append(_first_str(item[ak]))
                break
    return " ".join(p for p in parts if p)


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k in ("annotations", "data", "questions", "items"):
            if k in data and isinstance(data[k], list):
                return data[k]
        return list(data.values())
    if isinstance(data, list):
        return data
    raise ValueError("unsupported JSON top-level type: %s" % type(data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="源标注 JSON")
    ap.add_argument("--output", required=True, help="输出 C4 式纯文本 JSON（list[str]）")
    ap.add_argument("--num", type=int, default=128, help="抽取条数（须能被剪枝 batch_size 整除）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", action="store_true", help="按 seed 随机采样；否则取前 num 条")
    ap.add_argument("--include-choices", action="store_true", help="把选项拼进文本（MC 数据集建议开）")
    ap.add_argument("--include-answer", action="store_true", help="把答案也拼进文本")
    ap.add_argument("--dedup", action="store_true", default=True)
    args = ap.parse_args()

    rows = load_rows(args.input)
    texts = []
    for it in rows:
        t = extract_text(it, args.include_choices, args.include_answer)
        if t:
            texts.append(t)
    if not texts:
        raise SystemExit("[FATAL] 没能从 %s 抽出任何文本；检查字段名。" % args.input)

    if args.dedup:
        seen, uniq = set(), []
        for t in texts:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        texts = uniq

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(texts)

    if len(texts) < args.num:
        print("[WARN] 唯一文本 %d < num %d，循环补齐到 num（保证可被 batch 整除）" % (len(texts), args.num))
        reps = (args.num // len(texts)) + 1
        out = (texts * reps)[: args.num]
    else:
        out = texts[: args.num]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[OK] %s -> %s  (%d 条, 去重后共 %d 条可选)" % (args.input, args.output, len(out), len(texts)))


if __name__ == "__main__":
    main()
