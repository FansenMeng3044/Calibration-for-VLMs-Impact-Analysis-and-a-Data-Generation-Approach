#!/usr/bin/env python3
"""Quantify how much each calibration/eval text set is truncated at max_txt_len.

BLIP2-FlanT5's encoder forward tokenizes the input text with
``truncation=True, max_length=model.max_txt_len`` (default 32). Long prompts --
multiple-choice items in MMBench / MMMU / MathVista -- get cut, which affects
both evaluation (the model may not see every option) and, when such a set is
used as a CALIBRATION SOURCE, the text-side statistic S_L and sim_sem that are
computed from the truncated text.

This script reproduces exactly that tokenization (same tokenizer, same text
field via extract_text, no template -- matching EncoderForward.tokenize) but
WITHOUT truncation, so it can report, per dataset:

  * fraction of samples that would be truncated (len > max_txt_len)
  * length distribution (mean / median / p90 / p95 / max), in tokens
  * among truncated samples, mean tokens dropped and mean fraction dropped

No model weights and no GPU are needed: only the FlanT5 tokenizer is loaded.
Point --max_samples/--shuffle/--seed at the SAME draw you calibrate with (e.g.
128, shuffle, seed 42) to measure what truncation actually happened during
calibration; omit --max_samples to characterize the whole set.

Usage:
  python scripts/blip2/check_calibration_truncation.py \
      --dataset "MMBench=/p/mmbench_calib.json" \
      --dataset "MMMU=/p/mmmu.json" \
      --dataset "OKVQA=/p/okvqa.json" \
      --dataset "MathVista=/p/mathvista.json" \
      --dataset "CC3M=/p/cc3m.json" \
      --dataset "C4=/p/c4_calib.json" \
      --max_txt_len 32 --max_samples 128 --shuffle --seed 42 \
      --out_dir /p/out/truncation_audit
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np

from split_joint_analysis_common import (
    AUTO_INPUT_FIELDS,
    ensure_dir,
    extract_text,
    load_rows,
    select_rows,
    setup_matplotlib,
    value_to_text,
    write_csv,
)


def row_text(row, field: str, idx: int) -> str:
    """Text for a row, whatever its container type.

    Calibration files vary: some are lists of dicts, some (a plain text corpus
    like C4) are lists of raw strings, some wrap the text in a list. extract_text
    only handles dicts, so dispatch by type here.
    """
    if isinstance(row, str):
        return row.strip()
    if isinstance(row, (list, tuple)):
        return value_to_text(row)
    if isinstance(row, dict):
        return extract_text(row, field, AUTO_INPUT_FIELDS, idx)
    return str(row).strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit text truncation of calibration/eval sets at max_txt_len.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", action="append", required=True, metavar="LABEL=PATH",
                   help="Repeatable. Path is json/jsonl/parquet with a text field.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_txt_len", type=int, default=32,
                   help="Same cap the encoder forward uses. BLIP2-FlanT5 default is 32.")
    p.add_argument("--tokenizer", default="google/flan-t5-xl",
                   help="HF id of the FlanT5 tokenizer BLIP2-T5 uses.")
    p.add_argument("--from_model", default=None,
                   help="Optional: model_name,model_type to pull model.t5_tokenizer "
                        "instead of a bare HF tokenizer (exact fidelity, heavier). "
                        "e.g. 'blip2_t5,pretrain_flant5xl'.")
    p.add_argument("--text_field", default="auto",
                   help="Row text field; 'auto' tries %s." % (list(AUTO_INPUT_FIELDS),))
    p.add_argument("--prompt_template", default=None,
                   help="Optional format string with one {} placeholder, if your "
                        "calibration wraps the text (EncoderForward does NOT, so "
                        "leave unset to match the pruning pass).")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Match your calibration draw (e.g. 128). Omit for the whole set.")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def load_tokenizer(args):
    if args.from_model:
        # Exact tokenizer instance the model would use (needs LAVIS + weights).
        from split_joint_analysis_common import load_blip2
        name, mtype = [s.strip() for s in args.from_model.split(",")]
        model = load_blip2(name, mtype, device="cpu")
        tok = model.t5_tokenizer
        return tok
    # Bare FlanT5 tokenizer -- identical vocab/merges to model.t5_tokenizer.
    try:
        from transformers import T5TokenizerFast
        return T5TokenizerFast.from_pretrained(args.tokenizer)
    except Exception:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(args.tokenizer)


def token_length(tok, text: str) -> int:
    """Tokens the encoder would see, counted the way max_length counts them.

    truncation=False so we see the FULL length; add_special_tokens=True so the
    T5 eos is included, matching how truncation to max_length counts tokens.
    """
    ids = tok(text, truncation=False, add_special_tokens=True)["input_ids"]
    return len(ids)


def parse_dataset(item: str) -> Tuple[str, str]:
    if "=" not in item:
        raise ValueError("Expected LABEL=PATH, got %r" % item)
    label, path = item.split("=", 1)
    return label.strip(), os.path.abspath(os.path.expanduser(path.strip().strip('"')))


def audit_one(tok, label: str, path: str, args) -> Tuple[dict, List[dict]]:
    rows_all = load_rows(path)
    rows, original_indices = select_rows(rows_all, args.max_samples, args.shuffle, args.seed)

    lengths: List[int] = []
    per_sample: List[dict] = []
    missing = 0
    for i, row in enumerate(rows):
        try:
            text = row_text(row, args.text_field, original_indices[i])
        except KeyError:
            missing += 1
            continue
        if not text:
            missing += 1
            continue
        if args.prompt_template:
            text = args.prompt_template.format(text)
        n = token_length(tok, text)
        lengths.append(n)
        per_sample.append({"dataset": label, "orig_index": original_indices[i], "tokens": n})

    arr = np.asarray(lengths, dtype=np.float64)
    cap = args.max_txt_len
    n = int(arr.size)
    if n == 0:
        raise ValueError("No usable text rows in %s (missing field in %d rows)." % (path, missing))

    trunc_mask = arr > cap
    dropped = np.clip(arr - cap, 0, None)          # tokens lost per sample
    trunc_lengths = arr[trunc_mask]
    summary = {
        "dataset": label,
        "n": n,
        "missing_text": missing,
        "max_txt_len": cap,
        "pct_truncated": round(100.0 * float(trunc_mask.mean()), 2),
        "mean_tokens": round(float(arr.mean()), 2),
        "median_tokens": round(float(np.median(arr)), 1),
        "p90_tokens": round(float(np.percentile(arr, 90)), 1),
        "p95_tokens": round(float(np.percentile(arr, 95)), 1),
        "max_tokens": int(arr.max()),
        # among truncated samples only:
        "trunc_mean_len": round(float(trunc_lengths.mean()), 1) if trunc_lengths.size else 0.0,
        "trunc_mean_dropped": round(float(dropped[trunc_mask].mean()), 1) if trunc_lengths.size else 0.0,
        # over ALL samples: mean fraction of the text that is discarded
        "mean_frac_dropped_pct": round(100.0 * float((dropped / np.maximum(arr, 1)).mean()), 2),
    }
    return summary, per_sample


def maybe_plot(plt, label: str, lengths: List[int], cap: int, out_dir: str) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.hist(lengths, bins=40, color="#4C78A8", alpha=0.85)
    ax.axvline(cap, color="#E4572E", linestyle="--", linewidth=1.5, label="max_txt_len=%d" % cap)
    ax.set_xlabel("input tokens (untruncated)")
    ax.set_ylabel("samples")
    ax.set_title(label)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_%s.pdf" % label), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    tok = load_tokenizer(args)
    plt = None if args.no_plots else setup_matplotlib()

    summaries: List[dict] = []
    all_samples: List[dict] = []
    for item in args.dataset:
        label, path = parse_dataset(item)
        summary, per_sample = audit_one(tok, label, path, args)
        summaries.append(summary)
        all_samples.extend(per_sample)
        if plt is not None:
            maybe_plot(plt, label, [s["tokens"] for s in per_sample], args.max_txt_len, args.out_dir)
        print("[%s] n=%d  truncated=%.1f%%  mean=%.1f  p95=%.1f  max=%d  "
              "(trunc: mean_len=%.1f, mean_dropped=%.1f)"
              % (label, summary["n"], summary["pct_truncated"], summary["mean_tokens"],
                 summary["p95_tokens"], summary["max_tokens"],
                 summary["trunc_mean_len"], summary["trunc_mean_dropped"]))

    write_csv(os.path.join(args.out_dir, "truncation_summary.csv"), summaries)
    write_csv(os.path.join(args.out_dir, "per_sample_lengths.csv"), all_samples)

    # A compact table you can paste straight into the limitation paragraph.
    print("\n=== truncation summary (max_txt_len=%d) ===" % args.max_txt_len)
    hdr = ("dataset", "n", "%trunc", "mean", "p95", "max", "drop/trunc")
    print("  %-10s %5s %7s %6s %6s %5s %10s" % hdr)
    for s in summaries:
        print("  %-10s %5d %6.1f%% %6.1f %6.1f %5d %10.1f"
              % (s["dataset"], s["n"], s["pct_truncated"], s["mean_tokens"],
                 s["p95_tokens"], s["max_tokens"], s["trunc_mean_dropped"]))
    print("\n[OK] wrote:", os.path.join(args.out_dir, "truncation_summary.csv"))


if __name__ == "__main__":
    main()
