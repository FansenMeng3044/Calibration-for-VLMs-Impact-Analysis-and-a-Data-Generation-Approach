#!/usr/bin/env python3
"""Stream C4 en shard from HF mirror; write 128 lines to JSON (no Hub API)."""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import urllib.request

DEFAULT_URL = (
    "https://hf-mirror.com/datasets/allenai/c4/resolve/main/en/"
    "c4-train.00000-of-01024.json.gz"
)
FALLBACK_URL = (
    "https://huggingface.co/datasets/allenai/c4/resolve/main/en/"
    "c4-train.00000-of-01024.json.gz"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="c4_calib_128.json")
    p.add_argument("--url", type=str, default=DEFAULT_URL)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--try-fallback", action="store_true")
    args = p.parse_args()

    def fetch(url: str):
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; c4-calib/1.0)"},
        )
        return urllib.request.urlopen(req, timeout=120)

    try:
        resp = fetch(args.url)
    except OSError as e:
        if args.try_fallback and "hf-mirror.com" in args.url:
            print(f"mirror failed ({e}); trying huggingface.co ...", file=sys.stderr)
            resp = fetch(FALLBACK_URL)
        else:
            raise

    texts = []
    with resp:
        dec = gzip.GzipFile(fileobj=resp)
        text_io = io.TextIOWrapper(dec, encoding="utf-8", errors="replace")
        for line in text_io:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"].strip())
            if len(texts) >= args.n:
                break

    if len(texts) < args.n:
        raise SystemExit(f"only got {len(texts)} lines (need {args.n})")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=0)
    print(f"wrote {args.out} n={len(texts)}", file=sys.stderr)


if __name__ == "__main__":
    main()
