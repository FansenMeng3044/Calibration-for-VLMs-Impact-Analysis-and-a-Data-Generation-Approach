#!/usr/bin/env python3
"""Extract the per-sample embedding that enters the T5 (LLM).

In BLIP2-T5 the LLM input is  cat([t5_proj(Q-Former output), embed_tokens(text)]).
This captures both parts per sample, mean-pooled to one vector each:

  visual_prefix : mean over the 32 query tokens of t5_proj(...)   [H]
                  -- the image's representation handed to the LLM; changes with
                     ViT/Q-Former pruning.
  text_embed    : mean over text tokens of embed_tokens(...)       [H]
                  -- unchanged by pruning (embed_tokens is not in a prunable block).

Two downstream uses (see analyze_llm_embeddings.py):
  * run the DENSE model on each dataset -> calib-vs-eval SEMANTIC similarity.
  * run DENSE + each pruned model on ONE eval set -> per-calibration LLM-input
    FIDELITY to dense.

Usage:
  python scripts/blip2/extract_llm_input_embeddings.py \
      --label OKVQA --calib_json /p/okvqa.json --images_dir /p/okvqa_images \
      --ckpt /p/blip2_pretrained_flant5xl.pth \
      --out_dir /p/out/llm_emb/dense_on_OKVQA \
      --max_samples 128 --batch_size 8
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from typing import Any, List

import numpy as np

from split_joint_analysis_common import (
    AUTO_INPUT_FIELDS, EncoderForward, build_vis_processor, ensure_dir, extract_text,
    iter_batches, load_batch_images, load_blip2, load_rows, select_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract per-sample LLM-input embeddings (visual prefix + text).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--label", required=True)
    p.add_argument("--calib_json", required=True)
    p.add_argument("--images_dir", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--input_mode", choices=["multimodal", "text_only"], default="multimodal")
    p.add_argument("--ckpt", default=None, help="Dense or a pruned checkpoint. Omit for pretrained.")
    p.add_argument("--model_name", default="blip2_t5")
    p.add_argument("--model_type", default="pretrain_flant5xl")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=128)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--image_field", default="image")
    p.add_argument("--text_field", default="auto")
    p.add_argument("--max_txt_len", type=int, default=None)
    p.add_argument("--fp32", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Missing runtime dependency: %s" % exc) from exc
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(args.out_dir)
    if args.input_mode == "multimodal" and not args.images_dir:
        raise SystemExit("--images_dir is required when --input_mode multimodal.")

    rows, original_indices = select_rows(
        load_rows(os.path.abspath(args.calib_json)), args.max_samples, args.shuffle, args.seed)
    print("label:", args.label, "| device:", args.device, "| samples:", len(rows),
          "| mode:", args.input_mode, "| ckpt:", args.ckpt or "pretrained")

    model = load_blip2(args.model_name, args.model_type, args.device, args.ckpt, args.max_txt_len)
    if args.fp32:
        model.float()
    forward = EncoderForward(model, torch, padding="max_length", fp32=args.fp32)
    vis_processor = build_vis_processor(args.image_size)

    visual_rows: List[np.ndarray] = []
    text_rows: List[np.ndarray] = []
    sample_index: List[int] = []

    for bi, (start, batch_rows) in enumerate(iter_batches(rows, args.batch_size)):
        texts = [extract_text(r, args.text_field, AUTO_INPUT_FIELDS, original_indices[start + i])
                 for i, r in enumerate(batch_rows)]
        if args.input_mode == "multimodal":
            images = load_batch_images(
                batch_rows, start, original_indices, os.path.abspath(args.images_dir),
                args.image_field, vis_processor, torch, Image).to(args.device)
            out = forward.run(images, texts, args.device)
            vt = out["visual_tokens"].to(torch.float32)             # [B, num_query, H]
            te = out["text_embeddings"].to(torch.float32)           # [B, T, H]
            tmask = out["text_mask"][:, out["num_query"]:].to(te.dtype)  # [B, T]
            v_pool = vt.mean(dim=1)                                  # [B, H]
        else:
            with torch.no_grad():
                input_tokens = model.t5_tokenizer(
                    list(texts),
                    padding="max_length",
                    truncation=True,
                    max_length=model.max_txt_len,
                    return_tensors="pt",
                ).to(args.device)
                amp = contextlib.nullcontext() if args.fp32 else model.maybe_autocast(dtype=torch.bfloat16)
                with amp:
                    te = model.t5_model.encoder.embed_tokens(input_tokens.input_ids).to(torch.float32)
            tmask = input_tokens.attention_mask.to(te.dtype)
            v_pool = torch.zeros((te.shape[0], te.shape[-1]), dtype=te.dtype, device=te.device)
        denom = tmask.sum(dim=1, keepdim=True).clamp_min(1.0)
        t_pool = (te * tmask.unsqueeze(-1)).sum(dim=1) / denom   # [B, H], masked mean

        visual_rows.append(v_pool.cpu().numpy())
        text_rows.append(t_pool.cpu().numpy())
        sample_index.extend(original_indices[start:start + len(batch_rows)])
        if bi % 5 == 0:
            print("  batch %d" % bi)

    visual = np.concatenate(visual_rows, axis=0)
    text = np.concatenate(text_rows, axis=0)
    np.savez_compressed(os.path.join(args.out_dir, "llm_input_embeddings.npz"),
                        visual_prefix=visual, text_embed=text,
                        sample_index=np.asarray(sample_index, dtype=np.int64))
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as h:
        json.dump({"label": args.label, "checkpoint": os.path.abspath(args.ckpt) if args.ckpt else "pretrained",
                   "calib_json": os.path.abspath(args.calib_json),
                   "images_dir": os.path.abspath(args.images_dir) if args.images_dir else None,
                   "input_mode": args.input_mode,
                   "text_field": args.text_field,
                   "num_samples": int(visual.shape[0]),
                   "hidden": int(visual.shape[1])}, h, indent=2)
    print("[OK] wrote %d samples -> %s" % (visual.shape[0], os.path.join(args.out_dir, "llm_input_embeddings.npz")))


if __name__ == "__main__":
    main()
