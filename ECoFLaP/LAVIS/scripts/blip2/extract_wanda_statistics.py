#!/usr/bin/env python3
"""Part 2 / Stage A -- extract the raw Wanda calibration statistic per dataset.

This is the foundational GPU pass. Everything downstream (statistic centrality,
calib-vs-eval crosslink, statistic structure) reads the NPZ this writes, on CPU,
so the expensive forward runs once per dataset and analysis is cheap to iterate.

For a given dataset it runs the dense BLIP2-T5 forward and accumulates, for every
prunable linear, the per-input-channel sum of squared activations -- i.e. the
quantity behind Wanda's ``scaler_row`` (Wanda scores |W_ij| * sqrt(sum_t X_tj^2 / T)).

Two things make this more than a re-run of the pruner:
  * T5 encoder statistics are split by token group (visual prefix / text / pad),
    so you can ask the Part-1 question (who dominates the statistic) per dataset.
  * ViT block statistics are captured too, because in Part 2 different
    calibrations use different images, so the ViT mask varies -- unlike Part 1.

Run once per dataset (each calibration set, and each eval set you want as a
reference distribution). The label you pass is how datasets are keyed downstream.

Usage:
  python scripts/blip2/extract_wanda_statistics.py \
      --label OKVQA \
      --calib_json /path/okvqa_calib.json --images_dir /path/okvqa_images \
      --ckpt /path/blip2_pretrained_flant5xl.pth \
      --out_dir /path/out/part2_stats/OKVQA \
      --max_samples 128 --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List

import numpy as np

from split_joint_analysis_common import (
    AUTO_INPUT_FIELDS,
    EncoderForward,
    build_vis_processor,
    ensure_dir,
    extract_text,
    iter_batches,
    load_batch_images,
    load_blip2,
    load_rows,
    select_rows,
)

T5_GROUPS = ("visual", "text", "pad")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract per-channel Wanda statistics for one dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--label", required=True, help="Dataset name, used as the key downstream.")
    p.add_argument("--calib_json", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ckpt", default=None, help="Dense checkpoint. Omit for pretrained blip2_t5.")
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
    p.add_argument("--no_vit", action="store_true", help="Skip ViT block statistics.")
    p.add_argument("--padding", choices=["longest", "max_length"], default="longest",
                   help="Keep 'longest' to reproduce what the pruner saw.")
    return p.parse_args()


def t5_encoder_linears(model, torch):
    out = []
    for index, block in enumerate(model.t5_model.encoder.block):
        for name, module in block.named_modules():
            if isinstance(module, torch.nn.Linear):
                out.append((index, name, module))
    return out


def vit_block_linears(model, torch):
    out = []
    encoder = getattr(model, "visual_encoder", None)
    if encoder is None:
        return out
    for name, module in encoder.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        m = re.search(r"blocks\.(\d+)\.", name)
        if not m:
            continue
        out.append((int(m.group(1)), name, module))
    return out


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

    rows, original_indices = select_rows(
        load_rows(os.path.abspath(args.calib_json)), args.max_samples, args.shuffle, args.seed
    )
    print("label:", args.label, "| device:", args.device, "| samples:", len(rows))

    model = load_blip2(args.model_name, args.model_type, args.device, args.ckpt, args.max_txt_len)

    t5_linears = t5_encoder_linears(model, torch)
    vit_linears = [] if args.no_vit else vit_block_linears(model, torch)
    print("hooked T5 encoder linears:", len(t5_linears), "| ViT block linears:", len(vit_linears))

    # accumulators: key -> per-channel sumsq (fp64, cpu)
    t5_sumsq: Dict[str, Dict[str, Any]] = {}
    for idx, name, module in t5_linears:
        key = "t5enc::%d::%s" % (idx, name)
        t5_sumsq[key] = {g: torch.zeros(module.weight.shape[1], dtype=torch.float64) for g in T5_GROUPS}
    vit_sumsq: Dict[str, Any] = {}
    for idx, name, module in vit_linears:
        key = "vit::%d::%s" % (idx, name)
        vit_sumsq[key] = torch.zeros(module.weight.shape[1], dtype=torch.float64)

    token_counts = {g: 0.0 for g in T5_GROUPS}
    vit_tokens = 0.0

    handles = []

    def t5_hook(key):
        def hook(_m, inputs, _o):
            x = inputs[0].detach()
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = x.to(torch.float32)
            sq = (x * x)
            masks = EncoderForward.current_masks()
            for g in T5_GROUPS:
                mask = masks[g].to(x.device).unsqueeze(-1).to(x.dtype)
                t5_sumsq[key][g] += (sq * mask).sum(dim=(0, 1)).double().cpu()
        return hook

    def vit_hook(key):
        def hook(_m, inputs, _o):
            nonlocal vit_tokens
            x = inputs[0].detach()
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = x.to(torch.float32)
            vit_sumsq[key] += (x * x).sum(dim=(0, 1)).double().cpu()
        return hook

    for idx, name, module in t5_linears:
        handles.append(module.register_forward_hook(t5_hook("t5enc::%d::%s" % (idx, name))))
    for idx, name, module in vit_linears:
        handles.append(module.register_forward_hook(vit_hook("vit::%d::%s" % (idx, name))))

    forward = EncoderForward(model, torch, padding=args.padding)
    vis_processor = build_vis_processor(args.image_size)

    try:
        for bi, (start, batch_rows) in enumerate(iter_batches(rows, args.batch_size)):
            images = load_batch_images(
                batch_rows, start, original_indices, os.path.abspath(args.images_dir),
                args.image_field, vis_processor, torch, Image,
            ).to(args.device)
            texts = [
                extract_text(row, args.text_field, AUTO_INPUT_FIELDS, original_indices[start + i])
                for i, row in enumerate(batch_rows)
            ]
            out = forward.run(images, texts, args.device)
            for g in T5_GROUPS:
                token_counts[g] += float(out["%s_mask" % g].sum().item())
            # ViT sees every patch of every image in the batch
            vit_tokens += float(images.shape[0])  # per-image count; per-token handled inside hook sums
            if bi % 5 == 0:
                print("  batch %d seq_len=%d" % (bi, out["seq_len"]))
    finally:
        for h in handles:
            h.remove()
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ViT token count: recover from any captured channel? We tracked images only.
    # The per-channel sumsq already integrates over all patch tokens; store the
    # exact token count by re-deriving from the first ViT accumulator is not
    # possible, so we record images and let downstream normalize per-channel by
    # its own total (cosine is scale-free anyway).
    save: Dict[str, np.ndarray] = {}
    for key, groups in t5_sumsq.items():
        for g in T5_GROUPS:
            save["%s::%s::sumsq" % (key, g)] = groups[g].numpy().astype(np.float64)
    for key, vec in vit_sumsq.items():
        save["%s::all::sumsq" % key] = vec.numpy().astype(np.float64)

    npz_path = os.path.join(args.out_dir, "wanda_statistics.npz")
    np.savez_compressed(npz_path, **save)

    meta = {
        "label": args.label,
        "calib_json": os.path.abspath(args.calib_json),
        "checkpoint": os.path.abspath(args.ckpt) if args.ckpt else "pretrained",
        "num_samples": len(rows),
        "padding": args.padding,
        "t5_token_counts": token_counts,
        "vit_images": vit_tokens,
        "t5_groups": list(T5_GROUPS),
        "num_t5_linears": len(t5_linears),
        "num_vit_linears": len(vit_linears),
        "statistic_definition": "per-input-channel sum over tokens of X^2 (Wanda scaler_row numerator)",
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as h:
        json.dump(meta, h, indent=2)

    total = sum(token_counts.values())
    print("\n=== T5 token census ===")
    for g in T5_GROUPS:
        print("  %-7s %10.0f (%.1f%%)" % (g, token_counts[g], 100.0 * token_counts[g] / max(total, 1.0)))
    print("[OK] wrote:", npz_path)


if __name__ == "__main__":
    main()
