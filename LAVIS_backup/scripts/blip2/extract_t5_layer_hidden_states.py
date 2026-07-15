#!/usr/bin/env python3
"""Extract pooled BLIP2-T5 encoder hidden states for every T5 encoder layer.

This is the layer-wise version of extract_llm_input_embeddings.py.  Instead of
only saving the embedding that enters T5, it runs the T5 encoder with
output_hidden_states=True and saves one pooled vector per sample and per layer.

Saved arrays:

  layer_both   : [N, L, H], mean over visual-query tokens + real text tokens
  layer_visual : [N, L, H], mean over the 32 visual-query tokens
  layer_text   : [N, L, H], mean over real text tokens

For a 24-layer T5 encoder, L=24.  Layer index 0 is the output after encoder
block 0, and layer index 23 is the final encoder output after the last block
and final layer norm.  The raw T5 input embedding is intentionally excluded
because it is already handled by extract_llm_input_embeddings.py.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from typing import Any, Dict, List, Sequence

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract pooled T5 encoder hidden states for every encoder layer.",
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
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=128)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--image_field", default="image")
    p.add_argument("--text_field", default="auto")
    p.add_argument("--max_txt_len", type=int, default=None)
    p.add_argument("--fp32", action="store_true")
    p.add_argument(
        "--parts",
        default="both",
        help="Space-separated pooled parts to save: any of 'both visual text'.",
    )
    p.add_argument(
        "--save_dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Storage dtype for saved layer arrays.",
    )
    return p.parse_args()


def parse_parts(value: str) -> List[str]:
    parts = [x.strip() for x in value.replace(",", " ").split() if x.strip()]
    valid = {"both", "visual", "text"}
    bad = [x for x in parts if x not in valid]
    if bad:
        raise SystemExit("Invalid --parts values: %s" % ", ".join(bad))
    if not parts:
        raise SystemExit("--parts must contain at least one of: both visual text")
    return parts


def masked_mean(x: Any, mask: Any) -> Any:
    denom = mask.to(x.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask.to(x.dtype).unsqueeze(-1)).sum(dim=1) / denom


def pool_layers(hidden_states: Sequence[Any], masks: Dict[str, Any], parts: Sequence[str], torch: Any) -> Dict[str, Any]:
    """Return part -> [B, L, H] for T5 block outputs.

    HuggingFace T5 hidden_states has length num_layers + 1:
      hidden_states[0]  = T5 encoder input embedding after dropout
      hidden_states[1:] = outputs after encoder blocks/final layer norm
    """
    layers = list(hidden_states[1:])
    out: Dict[str, List[Any]] = {part: [] for part in parts}
    both_mask = masks["both"]
    for h in layers:
        h = h.to(torch.float32)
        if "both" in out:
            out["both"].append(masked_mean(h, both_mask))
        if "visual" in out:
            out["visual"].append(masked_mean(h, masks["visual"]))
        if "text" in out:
            out["text"].append(masked_mean(h, masks["text"]))
    return {part: torch.stack(values, dim=1) for part, values in out.items()}


def run_text_only_hidden_states(model: Any, texts: Sequence[str], device: str, torch: Any, fp32: bool):
    with torch.no_grad():
        input_tokens = model.t5_tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=model.max_txt_len,
            return_tensors="pt",
        ).to(device)
        amp = contextlib.nullcontext() if fp32 else model.maybe_autocast(dtype=torch.bfloat16)
        with amp:
            text_embeddings = model.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            encoder_outputs = model.t5_model.encoder(
                inputs_embeds=text_embeddings,
                attention_mask=input_tokens.attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
    text_mask = input_tokens.attention_mask.bool()
    visual_mask = torch.zeros_like(text_mask, dtype=torch.bool)
    return encoder_outputs.hidden_states, {
        "both": text_mask,
        "text": text_mask,
        "visual": visual_mask,
    }, input_tokens


def main() -> None:
    args = parse_args()
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Missing runtime dependency: %s" % exc) from exc

    parts = parse_parts(args.parts)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.input_mode == "multimodal" and not args.images_dir:
        raise SystemExit("--images_dir is required when --input_mode multimodal.")
    ensure_dir(args.out_dir)

    rows, original_indices = select_rows(
        load_rows(os.path.abspath(args.calib_json)), args.max_samples, args.shuffle, args.seed
    )
    print(
        "label:", args.label,
        "| device:", args.device,
        "| samples:", len(rows),
        "| mode:", args.input_mode,
        "| parts:", ",".join(parts),
        "| ckpt:", args.ckpt or "pretrained",
    )

    model = load_blip2(args.model_name, args.model_type, args.device, args.ckpt, args.max_txt_len)
    if args.fp32:
        model.float()
    forward = EncoderForward(model, torch, padding="max_length", fp32=args.fp32)
    vis_processor = build_vis_processor(args.image_size)

    part_rows: Dict[str, List[np.ndarray]] = {part: [] for part in parts}
    sample_index: List[int] = []
    layer_count = None
    hidden_size = None

    for bi, (start, batch_rows) in enumerate(iter_batches(rows, args.batch_size)):
        texts = [
            extract_text(r, args.text_field, AUTO_INPUT_FIELDS, original_indices[start + i])
            for i, r in enumerate(batch_rows)
        ]
        if args.input_mode == "multimodal":
            images = load_batch_images(
                batch_rows,
                start,
                original_indices,
                os.path.abspath(args.images_dir),
                args.image_field,
                vis_processor,
                torch,
                Image,
            ).to(args.device)
            out = forward.run(images, texts, args.device, output_hidden_states=True)
            hidden_states = out["encoder_hidden_states"]
            masks = {
                "visual": out["visual_mask"],
                "text": out["text_mask"],
                "both": out["encoder_attention"].bool(),
            }
        else:
            hidden_states, masks, _ = run_text_only_hidden_states(
                model, texts, args.device, torch, args.fp32
            )

        pooled = pool_layers(hidden_states, masks, parts, torch)
        for part in parts:
            arr = pooled[part].detach().cpu().numpy()
            if args.save_dtype == "float16":
                arr = arr.astype(np.float16)
            else:
                arr = arr.astype(np.float32)
            part_rows[part].append(arr)
        if layer_count is None:
            first = next(iter(pooled.values()))
            layer_count = int(first.shape[1])
            hidden_size = int(first.shape[2])
        sample_index.extend(original_indices[start:start + len(batch_rows)])
        if bi % 5 == 0:
            print("  batch %d" % bi)

    arrays = {
        "layer_%s" % part: np.concatenate(values, axis=0)
        for part, values in part_rows.items()
    }
    arrays["sample_index"] = np.asarray(sample_index, dtype=np.int64)
    arrays["layer_index"] = np.arange(int(layer_count or 0), dtype=np.int64)
    out_npz = os.path.join(args.out_dir, "t5_layer_hidden_states.npz")
    np.savez_compressed(out_npz, **arrays)

    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "label": args.label,
                "checkpoint": os.path.abspath(args.ckpt) if args.ckpt else "pretrained",
                "calib_json": os.path.abspath(args.calib_json),
                "images_dir": os.path.abspath(args.images_dir) if args.images_dir else None,
                "input_mode": args.input_mode,
                "text_field": args.text_field,
                "parts": parts,
                "num_samples": len(sample_index),
                "num_layers": int(layer_count or 0),
                "hidden": int(hidden_size or 0),
                "save_dtype": args.save_dtype,
            },
            handle,
            indent=2,
        )
    print("[OK] wrote %d samples x %d layers -> %s" % (len(sample_index), int(layer_count or 0), out_npz))


if __name__ == "__main__":
    main()
