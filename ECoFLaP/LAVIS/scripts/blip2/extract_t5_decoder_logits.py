#!/usr/bin/env python3
"""Extract teacher-forced T5 decoder hidden states and compact logit summaries.

This script is the decoder-side counterpart of extract_t5_layer_hidden_states.py.
It uses ground-truth output text with teacher forcing, so dense and pruned
models use the same decoder token sequence and can be compared layer-by-layer.

Saved arrays:

  decoder_layer      : [N, L, H], mean-pooled decoder hidden state per layer
  gold_logprob       : [N, T], log p(gold token | prefix, image/question)
  gold_logit         : [N, T], raw logit of each gold token
  decoder_mask       : [N, T], 1 for non-padding target positions
  argmax_ids         : [N, T], top-1 token id at every teacher-forced position
  topk_ids           : [N, T, K], top-K token ids by log probability
  topk_logprobs      : [N, T, K], corresponding top-K log probabilities

It intentionally does not save full vocabulary logits because that would be
huge for Flan-T5-XL.  The analysis script compares gold-token likelihood,
top-1 agreement, and top-K overlap as compact logit-fidelity proxies.
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
    AUTO_OUTPUT_FIELDS,
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
        description="Extract teacher-forced T5 decoder hidden states and compact logit summaries.",
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
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_samples", type=int, default=128)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--image_field", default="image")
    p.add_argument("--text_field", default="auto")
    p.add_argument("--output_field", default="auto")
    p.add_argument("--max_txt_len", type=int, default=None)
    p.add_argument(
        "--max_output_len",
        type=int,
        default=None,
        help="Tokenizer max length for decoder targets. Defaults to model.max_txt_len.",
    )
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--fp32", action="store_true")
    p.add_argument(
        "--save_dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Storage dtype for decoder hidden/logprob arrays.",
    )
    return p.parse_args()


def masked_mean(x: Any, mask: Any) -> Any:
    denom = mask.to(x.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask.to(x.dtype).unsqueeze(-1)).sum(dim=1) / denom


def pool_decoder_layers(hidden_states: Sequence[Any], decoder_mask: Any, torch: Any) -> Any:
    layers = list(hidden_states[1:])
    pooled = [masked_mean(h.to(torch.float32), decoder_mask) for h in layers]
    return torch.stack(pooled, dim=1)


def run_text_only_encoder(model: Any, texts: Sequence[str], device: str, torch: Any, fp32: bool):
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
                return_dict=True,
            )
    return encoder_outputs.last_hidden_state, input_tokens.attention_mask


def target_tokens(model: Any, outputs: Sequence[str], device: str, max_len: int):
    return model.t5_tokenizer(
        list(outputs),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)


def to_numpy(x: Any, save_dtype: str) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    if save_dtype == "float16":
        return arr.astype(np.float16)
    return arr.astype(np.float32)


def main() -> None:
    args = parse_args()
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Missing runtime dependency: %s" % exc) from exc

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.input_mode == "multimodal" and not args.images_dir:
        raise SystemExit("--images_dir is required when --input_mode multimodal.")
    if args.top_k <= 0:
        raise SystemExit("--top_k must be positive.")
    ensure_dir(args.out_dir)

    rows, original_indices = select_rows(
        load_rows(os.path.abspath(args.calib_json)), args.max_samples, args.shuffle, args.seed
    )
    print(
        "label:", args.label,
        "| device:", args.device,
        "| samples:", len(rows),
        "| mode:", args.input_mode,
        "| ckpt:", args.ckpt or "pretrained",
    )

    model = load_blip2(args.model_name, args.model_type, args.device, args.ckpt, args.max_txt_len)
    if args.fp32:
        model.float()
    forward = EncoderForward(model, torch, padding="max_length", fp32=args.fp32)
    vis_processor = build_vis_processor(args.image_size)
    max_output_len = args.max_output_len or model.max_txt_len
    top_k = min(int(args.top_k), int(model.t5_model.config.vocab_size))

    decoder_layers: List[np.ndarray] = []
    gold_logprobs: List[np.ndarray] = []
    gold_logits: List[np.ndarray] = []
    decoder_masks: List[np.ndarray] = []
    argmax_ids: List[np.ndarray] = []
    topk_ids_rows: List[np.ndarray] = []
    topk_logprobs_rows: List[np.ndarray] = []
    seq_nll_rows: List[np.ndarray] = []
    sample_index: List[int] = []
    layer_count = None
    hidden_size = None

    for bi, (start, batch_rows) in enumerate(iter_batches(rows, args.batch_size)):
        input_texts = [
            extract_text(r, args.text_field, AUTO_INPUT_FIELDS, original_indices[start + i])
            for i, r in enumerate(batch_rows)
        ]
        output_texts = [
            extract_text(r, args.output_field, AUTO_OUTPUT_FIELDS, original_indices[start + i])
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
            enc = forward.run(images, input_texts, args.device)
            encoder_hidden = enc["encoder_hidden"]
            encoder_attention = enc["encoder_attention"]
        else:
            encoder_hidden, encoder_attention = run_text_only_encoder(
                model, input_texts, args.device, torch, args.fp32
            )

        output_tokens = target_tokens(model, output_texts, args.device, max_output_len)
        decoder_mask = output_tokens.attention_mask.bool()
        labels = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == model.t5_tokenizer.pad_token_id, -100
        )
        decoder_input_ids = model.t5_model._shift_right(labels)

        with torch.no_grad():
            amp = contextlib.nullcontext() if args.fp32 else model.maybe_autocast(dtype=torch.bfloat16)
            with amp:
                outputs = model.t5_model(
                    encoder_outputs=(encoder_hidden,),
                    attention_mask=encoder_attention,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=output_tokens.attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
            logits = outputs.logits.to(torch.float32)
            log_probs = torch.log_softmax(logits, dim=-1)
            gather_ids = output_tokens.input_ids.clamp_min(0).unsqueeze(-1)
            gold_lp = log_probs.gather(-1, gather_ids).squeeze(-1)
            gold_lg = logits.gather(-1, gather_ids).squeeze(-1)
            gold_lp = gold_lp.masked_fill(~decoder_mask, 0.0)
            gold_lg = gold_lg.masked_fill(~decoder_mask, 0.0)
            seq_nll = -(gold_lp * decoder_mask.to(gold_lp.dtype)).sum(dim=1)
            top_values, top_indices = torch.topk(log_probs, k=top_k, dim=-1)
            pooled = pool_decoder_layers(outputs.decoder_hidden_states, decoder_mask, torch)

        decoder_layers.append(to_numpy(pooled, args.save_dtype))
        gold_logprobs.append(to_numpy(gold_lp, args.save_dtype))
        gold_logits.append(to_numpy(gold_lg, args.save_dtype))
        decoder_masks.append(decoder_mask.detach().cpu().numpy().astype(np.uint8))
        argmax_ids.append(torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int32))
        topk_ids_rows.append(top_indices.detach().cpu().numpy().astype(np.int32))
        topk_logprobs_rows.append(to_numpy(top_values, args.save_dtype))
        seq_nll_rows.append(to_numpy(seq_nll, "float32"))
        if layer_count is None:
            layer_count = int(pooled.shape[1])
            hidden_size = int(pooled.shape[2])
        sample_index.extend(original_indices[start:start + len(batch_rows)])
        if bi % 5 == 0:
            print("  batch %d" % bi)

    arrays: Dict[str, np.ndarray] = {
        "decoder_layer": np.concatenate(decoder_layers, axis=0),
        "gold_logprob": np.concatenate(gold_logprobs, axis=0),
        "gold_logit": np.concatenate(gold_logits, axis=0),
        "decoder_mask": np.concatenate(decoder_masks, axis=0),
        "argmax_ids": np.concatenate(argmax_ids, axis=0),
        "topk_ids": np.concatenate(topk_ids_rows, axis=0),
        "topk_logprobs": np.concatenate(topk_logprobs_rows, axis=0),
        "sequence_nll": np.concatenate(seq_nll_rows, axis=0),
        "sample_index": np.asarray(sample_index, dtype=np.int64),
        "layer_index": np.arange(int(layer_count or 0), dtype=np.int64),
    }
    out_npz = os.path.join(args.out_dir, "t5_decoder_logits.npz")
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
                "output_field": args.output_field,
                "num_samples": len(sample_index),
                "num_decoder_layers": int(layer_count or 0),
                "hidden": int(hidden_size or 0),
                "max_output_len": int(max_output_len),
                "top_k": int(top_k),
                "save_dtype": args.save_dtype,
            },
            handle,
            indent=2,
        )
    print("[OK] wrote %d samples x %d decoder layers -> %s" % (len(sample_index), int(layer_count or 0), out_npz))


if __name__ == "__main__":
    main()
