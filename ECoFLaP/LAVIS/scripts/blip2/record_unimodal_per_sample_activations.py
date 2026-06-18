#!/usr/bin/env python3
"""Record per-sample unimodal BLIP2-T5 activation distributions.

Two intentionally single-modality paths are supported:

  - t5_text_only: C4 or other text rows go only through Flan-T5. No image,
    ViT, Q-Former, or BLIP2 visual query is used.
  - vit_image_only: image rows go only through BLIP2's visual encoder and
    ln_vision. No Q-Former or T5 is used.

The output format mirrors record_multimodal_per_sample_ffn_activations.py:
CSV/JSON summaries, NPZ neuron arrays, and per-layer heatmaps. This makes the
single-modality runs directly comparable with multimodal runs at the same
module/layer granularity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LAVIS_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _LAVIS_ROOT not in sys.path:
    sys.path.insert(0, _LAVIS_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from record_multimodal_per_sample_ffn_activations import (  # noqa: E402
    CaptureSpec,
    WholeModelCapture,
    add_neuron_coverage,
    ensure_dir,
    iter_batches,
    load_rows,
    make_plots,
    resolve_image_path,
    sanitize_key,
    select_rows,
    summarize_sample_group,
    tokenize_with_length_stats,
    value_to_text,
    write_csv,
)


AUTO_TEXT_FIELDS = ("text", "caption", "text_input", "output", "question", "prompt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record per-sample unimodal BLIP2-T5 activations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_mode",
        choices=["t5_text_only", "vit_image_only"],
        required=True,
        help="Single-modality path to run.",
    )
    parser.add_argument("--calib_json", required=True)
    parser.add_argument(
        "--images_dir",
        default=None,
        help="Required for vit_image_only. Ignored for t5_text_only.",
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="blip2_t5")
    parser.add_argument("--model_type", default="pretrain_flant5xl")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--image_field", default="image")
    parser.add_argument(
        "--text_field",
        default="auto",
        help="Text field for C4 dictionaries. auto tries text/caption/text_input/output/question/prompt.",
    )
    parser.add_argument(
        "--no_decoder",
        action="store_true",
        help="For t5_text_only, record only T5 encoder activations.",
    )
    parser.add_argument(
        "--no_t5_ffn_neurons",
        action="store_true",
        help="For t5_text_only, skip DenseReluDense.wo pre-input FFN neuron captures.",
    )
    parser.add_argument(
        "--no_vit_ln",
        action="store_true",
        help="For vit_image_only, do not record ln_vision output as the last visual-token layer.",
    )
    parser.add_argument("--max_txt_len", type=int, default=None)
    parser.add_argument(
        "--mad_multiplier",
        type=float,
        default=3.0,
        help="Per-module neuron threshold = median + multiplier * MAD.",
    )
    parser.add_argument("--heatmap_max_samples", type=int, default=128)
    parser.add_argument("--log_every", type=int, default=20)
    return parser.parse_args()


def extract_text(row: Any, field: str, row_index: int) -> Tuple[str, str]:
    if isinstance(row, str):
        text = row.strip()
        if not text:
            raise ValueError("Row %d is empty text." % row_index)
        return text, "string"
    if not isinstance(row, dict):
        raise TypeError("Row %d must be a string or JSON object for text-only mode." % row_index)
    fields = [field] if field != "auto" else list(AUTO_TEXT_FIELDS)
    selected = next((name for name in fields if name in row and value_to_text(row.get(name))), None)
    if selected is None:
        raise KeyError(
            "Row %d has none of these non-empty text fields: %s"
            % (row_index, ", ".join(fields))
        )
    return value_to_text(row.get(selected)), selected


def build_unimodal_capture_specs(args: argparse.Namespace, model: Any) -> List[CaptureSpec]:
    specs: List[CaptureSpec] = []
    if args.input_mode == "vit_image_only":
        for i, block in enumerate(model.visual_encoder.blocks):
            specs.append(
                CaptureSpec("vit.block.%02d.output" % i, "vit", i, "block_output", block, "post")
            )
        if not args.no_vit_ln:
            specs.append(
                CaptureSpec(
                    "vit.ln_vision.output",
                    "vit",
                    len(model.visual_encoder.blocks),
                    "block_output",
                    model.ln_vision,
                    "post",
                )
            )
        return specs

    for i, block in enumerate(model.t5_model.encoder.block):
        specs.append(
            CaptureSpec("t5.encoder.block.%02d.output" % i, "t5_encoder", i, "block_output", block, "post")
        )
        if not args.no_t5_ffn_neurons:
            specs.append(
                CaptureSpec(
                    "t5.encoder.ffn.%02d.neuron" % i,
                    "t5_encoder",
                    i,
                    "ffn_neuron",
                    block.layer[-1].DenseReluDense.wo,
                    "pre",
                )
            )
    if not args.no_decoder:
        for i, block in enumerate(model.t5_model.decoder.block):
            specs.append(
                CaptureSpec("t5.decoder.block.%02d.output" % i, "t5_decoder", i, "block_output", block, "post")
            )
            if not args.no_t5_ffn_neurons:
                specs.append(
                    CaptureSpec(
                        "t5.decoder.ffn.%02d.neuron" % i,
                        "t5_decoder",
                        i,
                        "ffn_neuron",
                        block.layer[-1].DenseReluDense.wo,
                        "pre",
                    )
                )
    return specs


def masks_for_spec(spec: CaptureSpec, tensor: Any, runtime_masks: Dict[str, Any], torch: Any) -> List[Tuple[str, Any]]:
    if tensor.dim() == 2:
        seq_len = 1
        bsz = int(tensor.shape[0])
    else:
        bsz = int(tensor.shape[0])
        seq_len = int(np.prod(tensor.shape[1:-1])) if tensor.dim() > 2 else 1
    device = tensor.device
    if spec.component == "t5_encoder":
        return [("text", runtime_masks["encoder_text"])]
    if spec.component == "t5_decoder":
        return [("decoder", runtime_masks["decoder_all"])]
    return [("all", torch.ones((bsz, seq_len), dtype=torch.bool, device=device))]


def collect_text_only_records(
    model: Any,
    rows: Sequence[Any],
    original_indices: Sequence[int],
    capture: WholeModelCapture,
    specs: Sequence[CaptureSpec],
    args: argparse.Namespace,
    torch: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[np.ndarray]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    arrays: Dict[str, List[np.ndarray]] = {}
    selected_text_fields: Dict[str, int] = {}
    truncated_samples = 0
    original_tokens = 0
    retained_tokens = 0
    sample_meta: List[Dict[str, Any]] = []

    for batch_index, (start, batch_rows) in enumerate(iter_batches(rows, args.batch_size)):
        texts: List[str] = []
        text_fields: List[str] = []
        for local_index, row in enumerate(batch_rows):
            original_index = original_indices[start + local_index]
            text, text_field = extract_text(row, args.text_field, original_index)
            texts.append(text)
            text_fields.append(text_field)
            selected_text_fields[text_field] = selected_text_fields.get(text_field, 0) + 1

        capture.clear()
        with torch.no_grad():
            with model.maybe_autocast(dtype=torch.bfloat16):
                input_tokens, original_lengths = tokenize_with_length_stats(
                    model.t5_tokenizer,
                    texts,
                    model.max_txt_len,
                    args.device,
                )
                text_embeddings = model.t5_model.encoder.embed_tokens(input_tokens.input_ids)
                encoder_attention = input_tokens.attention_mask
                model.temp_label = torch.zeros_like(encoder_attention, dtype=torch.bool)
                if args.no_decoder:
                    model.t5_model.encoder(
                        inputs_embeds=text_embeddings,
                        attention_mask=encoder_attention,
                        return_dict=True,
                    )
                    decoder_attention = None
                else:
                    target_tokens, _target_original_lengths = tokenize_with_length_stats(
                        model.t5_tokenizer,
                        texts,
                        model.max_txt_len,
                        args.device,
                    )
                    targets = target_tokens.input_ids.masked_fill(
                        target_tokens.input_ids == model.t5_tokenizer.pad_token_id,
                        -100,
                    )
                    model.t5_model(
                        inputs_embeds=text_embeddings,
                        attention_mask=encoder_attention,
                        decoder_attention_mask=target_tokens.attention_mask,
                        labels=targets,
                        return_dict=True,
                    )
                    decoder_attention = target_tokens.attention_mask.bool()

        retained_lengths = input_tokens.attention_mask.detach().sum(dim=1).cpu().tolist()
        original_tokens += int(sum(original_lengths))
        retained_tokens += int(sum(retained_lengths))
        truncated_samples += sum(int(o > r) for o, r in zip(original_lengths, retained_lengths))
        runtime_masks = {"encoder_text": encoder_attention.bool()}
        if decoder_attention is not None:
            runtime_masks["decoder_all"] = decoder_attention

        for local_index, text in enumerate(texts):
            sample_meta.append(
                {
                    "sample_id": int(start + local_index),
                    "row_index": int(original_indices[start + local_index]),
                    "text_field": text_fields[local_index],
                    "text_preview": text.replace("\n", " ")[:180],
                    "original_tokens": int(original_lengths[local_index]),
                    "retained_tokens": int(retained_lengths[local_index]),
                    "truncated": bool(original_lengths[local_index] > retained_lengths[local_index]),
                }
            )

        for spec in specs:
            tensor = capture.values.get(spec.key)
            if tensor is None:
                if spec.component == "t5_decoder" and args.no_decoder:
                    continue
                raise RuntimeError("Missing captured activation: %s" % spec.key)
            for token_group, mask in masks_for_spec(spec, tensor, runtime_masks, torch):
                array_key = "%s|%s" % (spec.key, token_group)
                arrays.setdefault(array_key, [])
                for local_index in range(len(batch_rows)):
                    stats, neuron_vector, token_count = summarize_sample_group(tensor, mask, local_index)
                    arrays[array_key].append(neuron_vector)
                    record = {
                        "sample_id": int(start + local_index),
                        "row_index": int(original_indices[start + local_index]),
                        "module_key": spec.key,
                        "array_key": array_key,
                        "component": spec.component,
                        "layer": spec.layer,
                        "activation_kind": spec.activation_kind,
                        "token_group": token_group,
                        "token_count": token_count,
                    }
                    record.update(stats)
                    records.append(record)

        if args.log_every > 0 and (batch_index + 1) % args.log_every == 0:
            print("Processed %d/%d text samples" % (min(start + len(batch_rows), len(rows)), len(rows)))

    metadata = {
        "input_mode": args.input_mode,
        "num_samples": len(sample_meta),
        "num_modules": len(specs),
        "include_decoder": not args.no_decoder,
        "include_t5_ffn_neurons": not args.no_t5_ffn_neurons,
        "model_max_txt_len": int(model.max_txt_len),
        "selected_text_fields": selected_text_fields,
        "truncated_samples": int(truncated_samples),
        "truncated_sample_rate": float(truncated_samples / max(1, len(sample_meta))),
        "original_tokens": int(original_tokens),
        "retained_tokens": int(retained_tokens),
        "sample_metadata": sample_meta,
    }
    return records, arrays, metadata


def collect_vit_image_only_records(
    model: Any,
    rows: Sequence[Any],
    original_indices: Sequence[int],
    images_dir: str,
    vis_processor: Any,
    capture: WholeModelCapture,
    specs: Sequence[CaptureSpec],
    args: argparse.Namespace,
    torch: Any,
    Image: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[np.ndarray]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    arrays: Dict[str, List[np.ndarray]] = {}
    sample_meta: List[Dict[str, Any]] = []

    for batch_index, (start, batch_rows) in enumerate(iter_batches(rows, args.batch_size)):
        images = []
        image_values: List[str] = []
        for local_index, row in enumerate(batch_rows):
            original_index = original_indices[start + local_index]
            if not isinstance(row, dict) or args.image_field not in row:
                raise KeyError("Row %d is missing image field %r." % (original_index, args.image_field))
            image_value = row[args.image_field]
            image_path = resolve_image_path(images_dir, image_value)
            if not os.path.isfile(image_path):
                raise FileNotFoundError("Image not found for row %d: %s" % (original_index, image_path))
            with Image.open(image_path) as image:
                images.append(vis_processor(image.convert("RGB")))
            image_values.append(str(image_value))

        image_tensor = torch.stack(images).to(args.device)
        capture.clear()
        with torch.no_grad():
            with model.maybe_autocast():
                model.ln_vision(model.visual_encoder(image_tensor))

        for local_index, image_value in enumerate(image_values):
            sample_meta.append(
                {
                    "sample_id": int(start + local_index),
                    "row_index": int(original_indices[start + local_index]),
                    "image": image_value,
                }
            )

        runtime_masks: Dict[str, Any] = {}
        for spec in specs:
            tensor = capture.values.get(spec.key)
            if tensor is None:
                raise RuntimeError("Missing captured activation: %s" % spec.key)
            for token_group, mask in masks_for_spec(spec, tensor, runtime_masks, torch):
                array_key = "%s|%s" % (spec.key, token_group)
                arrays.setdefault(array_key, [])
                for local_index in range(len(batch_rows)):
                    stats, neuron_vector, token_count = summarize_sample_group(tensor, mask, local_index)
                    arrays[array_key].append(neuron_vector)
                    record = {
                        "sample_id": int(start + local_index),
                        "row_index": int(original_indices[start + local_index]),
                        "module_key": spec.key,
                        "array_key": array_key,
                        "component": spec.component,
                        "layer": spec.layer,
                        "activation_kind": spec.activation_kind,
                        "token_group": token_group,
                        "token_count": token_count,
                    }
                    record.update(stats)
                    records.append(record)

        if args.log_every > 0 and (batch_index + 1) % args.log_every == 0:
            print("Processed %d/%d image samples" % (min(start + len(batch_rows), len(rows)), len(rows)))

    metadata = {
        "input_mode": args.input_mode,
        "num_samples": len(sample_meta),
        "num_modules": len(specs),
        "include_vit_ln": not args.no_vit_ln,
        "sample_metadata": sample_meta,
    }
    return records, arrays, metadata


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max_samples must be >= 1")
    if args.max_txt_len is not None and args.max_txt_len < 2:
        raise ValueError("--max_txt_len must be >= 2")
    if args.input_mode == "vit_image_only" and not args.images_dir:
        raise ValueError("--images_dir is required for --input_mode vit_image_only")

    try:
        import torch
        from PIL import Image
        from lavis.models import load_model
        from lavis.processors import load_processor
    except ImportError as exc:
        raise SystemExit("Missing LAVIS runtime dependency: %s" % exc) from exc

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)
    rows, original_indices = select_rows(
        load_rows(os.path.abspath(args.calib_json)),
        args.max_samples,
        args.shuffle,
        args.seed,
    )

    print("input_mode:", args.input_mode)
    print("device:", args.device)
    print("samples:", len(rows))
    model = load_model(
        args.model_name,
        args.model_type,
        is_eval=True,
        device=args.device,
        checkpoint=args.ckpt,
    )
    model.eval()
    if args.max_txt_len is not None:
        model.max_txt_len = args.max_txt_len
    if args.input_mode == "t5_text_only":
        print("model.max_txt_len:", model.max_txt_len)

    specs = build_unimodal_capture_specs(args, model)
    print("captured modules:", len(specs))
    capture = WholeModelCapture(specs)
    try:
        if args.input_mode == "t5_text_only":
            records, arrays, metadata = collect_text_only_records(
                model,
                rows,
                original_indices,
                capture,
                specs,
                args,
                torch,
            )
        else:
            vis_processor = load_processor("blip_image_eval").build(image_size=args.image_size)
            records, arrays, metadata = collect_vit_image_only_records(
                model,
                rows,
                original_indices,
                os.path.abspath(args.images_dir),
                vis_processor,
                capture,
                specs,
                args,
                torch,
                Image,
            )
    finally:
        capture.close()

    saved_arrays = add_neuron_coverage(records, arrays, args.mad_multiplier)
    csv_path = os.path.join(out_dir, "unimodal_per_sample_activation_summary.csv")
    json_path = os.path.join(out_dir, "unimodal_per_sample_activation_summary.json")
    npz_path = os.path.join(out_dir, "unimodal_per_sample_activation_arrays.npz")
    write_csv(records, csv_path)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "calib_json": os.path.abspath(args.calib_json),
                "images_dir": os.path.abspath(args.images_dir) if args.images_dir else None,
                "checkpoint": os.path.abspath(args.ckpt),
                "input_mode": args.input_mode,
                "activation_definition": {
                    "block_output": "module forward output hidden states",
                    "ffn_neuron": "T5 DenseReluDense.wo pre-input intermediate FFN neuron activation",
                },
                "neuron_active_definition": "sample mean absolute activation over tokens > median + mad_multiplier * MAD for that module/group/neuron",
                "mad_multiplier": float(args.mad_multiplier),
                "metadata": metadata,
                "records": records,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    for key, chunks in arrays.items():
        saved_arrays[sanitize_key(key) + "__sample_count"] = np.asarray([len(chunks)], dtype=np.int64)
    np.savez_compressed(npz_path, **saved_arrays)
    plot_paths = make_plots(out_dir, records, args.heatmap_max_samples)

    if metadata.get("truncated_samples"):
        print(
            "[WARN] text truncated for %d/%d samples at max_txt_len=%d"
            % (
                metadata["truncated_samples"],
                metadata["num_samples"],
                metadata.get("model_max_txt_len", -1),
            )
        )
    for path in (csv_path, json_path, npz_path, *plot_paths):
        print("[OK] wrote:", path)


if __name__ == "__main__":
    main()
