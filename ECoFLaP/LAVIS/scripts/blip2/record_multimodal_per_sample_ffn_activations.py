#!/usr/bin/env python3
"""Record per-sample whole-model BLIP2-T5 activation distributions.

This script compares how different multimodal samples activate the full BLIP2-T5
path: ViT blocks, Q-Former layers, T5 encoder blocks, and optionally T5 decoder
blocks. It also records T5 encoder/decoder FFN intermediate neuron activations
by hooking the input to each ``DenseReluDense.wo`` projection.

Definitions:
  - block activation: layer output hidden states, summarized per sample.
  - T5 FFN neuron activation: DenseReluDense.wo pre-input, i.e. the FFN hidden
    neuron vector before projection back to model hidden size.
  - neuron coverage: for each module and token group, a sample activates a
    neuron when its mean absolute activation over that group's tokens is above
    median(sample means) + mad_multiplier * MAD(sample means).

For T5 encoder activations, visual-query positions and text-token positions are
also summarized separately. Decoder activations require output text; the script
automatically tries text_output, answer, caption, text, then question.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_LAVIS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _LAVIS_ROOT not in sys.path:
    sys.path.insert(0, _LAVIS_ROOT)


AUTO_INPUT_FIELDS = ("question", "caption", "text_input", "text", "prompt")
AUTO_OUTPUT_FIELDS = ("text_output", "answer", "caption", "text", "question")


@dataclass
class CaptureSpec:
    key: str
    component: str
    layer: int
    activation_kind: str
    module: Any
    hook_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record per-sample whole-model BLIP2-T5 activations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--calib_json", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="blip2_t5")
    parser.add_argument("--model_type", default="pretrain_flant5xl")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--image_field", default="image")
    parser.add_argument(
        "--text_field",
        default="auto",
        help="Encoder input text field. auto tries question/caption/text_input/text/prompt.",
    )
    parser.add_argument(
        "--output_field",
        default="auto",
        help="Decoder target field. auto tries text_output/answer/caption/text/question.",
    )
    parser.add_argument(
        "--no_decoder",
        action="store_true",
        help="Skip T5 decoder forward and decoder activation hooks.",
    )
    parser.add_argument(
        "--no_t5_ffn_neurons",
        action="store_true",
        help="Skip T5 DenseReluDense.wo pre-input neuron captures.",
    )
    parser.add_argument("--max_txt_len", type=int, default=None)
    parser.add_argument(
        "--mad_multiplier",
        type=float,
        default=3.0,
        help="Per-module neuron threshold = median + multiplier * MAD.",
    )
    parser.add_argument("--heatmap_max_samples", type=int, default=128)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_rows(path: str) -> List[Any]:
    if path.lower().endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError("%s must contain a non-empty JSON list or JSONL." % path)
    return rows


def select_rows(
    rows: Sequence[Any],
    max_samples: Optional[int],
    shuffle: bool,
    seed: int,
) -> Tuple[List[Any], List[int]]:
    indices = np.arange(len(rows))
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
    if max_samples is not None:
        indices = indices[:max_samples]
    return [rows[int(i)] for i in indices], [int(i) for i in indices]


def iter_batches(rows: Sequence[Any], batch_size: int) -> Iterable[Tuple[int, List[Any]]]:
    for start in range(0, len(rows), batch_size):
        yield start, list(rows[start : start + batch_size])


def resolve_image_path(images_dir: str, image_value: Any) -> str:
    path = os.path.expanduser(str(image_value))
    return path if os.path.isabs(path) else os.path.join(images_dir, path)


def value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value if v is not None).strip()
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values() if v is not None).strip()
    return str(value).strip()


def extract_text(row: Any, field: str, auto_fields: Sequence[str], row_index: int) -> Tuple[str, str]:
    if not isinstance(row, dict):
        raise TypeError("Row %d must be a JSON object." % row_index)
    fields = [field] if field != "auto" else list(auto_fields)
    selected = next((name for name in fields if name in row and value_to_text(row.get(name))), None)
    if selected is None:
        raise KeyError(
            "Row %d has none of these non-empty fields: %s"
            % (row_index, ", ".join(fields))
        )
    return value_to_text(row.get(selected)), selected


def tokenize_with_length_stats(tokenizer: Any, texts: Sequence[str], max_txt_len: int, device: str) -> Tuple[Any, List[int]]:
    full = tokenizer(
        list(texts),
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )
    original_lengths = [len(ids) for ids in full["input_ids"]]
    tokens = tokenizer(
        list(texts),
        padding="longest",
        truncation=True,
        max_length=max_txt_len,
        return_tensors="pt",
    ).to(device)
    return tokens, original_lengths


def first_tensor(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class WholeModelCapture:
    def __init__(self, specs: Sequence[CaptureSpec]):
        self.specs = list(specs)
        self.values: Dict[str, Any] = {}
        self.handles = []
        for spec in self.specs:
            if spec.hook_type == "pre":
                handle = spec.module.register_forward_pre_hook(self._make_pre_hook(spec.key))
            else:
                handle = spec.module.register_forward_hook(self._make_post_hook(spec.key))
            self.handles.append(handle)

    def _make_pre_hook(self, key: str):
        def hook(_module: Any, inputs: Tuple[Any, ...]) -> None:
            tensor = first_tensor(inputs)
            if tensor is not None:
                self.values[key] = tensor.detach()

        return hook

    def _make_post_hook(self, key: str):
        def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> None:
            tensor = first_tensor(output)
            if tensor is not None:
                self.values[key] = tensor.detach()

        return hook

    def clear(self) -> None:
        self.values = {}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def build_capture_specs(model: Any, include_decoder: bool, include_t5_ffn: bool) -> List[CaptureSpec]:
    specs: List[CaptureSpec] = []
    for i, block in enumerate(model.visual_encoder.blocks):
        specs.append(CaptureSpec("vit.block.%02d.output" % i, "vit", i, "block_output", block, "post"))
    for i, layer in enumerate(model.Qformer.bert.encoder.layer):
        specs.append(CaptureSpec("qformer.layer.%02d.output" % i, "qformer", i, "block_output", layer, "post"))
    for i, block in enumerate(model.t5_model.encoder.block):
        specs.append(CaptureSpec("t5.encoder.block.%02d.output" % i, "t5_encoder", i, "block_output", block, "post"))
        if include_t5_ffn:
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
    if include_decoder:
        for i, block in enumerate(model.t5_model.decoder.block):
            specs.append(CaptureSpec("t5.decoder.block.%02d.output" % i, "t5_decoder", i, "block_output", block, "post"))
            if include_t5_ffn:
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


def summarize_abs_values(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "mean_abs": 0.0,
            "std_abs": 0.0,
            "p50_abs": 0.0,
            "p90_abs": 0.0,
            "p95_abs": 0.0,
            "p99_abs": 0.0,
            "max_abs": 0.0,
        }
    return {
        "mean_abs": float(np.mean(values)),
        "std_abs": float(np.std(values)),
        "p50_abs": float(np.percentile(values, 50)),
        "p90_abs": float(np.percentile(values, 90)),
        "p95_abs": float(np.percentile(values, 95)),
        "p99_abs": float(np.percentile(values, 99)),
        "max_abs": float(np.max(values)),
    }


def safe_tensor_for_tokens(tensor: Any) -> Any:
    if tensor.dim() == 2:
        return tensor.unsqueeze(1)
    if tensor.dim() >= 3:
        return tensor.reshape(tensor.shape[0], -1, tensor.shape[-1])
    raise ValueError("Unsupported activation tensor shape: %s" % (tuple(tensor.shape),))


def summarize_sample_group(tensor: Any, mask: Any, sample_index: int) -> Tuple[Dict[str, float], np.ndarray, int]:
    tensor = safe_tensor_for_tokens(tensor)
    selected = tensor[sample_index][mask[sample_index].bool()]
    token_count = int(mask[sample_index].bool().sum().item())
    if selected.numel() == 0:
        neurons = np.zeros((tensor.shape[-1],), dtype=np.float32)
        return summarize_abs_values(np.zeros((0,), dtype=np.float32)), neurons, token_count
    abs_tensor = selected.detach().float().abs()
    neurons = abs_tensor.mean(dim=0).cpu().numpy().astype(np.float32)
    values = abs_tensor.reshape(-1).cpu().numpy().astype(np.float32)
    return summarize_abs_values(values), neurons, token_count


def masks_for_spec(spec: CaptureSpec, tensor: Any, runtime_masks: Dict[str, Any], torch: Any) -> List[Tuple[str, Any]]:
    tensor = safe_tensor_for_tokens(tensor)
    bsz, seq_len = int(tensor.shape[0]), int(tensor.shape[1])
    device = tensor.device
    if spec.component == "t5_encoder":
        return [
            ("all", runtime_masks["encoder_all"]),
            ("visual", runtime_masks["encoder_visual"]),
            ("text", runtime_masks["encoder_text"]),
        ]
    if spec.component == "t5_decoder":
        return [("decoder", runtime_masks["decoder_all"])]
    return [("all", torch.ones((bsz, seq_len), dtype=torch.bool, device=device))]


def collect_records(
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
    spec_by_key = {spec.key: spec for spec in specs}
    selected_input_fields: Dict[str, int] = {}
    selected_output_fields: Dict[str, int] = {}
    truncated_input_samples = 0
    truncated_output_samples = 0
    original_input_tokens = 0
    retained_input_tokens = 0
    original_output_tokens = 0
    retained_output_tokens = 0
    sample_meta: List[Dict[str, Any]] = []

    for batch_index, (start, batch_rows) in enumerate(iter_batches(rows, args.batch_size)):
        images = []
        input_texts = []
        output_texts = []
        image_values = []
        input_fields = []
        output_fields = []
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
            input_text, input_field = extract_text(row, args.text_field, AUTO_INPUT_FIELDS, original_index)
            output_text, output_field = extract_text(row, args.output_field, AUTO_OUTPUT_FIELDS, original_index)
            input_texts.append(input_text)
            output_texts.append(output_text)
            input_fields.append(input_field)
            output_fields.append(output_field)
            image_values.append(str(image_value))
            selected_input_fields[input_field] = selected_input_fields.get(input_field, 0) + 1
            selected_output_fields[output_field] = selected_output_fields.get(output_field, 0) + 1

        image_tensor = torch.stack(images).to(args.device)
        capture.clear()
        with torch.no_grad():
            with model.maybe_autocast():
                image_hidden = model.ln_vision(model.visual_encoder(image_tensor))
                image_atts = torch.ones(
                    image_hidden.size()[:-1],
                    dtype=torch.long,
                    device=image_hidden.device,
                )
                query_tokens = model.query_tokens.expand(image_hidden.shape[0], -1, -1)
                query_output = model.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_hidden,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                visual_tokens = model.t5_proj(query_output.last_hidden_state)

            with model.maybe_autocast(dtype=torch.bfloat16):
                input_tokens, input_original_lengths = tokenize_with_length_stats(
                    model.t5_tokenizer,
                    input_texts,
                    model.max_txt_len,
                    args.device,
                )
                visual_attention = torch.ones(
                    visual_tokens.size()[:-1],
                    dtype=torch.long,
                    device=visual_tokens.device,
                )
                encoder_attention = torch.cat([visual_attention, input_tokens.attention_mask], dim=1)
                input_embeddings = model.t5_model.encoder.embed_tokens(input_tokens.input_ids)
                encoder_embeddings = torch.cat([visual_tokens, input_embeddings], dim=1)
                num_query = int(visual_tokens.shape[1])
                encoder_visual = torch.zeros_like(encoder_attention, dtype=torch.bool)
                encoder_visual[:, :num_query] = True
                encoder_text = torch.zeros_like(encoder_attention, dtype=torch.bool)
                encoder_text[:, num_query:] = input_tokens.attention_mask.bool()
                model.temp_label = encoder_visual

                if args.no_decoder:
                    model.t5_model.encoder(
                        inputs_embeds=encoder_embeddings,
                        attention_mask=encoder_attention,
                        return_dict=True,
                    )
                    output_original_lengths = [0 for _ in batch_rows]
                    output_retained_lengths = [0 for _ in batch_rows]
                    decoder_attention = None
                else:
                    output_tokens, output_original_lengths = tokenize_with_length_stats(
                        model.t5_tokenizer,
                        output_texts,
                        model.max_txt_len,
                        args.device,
                    )
                    targets = output_tokens.input_ids.masked_fill(
                        output_tokens.input_ids == model.t5_tokenizer.pad_token_id,
                        -100,
                    )
                    model.t5_model(
                        inputs_embeds=encoder_embeddings,
                        attention_mask=encoder_attention,
                        decoder_attention_mask=output_tokens.attention_mask,
                        labels=targets,
                        return_dict=True,
                    )
                    decoder_attention = output_tokens.attention_mask.bool()
                    output_retained_lengths = decoder_attention.detach().sum(dim=1).cpu().tolist()

        input_retained_lengths = input_tokens.attention_mask.detach().sum(dim=1).cpu().tolist()
        original_input_tokens += int(sum(input_original_lengths))
        retained_input_tokens += int(sum(input_retained_lengths))
        original_output_tokens += int(sum(output_original_lengths))
        retained_output_tokens += int(sum(output_retained_lengths))
        truncated_input_samples += sum(
            int(o > r) for o, r in zip(input_original_lengths, input_retained_lengths)
        )
        truncated_output_samples += sum(
            int(o > r) for o, r in zip(output_original_lengths, output_retained_lengths)
        )

        runtime_masks = {
            "encoder_all": encoder_attention.bool(),
            "encoder_visual": encoder_visual,
            "encoder_text": encoder_text,
        }
        if decoder_attention is not None:
            runtime_masks["decoder_all"] = decoder_attention

        for local_index, _row in enumerate(batch_rows):
            sample_id = len(sample_meta)
            sample_meta.append(
                {
                    "sample_id": sample_id,
                    "row_index": int(original_indices[start + local_index]),
                    "image": image_values[local_index],
                    "input_field": input_fields[local_index],
                    "output_field": output_fields[local_index],
                    "input_preview": input_texts[local_index].replace("\n", " ")[:180],
                    "output_preview": output_texts[local_index].replace("\n", " ")[:120],
                    "input_original_tokens": int(input_original_lengths[local_index]),
                    "input_retained_tokens": int(input_retained_lengths[local_index]),
                    "input_truncated": bool(input_original_lengths[local_index] > input_retained_lengths[local_index]),
                    "output_original_tokens": int(output_original_lengths[local_index]),
                    "output_retained_tokens": int(output_retained_lengths[local_index]),
                    "output_truncated": bool(output_original_lengths[local_index] > output_retained_lengths[local_index]),
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
                if array_key not in arrays:
                    arrays[array_key] = []
                for local_index in range(len(batch_rows)):
                    stats, neuron_vector, token_count = summarize_sample_group(tensor, mask, local_index)
                    arrays[array_key].append(neuron_vector)
                    sample_id = start + local_index
                    record = {
                        "sample_id": int(sample_id),
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
            print("Processed %d/%d samples" % (min(start + len(batch_rows), len(rows)), len(rows)))

    metadata = {
        "num_samples": len(sample_meta),
        "num_modules": len(specs),
        "include_decoder": not args.no_decoder,
        "include_t5_ffn_neurons": not args.no_t5_ffn_neurons,
        "model_max_txt_len": int(model.max_txt_len),
        "selected_input_fields": selected_input_fields,
        "selected_output_fields": selected_output_fields,
        "truncated_input_samples": int(truncated_input_samples),
        "truncated_input_sample_rate": float(truncated_input_samples / max(1, len(sample_meta))),
        "truncated_output_samples": int(truncated_output_samples),
        "truncated_output_sample_rate": float(truncated_output_samples / max(1, len(sample_meta))),
        "original_input_tokens": int(original_input_tokens),
        "retained_input_tokens": int(retained_input_tokens),
        "original_output_tokens": int(original_output_tokens),
        "retained_output_tokens": int(retained_output_tokens),
        "sample_metadata": sample_meta,
    }
    return records, arrays, metadata


def add_neuron_coverage(records: List[Dict[str, Any]], arrays: Dict[str, List[np.ndarray]], mad_multiplier: float) -> Dict[str, np.ndarray]:
    record_positions: Dict[str, List[int]] = {}
    for index, row in enumerate(records):
        record_positions.setdefault(row["array_key"], []).append(index)

    saved_arrays: Dict[str, np.ndarray] = {}
    for key, chunks in arrays.items():
        matrix = np.stack(chunks, axis=0).astype(np.float32)
        median = np.median(matrix, axis=0)
        mad = np.median(np.abs(matrix - median), axis=0)
        threshold = median + mad_multiplier * mad
        active = matrix > threshold[None, :]
        coverage = active.mean(axis=1)
        for rec_index, cov in zip(record_positions.get(key, []), coverage):
            records[rec_index]["neuron_coverage"] = float(cov)
            records[rec_index]["active_neuron_count"] = int(round(float(cov) * matrix.shape[1]))
            records[rec_index]["num_neurons"] = int(matrix.shape[1])
        saved_arrays[sanitize_key(key) + "__mean_abs_per_neuron"] = matrix
        saved_arrays[sanitize_key(key) + "__active"] = active
        saved_arrays[sanitize_key(key) + "__threshold"] = threshold.astype(np.float32)
    return saved_arrays


def sanitize_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "__", key).strip("_")


def write_csv(records: Sequence[Dict[str, Any]], path: str) -> None:
    fieldnames = [
        "sample_id",
        "row_index",
        "module_key",
        "component",
        "layer",
        "activation_kind",
        "token_group",
        "token_count",
        "num_neurons",
        "active_neuron_count",
        "neuron_coverage",
        "mean_abs",
        "std_abs",
        "p50_abs",
        "p90_abs",
        "p95_abs",
        "p99_abs",
        "max_abs",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({name: row.get(name) for name in fieldnames})


def setup_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except (ImportError, RuntimeError) as exc:
        raise SystemExit("A compatible matplotlib/NumPy installation is required.") from exc


def pivot_matrix(records: Sequence[Dict[str, Any]], component: str, activation_kind: str, token_group: str, value_key: str) -> Tuple[List[str], np.ndarray]:
    rows = [
        row
        for row in records
        if row["component"] == component
        and row["activation_kind"] == activation_kind
        and row["token_group"] == token_group
    ]
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    sample_ids = sorted({int(row["sample_id"]) for row in rows})
    module_keys = sorted({row["module_key"] for row in rows})
    sample_to_col = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    module_to_row = {module_key: idx for idx, module_key in enumerate(module_keys)}
    matrix = np.full((len(module_keys), len(sample_ids)), np.nan, dtype=np.float32)
    for row in rows:
        matrix[module_to_row[row["module_key"]], sample_to_col[int(row["sample_id"])]] = float(row[value_key])
    return module_keys, matrix


def plot_matrix(plt: Any, matrix: np.ndarray, ylabels: Sequence[str], title: str, path: str, max_samples: int) -> None:
    if matrix.size == 0:
        return
    view = matrix[:, :max_samples]
    fig_height = max(4.8, min(18.0, 0.26 * len(ylabels) + 2.8))
    fig, ax = plt.subplots(figsize=(11.2, fig_height))
    im = ax.imshow(view, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Module / layer")
    ax.set_title(title)
    if len(ylabels) <= 80:
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=6)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_plots(out_dir: str, records: Sequence[Dict[str, Any]], max_samples: int) -> List[str]:
    plt = setup_matplotlib()
    saved: List[str] = []
    combos = [
        ("vit", "block_output", "all"),
        ("qformer", "block_output", "all"),
        ("t5_encoder", "block_output", "visual"),
        ("t5_encoder", "block_output", "text"),
        ("t5_encoder", "ffn_neuron", "visual"),
        ("t5_encoder", "ffn_neuron", "text"),
        ("t5_decoder", "block_output", "decoder"),
        ("t5_decoder", "ffn_neuron", "decoder"),
    ]
    for component, kind, group in combos:
        for value_key in ("mean_abs", "neuron_coverage"):
            labels, matrix = pivot_matrix(records, component, kind, group, value_key)
            if matrix.size == 0:
                continue
            path = os.path.join(
                out_dir,
                "%s_%s_%s_%s_heatmap.png" % (component, kind, group, value_key),
            )
            plot_matrix(
                plt,
                matrix,
                labels,
                "%s %s %s %s" % (component, kind, group, value_key),
                path,
                max_samples,
            )
            saved.append(path)

    profile_rows = [
        row
        for row in records
        if row["activation_kind"] == "block_output" and row["token_group"] in ("all", "visual", "text", "decoder")
    ]
    if profile_rows:
        sample_ids = sorted({int(row["sample_id"]) for row in profile_rows})
        features_by_sample: Dict[int, List[float]] = {i: [] for i in sample_ids}
        for row in sorted(profile_rows, key=lambda r: (r["component"], r["layer"], r["token_group"], r["sample_id"])):
            features_by_sample[int(row["sample_id"])].append(float(row["mean_abs"]))
            features_by_sample[int(row["sample_id"])].append(float(row["neuron_coverage"]))
        min_len = min(len(v) for v in features_by_sample.values())
        features = np.asarray([features_by_sample[i][:min_len] for i in sample_ids], dtype=np.float32)
        centered = features - features.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(centered, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        sim = (centered / denom) @ (centered / denom).T
        path = os.path.join(out_dir, "whole_model_sample_activation_profile_similarity.png")
        plot_matrix(plt, sim, [str(i) for i in sample_ids], "Sample Similarity From Whole-Model Activation Profiles", path, max_samples)
        saved.append(path)

    return saved


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max_samples must be >= 1")
    if args.max_txt_len is not None and args.max_txt_len < 2:
        raise ValueError("--max_txt_len must be >= 2")
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
    print("model.max_txt_len:", model.max_txt_len)

    specs = build_capture_specs(
        model,
        include_decoder=not args.no_decoder,
        include_t5_ffn=not args.no_t5_ffn_neurons,
    )
    print("captured modules:", len(specs))
    capture = WholeModelCapture(specs)
    vis_processor = load_processor("blip_image_eval").build(image_size=args.image_size)
    try:
        records, arrays, metadata = collect_records(
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
    csv_path = os.path.join(out_dir, "whole_model_per_sample_activation_summary.csv")
    json_path = os.path.join(out_dir, "whole_model_per_sample_activation_summary.json")
    npz_path = os.path.join(out_dir, "whole_model_per_sample_activation_arrays.npz")
    write_csv(records, csv_path)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "calib_json": os.path.abspath(args.calib_json),
                "images_dir": os.path.abspath(args.images_dir),
                "checkpoint": os.path.abspath(args.ckpt),
                "input_text_field": args.text_field,
                "output_text_field": args.output_field,
                "whole_model_components": ["vit", "qformer", "t5_encoder"] + ([] if args.no_decoder else ["t5_decoder"]),
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
    np.savez_compressed(npz_path, **saved_arrays)
    plot_paths = make_plots(out_dir, records, args.heatmap_max_samples)

    if metadata["truncated_input_samples"]:
        print(
            "[WARN] input text truncated for %d/%d samples at max_txt_len=%d"
            % (metadata["truncated_input_samples"], metadata["num_samples"], metadata["model_max_txt_len"])
        )
    if metadata["truncated_output_samples"]:
        print(
            "[WARN] output text truncated for %d/%d samples at max_txt_len=%d"
            % (metadata["truncated_output_samples"], metadata["num_samples"], metadata["model_max_txt_len"])
        )
    for path in (csv_path, json_path, npz_path, *plot_paths):
        print("[OK] wrote:", path)


if __name__ == "__main__":
    main()
