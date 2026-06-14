#!/usr/bin/env python3
"""Analyze visual-position and text-position T5 FFN activations.

The input is one multimodal dataset whose rows contain an image and either a
question or caption. Each batch follows the normal BLIP2 encoder path:

    image -> ViT -> Q-Former -> t5_proj
    [visual query tokens, text tokens] -> T5 encoder

For every T5 encoder layer, the script captures the input to the FFN ``wo``
projection. In gated Flan-T5 blocks this is the true intermediate FFN
activation:

    GELU(wi_0(RMSNorm(x))) * wi_1(RMSNorm(x))

The first ``num_query_tokens`` sequence positions are analyzed as visual
positions and the remaining non-padding positions as text positions. These are
position-attributed multimodal activations: after each layer's self-attention,
both position groups may already contain information from both modalities.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_LAVIS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _LAVIS_ROOT not in sys.path:
    sys.path.insert(0, _LAVIS_ROOT)


AUTO_TEXT_FIELDS = ("question", "caption", "text_input", "text", "prompt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze layer-wise T5 encoder FFN neuron activation for one "
            "multimodal image-text dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--calib_json",
        required=True,
        help="JSON list or JSONL containing image-text samples.",
    )
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="blip2_t5")
    parser.add_argument("--model_type", default="pretrain_flant5xl")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--image_field", default="image")
    parser.add_argument(
        "--text_field",
        default="auto",
        help=(
            "Input text field. auto tries: %s. The answer field is never used."
            % ", ".join(AUTO_TEXT_FIELDS)
        ),
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--max_txt_len",
        type=int,
        default=None,
        help=(
            "T5 text-token limit. None keeps the model default (normally 32). "
            "Use a larger value for long MMBench/MMMU/MathVista prompts."
        ),
    )
    parser.add_argument(
        "--tokens_per_group",
        type=int,
        default=512,
        help=(
            "Balanced number of visual-position and text-position tokens kept "
            "for every layer."
        ),
    )
    parser.add_argument(
        "--mad_multiplier",
        type=float,
        default=3.0,
        help="Per-neuron threshold: median(|a|) + multiplier * MAD(|a|).",
    )
    parser.add_argument(
        "--min_activation_rate",
        type=float,
        default=0.2,
        help=(
            "A neuron is covered when this fraction of sampled tokens exceeds "
            "its shared per-neuron threshold."
        ),
    )
    parser.add_argument("--frequency_bins", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=20)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_rows(path: str) -> List[Any]:
    if path.lower().endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "%s line %d is invalid JSON." % (path, line_number)
                    ) from exc
    else:
        with open(path, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
    if not isinstance(rows, list):
        raise TypeError("%s must contain a JSON list or JSONL rows." % path)
    if not rows:
        raise ValueError("%s contains no rows." % path)
    return rows


def shuffled_rows(
    rows: Sequence[Any],
    max_samples: Optional[int],
    seed: int,
) -> List[Any]:
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(rows))
    if max_samples is not None:
        indices = indices[:max_samples]
    return [rows[int(index)] for index in indices]


def iter_batches(
    rows: Sequence[Any],
    batch_size: int,
) -> Iterable[Tuple[int, List[Any]]]:
    for start in range(0, len(rows), batch_size):
        yield start, list(rows[start : start + batch_size])


def resolve_image_path(images_dir: str, image_value: Any) -> str:
    path = os.path.expanduser(str(image_value))
    return path if os.path.isabs(path) else os.path.join(images_dir, path)


def extract_text(
    row: Any,
    text_field: str,
    row_index: int,
) -> Tuple[str, str]:
    if not isinstance(row, dict):
        raise TypeError("Row %d must be a JSON object." % row_index)
    fields = [text_field] if text_field != "auto" else list(AUTO_TEXT_FIELDS)
    selected = next((field for field in fields if field in row), None)
    if selected is None:
        raise KeyError(
            "Row %d has none of these text fields: %s"
            % (row_index, ", ".join(fields))
        )
    text = str(row.get(selected, "")).strip()
    if not text:
        raise ValueError("Row %d has empty text field %r." % (row_index, selected))
    return text, selected


def get_ffn_wo_modules(model: Any) -> List[Any]:
    modules = []
    for layer_index, block in enumerate(model.t5_model.encoder.block):
        ffn = block.layer[-1]
        dense = getattr(ffn, "DenseReluDense", None)
        wo = getattr(dense, "wo", None)
        if wo is None:
            raise AttributeError(
                "T5 encoder layer %d has no DenseReluDense.wo module."
                % layer_index
            )
        modules.append(wo)
    return modules


class FFNActivationCapture:
    def __init__(self, modules: Sequence[Any]):
        self.values: List[Any] = [None] * len(modules)
        self.handles = [
            module.register_forward_pre_hook(self._make_hook(index))
            for index, module in enumerate(modules)
        ]

    def _make_hook(self, index: int):
        def hook(_module: Any, inputs: Tuple[Any, ...]) -> None:
            if not inputs:
                raise RuntimeError("FFN wo hook received no input.")
            self.values[index] = inputs[0].detach()

        return hook

    def clear(self) -> None:
        self.values = [None] * len(self.values)

    def require_all(self) -> List[Any]:
        missing = [index for index, value in enumerate(self.values) if value is None]
        if missing:
            raise RuntimeError("Missing FFN activations for layers: %s" % missing)
        return self.values

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def choose_balanced_position_tokens(
    visual_mask: Any,
    text_mask: Any,
    remaining: int,
    rng: np.random.RandomState,
    torch: Any,
) -> Tuple[Any, Any]:
    visual_cpu = visual_mask.detach().bool().cpu()
    text_cpu = text_mask.detach().bool().cpu()
    if visual_cpu.shape != text_cpu.shape:
        raise ValueError("Visual and text position masks must have the same shape.")

    visual_selected: List[int] = []
    text_selected: List[int] = []
    sequence_length = int(visual_cpu.shape[1])
    for sample_index in range(int(visual_cpu.shape[0])):
        if len(visual_selected) >= remaining:
            break
        visual_positions = torch.nonzero(
            visual_cpu[sample_index], as_tuple=False
        ).flatten().numpy()
        text_positions = torch.nonzero(
            text_cpu[sample_index], as_tuple=False
        ).flatten().numpy()
        count = min(
            len(visual_positions),
            len(text_positions),
            remaining - len(visual_selected),
        )
        if count <= 0:
            continue
        if len(visual_positions) > count:
            visual_positions = np.sort(
                rng.choice(visual_positions, size=count, replace=False)
            )
        if len(text_positions) > count:
            text_positions = np.sort(
                rng.choice(text_positions, size=count, replace=False)
            )
        offset = sample_index * sequence_length
        visual_selected.extend((visual_positions + offset).tolist())
        text_selected.extend((text_positions + offset).tolist())

    return (
        torch.as_tensor(visual_selected, dtype=torch.long),
        torch.as_tensor(text_selected, dtype=torch.long),
    )


def append_selected_layer_tokens(
    stores: List[List[np.ndarray]],
    captures: Sequence[Any],
    selected: Any,
) -> int:
    if len(selected) == 0:
        return 0
    for layer_index, activation in enumerate(captures):
        flat = activation.detach().reshape(-1, activation.shape[-1])
        sampled = flat[selected.to(flat.device)].float().cpu().numpy()
        stores[layer_index].append(sampled.astype(np.float16))
    return int(len(selected))


def tokenize_with_length_stats(
    tokenizer: Any,
    texts: Sequence[str],
    max_txt_len: int,
    device: str,
) -> Tuple[Any, List[int]]:
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


def collect_multimodal_ffn_activations(
    model: Any,
    rows: Sequence[Any],
    images_dir: str,
    vis_processor: Any,
    capture: FFNActivationCapture,
    args: argparse.Namespace,
    torch: Any,
    Image: Any,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
    num_layers = len(capture.values)
    visual_stores: List[List[np.ndarray]] = [[] for _ in range(num_layers)]
    text_stores: List[List[np.ndarray]] = [[] for _ in range(num_layers)]
    visual_count = 0
    text_count = 0
    processed_samples = 0
    truncated_samples = 0
    original_text_tokens = 0
    retained_text_tokens = 0
    selected_fields: Dict[str, int] = {}
    position_rng = np.random.RandomState(args.seed + 101)

    for batch_index, (row_start, batch_rows) in enumerate(
        iter_batches(rows, args.batch_size)
    ):
        images = []
        texts = []
        for local_index, row in enumerate(batch_rows):
            row_index = row_start + local_index
            if not isinstance(row, dict) or args.image_field not in row:
                raise KeyError(
                    "Row %d is missing image field %r."
                    % (row_index, args.image_field)
                )
            image_path = resolve_image_path(images_dir, row[args.image_field])
            if not os.path.isfile(image_path):
                raise FileNotFoundError(
                    "Image not found for row %d: %s" % (row_index, image_path)
                )
            with Image.open(image_path) as image:
                images.append(vis_processor(image.convert("RGB")))
            text, selected_field = extract_text(row, args.text_field, row_index)
            texts.append(text)
            selected_fields[selected_field] = selected_fields.get(selected_field, 0) + 1

        image_tensor = torch.stack(images).to(args.device)
        capture.clear()
        with torch.no_grad():
            with model.maybe_autocast():
                image_hidden = model.ln_vision(model.visual_encoder(image_tensor))
                image_mask = torch.ones(
                    image_hidden.size()[:-1],
                    dtype=torch.long,
                    device=image_hidden.device,
                )
                query_tokens = model.query_tokens.expand(
                    image_hidden.shape[0], -1, -1
                )
                query_output = model.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_hidden,
                    encoder_attention_mask=image_mask,
                    return_dict=True,
                )
                visual_tokens = model.t5_proj(query_output.last_hidden_state)

            with model.maybe_autocast(dtype=torch.bfloat16):
                text_tokens, batch_original_lengths = tokenize_with_length_stats(
                    model.t5_tokenizer,
                    texts,
                    model.max_txt_len,
                    args.device,
                )
                text_embeddings = model.t5_model.encoder.embed_tokens(
                    text_tokens.input_ids
                )
                visual_attention = torch.ones(
                    visual_tokens.size()[:-1],
                    dtype=torch.long,
                    device=visual_tokens.device,
                )
                encoder_attention = torch.cat(
                    [visual_attention, text_tokens.attention_mask],
                    dim=1,
                )
                encoder_embeddings = torch.cat(
                    [visual_tokens, text_embeddings],
                    dim=1,
                )
                num_query = int(visual_tokens.shape[1])
                visual_position_mask = torch.zeros_like(
                    encoder_attention, dtype=torch.bool
                )
                visual_position_mask[:, :num_query] = True
                text_position_mask = torch.zeros_like(
                    encoder_attention, dtype=torch.bool
                )
                text_position_mask[:, num_query:] = (
                    text_tokens.attention_mask.bool()
                )
                model.temp_label = visual_position_mask
                model.t5_model.encoder(
                    inputs_embeds=encoder_embeddings,
                    attention_mask=encoder_attention,
                    return_dict=True,
                )

        captures = capture.require_all()
        visual_selected, text_selected = choose_balanced_position_tokens(
            visual_position_mask,
            text_position_mask,
            args.tokens_per_group - visual_count,
            position_rng,
            torch,
        )
        visual_added = append_selected_layer_tokens(
            visual_stores,
            captures,
            visual_selected,
        )
        text_added = append_selected_layer_tokens(
            text_stores,
            captures,
            text_selected,
        )
        if visual_added != text_added:
            raise RuntimeError("Balanced position sampling returned unequal counts.")
        visual_count += visual_added
        text_count += text_added
        processed_samples += len(batch_rows)
        original_text_tokens += int(sum(batch_original_lengths))
        batch_retained_lengths = (
            text_tokens.attention_mask.detach().sum(dim=1).cpu().tolist()
        )
        retained_text_tokens += int(sum(batch_retained_lengths))
        truncated_samples += sum(
            int(original > retained)
            for original, retained in zip(
                batch_original_lengths, batch_retained_lengths
            )
        )

        if args.log_every > 0 and (batch_index + 1) % args.log_every == 0:
            print(
                "Processed %d samples; visual tokens=%d/%d, text tokens=%d/%d"
                % (
                    processed_samples,
                    visual_count,
                    args.tokens_per_group,
                    text_count,
                    args.tokens_per_group,
                )
            )
        if (
            visual_count >= args.tokens_per_group
            and text_count >= args.tokens_per_group
        ):
            break

    if visual_count < args.tokens_per_group:
        raise RuntimeError(
            "Dataset supplied only %d visual query tokens; requested %d."
            % (visual_count, args.tokens_per_group)
        )
    if text_count < args.tokens_per_group:
        raise RuntimeError(
            "Dataset supplied only %d valid text tokens; requested %d."
            % (text_count, args.tokens_per_group)
        )

    visual_layers = [np.concatenate(chunks, axis=0) for chunks in visual_stores]
    text_layers = [np.concatenate(chunks, axis=0) for chunks in text_stores]
    metadata = {
        "processed_samples": processed_samples,
        "visual_tokens_sampled": visual_count,
        "text_tokens_sampled": text_count,
        "num_query_tokens_per_sample": int(model.query_tokens.shape[1]),
        "model_max_txt_len": int(model.max_txt_len),
        "truncated_samples": int(truncated_samples),
        "truncated_sample_rate": float(truncated_samples / processed_samples),
        "original_text_tokens": int(original_text_tokens),
        "retained_text_tokens": int(retained_text_tokens),
        "retained_text_token_rate": float(
            retained_text_tokens / original_text_tokens
        ),
        "selected_text_fields": selected_fields,
    }
    return visual_layers, text_layers, metadata


def analyze_layer_activations(
    visual_activations: np.ndarray,
    text_activations: np.ndarray,
    mad_multiplier: float,
    min_activation_rate: float,
) -> Dict[str, np.ndarray]:
    if visual_activations.shape != text_activations.shape:
        raise ValueError(
            "Balanced visual/text activations must have the same shape, got %s and %s."
            % (visual_activations.shape, text_activations.shape)
        )
    visual_magnitude = np.abs(
        visual_activations.astype(np.float32, copy=False)
    )
    text_magnitude = np.abs(text_activations.astype(np.float32, copy=False))
    joint = np.concatenate([visual_magnitude, text_magnitude], axis=0)
    median = np.median(joint, axis=0)
    mad = np.median(np.abs(joint - median), axis=0)
    thresholds = median + mad_multiplier * mad
    visual_frequency = np.mean(visual_magnitude > thresholds, axis=0)
    text_frequency = np.mean(text_magnitude > thresholds, axis=0)
    visual_covered = visual_frequency >= min_activation_rate
    text_covered = text_frequency >= min_activation_rate
    return {
        "threshold": thresholds.astype(np.float32),
        "joint_median_abs": median.astype(np.float32),
        "joint_mad_abs": mad.astype(np.float32),
        "visual_frequency": visual_frequency.astype(np.float32),
        "text_frequency": text_frequency.astype(np.float32),
        "visual_mean_abs": np.mean(visual_magnitude, axis=0).astype(np.float32),
        "text_mean_abs": np.mean(text_magnitude, axis=0).astype(np.float32),
        "visual_covered": visual_covered,
        "text_covered": text_covered,
    }


def summarize_layers(
    analyses: Sequence[Dict[str, np.ndarray]],
) -> List[Dict[str, Any]]:
    rows = []
    for layer_index, values in enumerate(analyses):
        visual = values["visual_covered"]
        text = values["text_covered"]
        both = visual & text
        visual_only = visual & ~text
        text_only = text & ~visual
        neither = ~visual & ~text
        union = visual | text
        union_count = int(np.sum(union))
        both_count = int(np.sum(both))
        rows.append(
            {
                "layer": layer_index,
                "num_neurons": int(len(visual)),
                "visual_covered_count": int(np.sum(visual)),
                "visual_covered_rate": float(np.mean(visual)),
                "text_covered_count": int(np.sum(text)),
                "text_covered_rate": float(np.mean(text)),
                "visual_only_count": int(np.sum(visual_only)),
                "visual_only_rate": float(np.mean(visual_only)),
                "text_only_count": int(np.sum(text_only)),
                "text_only_rate": float(np.mean(text_only)),
                "both_count": both_count,
                "both_rate": float(np.mean(both)),
                "neither_count": int(np.sum(neither)),
                "neither_rate": float(np.mean(neither)),
                "union_count": union_count,
                "union_rate": float(np.mean(union)),
                "jaccard": float(both_count / union_count) if union_count else 1.0,
                "visual_mean_activation_frequency": float(
                    np.mean(values["visual_frequency"])
                ),
                "text_mean_activation_frequency": float(
                    np.mean(values["text_frequency"])
                ),
                "visual_mean_absolute_activation": float(
                    np.mean(values["visual_mean_abs"])
                ),
                "text_mean_absolute_activation": float(
                    np.mean(values["text_mean_abs"])
                ),
            }
        )
    return rows


def setup_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except (ImportError, RuntimeError) as exc:
        raise SystemExit(
            "A compatible matplotlib/NumPy installation is required."
        ) from exc


def write_summary_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(
    out_dir: str,
    summaries: Sequence[Dict[str, Any]],
    analyses: Sequence[Dict[str, np.ndarray]],
    min_activation_rate: float,
    frequency_bins: int,
) -> List[str]:
    plt = setup_matplotlib()
    layers = np.asarray([row["layer"] for row in summaries])
    visual_rate = 100.0 * np.asarray(
        [row["visual_covered_rate"] for row in summaries]
    )
    text_rate = 100.0 * np.asarray(
        [row["text_covered_rate"] for row in summaries]
    )
    union_rate = 100.0 * np.asarray([row["union_rate"] for row in summaries])
    saved = []

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    ax.plot(
        layers,
        visual_rate,
        marker="o",
        linewidth=1.8,
        markersize=4,
        color="#4C91AC",
        label="Visual-token positions",
    )
    ax.plot(
        layers,
        text_rate,
        marker="s",
        linewidth=1.8,
        markersize=4,
        color="#5B9B52",
        label="Text-token positions",
    )
    ax.plot(
        layers,
        union_rate,
        marker="^",
        linewidth=1.4,
        markersize=4,
        linestyle="--",
        color="#7A6F9B",
        label="Union",
    )
    ax.set_xticks(layers)
    ax.set_ylim(0, 102)
    ax.set_xlabel("T5 Encoder Layer")
    ax.set_ylabel("Covered FFN Neurons (%)")
    ax.set_title(
        "Multimodal T5 FFN Neuron Coverage by Token Position\n"
        "(Activation frequency >= %.1f%%)" % (100.0 * min_activation_rate)
    )
    ax.legend(frameon=True)
    ax.grid(alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    path = os.path.join(out_dir, "multimodal_t5_ffn_layerwise_coverage.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    visual_only = 100.0 * np.asarray(
        [row["visual_only_rate"] for row in summaries]
    )
    text_only = 100.0 * np.asarray(
        [row["text_only_rate"] for row in summaries]
    )
    both = 100.0 * np.asarray([row["both_rate"] for row in summaries])
    neither = 100.0 * np.asarray([row["neither_rate"] for row in summaries])
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    ax.bar(layers, visual_only, color="#8FC5DA", label="Visual positions only")
    ax.bar(
        layers,
        text_only,
        bottom=visual_only,
        color="#93CD81",
        label="Text positions only",
    )
    ax.bar(
        layers,
        both,
        bottom=visual_only + text_only,
        color="#8B9E78",
        label="Both",
    )
    ax.bar(
        layers,
        neither,
        bottom=visual_only + text_only + both,
        color="#C9C9C9",
        label="Neither",
    )
    ax.set_xticks(layers)
    ax.set_ylim(0, 100)
    ax.set_xlabel("T5 Encoder Layer")
    ax.set_ylabel("FFN Neurons (%)")
    ax.set_title("Visual/Text Position FFN Coverage Overlap")
    ax.legend(ncol=4, frameon=True, loc="upper center")
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    path = os.path.join(out_dir, "multimodal_t5_ffn_coverage_overlap.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    jaccard = np.asarray([row["jaccard"] for row in summaries])
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.plot(layers, jaccard, marker="o", color="#6B5F89", linewidth=1.8)
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("T5 Encoder Layer")
    ax.set_ylabel("Covered-Neuron Jaccard")
    ax.set_title("Visual/Text Position Neuron-Set Similarity")
    ax.grid(alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    path = os.path.join(out_dir, "multimodal_t5_ffn_coverage_jaccard.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    visual_abs = np.asarray(
        [row["visual_mean_absolute_activation"] for row in summaries]
    )
    text_abs = np.asarray(
        [row["text_mean_absolute_activation"] for row in summaries]
    )
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    ax.plot(
        layers,
        visual_abs,
        marker="o",
        color="#4C91AC",
        label="Visual-token positions",
    )
    ax.plot(
        layers,
        text_abs,
        marker="s",
        color="#5B9B52",
        label="Text-token positions",
    )
    ax.set_xticks(layers)
    ax.set_xlabel("T5 Encoder Layer")
    ax.set_ylabel("Mean Absolute FFN Activation")
    ax.set_title("Layer-Wise FFN Activation Magnitude")
    ax.legend(frameon=True)
    ax.grid(alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    path = os.path.join(out_dir, "multimodal_t5_ffn_activation_magnitude.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    visual_frequencies = 100.0 * np.concatenate(
        [values["visual_frequency"] for values in analyses]
    )
    text_frequencies = 100.0 * np.concatenate(
        [values["text_frequency"] for values in analyses]
    )
    max_frequency = max(
        float(np.percentile(visual_frequencies, 99.5)),
        float(np.percentile(text_frequencies, 99.5)),
        100.0 * min_activation_rate,
    )
    max_frequency = min(max(max_frequency, 1.0), 100.0)
    bins = np.linspace(0.0, max_frequency, frequency_bins + 1)
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    ax.hist(
        visual_frequencies,
        bins=bins,
        density=True,
        alpha=0.62,
        color="#8FC5DA",
        edgecolor="#567B89",
        linewidth=0.35,
        label="Visual-token positions",
    )
    ax.hist(
        text_frequencies,
        bins=bins,
        density=True,
        alpha=0.58,
        color="#93CD81",
        edgecolor="#527A48",
        linewidth=0.35,
        label="Text-token positions",
    )
    ax.axvline(
        100.0 * min_activation_rate,
        color="#7A6F9B",
        linestyle="--",
        linewidth=1.4,
        label="Coverage threshold",
    )
    ax.set_xlim(0, max_frequency)
    ax.set_xlabel("Per-Neuron Activated-Token Frequency (%)")
    ax.set_ylabel("Density")
    ax.set_title("Multimodal T5 FFN Activation Frequency Distribution")
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    path = os.path.join(out_dir, "multimodal_t5_ffn_frequency_distribution.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    return saved


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.tokens_per_group < 2:
        raise ValueError("--tokens_per_group must be >= 2")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max_samples must be >= 1")
    if args.max_txt_len is not None and args.max_txt_len < 2:
        raise ValueError("--max_txt_len must be >= 2")
    if args.mad_multiplier < 0:
        raise ValueError("--mad_multiplier must be >= 0")
    if not 0.0 < args.min_activation_rate <= 1.0:
        raise ValueError("--min_activation_rate must be in (0, 1]")
    if args.frequency_bins < 2:
        raise ValueError("--frequency_bins must be >= 2")

    try:
        import torch
        from PIL import Image
        from lavis.models import load_model
        from lavis.processors import load_processor
    except ImportError as exc:
        raise SystemExit(
            "Missing LAVIS runtime dependency: %s. Run in the LAVIS environment."
            % exc
        ) from exc

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    calib_json = os.path.abspath(args.calib_json)
    images_dir = os.path.abspath(args.images_dir)
    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)
    rows = shuffled_rows(
        load_rows(calib_json),
        args.max_samples,
        args.seed,
    )

    print("device:", args.device)
    print("dataset rows selected:", len(rows))
    print("text field:", args.text_field)
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
    vis_processor = load_processor("blip_image_eval").build(
        image_size=args.image_size
    )

    capture = FFNActivationCapture(get_ffn_wo_modules(model))
    try:
        visual_layers, text_layers, collection_metadata = (
            collect_multimodal_ffn_activations(
                model,
                rows,
                images_dir,
                vis_processor,
                capture,
                args,
                torch,
                Image,
            )
        )
    finally:
        capture.close()

    analyses = [
        analyze_layer_activations(
            visual,
            text,
            args.mad_multiplier,
            args.min_activation_rate,
        )
        for visual, text in zip(visual_layers, text_layers)
    ]
    summaries = summarize_layers(analyses)

    csv_path = os.path.join(
        out_dir, "multimodal_t5_ffn_neuron_coverage_summary.csv"
    )
    write_summary_csv(csv_path, summaries)
    npz_path = os.path.join(
        out_dir, "multimodal_t5_ffn_neuron_coverage_metrics.npz"
    )
    np.savez_compressed(
        npz_path,
        layer=np.arange(len(analyses), dtype=np.int64),
        threshold=np.stack([value["threshold"] for value in analyses]),
        joint_median_abs=np.stack(
            [value["joint_median_abs"] for value in analyses]
        ),
        joint_mad_abs=np.stack(
            [value["joint_mad_abs"] for value in analyses]
        ),
        visual_activation_frequency=np.stack(
            [value["visual_frequency"] for value in analyses]
        ),
        text_activation_frequency=np.stack(
            [value["text_frequency"] for value in analyses]
        ),
        visual_mean_absolute_activation=np.stack(
            [value["visual_mean_abs"] for value in analyses]
        ),
        text_mean_absolute_activation=np.stack(
            [value["text_mean_abs"] for value in analyses]
        ),
        visual_covered=np.stack(
            [value["visual_covered"] for value in analyses]
        ),
        text_covered=np.stack(
            [value["text_covered"] for value in analyses]
        ),
    )
    plot_paths = make_plots(
        out_dir,
        summaries,
        analyses,
        args.min_activation_rate,
        args.frequency_bins,
    )

    summary = {
        "calib_json": calib_json,
        "images_dir": images_dir,
        "checkpoint": os.path.abspath(args.ckpt),
        "model_name": args.model_name,
        "model_type": args.model_type,
        "input_definition": (
            "[BLIP2 visual query tokens, tokenized dataset text] passed "
            "together through the T5 encoder"
        ),
        "position_interpretation": (
            "visual/text position groups are contextualized multimodal "
            "activations after self-attention, not independent unimodal runs"
        ),
        "answer_used_as_input": False,
        "num_encoder_layers": len(analyses),
        "ffn_neurons_per_layer": int(len(analyses[0]["threshold"])),
        "collection": collection_metadata,
        "threshold_definition": "median(|a|) + mad_multiplier * MAD(|a|)",
        "threshold_reference": (
            "balanced visual-position and text-position tokens from the same "
            "multimodal forwards"
        ),
        "mad_multiplier": float(args.mad_multiplier),
        "neuron_coverage_definition": (
            "fraction of sampled position-group tokens with absolute FFN "
            "activation above the shared per-neuron threshold >= "
            "min_activation_rate"
        ),
        "min_activation_rate": float(args.min_activation_rate),
        "layers": summaries,
    }
    json_path = os.path.join(
        out_dir, "multimodal_t5_ffn_neuron_coverage_summary.json"
    )
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    if collection_metadata["truncated_samples"]:
        print(
            "[WARN] %d/%d processed samples were truncated at %d text tokens."
            % (
                collection_metadata["truncated_samples"],
                collection_metadata["processed_samples"],
                collection_metadata["model_max_txt_len"],
            )
        )
    for path in (csv_path, npz_path, json_path, *plot_paths):
        print("[OK] wrote:", path)


if __name__ == "__main__":
    main()
