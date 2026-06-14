#!/usr/bin/env python3
"""Compare image and C4 activation coverage inside T5 encoder FFN neurons.

This records the input to each encoder FFN ``wo`` projection. For gated T5
blocks, that tensor is:

    GELU(wi_0(RMSNorm(x))) * wi_1(RMSNorm(x))

The script runs image-derived visual query tokens and C4 text tokens through
the same T5 encoder, samples the same number of tokens from each modality, and
creates layer-wise FFN neuron coverage, overlap, and activation-frequency plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


_LAVIS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _LAVIS_ROOT not in sys.path:
    sys.path.insert(0, _LAVIS_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze layer-wise T5 encoder FFN neuron coverage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image_json", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--text_json", required=True)
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
        help="C4 dictionary field, or auto to try text/caption/text_input/output.",
    )
    parser.add_argument("--max_image_samples", type=int, default=None)
    parser.add_argument("--max_text_samples", type=int, default=None)
    parser.add_argument("--max_txt_len", type=int, default=None)
    parser.add_argument(
        "--tokens_per_modality",
        type=int,
        default=512,
        help="Balanced token sample count retained for every layer and modality.",
    )
    parser.add_argument(
        "--mad_multiplier",
        type=float,
        default=3.0,
        help="Per-neuron threshold is median(|a|) + multiplier * MAD(|a|).",
    )
    parser.add_argument(
        "--min_activation_rate",
        type=float,
        default=0.01,
        help="A neuron is covered when at least this fraction of tokens exceed its threshold.",
    )
    parser.add_argument("--frequency_bins", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=20)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json_list(path: str) -> List[Any]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise TypeError("%s must contain a JSON list." % path)
    if not rows:
        raise ValueError("%s contains no rows." % path)
    return rows


def extract_text(row: Any, text_field: str, row_index: int) -> str:
    if isinstance(row, str):
        text = row
    elif isinstance(row, dict):
        fields = (
            [text_field]
            if text_field != "auto"
            else ["text", "caption", "text_input", "output"]
        )
        selected = next((field for field in fields if field in row), None)
        if selected is None:
            raise KeyError(
                "Text row %d has none of these fields: %s"
                % (row_index, ", ".join(fields))
            )
        text = row[selected]
    else:
        raise TypeError("Text row %d must be a string or object." % row_index)
    text = str(text).strip()
    if not text:
        raise ValueError("Text row %d is empty." % row_index)
    return text


def resolve_image_path(images_dir: str, image_value: Any) -> str:
    path = str(image_value)
    return path if os.path.isabs(path) else os.path.join(images_dir, path)


def iter_batches(rows: Sequence[Any], batch_size: int) -> Iterable[List[Any]]:
    for start in range(0, len(rows), batch_size):
        yield list(rows[start : start + batch_size])


def shuffled_rows(
    rows: Sequence[Any],
    max_samples: Any,
    seed: int,
) -> List[Any]:
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(rows))
    if max_samples is not None:
        indices = indices[: int(max_samples)]
    return [rows[int(index)] for index in indices]


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


def choose_valid_tokens(
    mask: Any,
    remaining: int,
    rng: np.random.RandomState,
    torch: Any,
) -> Any:
    valid = torch.nonzero(mask.detach().bool().cpu().reshape(-1), as_tuple=False)
    valid = valid.flatten()
    if len(valid) <= remaining:
        return valid
    selected = np.sort(rng.choice(len(valid), size=remaining, replace=False))
    return valid[torch.as_tensor(selected, dtype=torch.long)]


def append_layer_tokens(
    stores: List[List[np.ndarray]],
    captures: Sequence[Any],
    mask: Any,
    remaining: int,
    rng: np.random.RandomState,
    torch: Any,
) -> int:
    selected = choose_valid_tokens(mask, remaining, rng, torch)
    if len(selected) == 0:
        return 0
    for layer_index, activation in enumerate(captures):
        flat = activation.detach().reshape(-1, activation.shape[-1])
        sampled = flat[selected.to(flat.device)].float().cpu().numpy()
        stores[layer_index].append(sampled.astype(np.float16))
    return int(len(selected))


def collect_image_ffn_activations(
    model: Any,
    rows: Sequence[Any],
    images_dir: str,
    vis_processor: Any,
    capture: FFNActivationCapture,
    args: argparse.Namespace,
    torch: Any,
    Image: Any,
) -> Tuple[List[np.ndarray], int]:
    num_layers = len(capture.values)
    stores: List[List[np.ndarray]] = [[] for _ in range(num_layers)]
    collected = 0
    rng = np.random.RandomState(args.seed + 101)

    for batch_index, batch_rows in enumerate(iter_batches(rows, args.batch_size)):
        images = []
        for row_index, row in enumerate(batch_rows):
            if not isinstance(row, dict) or args.image_field not in row:
                raise KeyError(
                    "Image batch row %d is missing field %r."
                    % (row_index, args.image_field)
                )
            image_path = resolve_image_path(images_dir, row[args.image_field])
            if not os.path.isfile(image_path):
                raise FileNotFoundError(image_path)
            image = Image.open(image_path).convert("RGB")
            images.append(vis_processor(image))

        capture.clear()
        image_tensor = torch.stack(images).to(args.device)
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
                t5_visual_tokens = model.t5_proj(query_output.last_hidden_state)
            with model.maybe_autocast(dtype=torch.bfloat16):
                t5_mask = torch.ones(
                    t5_visual_tokens.size()[:-1],
                    dtype=torch.long,
                    device=t5_visual_tokens.device,
                )
                model.temp_label = torch.ones_like(t5_mask, dtype=torch.bool)
                model.t5_model.encoder(
                    inputs_embeds=t5_visual_tokens,
                    attention_mask=t5_mask,
                    return_dict=True,
                )

        remaining = args.tokens_per_modality - collected
        added = append_layer_tokens(
            stores,
            capture.require_all(),
            t5_mask,
            remaining,
            rng,
            torch,
        )
        collected += added
        if args.log_every > 0 and (batch_index + 1) % args.log_every == 0:
            print(
                "Image FFN tokens: %d/%d"
                % (collected, args.tokens_per_modality)
            )
        if collected >= args.tokens_per_modality:
            break

    if collected < args.tokens_per_modality:
        raise RuntimeError(
            "Image data supplied only %d valid query tokens; requested %d."
            % (collected, args.tokens_per_modality)
        )
    return [np.concatenate(chunks, axis=0) for chunks in stores], collected


def collect_text_ffn_activations(
    model: Any,
    rows: Sequence[Any],
    capture: FFNActivationCapture,
    args: argparse.Namespace,
    torch: Any,
) -> Tuple[List[np.ndarray], int]:
    num_layers = len(capture.values)
    stores: List[List[np.ndarray]] = [[] for _ in range(num_layers)]
    collected = 0
    rng = np.random.RandomState(args.seed + 202)

    for batch_index, batch_rows in enumerate(iter_batches(rows, args.batch_size)):
        texts = [
            extract_text(row, args.text_field, batch_index * args.batch_size + index)
            for index, row in enumerate(batch_rows)
        ]
        capture.clear()
        device = next(model.t5_model.parameters()).device
        with torch.no_grad():
            with model.maybe_autocast(dtype=torch.bfloat16):
                tokens = model.t5_tokenizer(
                    texts,
                    padding="longest",
                    truncation=True,
                    max_length=model.max_txt_len,
                    return_tensors="pt",
                ).to(device)
                text_embeddings = model.t5_model.encoder.embed_tokens(
                    tokens.input_ids
                )
                model.temp_label = torch.zeros_like(
                    tokens.attention_mask, dtype=torch.bool
                )
                model.t5_model.encoder(
                    inputs_embeds=text_embeddings,
                    attention_mask=tokens.attention_mask,
                    return_dict=True,
                )

        remaining = args.tokens_per_modality - collected
        added = append_layer_tokens(
            stores,
            capture.require_all(),
            tokens.attention_mask,
            remaining,
            rng,
            torch,
        )
        collected += added
        if args.log_every > 0 and (batch_index + 1) % args.log_every == 0:
            print(
                "C4 FFN tokens: %d/%d"
                % (collected, args.tokens_per_modality)
            )
        if collected >= args.tokens_per_modality:
            break

    if collected < args.tokens_per_modality:
        raise RuntimeError(
            "C4 data supplied only %d valid tokens; requested %d."
            % (collected, args.tokens_per_modality)
        )
    return [np.concatenate(chunks, axis=0) for chunks in stores], collected


def analyze_layer_activations(
    image_activations: np.ndarray,
    text_activations: np.ndarray,
    mad_multiplier: float,
    min_activation_rate: float,
) -> Dict[str, np.ndarray]:
    if image_activations.shape != text_activations.shape:
        raise ValueError(
            "Balanced image/text activations must have the same shape, got %s and %s."
            % (image_activations.shape, text_activations.shape)
        )
    image_magnitude = np.abs(image_activations.astype(np.float32, copy=False))
    text_magnitude = np.abs(text_activations.astype(np.float32, copy=False))
    joint = np.concatenate([image_magnitude, text_magnitude], axis=0)
    median = np.median(joint, axis=0)
    mad = np.median(np.abs(joint - median), axis=0)
    thresholds = median + mad_multiplier * mad
    image_frequency = np.mean(image_magnitude > thresholds, axis=0)
    text_frequency = np.mean(text_magnitude > thresholds, axis=0)
    image_covered = image_frequency >= min_activation_rate
    text_covered = text_frequency >= min_activation_rate
    return {
        "threshold": thresholds.astype(np.float32),
        "joint_median_abs": median.astype(np.float32),
        "joint_mad_abs": mad.astype(np.float32),
        "image_frequency": image_frequency.astype(np.float32),
        "text_frequency": text_frequency.astype(np.float32),
        "image_covered": image_covered,
        "text_covered": text_covered,
    }


def summarize_layers(
    analyses: Sequence[Dict[str, np.ndarray]],
) -> List[Dict[str, Any]]:
    rows = []
    for layer_index, values in enumerate(analyses):
        image = values["image_covered"]
        text = values["text_covered"]
        both = image & text
        image_only = image & ~text
        text_only = text & ~image
        neither = ~image & ~text
        union = image | text
        union_count = int(np.sum(union))
        both_count = int(np.sum(both))
        num_neurons = int(len(image))
        rows.append(
            {
                "layer": layer_index,
                "num_neurons": num_neurons,
                "image_covered_count": int(np.sum(image)),
                "image_covered_rate": float(np.mean(image)),
                "c4_text_covered_count": int(np.sum(text)),
                "c4_text_covered_rate": float(np.mean(text)),
                "image_only_count": int(np.sum(image_only)),
                "image_only_rate": float(np.mean(image_only)),
                "c4_text_only_count": int(np.sum(text_only)),
                "c4_text_only_rate": float(np.mean(text_only)),
                "both_count": both_count,
                "both_rate": float(np.mean(both)),
                "neither_count": int(np.sum(neither)),
                "neither_rate": float(np.mean(neither)),
                "union_count": union_count,
                "union_rate": float(np.mean(union)),
                "jaccard": float(both_count / union_count) if union_count else 1.0,
                "image_mean_activation_frequency": float(
                    np.mean(values["image_frequency"])
                ),
                "c4_text_mean_activation_frequency": float(
                    np.mean(values["text_frequency"])
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
    image_rate = 100.0 * np.asarray(
        [row["image_covered_rate"] for row in summaries]
    )
    text_rate = 100.0 * np.asarray(
        [row["c4_text_covered_rate"] for row in summaries]
    )
    union_rate = 100.0 * np.asarray(
        [row["union_rate"] for row in summaries]
    )
    saved = []

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    ax.plot(
        layers,
        image_rate,
        marker="o",
        linewidth=1.8,
        markersize=4,
        color="#4C91AC",
        label="Image Visual Query",
    )
    ax.plot(
        layers,
        text_rate,
        marker="s",
        linewidth=1.8,
        markersize=4,
        color="#5B9B52",
        label="C4 Text",
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
        "Layer-Wise T5 FFN Neuron Coverage\n"
        "(Token Activation Frequency >= %.2f%%)"
        % (100.0 * min_activation_rate)
    )
    ax.legend(frameon=True)
    ax.grid(alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    path = os.path.join(out_dir, "t5_ffn_layerwise_neuron_coverage.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    image_only = 100.0 * np.asarray(
        [row["image_only_rate"] for row in summaries]
    )
    text_only = 100.0 * np.asarray(
        [row["c4_text_only_rate"] for row in summaries]
    )
    both = 100.0 * np.asarray([row["both_rate"] for row in summaries])
    neither = 100.0 * np.asarray([row["neither_rate"] for row in summaries])
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    ax.bar(layers, image_only, color="#8FC5DA", label="Image Only")
    ax.bar(
        layers,
        text_only,
        bottom=image_only,
        color="#93CD81",
        label="C4 Only",
    )
    ax.bar(
        layers,
        both,
        bottom=image_only + text_only,
        color="#8B9E78",
        label="Both",
    )
    ax.bar(
        layers,
        neither,
        bottom=image_only + text_only + both,
        color="#C9C9C9",
        label="Neither",
    )
    ax.set_xticks(layers)
    ax.set_ylim(0, 100)
    ax.set_xlabel("T5 Encoder Layer")
    ax.set_ylabel("FFN Neurons (%)")
    ax.set_title("Layer-Wise T5 FFN Neuron Coverage Overlap")
    ax.legend(ncol=4, frameon=True, loc="upper center")
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    path = os.path.join(out_dir, "t5_ffn_layerwise_coverage_overlap.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    image_frequencies = (
        100.0
        * np.concatenate([values["image_frequency"] for values in analyses])
    )
    text_frequencies = (
        100.0
        * np.concatenate([values["text_frequency"] for values in analyses])
    )
    max_frequency = max(
        float(np.percentile(image_frequencies, 99.5)),
        float(np.percentile(text_frequencies, 99.5)),
        100.0 * min_activation_rate,
    )
    max_frequency = min(max(max_frequency, 1.0), 100.0)
    bins = np.linspace(0.0, max_frequency, frequency_bins + 1)
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    ax.hist(
        image_frequencies,
        bins=bins,
        density=True,
        alpha=0.62,
        color="#8FC5DA",
        edgecolor="#567B89",
        linewidth=0.35,
        label="Image Visual Query",
    )
    ax.hist(
        text_frequencies,
        bins=bins,
        density=True,
        alpha=0.58,
        color="#93CD81",
        edgecolor="#527A48",
        linewidth=0.35,
        label="C4 Text",
    )
    ax.axvline(
        100.0 * min_activation_rate,
        color="#7A6F9B",
        linestyle="--",
        linewidth=1.4,
        label="Coverage Frequency Threshold",
    )
    ax.set_xlim(0, max_frequency)
    ax.set_xlabel("Per-Neuron Activated Token Frequency (%)")
    ax.set_ylabel("Density")
    ax.set_title("T5 Encoder FFN Activation Frequency Distribution")
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    path = os.path.join(out_dir, "t5_ffn_activation_frequency_distribution.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    return saved


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.tokens_per_modality < 2:
        raise ValueError("--tokens_per_modality must be >= 2")
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
    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)
    image_rows = shuffled_rows(
        load_json_list(os.path.abspath(args.image_json)),
        args.max_image_samples,
        args.seed,
    )
    text_rows = shuffled_rows(
        load_json_list(os.path.abspath(args.text_json)),
        args.max_text_samples,
        args.seed + 1,
    )

    print("device:", args.device)
    print("image rows:", len(image_rows))
    print("C4 rows:", len(text_rows))
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
    vis_processor = load_processor("blip_image_eval").build(
        image_size=args.image_size
    )

    ffn_modules = get_ffn_wo_modules(model)
    capture = FFNActivationCapture(ffn_modules)
    try:
        image_layers, image_token_count = collect_image_ffn_activations(
            model,
            image_rows,
            os.path.abspath(args.images_dir),
            vis_processor,
            capture,
            args,
            torch,
            Image,
        )
        text_layers, text_token_count = collect_text_ffn_activations(
            model,
            text_rows,
            capture,
            args,
            torch,
        )
    finally:
        capture.close()

    if image_token_count != text_token_count:
        raise RuntimeError("Image and text token samples are not balanced.")
    analyses = [
        analyze_layer_activations(
            image,
            text,
            args.mad_multiplier,
            args.min_activation_rate,
        )
        for image, text in zip(image_layers, text_layers)
    ]
    summaries = summarize_layers(analyses)

    csv_path = os.path.join(out_dir, "t5_ffn_neuron_coverage_summary.csv")
    write_summary_csv(csv_path, summaries)
    npz_payload: Dict[str, Any] = {
        "layer": np.arange(len(analyses), dtype=np.int64),
        "threshold": np.stack([value["threshold"] for value in analyses]),
        "joint_median_abs": np.stack(
            [value["joint_median_abs"] for value in analyses]
        ),
        "joint_mad_abs": np.stack(
            [value["joint_mad_abs"] for value in analyses]
        ),
        "image_activation_frequency": np.stack(
            [value["image_frequency"] for value in analyses]
        ),
        "c4_text_activation_frequency": np.stack(
            [value["text_frequency"] for value in analyses]
        ),
        "image_covered": np.stack(
            [value["image_covered"] for value in analyses]
        ),
        "c4_text_covered": np.stack(
            [value["text_covered"] for value in analyses]
        ),
    }
    npz_path = os.path.join(out_dir, "t5_ffn_neuron_coverage_metrics.npz")
    np.savez_compressed(npz_path, **npz_payload)
    plot_paths = make_plots(
        out_dir,
        summaries,
        analyses,
        args.min_activation_rate,
        args.frequency_bins,
    )

    summary = {
        "image_json": os.path.abspath(args.image_json),
        "images_dir": os.path.abspath(args.images_dir),
        "text_json": os.path.abspath(args.text_json),
        "checkpoint": os.path.abspath(args.ckpt),
        "num_encoder_layers": len(analyses),
        "ffn_neurons_per_layer": int(len(analyses[0]["threshold"])),
        "tokens_per_modality": image_token_count,
        "threshold_definition": "median(|a|) + mad_multiplier * MAD(|a|)",
        "mad_multiplier": float(args.mad_multiplier),
        "neuron_coverage_definition": (
            "fraction of tokens with |FFN activation| above the shared "
            "per-neuron threshold >= min_activation_rate"
        ),
        "min_activation_rate": float(args.min_activation_rate),
        "layers": summaries,
    }
    json_path = os.path.join(out_dir, "t5_ffn_neuron_coverage_summary.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    for path in (csv_path, npz_path, json_path, *plot_paths):
        print("[OK] wrote:", path)


if __name__ == "__main__":
    main()
