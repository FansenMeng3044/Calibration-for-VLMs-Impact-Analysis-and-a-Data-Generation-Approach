#!/usr/bin/env python3
"""Record BLIP2-T5 activation distributions and make static plots.

Supported input modes:
  - multimodal: image + question + answer through the full BLIP2-T5 path
  - t5_c4_text: C4 text through T5 only, using the text as its own target
  - vit_image_only: image through visual_encoder + ln_vision only

Multimodal example:
  python scripts/blip2/record_blip2_activation_distributions.py \
    --calib_json /data/data2/mfs/MMMU_calibration/mmmu_calibration_train.json \
    --images_dir /data/data2/mfs/MMMU_calibration/images \
    --ckpt /data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth \
    --out_dir /data/data2/mfs/MMMU_calibration/activation_original \
    --batch_size 4 \
    --max_samples 128

Text-only C4 example:
  python scripts/blip2/record_blip2_activation_distributions.py \
    --input_mode t5_c4_text \
    --calib_json /data/data2/mfs/c4_calib_128.json \
    --ckpt /data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth \
    --out_dir /data/data2/mfs/c4_activation

Image-only example:
  python scripts/blip2/record_blip2_activation_distributions.py \
    --input_mode vit_image_only \
    --record_shared_t5_space \
    --calib_json /data/data2/mfs/CC3M_calib_128/cc3m_calib_128.json \
    --images_dir /data/data2/mfs/CC3M_calib_128/images \
    --ckpt /data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth \
    --out_dir /data/data2/mfs/cc3m_image_activation

Recorded tensors:
  - T5 text embedding activation
  - ViT first-block input activation
  - T5 encoder first-block input activation
  - Final logits and probability-distribution summaries
  - T5 decoder last hidden states
  - T5 encoder last hidden states

The static plots also include a visual-vs-text t-SNE projection built from
individual visual-query and non-padding text-token vectors in the shared T5
first-block input space, plus an overlaid histogram comparing their
activation-value distributions.

By default the script stores compact per-sample statistics and pooled activation
vectors. Use --save_full_tensors only when you really want the full tensors,
because full logits/probabilities are large.
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


ACTIVATION_LABELS = {
    "t5_text_embedding": "T5 Text Embedding",
    "vit_first_block_input": "ViT First-Block Input",
    "vit_last_hidden": "ViT Last Hidden (After ln_vision)",
    "t5_visual_query_input": "Visual Query Input to T5",
    "t5_first_block_input": "T5 First-Block Input",
    "t5_encoder_last_hidden": "T5 Encoder Last Hidden",
    "t5_decoder_last_hidden": "T5 Decoder Last Hidden",
    "final_logits": "Final Logits",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Record multimodal or unimodal BLIP2-T5 calibration activations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--input_mode",
        choices=["multimodal", "t5_c4_text", "vit_image_only"],
        default="multimodal",
        help="Model path used to record activations.",
    )
    ap.add_argument(
        "--calib_json",
        required=True,
        help="JSON list. C4 may be a list of strings or dictionaries.",
    )
    ap.add_argument(
        "--images_dir",
        default=None,
        help="Image directory. Defaults to sibling images/ next to --calib_json.",
    )
    ap.add_argument(
        "--ckpt",
        required=True,
        help="Original or pruned BLIP2 checkpoint to inspect.",
    )
    ap.add_argument("--out_dir", required=True, help="Directory for JSONL/NPZ/PNG outputs.")
    ap.add_argument("--model_name", default="blip2_t5")
    ap.add_argument("--model_type", default="pretrain_flant5xl")
    ap.add_argument("--device", default=None, help="Defaults to cuda when available, otherwise cpu.")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--question_field", default="question")
    ap.add_argument("--image_field", default="image")
    ap.add_argument("--answer_field", default="answer")
    ap.add_argument(
        "--text_field",
        default="auto",
        help="C4 dictionary field, or auto to try text/caption/text_input/output.",
    )
    ap.add_argument(
        "--record_shared_t5_space",
        action="store_true",
        help=(
            "In vit_image_only mode, additionally run Q-Former + t5_proj and save "
            "visual query tokens in the shared T5 input space."
        ),
    )
    ap.add_argument(
        "--max_txt_len",
        type=int,
        default=None,
        help="Optional override for model.max_txt_len before tokenization.",
    )
    ap.add_argument("--top_k", type=int, default=5, help="Top-k output tokens saved per answer position.")
    ap.add_argument(
        "--top_k_positions",
        type=int,
        default=3,
        help="Number of valid answer-token positions with top-k details in JSONL.",
    )
    ap.add_argument(
        "--save_full_tensors",
        action="store_true",
        help="Also save full activation/logit/probability tensors per batch as .pt files.",
    )
    ap.add_argument(
        "--tsne_perplexity",
        type=float,
        default=30.0,
        help="Requested perplexity for the visual-vs-text activation t-SNE.",
    )
    ap.add_argument(
        "--tsne_max_points_per_modality",
        type=int,
        default=5000,
        help="Maximum token vectors per modality used by t-SNE.",
    )
    ap.add_argument("--tsne_random_state", type=int, default=42)
    ap.add_argument(
        "--activation_histogram_bins",
        type=int,
        default=80,
        help="Number of bins in the visual-vs-text activation histogram.",
    )
    ap.add_argument("--log_every", type=int, default=20)
    return ap.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_rows(path: str, max_samples: Optional[int]) -> List[Any]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise TypeError("Input calibration JSON must be a list.")
    return rows if max_samples is None else rows[:max_samples]


def row_value(row: Any, field: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(field, default)
    return default


def extract_text(row: Any, args: argparse.Namespace, row_index: int) -> str:
    if isinstance(row, str):
        text = row
    elif isinstance(row, dict):
        fields = (
            [args.text_field]
            if args.text_field != "auto"
            else ["text", "caption", "text_input", "output"]
        )
        selected = next((field for field in fields if field in row), None)
        if selected is None:
            raise KeyError(
                "Row %d has no C4 text field. Tried: %s"
                % (row_index, ", ".join(fields))
            )
        text = row[selected]
    else:
        raise TypeError("Row %d must be a string or JSON object." % row_index)

    text = str(text).strip()
    if not text:
        raise ValueError("Row %d contains empty text." % row_index)
    return text


def resolve_image_path(images_dir: str, image_value: Any) -> str:
    image_name = str(image_value)
    if os.path.isabs(image_name):
        return image_name
    return os.path.join(images_dir, image_name)


def answer_to_text(answer: Any) -> str:
    if isinstance(answer, list):
        if not answer:
            return ""
        return str(answer[0])
    if answer is None:
        return ""
    return str(answer)


def iter_batches(rows: Sequence[Any], batch_size: int) -> Iterable[Tuple[int, List[Any]]]:
    for start in range(0, len(rows), batch_size):
        yield start, list(rows[start : start + batch_size])


def sequence_stats(tensor: Any, mask: Optional[Any] = None) -> Dict[str, np.ndarray]:
    """Per-sample stats for sequence-shaped tensors [B, L, D]."""
    import torch

    x = tensor.detach().float()
    if x.dim() != 3:
        raise ValueError("Expected [B, L, D], got shape %s" % (tuple(x.shape),))

    if mask is None:
        flat = x.reshape(x.shape[0], -1)
        token_count = torch.full((x.shape[0],), x.shape[1], device=x.device, dtype=torch.float32)
        mean = flat.mean(dim=1)
        std = flat.std(dim=1, unbiased=False)
        mean_abs = flat.abs().mean(dim=1)
        rms = torch.sqrt(torch.square(flat).mean(dim=1).clamp_min(0))
    else:
        m = mask.detach().float().to(x.device)
        m = m[:, : x.shape[1]]
        m_expanded = m.unsqueeze(-1)
        denom = (m.sum(dim=1) * x.shape[-1]).clamp_min(1.0)
        token_count = m.sum(dim=1)
        mean = (x * m_expanded).sum(dim=(1, 2)) / denom
        mean_sq = (torch.square(x) * m_expanded).sum(dim=(1, 2)) / denom
        std = torch.sqrt((mean_sq - torch.square(mean)).clamp_min(0))
        mean_abs = (x.abs() * m_expanded).sum(dim=(1, 2)) / denom
        rms = torch.sqrt(mean_sq.clamp_min(0))

    return {
        "mean": mean.cpu().numpy(),
        "std": std.cpu().numpy(),
        "mean_abs": mean_abs.cpu().numpy(),
        "rms": rms.cpu().numpy(),
        "token_count": token_count.cpu().numpy(),
    }


def masked_sequence_mean(tensor: Any, mask: Optional[Any] = None) -> np.ndarray:
    """Return per-sample pooled vectors [B, D]."""
    import torch

    x = tensor.detach().float()
    if x.dim() != 3:
        raise ValueError("Expected [B, L, D], got shape %s" % (tuple(x.shape),))
    if mask is None:
        return x.mean(dim=1).cpu().numpy()

    m = mask.detach().float().to(x.device)
    m = m[:, : x.shape[1]]
    denom = m.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
    pooled = (x * m.unsqueeze(-1)).sum(dim=1) / denom
    return pooled.cpu().numpy()


def masked_mean(values: Any, mask: Any) -> np.ndarray:
    import torch

    v = values.detach().float()
    m = mask.detach().float().to(v.device)
    m = m[:, : v.shape[1]]
    return ((v * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)).cpu().numpy()


def decode_token(tokenizer: Any, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
    return str(token)


def output_distribution_stats(
    logits: Any,
    decoder_mask: Any,
    tokenizer: Any,
    top_k: int,
    top_k_positions: int,
) -> Tuple[Dict[str, np.ndarray], List[List[Dict[str, Any]]], Any]:
    import torch

    logits_f = logits.detach().float()
    probs = torch.softmax(logits_f, dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))
    entropy = -(probs * log_probs).sum(dim=-1)

    k = min(max(top_k, 1), logits_f.shape[-1])
    prob_values, token_ids = torch.topk(probs, k=k, dim=-1)
    logit_values = torch.gather(logits_f, -1, token_ids)

    top1_prob = prob_values[..., 0]
    if k >= 2:
        prob_margin = prob_values[..., 0] - prob_values[..., 1]
        logit_margin = logit_values[..., 0] - logit_values[..., 1]
    else:
        prob_margin = torch.zeros_like(top1_prob)
        logit_margin = torch.zeros_like(top1_prob)

    stats = {
        "entropy_mean": masked_mean(entropy, decoder_mask),
        "top1_prob_mean": masked_mean(top1_prob, decoder_mask),
        "prob_margin_mean": masked_mean(prob_margin, decoder_mask),
        "logit_margin_mean": masked_mean(logit_margin, decoder_mask),
    }

    topk_records: List[List[Dict[str, Any]]] = []
    mask_cpu = decoder_mask.detach().cpu()
    token_ids_cpu = token_ids.detach().cpu()
    prob_values_cpu = prob_values.detach().cpu()
    logit_values_cpu = logit_values.detach().cpu()
    for sample_idx in range(logits_f.shape[0]):
        sample_records: List[Dict[str, Any]] = []
        valid_positions = torch.nonzero(mask_cpu[sample_idx], as_tuple=False).flatten().tolist()
        for pos in valid_positions[:top_k_positions]:
            ids = token_ids_cpu[sample_idx, pos].tolist()
            sample_records.append(
                {
                    "position": int(pos),
                    "token_ids": [int(i) for i in ids],
                    "tokens": [decode_token(tokenizer, int(i)) for i in ids],
                    "probabilities": [float(v) for v in prob_values_cpu[sample_idx, pos].tolist()],
                    "logits": [float(v) for v in logit_values_cpu[sample_idx, pos].tolist()],
                }
            )
        topk_records.append(sample_records)

    return stats, topk_records, probs


def make_row_record(
    row_index: int,
    row: Any,
    image_path: Optional[str],
    input_text: Optional[str],
    activation_stats: Dict[str, Dict[str, np.ndarray]],
    output_stats: Optional[Dict[str, np.ndarray]],
    topk_records: Optional[List[Dict[str, Any]]],
    local_idx: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    stat_record = {}
    for name, stats in activation_stats.items():
        stat_record[name] = {
            "label": ACTIVATION_LABELS.get(name, name),
            "mean": float(stats["mean"][local_idx]),
            "std": float(stats["std"][local_idx]),
            "mean_abs": float(stats["mean_abs"][local_idx]),
            "rms": float(stats["rms"][local_idx]),
            "token_count": float(stats["token_count"][local_idx]),
        }

    if output_stats is not None:
        stat_record["final_probability_distribution"] = {
            "label": "Final Probability Distribution",
            "entropy_mean": float(output_stats["entropy_mean"][local_idx]),
            "top1_prob_mean": float(output_stats["top1_prob_mean"][local_idx]),
            "prob_margin_mean": float(output_stats["prob_margin_mean"][local_idx]),
            "logit_margin_mean": float(output_stats["logit_margin_mean"][local_idx]),
            "top_tokens": topk_records or [],
        }

    return {
        "row_index": row_index,
        "input_mode": args.input_mode,
        "image": row_value(row, args.image_field),
        "image_path": image_path,
        "text": input_text,
        "question": row_value(row, args.question_field),
        "answer": row_value(row, args.answer_field),
        "stats": stat_record,
    }


def flatten_record_for_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {
        "row_index": record["row_index"],
        "input_mode": record["input_mode"],
        "image": record["image"],
    }
    for name, values in record["stats"].items():
        for key, value in values.items():
            if key in ("label", "top_tokens"):
                continue
            flat["%s.%s" % (name, key)] = value
    return flat


def save_csv(path: str, records: Sequence[Dict[str, Any]]) -> None:
    rows = [flatten_record_for_csv(r) for r in records]
    fieldnames = sorted({k for row in rows for k in row.keys()})
    if "row_index" in fieldnames:
        fieldnames.remove("row_index")
        fieldnames.insert(0, "row_index")
    if "input_mode" in fieldnames:
        fieldnames.remove("input_mode")
        fieldnames.insert(1, "input_mode")
    if "image" in fieldnames:
        fieldnames.remove("image")
        fieldnames.insert(2, "image")

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summary(records: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_samples": len(records),
        "model_name": args.model_name,
        "model_type": args.model_type,
        "checkpoint": os.path.abspath(args.ckpt),
        "input_mode": args.input_mode,
        "record_shared_t5_space": bool(args.record_shared_t5_space),
        "metrics": {},
    }
    if not records:
        return summary

    metric_values: Dict[str, List[float]] = {}
    for record in records:
        for name, values in record["stats"].items():
            for key, value in values.items():
                if key in ("label", "top_tokens"):
                    continue
                metric_values.setdefault("%s.%s" % (name, key), []).append(float(value))

    for key, values in metric_values.items():
        arr = np.asarray(values, dtype=np.float64)
        summary["metrics"][key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return summary


def setup_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except (ImportError, RuntimeError) as exc:
        raise SystemExit(
            "A compatible matplotlib/NumPy installation is required for static visualization. "
            "Install the project requirements or run: pip install 'numpy<2' matplotlib"
        ) from exc


def make_plots(out_dir: str, records: Sequence[Dict[str, Any]]) -> List[str]:
    plt = setup_matplotlib()
    plot_dir = os.path.join(out_dir, "plots")
    ensure_dir(plot_dir)
    saved: List[str] = []

    if not records:
        return saved

    activation_names = [
        name
        for name in ACTIVATION_LABELS
        if any(name in record["stats"] for record in records)
    ]
    if not activation_names:
        return saved
    mean_abs_values = []
    rms_values = []
    box_data = []
    box_labels = []
    for name in activation_names:
        vals = np.asarray(
            [r["stats"][name]["mean_abs"] for r in records if name in r["stats"]],
            dtype=np.float64,
        )
        rms = np.asarray(
            [r["stats"][name]["rms"] for r in records if name in r["stats"]],
            dtype=np.float64,
        )
        if vals.size:
            mean_abs_values.append(float(np.mean(vals)))
            rms_values.append(float(np.mean(rms)))
            box_data.append(vals)
            box_labels.append(ACTIVATION_LABELS[name])

    x = np.arange(len(box_labels))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.18, mean_abs_values, width=0.36, label="Mean Absolute Activation")
    ax.bar(x + 0.18, rms_values, width=0.36, label="RMS Activation")
    ax.set_xticks(x)
    ax.set_xticklabels(box_labels, rotation=25, ha="right")
    ax.set_ylabel("Activation Magnitude")
    ax.set_title("Activation Magnitude Summary")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(plot_dir, "activation_magnitude_summary.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(box_data, labels=box_labels, showfliers=False)
    ax.set_ylabel("Per-Sample Mean Absolute Activation")
    ax.set_title("Per-Sample Activation Distribution")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    path = os.path.join(plot_dir, "activation_mean_abs_boxplot.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    saved.append(path)

    ncols = 3
    nrows = int(np.ceil(len(activation_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes_flat = np.asarray(axes).reshape(-1)
    for ax, name in zip(axes_flat, activation_names):
        vals = np.asarray(
            [r["stats"][name]["mean_abs"] for r in records if name in r["stats"]],
            dtype=np.float64,
        )
        ax.hist(vals, bins=30, color="#3B82F6", alpha=0.85)
        ax.set_title(ACTIVATION_LABELS[name])
        ax.set_xlabel("Mean Absolute Activation")
        ax.set_ylabel("Sample Count")
    for ax in axes_flat[len(activation_names) :]:
        ax.axis("off")
    fig.suptitle("Activation Histograms", y=1.02)
    fig.tight_layout()
    path = os.path.join(plot_dir, "activation_histograms.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    if all("final_probability_distribution" in r["stats"] for r in records):
        output_metrics = [
            ("entropy_mean", "Entropy"),
            ("top1_prob_mean", "Top-1 Probability"),
            ("prob_margin_mean", "Probability Margin"),
            ("logit_margin_mean", "Logit Margin"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes_flat = axes.reshape(-1)
        for ax, (metric, label) in zip(axes_flat, output_metrics):
            vals = np.asarray(
                [
                    r["stats"]["final_probability_distribution"][metric]
                    for r in records
                ],
                dtype=np.float64,
            )
            ax.hist(vals, bins=30, color="#10B981", alpha=0.85)
            ax.set_title(label)
            ax.set_xlabel(label)
            ax.set_ylabel("Sample Count")
        fig.suptitle("Final Probability Distribution Metrics", y=1.02)
        fig.tight_layout()
        path = os.path.join(plot_dir, "final_probability_distribution_metrics.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved


def make_visual_text_tsne(
    out_dir: str,
    visual_vectors: np.ndarray,
    text_vectors: np.ndarray,
    visual_row_indices: np.ndarray,
    visual_token_positions: np.ndarray,
    text_row_indices: np.ndarray,
    text_token_positions: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Optional[str], Optional[str]]:
    """Plot token-level visual-query and text-token activations."""
    if visual_vectors.ndim != 2 or text_vectors.ndim != 2:
        raise ValueError(
            "Visual and text token vectors must be 2D, got %s and %s."
            % (visual_vectors.shape, text_vectors.shape)
        )
    if visual_vectors.shape[1] != text_vectors.shape[1]:
        raise ValueError(
            "Visual and text token vectors must have the same hidden size, got %d and %d."
            % (visual_vectors.shape[1], text_vectors.shape[1])
        )
    if visual_vectors.shape[0] < 2 or text_vectors.shape[0] < 2:
        print("[WARN] t-SNE requires at least two tokens per modality; skipping t-SNE plot.")
        return None, None

    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
    except (ImportError, RuntimeError) as exc:
        raise SystemExit(
            "A compatible scikit-learn/SciPy/NumPy installation is required for t-SNE. "
            "Install the project requirements or run: pip install 'numpy<2' scipy scikit-learn"
        ) from exc

    rng = np.random.RandomState(args.tsne_random_state)
    def select_indices(num_points: int) -> np.ndarray:
        max_points = min(args.tsne_max_points_per_modality, num_points)
        if max_points < num_points:
            return np.sort(rng.choice(num_points, size=max_points, replace=False))
        return np.arange(num_points)

    visual_selected_indices = select_indices(visual_vectors.shape[0])
    text_selected_indices = select_indices(text_vectors.shape[0])
    visual_selected = visual_vectors[visual_selected_indices].astype(np.float32, copy=False)
    text_selected = text_vectors[text_selected_indices].astype(np.float32, copy=False)
    selected_visual_rows = visual_row_indices[visual_selected_indices]
    selected_visual_positions = visual_token_positions[visual_selected_indices]
    selected_text_rows = text_row_indices[text_selected_indices]
    selected_text_positions = text_token_positions[text_selected_indices]
    features = np.concatenate([visual_selected, text_selected], axis=0)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = StandardScaler().fit_transform(features)

    num_points = features.shape[0]
    perplexity = min(float(args.tsne_perplexity), float(num_points - 1))
    perplexity = max(perplexity, 1.0)
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate=200.0,
        random_state=args.tsne_random_state,
    ).fit_transform(features)

    num_visual = len(visual_selected_indices)
    visual_xy = embedding[:num_visual]
    text_xy = embedding[num_visual:]
    print(
        "t-SNE token points: Visual=%d/%d Text=%d/%d"
        % (
            len(visual_selected_indices),
            visual_vectors.shape[0],
            len(text_selected_indices),
            text_vectors.shape[0],
        )
    )

    plt = setup_matplotlib()
    plot_dir = os.path.join(out_dir, "plots")
    ensure_dir(plot_dir)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(
        visual_xy[:, 0],
        visual_xy[:, 1],
        s=10,
        c="#B9DCEA",
        edgecolors="#5F7F8D",
        linewidths=0.2,
        alpha=0.62,
        label="Visual",
    )
    ax.scatter(
        text_xy[:, 0],
        text_xy[:, 1],
        s=10,
        c="#A9D99B",
        edgecolors="#527A48",
        linewidths=0.2,
        alpha=0.62,
        label="Text",
    )
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title(
        "Token-Level Visual and Text Activation Representations\n"
        "(T5 First-Block Input; Visual=%d, Text=%d)"
        % (len(visual_xy), len(text_xy))
    )
    ax.legend(frameon=True)
    ax.grid(alpha=0.18, linewidth=0.6)
    fig.tight_layout()
    plot_path = os.path.join(plot_dir, "visual_text_activation_tsne.png")
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(out_dir, "visual_text_activation_tsne.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_index",
                "modality",
                "token_position",
                "tsne_dimension_1",
                "tsne_dimension_2",
            ],
        )
        writer.writeheader()
        modality_data = (
            (
                "Visual",
                visual_xy,
                selected_visual_rows,
                selected_visual_positions,
            ),
            (
                "Text",
                text_xy,
                selected_text_rows,
                selected_text_positions,
            ),
        )
        for modality, coords, rows, positions in modality_data:
            for row_index, token_position, xy in zip(rows, positions, coords):
                writer.writerow(
                    {
                        "row_index": int(row_index),
                        "modality": modality,
                        "token_position": int(token_position),
                        "tsne_dimension_1": float(xy[0]),
                        "tsne_dimension_2": float(xy[1]),
                    }
                )

    return plot_path, csv_path


def make_visual_text_activation_histogram(
    out_dir: str,
    visual_vectors: np.ndarray,
    text_vectors: np.ndarray,
    num_bins: int,
) -> Tuple[str, str]:
    """Plot activation values from token-level visual and text representations."""
    visual_values = np.nan_to_num(
        visual_vectors.astype(np.float32, copy=False).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    text_values = np.nan_to_num(
        text_vectors.astype(np.float32, copy=False).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    combined = np.concatenate([visual_values, text_values])
    lower, upper = np.percentile(combined, [0.5, 99.5])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        lower = float(np.min(combined))
        upper = float(np.max(combined))
    if lower >= upper:
        lower -= 0.5
        upper += 0.5

    bin_edges = np.linspace(lower, upper, num_bins + 1)
    visual_density, _ = np.histogram(visual_values, bins=bin_edges, density=True)
    text_density, _ = np.histogram(text_values, bins=bin_edges, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    plt = setup_matplotlib()
    plot_dir = os.path.join(out_dir, "plots")
    ensure_dir(plot_dir)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.hist(
        visual_values,
        bins=bin_edges,
        density=True,
        color="#8FC5DA",
        edgecolor="#567B89",
        linewidth=0.35,
        alpha=0.62,
        label="Visual",
    )
    ax.hist(
        text_values,
        bins=bin_edges,
        density=True,
        color="#93CD81",
        edgecolor="#527A48",
        linewidth=0.35,
        alpha=0.58,
        label="Text",
    )
    ax.axvline(
        float(np.mean(visual_values)),
        color="#416D7E",
        linestyle="--",
        linewidth=1.3,
        label="Visual Mean",
    )
    ax.axvline(
        float(np.mean(text_values)),
        color="#477A3F",
        linestyle="--",
        linewidth=1.3,
        label="Text Mean",
    )
    ax.set_xlim(lower, upper)
    ax.set_xlabel("Activation Value")
    ax.set_ylabel("Density")
    ax.set_title("Token-Level Visual and Text Activation Value Distributions\n(T5 First-Block Input)")
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    fig.tight_layout()
    plot_path = os.path.join(plot_dir, "visual_text_activation_histogram.png")
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(out_dir, "visual_text_activation_histogram.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bin_left",
                "bin_right",
                "bin_center",
                "visual_density",
                "text_density",
            ],
        )
        writer.writeheader()
        for idx in range(num_bins):
            writer.writerow(
                {
                    "bin_left": float(bin_edges[idx]),
                    "bin_right": float(bin_edges[idx + 1]),
                    "bin_center": float(bin_centers[idx]),
                    "visual_density": float(visual_density[idx]),
                    "text_density": float(text_density[idx]),
                }
            )

    return plot_path, csv_path


def make_single_modality_tsne(
    out_dir: str,
    token_vectors: np.ndarray,
    row_indices: np.ndarray,
    token_positions: np.ndarray,
    modality: str,
    activation_label: str,
    args: argparse.Namespace,
) -> Tuple[Optional[str], Optional[str]]:
    """Plot token-level activations for one modality."""
    if token_vectors.ndim != 2:
        raise ValueError("Token vectors must be 2D, got %s." % (token_vectors.shape,))
    if token_vectors.shape[0] < 2:
        print("[WARN] t-SNE requires at least two tokens; skipping %s plot." % modality)
        return None, None

    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
    except (ImportError, RuntimeError) as exc:
        raise SystemExit(
            "A compatible scikit-learn/SciPy/NumPy installation is required for t-SNE. "
            "Install the project requirements or run: pip install 'numpy<2' scipy scikit-learn"
        ) from exc

    rng = np.random.RandomState(args.tsne_random_state)
    num_selected = min(args.tsne_max_points_per_modality, token_vectors.shape[0])
    if num_selected < token_vectors.shape[0]:
        selected = np.sort(
            rng.choice(token_vectors.shape[0], size=num_selected, replace=False)
        )
    else:
        selected = np.arange(token_vectors.shape[0])

    features = token_vectors[selected].astype(np.float32, copy=False)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = StandardScaler().fit_transform(features)
    perplexity = min(float(args.tsne_perplexity), float(features.shape[0] - 1))
    perplexity = max(perplexity, 1.0)
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate=200.0,
        random_state=args.tsne_random_state,
    ).fit_transform(features)

    selected_rows = row_indices[selected]
    selected_positions = token_positions[selected]
    slug = modality.lower()
    print(
        "t-SNE token points: %s=%d/%d"
        % (modality, len(selected), token_vectors.shape[0])
    )

    plt = setup_matplotlib()
    plot_dir = os.path.join(out_dir, "plots")
    ensure_dir(plot_dir)
    color = "#93CD81" if modality == "Text" else "#8FC5DA"
    edge = "#527A48" if modality == "Text" else "#567B89"
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=10,
        c=color,
        edgecolors=edge,
        linewidths=0.2,
        alpha=0.62,
    )
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title(
        "Token-Level %s Activation Representations\n(%s; Tokens=%d)"
        % (modality, activation_label, len(selected))
    )
    ax.grid(alpha=0.18, linewidth=0.6)
    fig.tight_layout()
    plot_path = os.path.join(plot_dir, "%s_token_activation_tsne.png" % slug)
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(out_dir, "%s_token_activation_tsne.csv" % slug)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_index",
                "modality",
                "token_position",
                "tsne_dimension_1",
                "tsne_dimension_2",
            ],
        )
        writer.writeheader()
        for row_index, token_position, xy in zip(
            selected_rows, selected_positions, embedding
        ):
            writer.writerow(
                {
                    "row_index": int(row_index),
                    "modality": modality,
                    "token_position": int(token_position),
                    "tsne_dimension_1": float(xy[0]),
                    "tsne_dimension_2": float(xy[1]),
                }
            )
    return plot_path, csv_path


def make_single_modality_histogram(
    out_dir: str,
    token_vectors: np.ndarray,
    modality: str,
    activation_label: str,
    num_bins: int,
) -> Tuple[str, str]:
    """Plot scalar activation values for one modality."""
    values = np.nan_to_num(
        token_vectors.astype(np.float32, copy=False).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    lower, upper = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        lower = float(np.min(values))
        upper = float(np.max(values))
    if lower >= upper:
        lower -= 0.5
        upper += 0.5

    bin_edges = np.linspace(lower, upper, num_bins + 1)
    density, _ = np.histogram(values, bins=bin_edges, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    slug = modality.lower()

    plt = setup_matplotlib()
    plot_dir = os.path.join(out_dir, "plots")
    ensure_dir(plot_dir)
    color = "#93CD81" if modality == "Text" else "#8FC5DA"
    edge = "#527A48" if modality == "Text" else "#567B89"
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.hist(
        values,
        bins=bin_edges,
        density=True,
        color=color,
        edgecolor=edge,
        linewidth=0.35,
        alpha=0.7,
    )
    ax.axvline(
        float(np.mean(values)),
        color=edge,
        linestyle="--",
        linewidth=1.3,
        label="%s Mean" % modality,
    )
    ax.set_xlim(lower, upper)
    ax.set_xlabel("Activation Value")
    ax.set_ylabel("Density")
    ax.set_title(
        "Token-Level %s Activation Value Distribution\n(%s)"
        % (modality, activation_label)
    )
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    fig.tight_layout()
    plot_path = os.path.join(plot_dir, "%s_token_activation_histogram.png" % slug)
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(out_dir, "%s_token_activation_histogram.csv" % slug)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bin_left", "bin_right", "bin_center", "density"],
        )
        writer.writeheader()
        for idx in range(num_bins):
            writer.writerow(
                {
                    "bin_left": float(bin_edges[idx]),
                    "bin_right": float(bin_edges[idx + 1]),
                    "bin_center": float(bin_centers[idx]),
                    "density": float(density[idx]),
                }
            )
    return plot_path, csv_path


def prepare_multimodal_batch(
    rows: Sequence[Dict[str, Any]],
    row_start: int,
    images_dir: str,
    vis_processor: Any,
    args: argparse.Namespace,
    torch: Any,
    Image: Any,
) -> Tuple[Any, List[str], List[str], List[int], List[str]]:
    images = []
    questions = []
    answers = []
    row_indices = []
    image_paths = []

    for local_idx, row in enumerate(rows):
        row_index = row_start + local_idx
        if not isinstance(row, dict):
            raise TypeError("Row %d must be a JSON object in multimodal mode." % row_index)
        if args.question_field not in row:
            raise KeyError("Row %d missing question field %r" % (row_index, args.question_field))
        if args.image_field not in row:
            raise KeyError("Row %d missing image field %r" % (row_index, args.image_field))

        image_path = resolve_image_path(images_dir, row[args.image_field])
        if not os.path.isfile(image_path):
            raise FileNotFoundError("Image not found for row %d: %s" % (row_index, image_path))

        image = Image.open(image_path).convert("RGB")
        images.append(vis_processor(image))
        questions.append(str(row.get(args.question_field, "")))
        answers.append(answer_to_text(row.get(args.answer_field)))
        row_indices.append(row_index)
        image_paths.append(image_path)

    image_tensor = torch.stack(images)
    return image_tensor, questions, answers, row_indices, image_paths


def prepare_text_batch(
    rows: Sequence[Any],
    row_start: int,
    args: argparse.Namespace,
) -> Tuple[List[str], List[int]]:
    texts = []
    row_indices = []
    for local_idx, row in enumerate(rows):
        row_index = row_start + local_idx
        texts.append(extract_text(row, args, row_index))
        row_indices.append(row_index)
    return texts, row_indices


def prepare_image_batch(
    rows: Sequence[Dict[str, Any]],
    row_start: int,
    images_dir: str,
    vis_processor: Any,
    args: argparse.Namespace,
    torch: Any,
    Image: Any,
) -> Tuple[Any, List[int], List[str]]:
    images = []
    row_indices = []
    image_paths = []
    for local_idx, row in enumerate(rows):
        row_index = row_start + local_idx
        if not isinstance(row, dict):
            raise TypeError("Row %d must be a JSON object in image-only mode." % row_index)
        if args.image_field not in row:
            raise KeyError("Row %d missing image field %r" % (row_index, args.image_field))
        image_path = resolve_image_path(images_dir, row[args.image_field])
        if not os.path.isfile(image_path):
            raise FileNotFoundError("Image not found for row %d: %s" % (row_index, image_path))
        image = Image.open(image_path).convert("RGB")
        images.append(vis_processor(image))
        row_indices.append(row_index)
        image_paths.append(image_path)
    return torch.stack(images), row_indices, image_paths


def forward_multimodal_and_capture(
    model: Any,
    image_tensor: Any,
    questions: Sequence[str],
    answers: Sequence[str],
    args: argparse.Namespace,
    torch: Any,
) -> Dict[str, Any]:
    captures: Dict[str, Any] = {}

    def capture_pre_hook(name: str):
        def hook(_module: Any, inputs: Tuple[Any, ...]) -> None:
            if inputs and getattr(inputs[0], "detach", None) is not None:
                captures[name] = inputs[0].detach()

        return hook

    handles = [
        model.visual_encoder.blocks[0].register_forward_pre_hook(
            capture_pre_hook("vit_first_block_input")
        ),
        model.t5_model.encoder.block[0].register_forward_pre_hook(
            capture_pre_hook("t5_first_block_input")
        ),
    ]

    try:
        image = image_tensor.to(args.device)
        with torch.no_grad():
            with model.maybe_autocast():
                image_embeds = model.ln_vision(model.visual_encoder(image))

            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = model.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_t5 = model.t5_proj(query_output.last_hidden_state)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

            with model.maybe_autocast(dtype=torch.bfloat16):
                input_tokens = model.t5_tokenizer(
                    list(questions),
                    padding="longest",
                    truncation=True,
                    max_length=model.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                output_tokens = model.t5_tokenizer(
                    list(answers),
                    padding="longest",
                    truncation=True,
                    max_length=model.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)

                text_embeds = model.t5_model.encoder.embed_tokens(input_tokens.input_ids)
                encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
                inputs_embeds = torch.cat([inputs_t5, text_embeds], dim=1)

                num_query = inputs_t5.shape[1]
                bsz, seq_len, _ = inputs_embeds.shape
                temp_label = torch.zeros((bsz, seq_len), dtype=torch.bool, device=inputs_embeds.device)
                temp_label[:, :num_query] = True
                model.temp_label = temp_label

                targets = output_tokens.input_ids.masked_fill(
                    output_tokens.input_ids == model.t5_tokenizer.pad_token_id,
                    -100,
                )

                outputs = model.t5_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_atts,
                    decoder_attention_mask=output_tokens.attention_mask,
                    return_dict=True,
                    labels=targets,
                    output_hidden_states=True,
                )

        encoder_last_hidden = getattr(outputs, "encoder_last_hidden_state", None)
        if encoder_last_hidden is None and getattr(outputs, "encoder_hidden_states", None) is not None:
            encoder_last_hidden = outputs.encoder_hidden_states[-1]
        decoder_hidden_states = getattr(outputs, "decoder_hidden_states", None)
        if decoder_hidden_states is None:
            raise RuntimeError("T5 output did not return decoder_hidden_states.")
        if encoder_last_hidden is None:
            raise RuntimeError("T5 output did not return encoder_last_hidden_state.")

        if "vit_first_block_input" not in captures:
            raise RuntimeError("Failed to capture ViT first-block input activation.")
        if "t5_first_block_input" not in captures:
            captures["t5_first_block_input"] = inputs_embeds.detach()

        return {
            "t5_text_embedding": text_embeds.detach(),
            "vit_first_block_input": captures["vit_first_block_input"],
            "t5_first_block_input": captures["t5_first_block_input"],
            "num_query_tokens": int(num_query),
            "t5_encoder_last_hidden": encoder_last_hidden.detach(),
            "t5_decoder_last_hidden": decoder_hidden_states[-1].detach(),
            "final_logits": outputs.logits.detach(),
            "input_attention_mask": input_tokens.attention_mask.detach(),
            "encoder_attention_mask": encoder_atts.detach(),
            "decoder_attention_mask": output_tokens.attention_mask.detach(),
        }
    finally:
        for handle in handles:
            handle.remove()


def forward_text_only_and_capture(
    model: Any,
    texts: Sequence[str],
    args: argparse.Namespace,
    torch: Any,
) -> Dict[str, Any]:
    captures: Dict[str, Any] = {}

    def capture_t5_input(_module: Any, inputs: Tuple[Any, ...]) -> None:
        if inputs and getattr(inputs[0], "detach", None) is not None:
            captures["t5_first_block_input"] = inputs[0].detach()

    handle = model.t5_model.encoder.block[0].register_forward_pre_hook(capture_t5_input)
    try:
        device = next(model.t5_model.parameters()).device
        with torch.no_grad():
            with model.maybe_autocast(dtype=torch.bfloat16):
                input_tokens = model.t5_tokenizer(
                    list(texts),
                    padding="longest",
                    truncation=True,
                    max_length=model.max_txt_len,
                    return_tensors="pt",
                ).to(device)
                output_tokens = model.t5_tokenizer(
                    list(texts),
                    padding="longest",
                    truncation=True,
                    max_length=model.max_txt_len,
                    return_tensors="pt",
                ).to(device)
                text_embeds = model.t5_model.encoder.embed_tokens(input_tokens.input_ids)
                bsz, seq_len = text_embeds.shape[:2]
                model.temp_label = torch.zeros(
                    (bsz, seq_len), dtype=torch.bool, device=device
                )
                targets = output_tokens.input_ids.masked_fill(
                    output_tokens.input_ids == model.t5_tokenizer.pad_token_id,
                    -100,
                )
                outputs = model.t5_model(
                    inputs_embeds=text_embeds,
                    attention_mask=input_tokens.attention_mask,
                    decoder_attention_mask=output_tokens.attention_mask,
                    return_dict=True,
                    labels=targets,
                    output_hidden_states=True,
                )

        encoder_last_hidden = getattr(outputs, "encoder_last_hidden_state", None)
        if encoder_last_hidden is None and getattr(outputs, "encoder_hidden_states", None) is not None:
            encoder_last_hidden = outputs.encoder_hidden_states[-1]
        decoder_hidden_states = getattr(outputs, "decoder_hidden_states", None)
        if encoder_last_hidden is None or decoder_hidden_states is None:
            raise RuntimeError("T5 text-only output did not return hidden states.")
        if "t5_first_block_input" not in captures:
            captures["t5_first_block_input"] = text_embeds.detach()

        return {
            "t5_text_embedding": text_embeds.detach(),
            "t5_first_block_input": captures["t5_first_block_input"],
            "t5_encoder_last_hidden": encoder_last_hidden.detach(),
            "t5_decoder_last_hidden": decoder_hidden_states[-1].detach(),
            "final_logits": outputs.logits.detach(),
            "input_attention_mask": input_tokens.attention_mask.detach(),
            "encoder_attention_mask": input_tokens.attention_mask.detach(),
            "decoder_attention_mask": output_tokens.attention_mask.detach(),
        }
    finally:
        handle.remove()


def forward_image_only_and_capture(
    model: Any,
    image_tensor: Any,
    args: argparse.Namespace,
    torch: Any,
) -> Dict[str, Any]:
    captures: Dict[str, Any] = {}

    def capture_vit_input(_module: Any, inputs: Tuple[Any, ...]) -> None:
        if inputs and getattr(inputs[0], "detach", None) is not None:
            captures["vit_first_block_input"] = inputs[0].detach()

    handle = model.visual_encoder.blocks[0].register_forward_pre_hook(capture_vit_input)
    try:
        image = image_tensor.to(args.device)
        with torch.no_grad():
            with model.maybe_autocast():
                vit_last_hidden = model.ln_vision(model.visual_encoder(image))
        if "vit_first_block_input" not in captures:
            raise RuntimeError("Failed to capture ViT first-block input activation.")
        result = {
            "vit_first_block_input": captures["vit_first_block_input"],
            "vit_last_hidden": vit_last_hidden.detach(),
        }
        if args.record_shared_t5_space:
            image_atts = torch.ones(
                vit_last_hidden.size()[:-1],
                dtype=torch.long,
                device=vit_last_hidden.device,
            )
            query_tokens = model.query_tokens.expand(
                vit_last_hidden.shape[0], -1, -1
            )
            with torch.no_grad():
                query_output = model.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=vit_last_hidden,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                result["t5_visual_query_input"] = model.t5_proj(
                    query_output.last_hidden_state
                ).detach()
        return result
    finally:
        handle.remove()


def append_metrics(
    metric_store: Dict[str, List[float]],
    activation_stats: Dict[str, Dict[str, np.ndarray]],
    output_stats: Optional[Dict[str, np.ndarray]],
) -> None:
    for name, stats in activation_stats.items():
        for key in ("mean", "std", "mean_abs", "rms", "token_count"):
            metric_store.setdefault("%s.%s" % (name, key), []).extend(
                [float(v) for v in stats[key].tolist()]
            )
    if output_stats is not None:
        for key, values in output_stats.items():
            metric_store.setdefault("final_probability_distribution.%s" % key, []).extend(
                [float(v) for v in values.tolist()]
            )


def save_full_tensor_batch(
    path: str,
    row_indices: Sequence[int],
    image_paths: Sequence[str],
    captured: Dict[str, Any],
    probs: Optional[Any],
    torch: Any,
) -> None:
    payload = {
        "row_indices": torch.tensor(list(row_indices), dtype=torch.long),
        "image_paths": list(image_paths),
    }
    for name, value in captured.items():
        if getattr(value, "detach", None) is not None:
            payload[name] = value.detach().cpu()
        else:
            payload[name] = value
    if probs is not None:
        payload["final_probabilities"] = probs.detach().cpu()
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.top_k < 1:
        raise ValueError("--top_k must be >= 1")
    if args.top_k_positions < 0:
        raise ValueError("--top_k_positions must be >= 0")
    if args.tsne_perplexity <= 0:
        raise ValueError("--tsne_perplexity must be > 0")
    if args.tsne_max_points_per_modality < 2:
        raise ValueError("--tsne_max_points_per_modality must be >= 2")
    if args.activation_histogram_bins < 2:
        raise ValueError("--activation_histogram_bins must be >= 2")
    if args.record_shared_t5_space and args.input_mode != "vit_image_only":
        raise ValueError(
            "--record_shared_t5_space is only valid with --input_mode vit_image_only"
        )

    try:
        import torch
        from PIL import Image
        from lavis.models import load_model
        from lavis.processors import load_processor
    except ImportError as exc:
        raise SystemExit(
            "Missing LAVIS runtime dependency: %s. Run this in the LAVIS environment."
            % exc
        ) from exc

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    calib_json = os.path.abspath(args.calib_json)
    images_dir = (
        os.path.abspath(args.images_dir)
        if args.images_dir
        else os.path.join(os.path.dirname(calib_json), "images")
    )
    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)

    rows = load_rows(calib_json, args.max_samples)
    if not rows:
        raise RuntimeError("No rows found in %s" % calib_json)

    print("Loaded %d rows from %s" % (len(rows), calib_json))
    print("input_mode:", args.input_mode)
    if args.input_mode != "t5_c4_text":
        print("images_dir:", images_dir)
    print("out_dir:", out_dir)
    print("device:", args.device)

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
    vis_processor = None
    if args.input_mode != "t5_c4_text":
        vis_processor = load_processor("blip_image_eval").build(image_size=args.image_size)

    records: List[Dict[str, Any]] = []
    metric_store: Dict[str, List[float]] = {}
    pooled_vectors: Dict[str, List[np.ndarray]] = {}
    visual_token_vectors: List[np.ndarray] = []
    visual_token_row_indices: List[np.ndarray] = []
    visual_token_positions: List[np.ndarray] = []
    text_token_vectors: List[np.ndarray] = []
    text_token_row_indices: List[np.ndarray] = []
    text_token_positions: List[np.ndarray] = []
    shared_visual_token_vectors: List[np.ndarray] = []
    shared_visual_token_row_indices: List[np.ndarray] = []
    shared_visual_token_positions: List[np.ndarray] = []
    row_index_values: List[int] = []
    image_values: List[str] = []
    text_values: List[str] = []
    question_values: List[str] = []
    answer_values: List[str] = []

    jsonl_path = os.path.join(out_dir, "activation_summary.jsonl")
    full_dir = os.path.join(out_dir, "full_tensors")
    if args.save_full_tensors:
        ensure_dir(full_dir)

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_f:
        for batch_start, batch_rows in iter_batches(rows, args.batch_size):
            output_stats: Optional[Dict[str, np.ndarray]] = None
            topk_records: Optional[List[List[Dict[str, Any]]]] = None
            probs: Optional[Any] = None

            if args.input_mode == "multimodal":
                (
                    image_tensor,
                    questions,
                    answers,
                    row_indices,
                    image_paths,
                ) = prepare_multimodal_batch(
                    batch_rows,
                    batch_start,
                    images_dir,
                    vis_processor,
                    args,
                    torch,
                    Image,
                )
                input_texts: List[Optional[str]] = list(questions)
                captured = forward_multimodal_and_capture(
                    model, image_tensor, questions, answers, args, torch
                )
                activation_masks = {
                    "t5_text_embedding": captured["input_attention_mask"],
                    "vit_first_block_input": None,
                    "t5_first_block_input": captured["encoder_attention_mask"],
                    "t5_encoder_last_hidden": captured["encoder_attention_mask"],
                    "t5_decoder_last_hidden": captured["decoder_attention_mask"],
                    "final_logits": captured["decoder_attention_mask"],
                }

                num_query = captured["num_query_tokens"]
                first_block_input = captured["t5_first_block_input"]
                visual_batch = first_block_input[:, :num_query, :].detach().float().cpu()
                visual_token_vectors.append(
                    visual_batch.reshape(-1, visual_batch.shape[-1]).numpy()
                )
                visual_token_row_indices.append(
                    np.repeat(np.asarray(row_indices, dtype=np.int64), num_query)
                )
                visual_token_positions.append(
                    np.tile(np.arange(num_query, dtype=np.int64), len(row_indices))
                )

                text_batch = first_block_input[:, num_query:, :].detach().float().cpu()
                text_mask = captured["input_attention_mask"].detach().bool().cpu()
                valid_text_positions = torch.nonzero(text_mask, as_tuple=False)
                text_token_vectors.append(text_batch[text_mask].numpy())
                text_token_row_indices.append(
                    np.asarray(row_indices, dtype=np.int64)[
                        valid_text_positions[:, 0].numpy()
                    ]
                )
                text_token_positions.append(
                    valid_text_positions[:, 1].numpy().astype(np.int64, copy=False)
                )
            elif args.input_mode == "t5_c4_text":
                texts, row_indices = prepare_text_batch(batch_rows, batch_start, args)
                image_paths = [None] * len(row_indices)
                input_texts = list(texts)
                captured = forward_text_only_and_capture(model, texts, args, torch)
                activation_masks = {
                    "t5_text_embedding": captured["input_attention_mask"],
                    "t5_first_block_input": captured["encoder_attention_mask"],
                    "t5_encoder_last_hidden": captured["encoder_attention_mask"],
                    "t5_decoder_last_hidden": captured["decoder_attention_mask"],
                    "final_logits": captured["decoder_attention_mask"],
                }

                text_batch = captured["t5_first_block_input"].detach().float().cpu()
                text_mask = captured["input_attention_mask"].detach().bool().cpu()
                valid_text_positions = torch.nonzero(text_mask, as_tuple=False)
                text_token_vectors.append(text_batch[text_mask].numpy())
                text_token_row_indices.append(
                    np.asarray(row_indices, dtype=np.int64)[
                        valid_text_positions[:, 0].numpy()
                    ]
                )
                text_token_positions.append(
                    valid_text_positions[:, 1].numpy().astype(np.int64, copy=False)
                )
            else:
                image_tensor, row_indices, image_paths = prepare_image_batch(
                    batch_rows,
                    batch_start,
                    images_dir,
                    vis_processor,
                    args,
                    torch,
                    Image,
                )
                input_texts = [None] * len(row_indices)
                captured = forward_image_only_and_capture(
                    model, image_tensor, args, torch
                )
                activation_masks = {
                    "vit_first_block_input": None,
                    "vit_last_hidden": None,
                }
                if "t5_visual_query_input" in captured:
                    activation_masks["t5_visual_query_input"] = None

                visual_batch = (
                    captured["vit_first_block_input"].detach().float().cpu()
                )
                num_visual_tokens = visual_batch.shape[1]
                visual_token_vectors.append(
                    visual_batch.reshape(-1, visual_batch.shape[-1]).numpy()
                )
                visual_token_row_indices.append(
                    np.repeat(
                        np.asarray(row_indices, dtype=np.int64), num_visual_tokens
                    )
                )
                visual_token_positions.append(
                    np.tile(
                        np.arange(num_visual_tokens, dtype=np.int64),
                        len(row_indices),
                    )
                )
                if "t5_visual_query_input" in captured:
                    shared_visual_batch = (
                        captured["t5_visual_query_input"].detach().float().cpu()
                    )
                    num_query_tokens = shared_visual_batch.shape[1]
                    shared_visual_token_vectors.append(
                        shared_visual_batch.reshape(
                            -1, shared_visual_batch.shape[-1]
                        ).numpy()
                    )
                    shared_visual_token_row_indices.append(
                        np.repeat(
                            np.asarray(row_indices, dtype=np.int64),
                            num_query_tokens,
                        )
                    )
                    shared_visual_token_positions.append(
                        np.tile(
                            np.arange(num_query_tokens, dtype=np.int64),
                            len(row_indices),
                        )
                    )

            activation_stats = {
                name: sequence_stats(captured[name], mask)
                for name, mask in activation_masks.items()
            }
            if "final_logits" in captured:
                output_stats, topk_records, probs = output_distribution_stats(
                    captured["final_logits"],
                    captured["decoder_attention_mask"],
                    model.t5_tokenizer,
                    args.top_k,
                    args.top_k_positions,
                )
            append_metrics(metric_store, activation_stats, output_stats)

            for name, mask in activation_masks.items():
                if name == "final_logits":
                    continue
                pooled_vectors.setdefault(name, []).append(
                    masked_sequence_mean(captured[name], mask)
                )

            for local_idx, row in enumerate(batch_rows):
                row_index = row_indices[local_idx]
                record = make_row_record(
                    row_index=row_index,
                    row=row,
                    image_path=image_paths[local_idx],
                    input_text=input_texts[local_idx],
                    activation_stats=activation_stats,
                    output_stats=output_stats,
                    topk_records=(
                        topk_records[local_idx]
                        if topk_records is not None
                        else None
                    ),
                    local_idx=local_idx,
                    args=args,
                )
                records.append(record)
                jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                row_index_values.append(row_index)
                image_values.append(str(row_value(row, args.image_field, "") or ""))
                text_values.append(str(input_texts[local_idx] or ""))
                question_values.append(str(row_value(row, args.question_field, "") or ""))
                answer_values.append(answer_to_text(row_value(row, args.answer_field)))

            jsonl_f.flush()

            if args.save_full_tensors:
                save_full_tensor_batch(
                    os.path.join(full_dir, "batch_%06d.pt" % batch_start),
                    row_indices,
                    [path or "" for path in image_paths],
                    captured,
                    probs,
                    torch,
                )

            done = batch_start + len(batch_rows)
            if args.log_every > 0 and (done == len(rows) or done % args.log_every == 0):
                print("Processed %d/%d rows" % (done, len(rows)))

    npz_payload: Dict[str, Any] = {
        "row_index": np.asarray(row_index_values, dtype=np.int64),
        "input_mode": np.asarray([args.input_mode] * len(row_index_values), dtype=str),
        "record_shared_t5_space": np.asarray(
            [bool(args.record_shared_t5_space)], dtype=np.bool_
        ),
        "image": np.asarray(image_values, dtype=str),
        "text": np.asarray(text_values, dtype=str),
        "question": np.asarray(question_values, dtype=str),
        "answer": np.asarray(answer_values, dtype=str),
    }
    concatenated_vectors = {
        name: np.concatenate(chunks, axis=0)
        for name, chunks in pooled_vectors.items()
    }
    for name, values in concatenated_vectors.items():
        npz_payload[name + "_pooled"] = values
    all_visual_token_vectors = (
        np.concatenate(visual_token_vectors, axis=0)
        if visual_token_vectors
        else None
    )
    all_visual_token_rows = (
        np.concatenate(visual_token_row_indices, axis=0)
        if visual_token_row_indices
        else None
    )
    all_visual_token_positions = (
        np.concatenate(visual_token_positions, axis=0)
        if visual_token_positions
        else None
    )
    all_text_token_vectors = (
        np.concatenate(text_token_vectors, axis=0)
        if text_token_vectors
        else None
    )
    all_text_token_rows = (
        np.concatenate(text_token_row_indices, axis=0)
        if text_token_row_indices
        else None
    )
    all_text_token_positions = (
        np.concatenate(text_token_positions, axis=0)
        if text_token_positions
        else None
    )
    all_shared_visual_token_vectors = (
        np.concatenate(shared_visual_token_vectors, axis=0)
        if shared_visual_token_vectors
        else None
    )
    all_shared_visual_token_rows = (
        np.concatenate(shared_visual_token_row_indices, axis=0)
        if shared_visual_token_row_indices
        else None
    )
    all_shared_visual_token_positions = (
        np.concatenate(shared_visual_token_positions, axis=0)
        if shared_visual_token_positions
        else None
    )
    if all_visual_token_vectors is not None:
        visual_prefix = (
            "vit_first_block_input"
            if args.input_mode == "vit_image_only"
            else "t5_visual_query_first_block_input"
        )
        npz_payload[visual_prefix + "_tokens"] = all_visual_token_vectors
        npz_payload[visual_prefix + "_token_row_index"] = all_visual_token_rows
        npz_payload[visual_prefix + "_token_position"] = all_visual_token_positions
    if all_text_token_vectors is not None:
        npz_payload["t5_text_first_block_input_tokens"] = all_text_token_vectors
        npz_payload["t5_text_token_row_index"] = all_text_token_rows
        npz_payload["t5_text_token_position"] = all_text_token_positions
    if all_shared_visual_token_vectors is not None:
        npz_payload[
            "t5_visual_query_first_block_input_tokens"
        ] = all_shared_visual_token_vectors
        npz_payload[
            "t5_visual_query_token_row_index"
        ] = all_shared_visual_token_rows
        npz_payload[
            "t5_visual_query_token_position"
        ] = all_shared_visual_token_positions
    for metric_name, values in metric_store.items():
        npz_payload[metric_name.replace(".", "__")] = np.asarray(values, dtype=np.float32)

    np.savez_compressed(os.path.join(out_dir, "activation_vectors_and_metrics.npz"), **npz_payload)
    save_csv(os.path.join(out_dir, "activation_summary.csv"), records)
    summary = aggregate_summary(records, args)
    with open(os.path.join(out_dir, "activation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    plot_paths = make_plots(out_dir, records)
    tsne_csv_path: Optional[str]
    if args.input_mode == "multimodal":
        tsne_plot_path, tsne_csv_path = make_visual_text_tsne(
            out_dir=out_dir,
            visual_vectors=all_visual_token_vectors,
            text_vectors=all_text_token_vectors,
            visual_row_indices=all_visual_token_rows,
            visual_token_positions=all_visual_token_positions,
            text_row_indices=all_text_token_rows,
            text_token_positions=all_text_token_positions,
            args=args,
        )
        if tsne_plot_path:
            plot_paths.append(tsne_plot_path)
        histogram_plot_path, histogram_csv_path = make_visual_text_activation_histogram(
            out_dir=out_dir,
            visual_vectors=all_visual_token_vectors,
            text_vectors=all_text_token_vectors,
            num_bins=args.activation_histogram_bins,
        )
    elif args.input_mode == "t5_c4_text":
        tsne_plot_path, tsne_csv_path = make_single_modality_tsne(
            out_dir,
            all_text_token_vectors,
            all_text_token_rows,
            all_text_token_positions,
            "Text",
            "T5 First-Block Input",
            args,
        )
        if tsne_plot_path:
            plot_paths.append(tsne_plot_path)
        histogram_plot_path, histogram_csv_path = make_single_modality_histogram(
            out_dir,
            all_text_token_vectors,
            "Text",
            "T5 First-Block Input",
            args.activation_histogram_bins,
        )
    else:
        tsne_plot_path, tsne_csv_path = make_single_modality_tsne(
            out_dir,
            all_visual_token_vectors,
            all_visual_token_rows,
            all_visual_token_positions,
            "Visual",
            "ViT First-Block Input",
            args,
        )
        if tsne_plot_path:
            plot_paths.append(tsne_plot_path)
        histogram_plot_path, histogram_csv_path = make_single_modality_histogram(
            out_dir,
            all_visual_token_vectors,
            "Visual",
            "ViT First-Block Input",
            args.activation_histogram_bins,
        )
    plot_paths.append(histogram_plot_path)
    print("[OK] wrote:", jsonl_path)
    print("[OK] wrote:", os.path.join(out_dir, "activation_vectors_and_metrics.npz"))
    print("[OK] wrote:", os.path.join(out_dir, "activation_summary.csv"))
    print("[OK] wrote:", os.path.join(out_dir, "activation_summary.json"))
    for path in plot_paths:
        print("[OK] plot:", path)
    if tsne_csv_path:
        print("[OK] wrote:", tsne_csv_path)
    print("[OK] wrote:", histogram_csv_path)
    if args.save_full_tensors:
        print("[OK] full tensors:", full_dir)


if __name__ == "__main__":
    main()
