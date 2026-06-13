#!/usr/bin/env python3
"""Record BLIP2-T5 input/output activations and make static plots.

This script is for comparing calibration input distributions, for example:

  python scripts/blip2/record_blip2_activation_distributions.py \
    --calib_json /data/data2/mfs/MMMU_calibration/mmmu_calibration_train.json \
    --images_dir /data/data2/mfs/MMMU_calibration/images \
    --ckpt /data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth \
    --out_dir /data/data2/mfs/MMMU_calibration/activation_original \
    --batch_size 4 \
    --max_samples 128

Recorded tensors:
  - T5 text embedding activation
  - ViT first-block input activation
  - T5 encoder first-block input activation
  - Final logits and probability-distribution summaries
  - T5 decoder last hidden states
  - T5 encoder last hidden states

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
    "t5_first_block_input": "T5 First-Block Input",
    "t5_encoder_last_hidden": "T5 Encoder Last Hidden",
    "t5_decoder_last_hidden": "T5 Decoder Last Hidden",
    "final_logits": "Final Logits",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Record BLIP2-T5 activation distributions for calibration JSON data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--calib_json", required=True, help="Input calibration JSON list.")
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
    ap.add_argument("--log_every", type=int, default=20)
    return ap.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_rows(path: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise TypeError("Input calibration JSON must be a list of objects.")
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError("Row %d is not a JSON object." % idx)
    return rows if max_samples is None else rows[:max_samples]


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


def iter_batches(rows: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[Tuple[int, List[Dict[str, Any]]]]:
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
    row: Dict[str, Any],
    image_path: str,
    activation_stats: Dict[str, Dict[str, np.ndarray]],
    output_stats: Dict[str, np.ndarray],
    topk_records: List[Dict[str, Any]],
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

    stat_record["final_probability_distribution"] = {
        "label": "Final Probability Distribution",
        "entropy_mean": float(output_stats["entropy_mean"][local_idx]),
        "top1_prob_mean": float(output_stats["top1_prob_mean"][local_idx]),
        "prob_margin_mean": float(output_stats["prob_margin_mean"][local_idx]),
        "logit_margin_mean": float(output_stats["logit_margin_mean"][local_idx]),
        "top_tokens": topk_records,
    }

    return {
        "row_index": row_index,
        "image": row.get(args.image_field),
        "image_path": image_path,
        "question": row.get(args.question_field),
        "answer": row.get(args.answer_field),
        "stats": stat_record,
    }


def flatten_record_for_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {
        "row_index": record["row_index"],
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
    if "image" in fieldnames:
        fieldnames.remove("image")
        fieldnames.insert(1, "image")

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
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for static visualization. Install it with: pip install matplotlib"
        ) from exc


def make_plots(out_dir: str, records: Sequence[Dict[str, Any]]) -> List[str]:
    plt = setup_matplotlib()
    plot_dir = os.path.join(out_dir, "plots")
    ensure_dir(plot_dir)
    saved: List[str] = []

    if not records:
        return saved

    activation_names = list(ACTIVATION_LABELS.keys())
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


def prepare_batch(
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


def forward_and_capture(
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


def append_metrics(
    metric_store: Dict[str, List[float]],
    activation_stats: Dict[str, Dict[str, np.ndarray]],
    output_stats: Dict[str, np.ndarray],
) -> None:
    for name, stats in activation_stats.items():
        for key in ("mean", "std", "mean_abs", "rms", "token_count"):
            metric_store.setdefault("%s.%s" % (name, key), []).extend(
                [float(v) for v in stats[key].tolist()]
            )
    for key, values in output_stats.items():
        metric_store.setdefault("final_probability_distribution.%s" % key, []).extend(
            [float(v) for v in values.tolist()]
        )


def save_full_tensor_batch(
    path: str,
    row_indices: Sequence[int],
    image_paths: Sequence[str],
    captured: Dict[str, Any],
    probs: Any,
    torch: Any,
) -> None:
    payload = {
        "row_indices": torch.tensor(list(row_indices), dtype=torch.long),
        "image_paths": list(image_paths),
        "t5_text_embedding": captured["t5_text_embedding"].detach().cpu(),
        "vit_first_block_input": captured["vit_first_block_input"].detach().cpu(),
        "t5_first_block_input": captured["t5_first_block_input"].detach().cpu(),
        "t5_encoder_last_hidden": captured["t5_encoder_last_hidden"].detach().cpu(),
        "t5_decoder_last_hidden": captured["t5_decoder_last_hidden"].detach().cpu(),
        "final_logits": captured["final_logits"].detach().cpu(),
        "final_probabilities": probs.detach().cpu(),
        "input_attention_mask": captured["input_attention_mask"].detach().cpu(),
        "encoder_attention_mask": captured["encoder_attention_mask"].detach().cpu(),
        "decoder_attention_mask": captured["decoder_attention_mask"].detach().cpu(),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.top_k < 1:
        raise ValueError("--top_k must be >= 1")
    if args.top_k_positions < 0:
        raise ValueError("--top_k_positions must be >= 0")

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
    vis_processor = load_processor("blip_image_eval").build(image_size=args.image_size)

    records: List[Dict[str, Any]] = []
    metric_store: Dict[str, List[float]] = {}
    pooled_vectors: Dict[str, List[np.ndarray]] = {
        "t5_text_embedding": [],
        "vit_first_block_input": [],
        "t5_first_block_input": [],
        "t5_encoder_last_hidden": [],
        "t5_decoder_last_hidden": [],
    }
    row_index_values: List[int] = []
    image_values: List[str] = []
    question_values: List[str] = []
    answer_values: List[str] = []

    jsonl_path = os.path.join(out_dir, "activation_summary.jsonl")
    full_dir = os.path.join(out_dir, "full_tensors")
    if args.save_full_tensors:
        ensure_dir(full_dir)

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_f:
        for batch_start, batch_rows in iter_batches(rows, args.batch_size):
            (
                image_tensor,
                questions,
                answers,
                row_indices,
                image_paths,
            ) = prepare_batch(
                batch_rows,
                batch_start,
                images_dir,
                vis_processor,
                args,
                torch,
                Image,
            )

            captured = forward_and_capture(model, image_tensor, questions, answers, args, torch)

            activation_stats = {
                "t5_text_embedding": sequence_stats(
                    captured["t5_text_embedding"],
                    captured["input_attention_mask"],
                ),
                "vit_first_block_input": sequence_stats(
                    captured["vit_first_block_input"],
                    None,
                ),
                "t5_first_block_input": sequence_stats(
                    captured["t5_first_block_input"],
                    captured["encoder_attention_mask"],
                ),
                "t5_encoder_last_hidden": sequence_stats(
                    captured["t5_encoder_last_hidden"],
                    captured["encoder_attention_mask"],
                ),
                "t5_decoder_last_hidden": sequence_stats(
                    captured["t5_decoder_last_hidden"],
                    captured["decoder_attention_mask"],
                ),
                "final_logits": sequence_stats(
                    captured["final_logits"],
                    captured["decoder_attention_mask"],
                ),
            }
            output_stats, topk_records, probs = output_distribution_stats(
                captured["final_logits"],
                captured["decoder_attention_mask"],
                model.t5_tokenizer,
                args.top_k,
                args.top_k_positions,
            )
            append_metrics(metric_store, activation_stats, output_stats)

            pooled_vectors["t5_text_embedding"].append(
                masked_sequence_mean(captured["t5_text_embedding"], captured["input_attention_mask"])
            )
            pooled_vectors["vit_first_block_input"].append(
                masked_sequence_mean(captured["vit_first_block_input"], None)
            )
            pooled_vectors["t5_first_block_input"].append(
                masked_sequence_mean(captured["t5_first_block_input"], captured["encoder_attention_mask"])
            )
            pooled_vectors["t5_encoder_last_hidden"].append(
                masked_sequence_mean(captured["t5_encoder_last_hidden"], captured["encoder_attention_mask"])
            )
            pooled_vectors["t5_decoder_last_hidden"].append(
                masked_sequence_mean(captured["t5_decoder_last_hidden"], captured["decoder_attention_mask"])
            )

            for local_idx, row in enumerate(batch_rows):
                row_index = row_indices[local_idx]
                record = make_row_record(
                    row_index=row_index,
                    row=row,
                    image_path=image_paths[local_idx],
                    activation_stats=activation_stats,
                    output_stats=output_stats,
                    topk_records=topk_records[local_idx],
                    local_idx=local_idx,
                    args=args,
                )
                records.append(record)
                jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                row_index_values.append(row_index)
                image_values.append(str(row.get(args.image_field)))
                question_values.append(str(row.get(args.question_field, "")))
                answer_values.append(answer_to_text(row.get(args.answer_field)))

            jsonl_f.flush()

            if args.save_full_tensors:
                save_full_tensor_batch(
                    os.path.join(full_dir, "batch_%06d.pt" % batch_start),
                    row_indices,
                    image_paths,
                    captured,
                    probs,
                    torch,
                )

            done = batch_start + len(batch_rows)
            if args.log_every > 0 and (done == len(rows) or done % args.log_every == 0):
                print("Processed %d/%d rows" % (done, len(rows)))

    npz_payload: Dict[str, Any] = {
        "row_index": np.asarray(row_index_values, dtype=np.int64),
        "image": np.asarray(image_values, dtype=str),
        "question": np.asarray(question_values, dtype=str),
        "answer": np.asarray(answer_values, dtype=str),
    }
    for name, chunks in pooled_vectors.items():
        npz_payload[name + "_pooled"] = np.concatenate(chunks, axis=0)
    for metric_name, values in metric_store.items():
        npz_payload[metric_name.replace(".", "__")] = np.asarray(values, dtype=np.float32)

    np.savez_compressed(os.path.join(out_dir, "activation_vectors_and_metrics.npz"), **npz_payload)
    save_csv(os.path.join(out_dir, "activation_summary.csv"), records)
    summary = aggregate_summary(records, args)
    with open(os.path.join(out_dir, "activation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    plot_paths = make_plots(out_dir, records)
    print("[OK] wrote:", jsonl_path)
    print("[OK] wrote:", os.path.join(out_dir, "activation_vectors_and_metrics.npz"))
    print("[OK] wrote:", os.path.join(out_dir, "activation_summary.csv"))
    print("[OK] wrote:", os.path.join(out_dir, "activation_summary.json"))
    for path in plot_paths:
        print("[OK] plot:", path)
    if args.save_full_tensors:
        print("[OK] full tensors:", full_dir)


if __name__ == "__main__":
    main()
