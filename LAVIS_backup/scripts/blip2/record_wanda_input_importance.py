#!/usr/bin/env python3
"""Record Wanda-style input activations and weight importance for BLIP2-T5.

This script is diagnostic only; it does not prune or modify the model.  It
records the input activation seen by each target Linear layer and computes the
same per-weight importance form used by Wanda:

    importance[o, i] = abs(weight[o, i]) * sqrt(input_activation_scale[i])

Supported paths:
  - cc3m_multimodal: image + text through BLIP2-T5.
  - t5_text_only: C4/text rows through T5 only.
  - vit_image_only: images through ViT + ln_vision only.

The target Linear layers are T5 attention q/k/v/o, T5 FFN wi/wi_0/wi_1/wo,
and ViT attention qkv/proj plus MLP fc1/fc2.
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


AUTO_TEXT_FIELDS = ("question", "caption", "text_input", "text", "prompt", "output")
AUTO_OUTPUT_FIELDS = ("text_output", "answer", "caption", "text", "question", "output")


@dataclass
class TargetInfo:
    name: str
    module: Any
    component: str
    layer: int
    role: str


class WandaInputCollector:
    """Accumulate input scales in the same shape used by Wanda."""

    def __init__(self, module: Any):
        import torch

        self.module = module
        self.device = module.weight.device
        self.columns = int(module.weight.shape[1])
        self.scaler_row = torch.zeros((self.columns,), device=self.device)
        self.token_sumsq = torch.zeros((self.columns,), device=self.device)
        self.nsamples = 0
        self.token_rows = 0

    def add(self, inp: Any) -> None:
        if inp is None:
            return
        if isinstance(inp, (list, tuple)):
            if not inp:
                return
            inp = inp[0]
        if not hasattr(inp, "detach"):
            return

        x = inp.detach()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        batch_size = int(x.shape[0])
        flat = x.reshape(-1, x.shape[-1]).float()
        if flat.shape[-1] != self.columns:
            raise RuntimeError(
                "Input dim mismatch for %s: got %d, expected %d."
                % (self.module.__class__.__name__, flat.shape[-1], self.columns)
            )

        # Match the repo's WrappedGPT accumulation: norm over all token rows,
        # normalized by sample count.  This intentionally keeps sequence length
        # in the scale, because that is what Wanda sees in this codebase.
        self.scaler_row *= self.nsamples / max(self.nsamples + batch_size, 1)
        self.nsamples += batch_size
        self.scaler_row += flat.t().norm(p=2, dim=1).pow(2) / max(self.nsamples, 1)

        self.token_sumsq += flat.pow(2).sum(dim=0)
        self.token_rows += int(flat.shape[0])

    def token_mean_input_sq(self) -> Any:
        return self.token_sumsq / max(self.token_rows, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record Wanda-style Linear input activations and importance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_mode",
        choices=["cc3m_multimodal", "t5_text_only", "vit_image_only"],
        required=True,
        help="Forward path to analyze.",
    )
    parser.add_argument("--calib_json", required=True)
    parser.add_argument("--images_dir", default=None)
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
        help="Encoder/text input field. auto tries question/caption/text_input/text/prompt/output.",
    )
    parser.add_argument(
        "--output_field",
        default="auto",
        help="Decoder target field for modes using T5 decoder. auto tries text_output/answer/caption/text/question/output.",
    )
    parser.add_argument("--no_decoder", action="store_true", help="Skip T5 decoder path.")
    parser.add_argument("--max_txt_len", type=int, default=None)
    parser.add_argument("--importance_hist_bins", type=int, default=80)
    parser.add_argument(
        "--wanda_sparsity",
        type=float,
        default=0.5,
        help="Sparsity ratio used when materializing Wanda pruned masks for comparison.",
    )
    parser.add_argument(
        "--mask_style",
        choices=["auto", "row", "global"],
        default="auto",
        help="Mask construction style. auto uses row-wise masks for T5 and global masks for ViT.",
    )
    parser.add_argument(
        "--save_wanda_mask",
        action="store_true",
        help="Save full Wanda pruned-mask arrays. This can be large for T5-XL.",
    )
    parser.add_argument(
        "--save_wanda_metric",
        action="store_true",
        help="Save full W_metric arrays. This is very large; disabled by default.",
    )
    parser.add_argument("--log_every", type=int, default=20)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_rows(path: str) -> List[Any]:
    rows: List[Any]
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


def select_rows(rows: Sequence[Any], max_samples: Optional[int], shuffle: bool, seed: int) -> Tuple[List[Any], List[int]]:
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


def value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value if v is not None).strip()
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values() if v is not None).strip()
    return str(value).strip()


def extract_text(row: Any, field: str, auto_fields: Sequence[str], row_index: int) -> Tuple[str, str]:
    if isinstance(row, str):
        text = row.strip()
        if not text:
            raise ValueError("Row %d is empty text." % row_index)
        return text, "string"
    if not isinstance(row, dict):
        raise TypeError("Row %d must be a string or JSON object." % row_index)
    fields = [field] if field != "auto" else list(auto_fields)
    selected = next((name for name in fields if name in row and value_to_text(row.get(name))), None)
    if selected is None:
        raise KeyError("Row %d has none of these non-empty fields: %s" % (row_index, ", ".join(fields)))
    return value_to_text(row.get(selected)), selected


def resolve_image_path(images_dir: str, image_value: Any) -> str:
    path = os.path.expanduser(str(image_value))
    return path if os.path.isabs(path) else os.path.join(images_dir, path)


def tokenize_with_length_stats(tokenizer: Any, texts: Sequence[str], max_txt_len: int, device: str) -> Tuple[Any, List[int]]:
    full = tokenizer(list(texts), padding=False, truncation=False, add_special_tokens=True)
    original_lengths = [len(ids) for ids in full["input_ids"]]
    tokens = tokenizer(
        list(texts),
        padding="longest",
        truncation=True,
        max_length=max_txt_len,
        return_tensors="pt",
    ).to(device)
    return tokens, original_lengths


def parse_t5_layer(name: str) -> Tuple[str, int, str]:
    match = re.match(r"t5_model\.(encoder|decoder)\.block\.(\d+)\.(.+)$", name)
    if not match:
        raise ValueError("Cannot parse T5 module name: %s" % name)
    stack = match.group(1)
    layer = int(match.group(2))
    tail = match.group(3)
    if ".SelfAttention." in tail:
        role = "self_attn_" + tail.rsplit(".", 1)[-1]
    elif ".EncDecAttention." in tail:
        role = "cross_attn_" + tail.rsplit(".", 1)[-1]
    elif ".DenseReluDense." in tail:
        role = "ffn_" + tail.rsplit(".", 1)[-1]
    else:
        role = tail.replace(".", "_")
    return "t5_" + stack, layer, role


def parse_vit_layer(name: str) -> Tuple[str, int, str]:
    match = re.match(r"visual_encoder\.blocks\.(\d+)\.(.+)$", name)
    if not match:
        raise ValueError("Cannot parse ViT module name: %s" % name)
    layer = int(match.group(1))
    tail = match.group(2)
    if tail == "attn.qkv":
        role = "attn_qkv"
    elif tail == "attn.proj":
        role = "attn_proj"
    elif tail == "mlp.fc1":
        role = "mlp_fc1"
    elif tail == "mlp.fc2":
        role = "mlp_fc2"
    else:
        role = tail.replace(".", "_")
    return "vit", layer, role


def is_target_t5(name: str) -> bool:
    if not re.match(r"t5_model\.(encoder|decoder)\.block\.\d+\.", name):
        return False
    wanted_suffixes = (
        ".SelfAttention.q",
        ".SelfAttention.k",
        ".SelfAttention.v",
        ".SelfAttention.o",
        ".EncDecAttention.q",
        ".EncDecAttention.k",
        ".EncDecAttention.v",
        ".EncDecAttention.o",
        ".DenseReluDense.wi",
        ".DenseReluDense.wi_0",
        ".DenseReluDense.wi_1",
        ".DenseReluDense.wo",
    )
    return name.endswith(wanted_suffixes)


def is_target_vit(name: str) -> bool:
    if not re.match(r"visual_encoder\.blocks\.\d+\.", name):
        return False
    return name.endswith(("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"))


def collect_target_linears(model: Any, input_mode: str, include_decoder: bool) -> List[TargetInfo]:
    import torch.nn as nn

    targets: List[TargetInfo] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if input_mode == "vit_image_only":
            if is_target_vit(name):
                component, layer, role = parse_vit_layer(name)
                targets.append(TargetInfo(name, module, component, layer, role))
            continue

        if input_mode == "t5_text_only":
            if is_target_t5(name):
                component, layer, role = parse_t5_layer(name)
                if component == "t5_decoder" and not include_decoder:
                    continue
                targets.append(TargetInfo(name, module, component, layer, role))
            continue

        if is_target_vit(name):
            component, layer, role = parse_vit_layer(name)
            targets.append(TargetInfo(name, module, component, layer, role))
        elif is_target_t5(name):
            component, layer, role = parse_t5_layer(name)
            if component == "t5_decoder" and not include_decoder:
                continue
            targets.append(TargetInfo(name, module, component, layer, role))

    targets.sort(key=lambda item: (item.component, item.layer, item.role, item.name))
    return targets


def attach_collectors(targets: Sequence[TargetInfo]) -> Tuple[Dict[str, WandaInputCollector], List[Any]]:
    collectors: Dict[str, WandaInputCollector] = {}
    handles: List[Any] = []
    for target in targets:
        collectors[target.name] = WandaInputCollector(target.module)

        def make_hook(module_name: str):
            def hook(_module: Any, inputs: Tuple[Any, ...], _output: Any) -> None:
                collectors[module_name].add(inputs)

            return hook

        handles.append(target.module.register_forward_hook(make_hook(target.name)))
    return collectors, handles


def tensor_stats(x: Any, prefix: str) -> Dict[str, float]:
    import torch

    flat = x.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {
            prefix + "_mean": 0.0,
            prefix + "_std": 0.0,
            prefix + "_p50": 0.0,
            prefix + "_p90": 0.0,
            prefix + "_p95": 0.0,
            prefix + "_p99": 0.0,
            prefix + "_max": 0.0,
        }
    quantiles = torch.quantile(
        flat,
        torch.tensor([0.5, 0.9, 0.95, 0.99], device=flat.device, dtype=flat.dtype),
    )
    return {
        prefix + "_mean": float(flat.mean().item()),
        prefix + "_std": float(flat.std(unbiased=False).item()),
        prefix + "_p50": float(quantiles[0].item()),
        prefix + "_p90": float(quantiles[1].item()),
        prefix + "_p95": float(quantiles[2].item()),
        prefix + "_p99": float(quantiles[3].item()),
        prefix + "_max": float(flat.max().item()),
    }


def infer_mask_style(target: TargetInfo, requested_style: str) -> str:
    if requested_style != "auto":
        return requested_style
    if target.component.startswith("t5_"):
        return "row"
    return "global"


def wanda_pruned_mask(importance: Any, sparsity: float, style: str) -> Any:
    import torch

    if sparsity <= 0:
        return torch.zeros_like(importance, dtype=torch.bool)
    if sparsity >= 1:
        return torch.ones_like(importance, dtype=torch.bool)

    mask = torch.zeros_like(importance, dtype=torch.bool)
    if style == "row":
        cols = int(importance.shape[1])
        k = int(cols * sparsity)
        if k <= 0:
            return mask
        if k >= cols:
            return torch.ones_like(importance, dtype=torch.bool)
        indices = torch.sort(importance, dim=-1, stable=True).indices[:, :k]
        mask.scatter_(1, indices, True)
        return mask

    flat = importance.reshape(-1)
    k = int(flat.numel() * sparsity)
    if k <= 0:
        return mask
    if k >= flat.numel():
        return torch.ones_like(importance, dtype=torch.bool)
    indices = torch.sort(flat, stable=True).indices[:k]
    mask.reshape(-1)[indices] = True
    return mask


def run_multimodal_batch(
    model: Any,
    batch_rows: Sequence[Any],
    original_indices: Sequence[int],
    images_dir: str,
    vis_processor: Any,
    args: argparse.Namespace,
    torch: Any,
    Image: Any,
    metadata: Dict[str, Any],
) -> None:
    images = []
    input_texts: List[str] = []
    output_texts: List[str] = []
    for local_index, row in enumerate(batch_rows):
        original_index = original_indices[local_index]
        if not isinstance(row, dict) or args.image_field not in row:
            raise KeyError("Row %d is missing image field %r." % (original_index, args.image_field))
        image_path = resolve_image_path(images_dir, row[args.image_field])
        if not os.path.isfile(image_path):
            raise FileNotFoundError("Image not found for row %d: %s" % (original_index, image_path))
        with Image.open(image_path) as image:
            images.append(vis_processor(image.convert("RGB")))
        text, text_field = extract_text(row, args.text_field, AUTO_TEXT_FIELDS, original_index)
        output, output_field = extract_text(row, args.output_field, AUTO_OUTPUT_FIELDS, original_index)
        input_texts.append(text)
        output_texts.append(output)
        metadata["selected_text_fields"][text_field] = metadata["selected_text_fields"].get(text_field, 0) + 1
        metadata["selected_output_fields"][output_field] = metadata["selected_output_fields"].get(output_field, 0) + 1

    image_tensor = torch.stack(images).to(args.device)
    with torch.no_grad():
        with model.maybe_autocast():
            image_hidden = model.ln_vision(model.visual_encoder(image_tensor))
            image_atts = torch.ones(image_hidden.size()[:-1], dtype=torch.long, device=image_hidden.device)
            query_tokens = model.query_tokens.expand(image_hidden.shape[0], -1, -1)
            query_output = model.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_hidden,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            visual_tokens = model.t5_proj(query_output.last_hidden_state)

        with model.maybe_autocast(dtype=torch.bfloat16):
            input_tokens, original_lengths = tokenize_with_length_stats(
                model.t5_tokenizer, input_texts, model.max_txt_len, args.device
            )
            visual_attention = torch.ones(visual_tokens.size()[:-1], dtype=torch.long, device=visual_tokens.device)
            encoder_attention = torch.cat([visual_attention, input_tokens.attention_mask], dim=1)
            input_embeddings = model.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            encoder_embeddings = torch.cat([visual_tokens, input_embeddings], dim=1)
            model.temp_label = torch.zeros_like(encoder_attention, dtype=torch.bool)
            model.temp_label[:, : visual_tokens.shape[1]] = True

            if args.no_decoder:
                model.t5_model.encoder(
                    inputs_embeds=encoder_embeddings,
                    attention_mask=encoder_attention,
                    return_dict=True,
                )
            else:
                target_tokens, target_original_lengths = tokenize_with_length_stats(
                    model.t5_tokenizer, output_texts, model.max_txt_len, args.device
                )
                targets = target_tokens.input_ids.masked_fill(
                    target_tokens.input_ids == model.t5_tokenizer.pad_token_id,
                    -100,
                )
                model.t5_model(
                    inputs_embeds=encoder_embeddings,
                    attention_mask=encoder_attention,
                    decoder_attention_mask=target_tokens.attention_mask,
                    labels=targets,
                    return_dict=True,
                )
                metadata["original_output_tokens"] += int(sum(target_original_lengths))
                metadata["retained_output_tokens"] += int(target_tokens.attention_mask.sum().item())
                metadata["truncated_output_samples"] += int(
                    sum(int(a > b) for a, b in zip(target_original_lengths, target_tokens.attention_mask.sum(dim=1).tolist()))
                )

    retained_lengths = input_tokens.attention_mask.sum(dim=1).detach().cpu().tolist()
    metadata["original_input_tokens"] += int(sum(original_lengths))
    metadata["retained_input_tokens"] += int(sum(retained_lengths))
    metadata["truncated_input_samples"] += int(sum(int(a > b) for a, b in zip(original_lengths, retained_lengths)))


def run_t5_text_batch(
    model: Any,
    batch_rows: Sequence[Any],
    original_indices: Sequence[int],
    args: argparse.Namespace,
    torch: Any,
    metadata: Dict[str, Any],
) -> None:
    texts: List[str] = []
    for local_index, row in enumerate(batch_rows):
        text, text_field = extract_text(row, args.text_field, AUTO_TEXT_FIELDS, original_indices[local_index])
        texts.append(text)
        metadata["selected_text_fields"][text_field] = metadata["selected_text_fields"].get(text_field, 0) + 1

    with torch.no_grad():
        with model.maybe_autocast(dtype=torch.bfloat16):
            input_tokens, original_lengths = tokenize_with_length_stats(
                model.t5_tokenizer, texts, model.max_txt_len, args.device
            )
            embeddings = model.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            attention = input_tokens.attention_mask
            model.temp_label = torch.zeros_like(attention, dtype=torch.bool)
            if args.no_decoder:
                model.t5_model.encoder(inputs_embeds=embeddings, attention_mask=attention, return_dict=True)
            else:
                target_tokens, target_original_lengths = tokenize_with_length_stats(
                    model.t5_tokenizer, texts, model.max_txt_len, args.device
                )
                targets = target_tokens.input_ids.masked_fill(
                    target_tokens.input_ids == model.t5_tokenizer.pad_token_id,
                    -100,
                )
                model.t5_model(
                    inputs_embeds=embeddings,
                    attention_mask=attention,
                    decoder_attention_mask=target_tokens.attention_mask,
                    labels=targets,
                    return_dict=True,
                )
                metadata["original_output_tokens"] += int(sum(target_original_lengths))
                metadata["retained_output_tokens"] += int(target_tokens.attention_mask.sum().item())

    retained_lengths = input_tokens.attention_mask.sum(dim=1).detach().cpu().tolist()
    metadata["original_input_tokens"] += int(sum(original_lengths))
    metadata["retained_input_tokens"] += int(sum(retained_lengths))
    metadata["truncated_input_samples"] += int(sum(int(a > b) for a, b in zip(original_lengths, retained_lengths)))


def run_vit_image_batch(
    model: Any,
    batch_rows: Sequence[Any],
    original_indices: Sequence[int],
    images_dir: str,
    vis_processor: Any,
    args: argparse.Namespace,
    torch: Any,
    Image: Any,
) -> None:
    images = []
    for local_index, row in enumerate(batch_rows):
        original_index = original_indices[local_index]
        if not isinstance(row, dict) or args.image_field not in row:
            raise KeyError("Row %d is missing image field %r." % (original_index, args.image_field))
        image_path = resolve_image_path(images_dir, row[args.image_field])
        if not os.path.isfile(image_path):
            raise FileNotFoundError("Image not found for row %d: %s" % (original_index, image_path))
        with Image.open(image_path) as image:
            images.append(vis_processor(image.convert("RGB")))
    image_tensor = torch.stack(images).to(args.device)
    with torch.no_grad():
        with model.maybe_autocast():
            model.ln_vision(model.visual_encoder(image_tensor))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_collectors(
    targets: Sequence[TargetInfo],
    collectors: Dict[str, WandaInputCollector],
    hist_bins: int,
    wanda_sparsity: float,
    mask_style: str,
    save_wanda_mask: bool,
    save_wanda_metric: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, np.ndarray]]:
    import torch

    module_rows: List[Dict[str, Any]] = []
    layer_acc: Dict[Tuple[str, int], Dict[str, float]] = {}
    arrays: Dict[str, np.ndarray] = {}

    for target in targets:
        collector = collectors[target.name]
        weight_abs = target.module.weight.detach().float().abs()
        scaler = collector.scaler_row.detach().float().clamp_min(0)
        token_mean_sq = collector.token_mean_input_sq().detach().float().clamp_min(0)
        importance = weight_abs * torch.sqrt(scaler.reshape(1, -1))
        weight_col_mean = weight_abs.mean(dim=0)
        importance_col_mean = importance.mean(dim=0)
        resolved_mask_style = infer_mask_style(target, mask_style)
        pruned_mask = wanda_pruned_mask(importance, wanda_sparsity, resolved_mask_style)

        row: Dict[str, Any] = {
            "component": target.component,
            "layer": target.layer,
            "role": target.role,
            "module_name": target.name,
            "weight_shape": "x".join(str(v) for v in target.module.weight.shape),
            "out_dim": int(target.module.weight.shape[0]),
            "in_dim": int(target.module.weight.shape[1]),
            "weight_numel": int(target.module.weight.numel()),
            "wanda_nsamples": int(collector.nsamples),
            "token_rows": int(collector.token_rows),
            "wanda_sparsity": float(wanda_sparsity),
            "wanda_mask_style": resolved_mask_style,
            "wanda_pruned_numel": int(pruned_mask.sum().item()),
            "wanda_pruned_fraction": float(pruned_mask.float().mean().item()) if pruned_mask.numel() else 0.0,
        }
        row.update(tensor_stats(scaler, "wanda_scaler"))
        row.update(tensor_stats(token_mean_sq, "token_mean_input_sq"))
        row.update(tensor_stats(weight_abs, "weight_abs"))
        row.update(tensor_stats(weight_col_mean, "weight_abs_col_mean"))
        row.update(tensor_stats(importance, "wanda_importance"))
        row.update(tensor_stats(importance_col_mean, "wanda_importance_col_mean"))

        hist_max = float(importance.detach().float().max().item()) if importance.numel() else 0.0
        if hist_max > 0:
            hist = torch.histc(importance.detach().float(), bins=hist_bins, min=0.0, max=hist_max)
        else:
            hist = torch.zeros((hist_bins,), device=importance.device)
        row["importance_hist_max"] = hist_max
        row["importance_hist_bins"] = hist_bins
        module_rows.append(row)

        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", target.name)
        arrays[safe_key + ".wanda_scaler"] = scaler.cpu().numpy().astype(np.float32)
        arrays[safe_key + ".token_mean_input_sq"] = token_mean_sq.cpu().numpy().astype(np.float32)
        arrays[safe_key + ".weight_abs_col_mean"] = weight_col_mean.cpu().numpy().astype(np.float32)
        arrays[safe_key + ".wanda_importance_col_mean"] = importance_col_mean.cpu().numpy().astype(np.float32)
        arrays[safe_key + ".wanda_importance_hist"] = hist.cpu().numpy().astype(np.float32)
        if save_wanda_mask:
            arrays[safe_key + ".wanda_pruned_mask"] = pruned_mask.cpu().numpy().astype(np.uint8)
        if save_wanda_metric:
            arrays[safe_key + ".wanda_metric"] = importance.cpu().numpy().astype(np.float32)

        layer_key = (target.component, target.layer)
        acc = layer_acc.setdefault(
            layer_key,
            {
                "component": target.component,
                "layer": target.layer,
                "modules": 0,
                "weight_numel": 0,
                "wanda_importance_mean_weighted_sum": 0.0,
                "weight_abs_mean_weighted_sum": 0.0,
                "wanda_scaler_mean_sum": 0.0,
                "token_mean_input_sq_mean_sum": 0.0,
                "wanda_importance_max": 0.0,
            },
        )
        acc["modules"] += 1
        acc["weight_numel"] += row["weight_numel"]
        acc["wanda_importance_mean_weighted_sum"] += row["wanda_importance_mean"] * row["weight_numel"]
        acc["weight_abs_mean_weighted_sum"] += row["weight_abs_mean"] * row["weight_numel"]
        acc["wanda_scaler_mean_sum"] += row["wanda_scaler_mean"]
        acc["token_mean_input_sq_mean_sum"] += row["token_mean_input_sq_mean"]
        acc["wanda_importance_max"] = max(acc["wanda_importance_max"], row["wanda_importance_max"])

    layer_rows: List[Dict[str, Any]] = []
    for (_component, _layer), acc in sorted(layer_acc.items(), key=lambda item: (item[0][0], item[0][1])):
        numel = max(int(acc["weight_numel"]), 1)
        modules = max(int(acc["modules"]), 1)
        layer_rows.append(
            {
                "component": acc["component"],
                "layer": int(acc["layer"]),
                "modules": int(acc["modules"]),
                "weight_numel": int(acc["weight_numel"]),
                "wanda_importance_mean": acc["wanda_importance_mean_weighted_sum"] / numel,
                "weight_abs_mean": acc["weight_abs_mean_weighted_sum"] / numel,
                "wanda_scaler_mean": acc["wanda_scaler_mean_sum"] / modules,
                "token_mean_input_sq_mean": acc["token_mean_input_sq_mean_sum"] / modules,
                "wanda_importance_max": acc["wanda_importance_max"],
            }
        )
    return module_rows, layer_rows, arrays


def make_plots(out_dir: str, layer_rows: Sequence[Dict[str, Any]], module_rows: Sequence[Dict[str, Any]]) -> List[str]:
    if not layer_rows:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print("[WARN] matplotlib unavailable, skipping plots: %s" % exc)
        return []

    paths: List[str] = []
    for metric, ylabel, filename in [
        ("wanda_importance_mean", "Mean Wanda importance", "layer_wanda_importance_mean.png"),
        ("wanda_scaler_mean", "Mean Wanda input scaler", "layer_wanda_input_scaler_mean.png"),
        ("weight_abs_mean", "Mean |weight|", "layer_weight_abs_mean.png"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 6))
        for component in sorted({row["component"] for row in layer_rows}):
            rows = [row for row in layer_rows if row["component"] == component]
            rows.sort(key=lambda row: int(row["layer"]))
            ax.plot([row["layer"] for row in rows], [row[metric] for row in rows], marker="o", label=component)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by Layer")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)

    roles = sorted({row["role"] for row in module_rows})
    if roles:
        role_means = []
        for role in roles:
            rows = [row for row in module_rows if row["role"] == role]
            total = sum(int(row["weight_numel"]) for row in rows)
            value = sum(float(row["wanda_importance_mean"]) * int(row["weight_numel"]) for row in rows) / max(total, 1)
            role_means.append(value)
        fig, ax = plt.subplots(figsize=(max(10, len(roles) * 0.5), 5))
        ax.bar(roles, role_means)
        ax.set_ylabel("Mean Wanda importance")
        ax.set_title("Wanda Importance by Linear Role")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        path = os.path.join(out_dir, "role_wanda_importance_mean.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)

    return paths


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    import torch
    from PIL import Image
    from lavis.models import load_model
    from lavis.processors import load_processor

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.max_txt_len is not None:
        # The model value is overridden after loading.
        pass
    if args.input_mode in {"cc3m_multimodal", "vit_image_only"} and not args.images_dir:
        raise ValueError("--images_dir is required for image modes.")

    rows_all = load_rows(args.calib_json)
    rows, original_indices = select_rows(rows_all, args.max_samples, args.shuffle, args.seed)

    model = load_model(
        args.model_name,
        args.model_type,
        is_eval=True,
        device=args.device,
        checkpoint=args.ckpt,
    )
    model.eval()
    if args.max_txt_len is not None:
        model.max_txt_len = int(args.max_txt_len)
    vis_processor = None
    if args.input_mode in {"cc3m_multimodal", "vit_image_only"}:
        vis_processor = load_processor("blip_image_eval").build(image_size=args.image_size)

    targets = collect_target_linears(model, args.input_mode, include_decoder=not args.no_decoder)
    if not targets:
        raise RuntimeError("No target Linear layers were found for input_mode=%s." % args.input_mode)
    collectors, handles = attach_collectors(targets)

    metadata: Dict[str, Any] = {
        "input_mode": args.input_mode,
        "calib_json": args.calib_json,
        "images_dir": args.images_dir,
        "ckpt": args.ckpt,
        "model_name": args.model_name,
        "model_type": args.model_type,
        "device": args.device,
        "num_rows_total": len(rows_all),
        "num_rows_used": len(rows),
        "batch_size": args.batch_size,
        "max_txt_len": model.max_txt_len,
        "target_linear_count": len(targets),
        "selected_text_fields": {},
        "selected_output_fields": {},
        "original_input_tokens": 0,
        "retained_input_tokens": 0,
        "truncated_input_samples": 0,
        "original_output_tokens": 0,
        "retained_output_tokens": 0,
        "truncated_output_samples": 0,
        "importance_definition": "abs(weight[o,i]) * sqrt(wanda_scaler[i])",
        "wanda_scaler_definition": "Matches WrappedGPT in lavis/compression/pruners/wanda_pruner.py: squared L2 norm of Linear input columns normalized by sample count.",
        "token_mean_input_sq_definition": "Squared Linear input activation averaged over flattened token rows; diagnostic only.",
        "wanda_sparsity": args.wanda_sparsity,
        "mask_style": args.mask_style,
        "save_wanda_mask": bool(args.save_wanda_mask),
        "save_wanda_metric": bool(args.save_wanda_metric),
        "wanda_mask_definition": "T5 auto masks match Wanda row-wise lowest W_metric per output row; ViT auto masks match Wanda global lowest W_metric per Linear.",
    }

    try:
        for batch_number, (start, batch_rows) in enumerate(iter_batches(rows, args.batch_size), start=1):
            batch_original_indices = original_indices[start : start + len(batch_rows)]
            if args.input_mode == "cc3m_multimodal":
                run_multimodal_batch(
                    model,
                    batch_rows,
                    batch_original_indices,
                    args.images_dir,
                    vis_processor,
                    args,
                    torch,
                    Image,
                    metadata,
                )
            elif args.input_mode == "t5_text_only":
                run_t5_text_batch(model, batch_rows, batch_original_indices, args, torch, metadata)
            elif args.input_mode == "vit_image_only":
                run_vit_image_batch(
                    model,
                    batch_rows,
                    batch_original_indices,
                    args.images_dir,
                    vis_processor,
                    args,
                    torch,
                    Image,
                )
            else:  # pragma: no cover
                raise ValueError(args.input_mode)

            processed = min(start + len(batch_rows), len(rows))
            if args.log_every and (processed % args.log_every == 0 or processed == len(rows)):
                print("Processed %d/%d rows" % (processed, len(rows)))
    finally:
        for handle in handles:
            handle.remove()

    module_rows, layer_rows, arrays = summarize_collectors(
        targets,
        collectors,
        args.importance_hist_bins,
        args.wanda_sparsity,
        args.mask_style,
        args.save_wanda_mask,
        args.save_wanda_metric,
    )
    module_csv = os.path.join(args.out_dir, "wanda_linear_input_importance_by_module.csv")
    layer_csv = os.path.join(args.out_dir, "wanda_linear_input_importance_by_layer.csv")
    write_csv(module_csv, module_rows)
    write_csv(layer_csv, layer_rows)

    with open(os.path.join(args.out_dir, "wanda_linear_input_importance_by_module.json"), "w", encoding="utf-8") as handle:
        json.dump(module_rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with open(os.path.join(args.out_dir, "wanda_linear_input_importance_by_layer.json"), "w", encoding="utf-8") as handle:
        json.dump(layer_rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with open(os.path.join(args.out_dir, "wanda_input_importance_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    np.savez_compressed(os.path.join(args.out_dir, "wanda_linear_input_importance_arrays.npz"), **arrays)
    plot_paths = make_plots(args.out_dir, layer_rows, module_rows)
    print("[OK] target Linear layers:", len(targets))
    print("[OK] wrote:", module_csv)
    print("[OK] wrote:", layer_csv)
    for path in plot_paths:
        print("[OK] plot:", path)


if __name__ == "__main__":
    main()
