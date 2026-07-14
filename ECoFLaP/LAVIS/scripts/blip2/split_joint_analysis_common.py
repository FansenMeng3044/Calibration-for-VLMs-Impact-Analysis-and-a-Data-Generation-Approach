#!/usr/bin/env python3
"""Shared loading / forward helpers for the split-vs-joint pruning analysis.

The three analysis scripts (sparsity allocation, Wanda token attribution,
pruned-activation drift) all need to build the *same* BLIP2-T5 encoder forward
so their numbers are comparable.  That forward is reproduced here once.

Token layout inside the T5 encoder (see blip2_t5.py):

    inputs_embeds = cat([t5_proj(qformer_out), embed_tokens(text_ids)], dim=1)

so encoder position 0..num_query-1 is the *visual prefix*, and num_query.. is
*text* (padded).  Wanda's ``WrappedGPT.add_batch`` flattens [B, T, C] over all
tokens, padding included, so padding is tracked as its own group.
"""

from __future__ import annotations

import csv
import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


VIT_PREFIX = "visual_encoder"
T5_PREFIX = "t5_model"

AUTO_INPUT_FIELDS = ("question", "caption", "text_input", "text", "prompt")
AUTO_OUTPUT_FIELDS = ("text_output", "answer", "caption", "text", "question")


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def load_rows(path: str) -> List[Any]:
    if path.lower().endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        for key in ("data", "annotations", "rows", "samples"):
            if key in data and isinstance(data[key], list):
                return data[key]
        raise ValueError("Could not find a sample list inside %s" % path)
    if not isinstance(data, list):
        raise ValueError("Unsupported JSON structure in %s" % path)
    return data


def select_rows(
    rows: Sequence[Any],
    max_samples: Optional[int],
    shuffle: bool,
    seed: int,
) -> Tuple[List[Any], List[int]]:
    indices = list(range(len(rows)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    if max_samples is not None:
        indices = indices[:max_samples]
    return [rows[i] for i in indices], indices


def iter_batches(rows: Sequence[Any], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield start, list(rows[start : start + batch_size])


def resolve_image_path(images_dir: str, image_value: Any) -> str:
    value = str(image_value)
    if os.path.isabs(value):
        return value
    return os.path.join(images_dir, value)


def value_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)) and value:
        return value_to_text(value[0])
    if value is None:
        return ""
    return str(value).strip()


def extract_text(row: Any, field: str, auto_fields: Sequence[str], row_index: int) -> str:
    if field != "auto":
        if field not in row:
            raise KeyError("Row %d is missing text field %r." % (row_index, field))
        return value_to_text(row[field])
    for candidate in auto_fields:
        if candidate in row:
            text = value_to_text(row[candidate])
            if text:
                return text
    raise KeyError(
        "Row %d has none of the auto text fields %s." % (row_index, list(auto_fields))
    )


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(os.path.dirname(path))
    if not rows:
        open(path, "w", encoding="utf-8").close()
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def setup_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print("[WARN] matplotlib unavailable; CSVs will still be written: %s" % exc)
        return None
    return plt


def parse_labeled_paths(items: Sequence[str]) -> "Dict[str, str]":
    """Parse ``label=/path/to.pth`` arguments, preserving order."""
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("Expected label=path, got %r" % item)
        label, path = item.split("=", 1)
        label = label.strip()
        path = os.path.abspath(os.path.expanduser(path.strip().strip('"')))
        if not label:
            raise ValueError("Empty label in %r" % item)
        if label in out:
            raise ValueError("Duplicate label %r" % label)
        out[label] = path
    return out


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------
def load_state_dict(path: str) -> Dict[str, Any]:
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "module"):
            inner = obj.get(key)
            if isinstance(inner, dict) and inner:
                return inner
    if not isinstance(obj, dict):
        raise ValueError("Unsupported checkpoint structure: %s" % path)
    return obj


def prunable_block_group(name: str, ndim: int) -> Optional[Tuple[str, str, int, str]]:
    """Replicate ``BLIPT5LayerWandaPruner.get_sparsity`` block grouping.

    Returns ``(model, submodel, block_index, group_key)`` or None when the
    tensor is not one the pruner would ever touch.
    """
    if ndim != 2:
        return None
    if ".block" not in name:
        return None
    if "relative_attention_bias" in name:
        return None

    parts = name.split(".")
    if name.startswith(T5_PREFIX + "."):
        # t5_model.<encoder|decoder>.block.<i>....
        if len(parts) < 4 or parts[2] != "block":
            return None
        try:
            index = int(parts[3])
        except ValueError:
            return None
        return "t5", parts[1], index, ".".join(parts[:4])
    if name.startswith(VIT_PREFIX + "."):
        # visual_encoder.blocks.<i>....
        if len(parts) < 3 or parts[1] != "blocks":
            return None
        try:
            index = int(parts[2])
        except ValueError:
            return None
        return "vit", "blocks", index, ".".join(parts[:3])
    return None


# --------------------------------------------------------------------------
# model + forward
# --------------------------------------------------------------------------
def load_blip2(
    model_name: str,
    model_type: str,
    device: str,
    checkpoint: Optional[str] = None,
    max_txt_len: Optional[int] = None,
):
    from lavis.models import load_model

    kwargs = dict(is_eval=True, device=device)
    if checkpoint:
        kwargs["checkpoint"] = checkpoint
    model = load_model(model_name, model_type, **kwargs)
    model.eval()
    if max_txt_len is not None:
        model.max_txt_len = max_txt_len
    return model


def build_vis_processor(image_size: int):
    from lavis.processors import load_processor

    return load_processor("blip_image_eval").build(image_size=image_size)


def load_batch_images(
    batch_rows: Sequence[Any],
    start: int,
    original_indices: Sequence[int],
    images_dir: str,
    image_field: str,
    vis_processor: Any,
    torch: Any,
    Image: Any,
):
    images = []
    for local_index, row in enumerate(batch_rows):
        original_index = original_indices[start + local_index]
        if not isinstance(row, dict) or image_field not in row:
            raise KeyError("Row %d is missing image field %r." % (original_index, image_field))
        image_path = resolve_image_path(images_dir, row[image_field])
        if not os.path.isfile(image_path):
            raise FileNotFoundError("Image not found for row %d: %s" % (original_index, image_path))
        with Image.open(image_path) as image:
            images.append(vis_processor(image.convert("RGB")))
    return torch.stack(images)


class EncoderForward:
    """Reproduce blip2_t5's encoder-side forward and expose the token groups.

    ``padding`` matters:

      * ``longest`` matches what the pruner actually saw during calibration, so
        use it when you are auditing the Wanda statistic (step 1).
      * ``max_length`` gives a fixed sequence length across checkpoints, which
        is what you need to compare activations position-by-position (step 2).
    """

    def __init__(self, model: Any, torch: Any, padding: str = "longest"):
        self.model = model
        self.torch = torch
        self.padding = padding

    def tokenize(self, texts: Sequence[str], device: str):
        return self.model.t5_tokenizer(
            list(texts),
            padding=self.padding,
            truncation=True,
            max_length=self.model.max_txt_len,
            return_tensors="pt",
        ).to(device)

    def run(self, image_tensor: Any, texts: Sequence[str], device: str) -> Dict[str, Any]:
        torch = self.torch
        model = self.model

        with torch.no_grad():
            with model.maybe_autocast():
                image_hidden = model.ln_vision(model.visual_encoder(image_tensor))
                image_atts = torch.ones(
                    image_hidden.size()[:-1], dtype=torch.long, device=image_hidden.device
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
                input_tokens = self.tokenize(texts, device)
                visual_attention = torch.ones(
                    visual_tokens.size()[:-1], dtype=torch.long, device=visual_tokens.device
                )
                encoder_attention = torch.cat(
                    [visual_attention, input_tokens.attention_mask], dim=1
                )
                text_embeddings = model.t5_model.encoder.embed_tokens(input_tokens.input_ids)
                encoder_embeddings = torch.cat([visual_tokens, text_embeddings], dim=1)

                num_query = int(visual_tokens.shape[1])
                total = int(encoder_attention.shape[1])

                visual_mask = torch.zeros(
                    (encoder_attention.shape[0], total), dtype=torch.bool, device=device
                )
                visual_mask[:, :num_query] = True
                text_mask = torch.zeros_like(visual_mask)
                text_mask[:, num_query:] = input_tokens.attention_mask.bool()
                pad_mask = torch.zeros_like(visual_mask)
                pad_mask[:, num_query:] = ~input_tokens.attention_mask.bool()

                # The pruner feeds every position (padding included) to add_batch.
                self.set_masks(
                    {"visual": visual_mask, "text": text_mask, "pad": pad_mask}
                )

                encoder_outputs = model.t5_model.encoder(
                    inputs_embeds=encoder_embeddings,
                    attention_mask=encoder_attention,
                    return_dict=True,
                )

        return {
            "encoder_hidden": encoder_outputs.last_hidden_state,
            "encoder_attention": encoder_attention,
            "input_tokens": input_tokens,
            "visual_mask": visual_mask,
            "text_mask": text_mask,
            "pad_mask": pad_mask,
            "num_query": num_query,
            "seq_len": total,
        }

    # Hooks read the current token grouping from here.
    _masks: Dict[str, Any] = {}

    def set_masks(self, masks: Dict[str, Any]) -> None:
        type(self)._masks = masks

    @classmethod
    def current_masks(cls) -> Dict[str, Any]:
        return cls._masks


def t5_encoder_linears(model: Any, torch: Any) -> "List[Tuple[int, str, str, Any]]":
    """Every nn.Linear the T5-encoder Wanda pass would prune.

    Returns ``(block_index, submodule_name, sparsity_key, module)`` where
    ``sparsity_key`` matches the pruner's
    ``f"{module_to_process}.{i}.{name}.weight"``.
    """
    out: List[Tuple[int, str, str, Any]] = []
    blocks = model.t5_model.encoder.block
    for index, block in enumerate(blocks):
        for name, module in block.named_modules():
            if isinstance(module, torch.nn.Linear):
                key = "%s.encoder.block.%d.%s.weight" % (T5_PREFIX, index, name)
                out.append((index, name, key, module))
    return out


def wanda_keep_mask(weight: Any, scaler: Any, sparsity: float, torch: Any) -> Any:
    """Rebuild Wanda's mask exactly as wanda_pruner does.

    ``W_metric = |W| * sqrt(scaler_row)``, then per output row sort along the
    input dim and drop the lowest ``sparsity`` fraction.  ``scaler`` only needs
    to be proportional to the pruner's ``scaler_row``: a positive per-layer
    constant cannot change the within-row ranking.

    Returns a bool tensor that is True for *kept* weights.
    """
    weight = weight.detach().to(torch.float32)
    scaler = scaler.detach().to(torch.float32).to(weight.device)
    metric = weight.abs() * torch.sqrt(scaler).reshape(1, -1)
    num_prune = int(metric.shape[1] * sparsity)
    keep = torch.ones_like(metric, dtype=torch.bool)
    if num_prune > 0:
        indices = torch.sort(metric, dim=-1, stable=True)[1][:, :num_prune]
        keep.scatter_(1, indices, False)
    return keep


def mask_agreement(a: Any, b: Any, torch: Any) -> Tuple[float, float]:
    """(overlap, iou) between two boolean *keep* masks.

    overlap = |A n B| / |A|  -- 1.0 when identical, keep_ratio when independent.
    iou     = |A n B| / |A u B|.
    """
    a = a.bool()
    b = b.bool()
    inter = float((a & b).sum().item())
    union = float((a | b).sum().item())
    size_a = float(a.sum().item())
    overlap = inter / size_a if size_a > 0 else float("nan")
    iou = inter / union if union > 0 else float("nan")
    return overlap, iou


def random_baselines(sparsity: float) -> Tuple[float, float]:
    """Expected (overlap, iou) for two independent keep masks."""
    keep = 1.0 - sparsity
    overlap = keep
    iou = keep / (2.0 - keep) if keep > 0 else 0.0
    return overlap, iou
