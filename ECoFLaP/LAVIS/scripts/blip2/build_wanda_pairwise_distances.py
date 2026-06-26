#!/usr/bin/env python3
"""Build pairwise Wanda W_metric/mask distances for calibration selection.

For each candidate sample, this script runs one forward pass through BLIP2-T5
and records each target Linear layer's Wanda scaler_row.  It then computes:

  D_wmetric: exact normalized Frobenius distance between per-sample W_metric
             matrices, using W_metric = abs(W) * sqrt(scaler_row), or a
             lighter row-level approximation selected by
             --wmetric_distance_level.
  D_mask:    sampled final-mask distance from per-sample Wanda pruning masks.
  D_final:   alpha * normalized(D_wmetric) + beta * normalized(D_mask).

The output NPZ can be passed directly to ppo_calibration_selection.py via
--distance_npz, or the separate D_wmetric/D_mask arrays can be passed there.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LAVIS_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _LAVIS_ROOT not in sys.path:
    sys.path.insert(0, _LAVIS_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from record_wanda_input_importance import (  # noqa: E402
    AUTO_OUTPUT_FIELDS,
    AUTO_TEXT_FIELDS,
    TargetInfo,
    collect_target_linears,
    extract_text,
    infer_mask_style,
    iter_batches,
    load_rows,
    resolve_image_path,
    select_rows,
    tokenize_with_length_stats,
    value_to_text,
    wanda_pruned_mask,
)


class SingleSampleWandaCollector:
    def __init__(self, module: Any):
        import torch

        self.module = module
        self.columns = int(module.weight.shape[1])
        self.device = module.weight.device
        self.scaler_row = torch.zeros((self.columns,), device=self.device)
        self.token_rows = 0

    def reset(self) -> None:
        self.scaler_row.zero_()
        self.token_rows = 0

    def add(self, inp: Any) -> None:
        if inp is None:
            return
        if isinstance(inp, (tuple, list)):
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
        flat = x.reshape(-1, x.shape[-1]).float()
        if flat.shape[-1] != self.columns:
            raise RuntimeError("Input dim mismatch: got %d expected %d" % (flat.shape[-1], self.columns))
        # For batch_size=1, this is the exact per-sample WrappedGPT scaler.
        self.scaler_row += flat.t().norm(p=2, dim=1).pow(2)
        self.token_rows += int(flat.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pairwise W_metric/mask distances for PPO calibration selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_mode",
        choices=["cc3m_multimodal", "t5_text_only", "vit_image_only"],
        required=True,
    )
    parser.add_argument("--calib_json", required=True)
    parser.add_argument("--images_dir", default=None)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="blip2_t5")
    parser.add_argument("--model_type", default="pretrain_flant5xl")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--image_field", default="image")
    parser.add_argument("--text_field", default="auto")
    parser.add_argument("--output_field", default="auto")
    parser.add_argument("--no_decoder", action="store_true")
    parser.add_argument("--max_txt_len", type=int, default=None)
    parser.add_argument("--component", choices=["all", "t5", "t5_encoder", "t5_decoder", "vit"], default="all")
    parser.add_argument("--wanda_sparsity", type=float, default=0.5)
    parser.add_argument("--mask_style", choices=["auto", "row", "global"], default="auto")
    parser.add_argument(
        "--mask_distance_mode",
        choices=["sampled", "none"],
        default="sampled",
        help="sampled computes pruned-mask IoU on sampled matrix positions.",
    )
    parser.add_argument(
        "--mask_sample_positions_per_module",
        type=int,
        default=1024,
        help="Number of weight positions sampled per Linear module for D_mask.",
    )
    parser.add_argument(
        "--wmetric_distance_level",
        choices=["wmetric", "weighted_row", "scaler_row"],
        default="wmetric",
        help=(
            "wmetric keeps the original exact W_metric Frobenius distance; "
            "weighted_row compares sqrt(scaler_row) with column mean-|W| weights; "
            "scaler_row compares only sqrt(scaler_row)."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--normalize_distances", choices=["percentile", "max", "none"], default="percentile")
    parser.add_argument("--normalization_percentile", type=float, default=95.0)
    parser.add_argument("--save_per_sample_scalers", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def component_allowed(component: str, selected: str) -> bool:
    if selected == "all":
        return True
    if selected == "t5":
        return component in {"t5_encoder", "t5_decoder"}
    return component == selected


def attach_collectors(targets: Sequence[TargetInfo]) -> Tuple[Dict[str, SingleSampleWandaCollector], List[Any]]:
    collectors: Dict[str, SingleSampleWandaCollector] = {}
    handles: List[Any] = []
    for target in targets:
        collectors[target.name] = SingleSampleWandaCollector(target.module)

        def make_hook(module_name: str):
            def hook(_module: Any, inputs: Tuple[Any, ...], _output: Any) -> None:
                collectors[module_name].add(inputs)

            return hook

        handles.append(target.module.register_forward_hook(make_hook(target.name)))
    return collectors, handles


def reset_collectors(collectors: Dict[str, SingleSampleWandaCollector]) -> None:
    for collector in collectors.values():
        collector.reset()


def forward_multimodal_one(
    model: Any,
    row: Any,
    original_index: int,
    images_dir: str,
    vis_processor: Any,
    args: argparse.Namespace,
    torch: Any,
    Image: Any,
    metadata: Dict[str, Any],
) -> None:
    if not isinstance(row, dict) or args.image_field not in row:
        raise KeyError("Row %d is missing image field %r." % (original_index, args.image_field))
    image_path = resolve_image_path(images_dir, row[args.image_field])
    if not os.path.isfile(image_path):
        raise FileNotFoundError("Image not found for row %d: %s" % (original_index, image_path))
    with Image.open(image_path) as image:
        image_tensor = vis_processor(image.convert("RGB")).unsqueeze(0).to(args.device)

    input_text, text_field = extract_text(row, args.text_field, AUTO_TEXT_FIELDS, original_index)
    output_text, output_field = extract_text(row, args.output_field, AUTO_OUTPUT_FIELDS, original_index)
    metadata["selected_text_fields"][text_field] = metadata["selected_text_fields"].get(text_field, 0) + 1
    metadata["selected_output_fields"][output_field] = metadata["selected_output_fields"].get(output_field, 0) + 1

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
                model.t5_tokenizer, [input_text], model.max_txt_len, args.device
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
                    model.t5_tokenizer, [output_text], model.max_txt_len, args.device
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

    retained_lengths = input_tokens.attention_mask.sum(dim=1).detach().cpu().tolist()
    metadata["original_input_tokens"] += int(sum(original_lengths))
    metadata["retained_input_tokens"] += int(sum(retained_lengths))
    metadata["truncated_input_samples"] += int(sum(int(a > b) for a, b in zip(original_lengths, retained_lengths)))


def forward_t5_text_one(
    model: Any,
    row: Any,
    original_index: int,
    args: argparse.Namespace,
    torch: Any,
    metadata: Dict[str, Any],
) -> None:
    text, text_field = extract_text(row, args.text_field, AUTO_TEXT_FIELDS, original_index)
    metadata["selected_text_fields"][text_field] = metadata["selected_text_fields"].get(text_field, 0) + 1
    with torch.no_grad():
        with model.maybe_autocast(dtype=torch.bfloat16):
            input_tokens, original_lengths = tokenize_with_length_stats(
                model.t5_tokenizer, [text], model.max_txt_len, args.device
            )
            embeddings = model.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            attention = input_tokens.attention_mask
            model.temp_label = torch.zeros_like(attention, dtype=torch.bool)
            if args.no_decoder:
                model.t5_model.encoder(inputs_embeds=embeddings, attention_mask=attention, return_dict=True)
            else:
                targets = input_tokens.input_ids.masked_fill(
                    input_tokens.input_ids == model.t5_tokenizer.pad_token_id,
                    -100,
                )
                model.t5_model(
                    inputs_embeds=embeddings,
                    attention_mask=attention,
                    decoder_attention_mask=attention,
                    labels=targets,
                    return_dict=True,
                )

    retained_lengths = input_tokens.attention_mask.sum(dim=1).detach().cpu().tolist()
    metadata["original_input_tokens"] += int(sum(original_lengths))
    metadata["retained_input_tokens"] += int(sum(retained_lengths))
    metadata["truncated_input_samples"] += int(sum(int(a > b) for a, b in zip(original_lengths, retained_lengths)))


def forward_vit_image_one(
    model: Any,
    row: Any,
    original_index: int,
    images_dir: str,
    vis_processor: Any,
    args: argparse.Namespace,
    torch: Any,
    Image: Any,
) -> None:
    if not isinstance(row, dict) or args.image_field not in row:
        raise KeyError("Row %d is missing image field %r." % (original_index, args.image_field))
    image_path = resolve_image_path(images_dir, row[args.image_field])
    if not os.path.isfile(image_path):
        raise FileNotFoundError("Image not found for row %d: %s" % (original_index, image_path))
    with Image.open(image_path) as image:
        image_tensor = vis_processor(image.convert("RGB")).unsqueeze(0).to(args.device)
    with torch.no_grad():
        with model.maybe_autocast():
            model.ln_vision(model.visual_encoder(image_tensor))


def normalize_distance(matrix: np.ndarray, mode: str, percentile: float) -> np.ndarray:
    if mode == "none":
        return matrix
    vals = matrix[np.triu_indices(matrix.shape[0], k=1)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return matrix.copy()
    scale = float(vals.max()) if mode == "max" else float(np.percentile(vals, percentile))
    if scale <= 0 or not math.isfinite(scale):
        return matrix.copy()
    return matrix / scale


def compute_wmetric_distance(
    targets: Sequence[TargetInfo],
    scalers: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    n = next(iter(scalers.values())).shape[0]
    dist_sq = np.zeros((n, n), dtype=np.float64)
    denom = 0.0
    rows: List[Dict[str, Any]] = []
    for target in targets:
        weight = target.module.weight.detach().float().abs()
        col_weight = weight.pow(2).sum(dim=0).cpu().numpy().astype(np.float64)
        g = np.sqrt(np.maximum(scalers[target.name].astype(np.float64), 0.0))
        gw = g * col_weight.reshape(1, -1)
        norm = (g * g * col_weight.reshape(1, -1)).sum(axis=1)
        module_dist_sq = np.maximum(norm[:, None] + norm[None, :] - 2.0 * np.matmul(gw, g.T), 0.0)
        dist_sq += module_dist_sq
        denom += float(weight.numel())
        rows.append(
            {
                "module_name": target.name,
                "component": target.component,
                "layer": target.layer,
                "role": target.role,
                "in_dim": int(weight.shape[1]),
                "out_dim": int(weight.shape[0]),
                "weight_numel": int(weight.numel()),
                "wmetric_dist_sq_mean": float(module_dist_sq.mean() / max(float(weight.numel()), 1.0)),
            }
        )
    d_wmetric = np.sqrt(np.maximum(dist_sq / max(denom, 1.0), 0.0))
    np.fill_diagonal(d_wmetric, 0.0)
    return d_wmetric, rows


def compute_row_level_distance(
    targets: Sequence[TargetInfo],
    scalers: Dict[str, np.ndarray],
    level: str,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if level not in {"weighted_row", "scaler_row"}:
        raise ValueError("Unsupported row-level distance: %s" % level)
    n = next(iter(scalers.values())).shape[0]
    dist_sq = np.zeros((n, n), dtype=np.float64)
    denom = 0.0
    rows: List[Dict[str, Any]] = []
    for target in targets:
        weight = target.module.weight.detach().float().abs()
        g = np.sqrt(np.maximum(scalers[target.name].astype(np.float64), 0.0))
        if level == "weighted_row":
            col_weight = weight.mean(dim=0).cpu().numpy().astype(np.float64)
            feature = g * col_weight.reshape(1, -1)
        else:
            feature = g

        norm = (feature * feature).sum(axis=1)
        module_dist_sq = np.maximum(norm[:, None] + norm[None, :] - 2.0 * np.matmul(feature, feature.T), 0.0)
        dist_sq += module_dist_sq
        denom += float(feature.shape[1])
        rows.append(
            {
                "module_name": target.name,
                "component": target.component,
                "layer": target.layer,
                "role": target.role,
                "distance_level": level,
                "in_dim": int(weight.shape[1]),
                "out_dim": int(weight.shape[0]),
                "weight_numel": int(weight.numel()),
                "row_feature_dim": int(feature.shape[1]),
                "row_dist_sq_mean": float(module_dist_sq.mean() / max(float(feature.shape[1]), 1.0)),
            }
        )
    d_row = np.sqrt(np.maximum(dist_sq / max(denom, 1.0), 0.0))
    np.fill_diagonal(d_row, 0.0)
    return d_row, rows


def sample_module_positions(target: TargetInfo, count: int, rng: np.random.RandomState) -> np.ndarray:
    rows = int(target.module.weight.shape[0])
    cols = int(target.module.weight.shape[1])
    total = rows * cols
    take = min(max(int(count), 0), total)
    if take <= 0:
        return np.zeros((0,), dtype=np.int64)
    return rng.choice(total, size=take, replace=False).astype(np.int64)


def module_mask_signature(
    target: TargetInfo,
    scaler: np.ndarray,
    flat_positions: np.ndarray,
    sparsity: float,
    mask_style: str,
) -> np.ndarray:
    import torch

    if flat_positions.size == 0:
        return np.zeros((0,), dtype=bool)
    device = target.module.weight.device
    weight_abs = target.module.weight.detach().float().abs()
    scaler_t = torch.as_tensor(scaler, dtype=torch.float32, device=device).clamp_min(0)
    importance = weight_abs * torch.sqrt(scaler_t.reshape(1, -1))
    pruned_mask = wanda_pruned_mask(importance, sparsity, mask_style)
    pos = torch.as_tensor(flat_positions, dtype=torch.long, device=device)
    values = pruned_mask.reshape(-1).index_select(0, pos)
    return values.detach().cpu().numpy().astype(bool)


def compute_mask_distance(
    targets: Sequence[TargetInfo],
    scalers: Dict[str, np.ndarray],
    sparsity: float,
    mask_style: str,
    positions_per_module: int,
    seed: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    n = next(iter(scalers.values())).shape[0]
    rng = np.random.RandomState(seed)
    signatures: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    for target_index, target in enumerate(targets):
        flat_positions = sample_module_positions(target, positions_per_module, rng)
        resolved_style = infer_mask_style(target, mask_style)
        module_sig = np.zeros((n, flat_positions.size), dtype=bool)
        for sample_index in range(n):
            module_sig[sample_index] = module_mask_signature(
                target,
                scalers[target.name][sample_index],
                flat_positions,
                sparsity,
                resolved_style,
            )
        signatures.append(module_sig)
        rows.append(
            {
                "module_name": target.name,
                "component": target.component,
                "layer": target.layer,
                "role": target.role,
                "sampled_positions": int(flat_positions.size),
                "mask_style": resolved_style,
                "mean_pruned_fraction_in_signature": float(module_sig.mean()) if module_sig.size else 0.0,
            }
        )
        if (target_index + 1) % 20 == 0 or target_index + 1 == len(targets):
            print("[mask] processed %d/%d modules" % (target_index + 1, len(targets)))

    if not signatures:
        return np.zeros((n, n), dtype=np.float64), rows
    sig = np.concatenate(signatures, axis=1)
    sig_f = sig.astype(np.float32)
    intersections = np.matmul(sig_f, sig_f.T)
    sums = sig_f.sum(axis=1)
    unions = sums[:, None] + sums[None, :] - intersections
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(unions > 0, intersections / unions, 1.0)
    d_mask = 1.0 - iou
    d_mask = 0.5 * (d_mask + d_mask.T)
    np.fill_diagonal(d_mask, 0.0)
    return d_mask.astype(np.float64), rows


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    rng = np.random.RandomState(args.seed)

    import torch
    from PIL import Image
    from lavis.models import load_model
    from lavis.processors import load_processor

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows_all = load_rows(args.calib_json)
    rows, original_indices = select_rows(rows_all, args.max_samples, args.shuffle, args.seed)
    if args.max_txt_len is not None and args.max_txt_len < 1:
        raise ValueError("--max_txt_len must be positive.")
    if args.input_mode != "t5_text_only" and not args.images_dir:
        raise ValueError("--images_dir is required for image modes.")

    print("loading model:", args.model_name, args.model_type)
    model = load_model(args.model_name, args.model_type, is_eval=True, device=args.device, checkpoint=args.ckpt)
    model.eval()
    if args.max_txt_len is not None:
        model.max_txt_len = int(args.max_txt_len)
    vis_processor = None
    if args.input_mode != "t5_text_only":
        vis_processor = load_processor("blip_image_eval").build(image_size=args.image_size)

    targets = collect_target_linears(model, args.input_mode, include_decoder=not args.no_decoder)
    targets = [target for target in targets if component_allowed(target.component, args.component)]
    if not targets:
        raise RuntimeError("No target Linear layers found for mode=%s component=%s" % (args.input_mode, args.component))
    print("[OK] target Linear layers:", len(targets))

    collectors, handles = attach_collectors(targets)
    scalers: Dict[str, List[np.ndarray]] = {target.name: [] for target in targets}
    sample_rows: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {
        "selected_text_fields": {},
        "selected_output_fields": {},
        "original_input_tokens": 0,
        "retained_input_tokens": 0,
        "truncated_input_samples": 0,
        "original_output_tokens": 0,
        "retained_output_tokens": 0,
    }

    try:
        for sample_pos, (row, original_index) in enumerate(zip(rows, original_indices)):
            reset_collectors(collectors)
            if args.input_mode == "cc3m_multimodal":
                forward_multimodal_one(
                    model,
                    row,
                    original_index,
                    args.images_dir,
                    vis_processor,
                    args,
                    torch,
                    Image,
                    metadata,
                )
            elif args.input_mode == "t5_text_only":
                forward_t5_text_one(model, row, original_index, args, torch, metadata)
            else:
                forward_vit_image_one(
                    model,
                    row,
                    original_index,
                    args.images_dir,
                    vis_processor,
                    args,
                    torch,
                    Image,
                )

            for target in targets:
                scalers[target.name].append(
                    collectors[target.name].scaler_row.detach().float().cpu().numpy().astype(np.float32)
                )
            sample_rows.append({"sample_position": sample_pos, "original_index": int(original_index)})
            if args.log_every and ((sample_pos + 1) % args.log_every == 0 or sample_pos + 1 == len(rows)):
                print("[samples] processed %d/%d" % (sample_pos + 1, len(rows)))
    finally:
        for handle in handles:
            handle.remove()

    scaler_arrays: Dict[str, np.ndarray] = {
        name: np.stack(values, axis=0).astype(np.float32) for name, values in scalers.items()
    }
    print("[distance] computing D_wmetric level=%s" % args.wmetric_distance_level)
    if args.wmetric_distance_level == "wmetric":
        d_wmetric, wmetric_rows = compute_wmetric_distance(targets, scaler_arrays)
    else:
        d_wmetric, wmetric_rows = compute_row_level_distance(
            targets,
            scaler_arrays,
            args.wmetric_distance_level,
        )

    if args.mask_distance_mode == "none":
        d_mask = np.zeros_like(d_wmetric)
        mask_rows: List[Dict[str, Any]] = []
    else:
        print("[distance] computing sampled D_mask")
        d_mask, mask_rows = compute_mask_distance(
            targets,
            scaler_arrays,
            args.wanda_sparsity,
            args.mask_style,
            args.mask_sample_positions_per_module,
            args.seed,
        )

    d_w_norm = normalize_distance(d_wmetric, args.normalize_distances, args.normalization_percentile)
    d_m_norm = normalize_distance(d_mask, args.normalize_distances, args.normalization_percentile)
    weight_sum = max(args.alpha, 0.0) + max(args.beta, 0.0)
    if weight_sum <= 0:
        raise ValueError("alpha/beta must contain at least one positive weight.")
    d_final = (max(args.alpha, 0.0) * d_w_norm + max(args.beta, 0.0) * d_m_norm) / weight_sum
    np.fill_diagonal(d_final, 0.0)

    payload: Dict[str, Any] = {
        "D_wmetric": d_wmetric.astype(np.float32),
        "D_mask": d_mask.astype(np.float32),
        "D_final": d_final.astype(np.float32),
        "candidate_indices": np.asarray(original_indices, dtype=np.int64),
    }
    if args.save_per_sample_scalers:
        for target in targets:
            payload[safe_key(target.name) + ".scaler_per_sample"] = scaler_arrays[target.name]

    npz_path = os.path.join(args.out_dir, "wanda_pairwise_distance_matrices.npz")
    np.savez_compressed(npz_path, **payload)
    write_json(os.path.join(args.out_dir, "candidate_indices.json"), [int(i) for i in original_indices])
    write_csv(os.path.join(args.out_dir, "wanda_pairwise_wmetric_modules.csv"), wmetric_rows)
    write_csv(os.path.join(args.out_dir, "wanda_pairwise_mask_modules.csv"), mask_rows)
    write_csv(os.path.join(args.out_dir, "wanda_pairwise_samples.csv"), sample_rows)
    write_json(
        os.path.join(args.out_dir, "wanda_pairwise_distance_metadata.json"),
        {
            "input_mode": args.input_mode,
            "calib_json": os.path.abspath(args.calib_json),
            "images_dir": os.path.abspath(args.images_dir) if args.images_dir else "",
            "ckpt": os.path.abspath(args.ckpt),
            "model_name": args.model_name,
            "model_type": args.model_type,
            "device": args.device,
            "samples": len(rows),
            "targets": len(targets),
            "component": args.component,
            "wanda_sparsity": args.wanda_sparsity,
            "wmetric_distance_level": args.wmetric_distance_level,
            "mask_distance_mode": args.mask_distance_mode,
            "mask_sample_positions_per_module": args.mask_sample_positions_per_module,
            "alpha": args.alpha,
            "beta": args.beta,
            "normalize_distances": args.normalize_distances,
            "normalization_percentile": args.normalization_percentile,
            "metadata": metadata,
            "outputs": {
                "distance_npz": os.path.abspath(npz_path),
            },
        },
    )
    print("[OK] wrote:", npz_path)
    print("[OK] candidate indices:", os.path.join(args.out_dir, "candidate_indices.json"))


if __name__ == "__main__":
    main()
