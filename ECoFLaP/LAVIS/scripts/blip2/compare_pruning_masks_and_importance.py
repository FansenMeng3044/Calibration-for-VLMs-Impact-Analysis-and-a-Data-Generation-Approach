#!/usr/bin/env python3
"""Compare pruning masks, layer sparsity, and optional importance scores.

This script is designed for the module-matched calibration experiment:

  - C4 text-only calibration pruning T5
  - CC3M multimodal calibration pruning BLIP2-T5
  - CC3M image-only calibration pruning ViT

It can compare:
  1. C4 T5 pruning mask vs CC3M-multimodal T5 pruning mask
  2. CC3M image-only ViT pruning mask vs CC3M-multimodal ViT pruning mask

If explicit mask files are unavailable, masks are inferred from pruned
checkpoints: a parameter is considered pruned where the base checkpoint is
non-zero and the pruned checkpoint value is zero. The script reports both
pruned-mask IoU and keep-mask IoU.

Optional importance-score files can also be supplied. They should be torch
.pth/.pt dictionaries, .npz archives, or JSON dictionaries keyed by parameter
name. Importance plots are skipped when files are not provided.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pruning masks and optional importance scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_ckpt", required=True, help="Dense BLIP2-T5 checkpoint/state_dict.")
    parser.add_argument("--c4_t5_ckpt", required=True, help="C4 text-only T5-pruned checkpoint.")
    parser.add_argument("--cc3m_multimodal_ckpt", required=True, help="CC3M multimodal pruned checkpoint.")
    parser.add_argument("--cc3m_image_vit_ckpt", required=True, help="CC3M image-only ViT-pruned checkpoint.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--c4_t5_importance", default=None)
    parser.add_argument("--cc3m_multimodal_importance", default=None)
    parser.add_argument("--cc3m_image_vit_importance", default=None)
    parser.add_argument(
        "--zero_tol",
        type=float,
        default=0.0,
        help="Absolute value <= zero_tol is treated as zero.",
    )
    parser.add_argument(
        "--include_bias",
        action="store_true",
        help="Include 1D bias/LayerNorm-like parameters in mask analysis.",
    )
    parser.add_argument(
        "--max_hist_values",
        type=int,
        default=300000,
        help="Maximum flattened values sampled per run for histogram plots.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except (ImportError, RuntimeError) as exc:
        raise SystemExit("A compatible matplotlib/NumPy installation is required.") from exc


def torch_load(path: str) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required to read checkpoints: %s" % exc) from exc
    return torch.load(path, map_location="cpu")


def unwrap_state_dict(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise TypeError("Expected checkpoint dict, got %s" % type(obj).__name__)
    for key in ("model", "state_dict", "module", "ema_state_dict"):
        if key in obj and isinstance(obj[key], dict):
            return unwrap_state_dict(obj[key])
    return obj


def load_state_dict(path: str) -> Dict[str, Any]:
    state = unwrap_state_dict(torch_load(path))
    out: Dict[str, Any] = {}
    for key, value in state.items():
        if hasattr(value, "shape"):
            out[canonical_key(str(key))] = value
    return out


def canonical_key(key: str) -> str:
    """Normalize common checkpoint wrappers while keeping model-internal names."""
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model.", "_orig_mod.", "blip2.", "base_model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True

    # Some T5-only checkpoints save the HF T5 module directly.
    if key.startswith(("encoder.", "decoder.", "shared.", "lm_head.")):
        key = "t5_model." + key
    return key


def tensor_to_numpy(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu()
        if hasattr(tensor, "float"):
            tensor = tensor.float()
        return tensor.numpy()
    return np.asarray(tensor)


def is_float_tensor(tensor: Any) -> bool:
    if hasattr(tensor, "is_floating_point"):
        return bool(tensor.is_floating_point())
    arr = np.asarray(tensor)
    return np.issubdtype(arr.dtype, np.floating)


def component_of_key(key: str) -> Optional[str]:
    key = canonical_key(key)
    if key.startswith("t5_model."):
        return "t5"
    if key.startswith("visual_encoder.") or key.startswith("ln_vision."):
        return "vit"
    if key.startswith("Qformer.") or key.startswith("t5_proj.") or key.startswith("query_tokens"):
        return "bridge"
    return None


def layer_id_for_key(key: str, component: str) -> str:
    key = canonical_key(key)
    if component == "t5":
        m = re.search(r"t5_model\.(encoder|decoder)\.block\.(\d+)", key)
        if m:
            return "t5_%s_%02d" % (m.group(1), int(m.group(2)))
        if key.startswith("t5_model.shared") or "embed_tokens" in key:
            return "t5_embedding"
        if key.startswith("t5_model.lm_head"):
            return "t5_lm_head"
        return "t5_other"
    if component == "vit":
        m = re.search(r"visual_encoder\.blocks\.(\d+)", key)
        if m:
            return "vit_block_%02d" % int(m.group(1))
        if key.startswith("visual_encoder.patch_embed"):
            return "vit_patch_embed"
        if key.startswith("visual_encoder.pos_embed"):
            return "vit_pos_embed"
        if key.startswith("visual_encoder.norm") or key.startswith("ln_vision"):
            return "vit_norm"
        return "vit_other"
    return "other"


def should_include_param(key: str, tensor: Any, component: str, include_bias: bool) -> bool:
    key = canonical_key(key)
    if component_of_key(key) != component:
        return False
    if not is_float_tensor(tensor):
        return False
    shape = tuple(tensor.shape)
    if not include_bias and len(shape) < 2:
        return False
    if key.endswith(".num_batches_tracked"):
        return False
    return True


def infer_masks(
    base_state: Dict[str, Any],
    pruned_state: Dict[str, Any],
    component: str,
    include_bias: bool,
    zero_tol: float,
) -> Dict[str, Dict[str, Any]]:
    masks: Dict[str, Dict[str, Any]] = {}
    for key, base_tensor in base_state.items():
        key = canonical_key(key)
        if key not in pruned_state:
            continue
        if not should_include_param(key, base_tensor, component, include_bias):
            continue
        if tuple(base_tensor.shape) != tuple(pruned_state[key].shape):
            continue
        base_arr = tensor_to_numpy(base_tensor)
        pruned_arr = tensor_to_numpy(pruned_state[key])
        valid = np.abs(base_arr) > zero_tol
        if not np.any(valid):
            continue
        keep = (np.abs(pruned_arr) > zero_tol) & valid
        pruned = (~keep) & valid
        masks[key] = {
            "keep": keep.astype(bool, copy=False),
            "pruned": pruned.astype(bool, copy=False),
            "valid": valid.astype(bool, copy=False),
            "layer": layer_id_for_key(key, component),
            "shape": tuple(base_arr.shape),
        }
    return masks


def checkpoint_diagnostics(name: str, state: Dict[str, Any], include_bias: bool) -> Dict[str, Any]:
    keys = sorted(state.keys())
    t5 = [
        key
        for key, value in state.items()
        if should_include_param(key, value, "t5", include_bias)
    ]
    vit = [
        key
        for key, value in state.items()
        if should_include_param(key, value, "vit", include_bias)
    ]
    return {
        "name": name,
        "num_tensors": len(keys),
        "num_t5_candidate_tensors": len(t5),
        "num_vit_candidate_tensors": len(vit),
        "first_keys": keys[:20],
        "first_t5_keys": sorted(t5)[:10],
        "first_vit_keys": sorted(vit)[:10],
    }


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray, valid: np.ndarray) -> Tuple[float, int, int]:
    a = mask_a[valid].astype(bool, copy=False)
    b = mask_b[valid].astype(bool, copy=False)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return (float(inter / union) if union else 1.0), inter, union


def compare_mask_sets(
    label_a: str,
    masks_a: Dict[str, Dict[str, Any]],
    label_b: str,
    masks_b: Dict[str, Dict[str, Any]],
    component: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    layer_acc: Dict[str, Dict[str, int]] = {}
    common = sorted(set(masks_a) & set(masks_b))
    for key in common:
        a = masks_a[key]
        b = masks_b[key]
        valid = a["valid"] & b["valid"]
        if not np.any(valid):
            continue
        pruned_iou, pruned_inter, pruned_union = mask_iou(a["pruned"], b["pruned"], valid)
        keep_iou, keep_inter, keep_union = mask_iou(a["keep"], b["keep"], valid)
        valid_count = int(valid.sum())
        a_pruned = int((a["pruned"] & valid).sum())
        b_pruned = int((b["pruned"] & valid).sum())
        row = {
            "comparison": "%s_vs_%s" % (label_a, label_b),
            "component": component,
            "parameter": key,
            "layer": a["layer"],
            "valid_parameters": valid_count,
            "%s_pruned_count" % label_a: a_pruned,
            "%s_pruned_count" % label_b: b_pruned,
            "%s_pruned_ratio" % label_a: float(a_pruned / valid_count),
            "%s_pruned_ratio" % label_b: float(b_pruned / valid_count),
            "%s_keep_ratio" % label_a: float((a["keep"] & valid).sum() / valid_count),
            "%s_keep_ratio" % label_b: float((b["keep"] & valid).sum() / valid_count),
            "pruned_mask_iou": pruned_iou,
            "keep_mask_iou": keep_iou,
            "pruned_intersection": pruned_inter,
            "pruned_union": pruned_union,
            "keep_intersection": keep_inter,
            "keep_union": keep_union,
        }
        rows.append(row)
        acc = layer_acc.setdefault(
            a["layer"],
            {
                "valid": 0,
                "a_pruned": 0,
                "b_pruned": 0,
                "pruned_inter": 0,
                "pruned_union": 0,
                "keep_inter": 0,
                "keep_union": 0,
            },
        )
        acc["valid"] += valid_count
        acc["a_pruned"] += a_pruned
        acc["b_pruned"] += b_pruned
        acc["pruned_inter"] += pruned_inter
        acc["pruned_union"] += pruned_union
        acc["keep_inter"] += keep_inter
        acc["keep_union"] += keep_union

    layer_rows: List[Dict[str, Any]] = []
    for layer, acc in sorted(layer_acc.items()):
        valid_count = acc["valid"]
        layer_rows.append(
            {
                "comparison": "%s_vs_%s" % (label_a, label_b),
                "component": component,
                "layer": layer,
                "valid_parameters": valid_count,
                "%s_pruned_count" % label_a: acc["a_pruned"],
                "%s_pruned_count" % label_b: acc["b_pruned"],
                "%s_pruned_ratio" % label_a: float(acc["a_pruned"] / valid_count),
                "%s_pruned_ratio" % label_b: float(acc["b_pruned"] / valid_count),
                "%s_keep_ratio" % label_a: float(1.0 - acc["a_pruned"] / valid_count),
                "%s_keep_ratio" % label_b: float(1.0 - acc["b_pruned"] / valid_count),
                "pruned_mask_iou": (
                    float(acc["pruned_inter"] / acc["pruned_union"])
                    if acc["pruned_union"]
                    else 1.0
                ),
                "keep_mask_iou": (
                    float(acc["keep_inter"] / acc["keep_union"])
                    if acc["keep_union"]
                    else 1.0
                ),
                "pruned_intersection": acc["pruned_inter"],
                "pruned_union": acc["pruned_union"],
                "keep_intersection": acc["keep_inter"],
                "keep_union": acc["keep_union"],
            }
        )
    return rows, layer_rows


def summarize_run_masks(label: str, masks: Dict[str, Dict[str, Any]], component: str) -> List[Dict[str, Any]]:
    acc: Dict[str, Dict[str, int]] = {}
    for key, item in masks.items():
        valid = item["valid"]
        layer = item["layer"]
        total = int(valid.sum())
        pruned = int((item["pruned"] & valid).sum())
        d = acc.setdefault(layer, {"valid": 0, "pruned": 0, "num_parameters": 0})
        d["valid"] += total
        d["pruned"] += pruned
        d["num_parameters"] += 1
    rows = []
    for layer, d in sorted(acc.items()):
        valid = d["valid"]
        pruned = d["pruned"]
        rows.append(
            {
                "run": label,
                "component": component,
                "layer": layer,
                "num_parameter_tensors": d["num_parameters"],
                "valid_parameters": valid,
                "pruned_parameters": pruned,
                "kept_parameters": valid - pruned,
                "pruned_ratio": float(pruned / valid) if valid else 0.0,
                "keep_ratio": float(1.0 - pruned / valid) if valid else 1.0,
            }
        )
    return rows


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
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


def layer_sort_key(layer: str) -> Tuple[str, int, str]:
    m = re.search(r"(\d+)$", layer)
    return (re.sub(r"\d+$", "", layer), int(m.group(1)) if m else -1, layer)


def plot_layer_keep_ratios(plt: Any, rows: Sequence[Dict[str, Any]], path: str, title: str) -> None:
    if not rows:
        return
    runs = sorted({str(row["run"]) for row in rows})
    layers = sorted({str(row["layer"]) for row in rows}, key=layer_sort_key)
    values = {run: {layer: np.nan for layer in layers} for run in runs}
    for row in rows:
        values[str(row["run"])][str(row["layer"])] = float(row["keep_ratio"])
    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(max(10.0, 0.36 * len(layers)), 5.4))
    for run in runs:
        ax.plot(x, [values[run][layer] for layer in layers], marker="o", linewidth=1.6, label=run)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Keep Ratio")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_layer_iou(plt: Any, rows: Sequence[Dict[str, Any]], path: str, title: str) -> None:
    if not rows:
        return
    layers = [str(row["layer"]) for row in sorted(rows, key=lambda r: layer_sort_key(str(r["layer"])))]
    pruned = [float(row["pruned_mask_iou"]) for row in sorted(rows, key=lambda r: layer_sort_key(str(r["layer"])))]
    keep = [float(row["keep_mask_iou"]) for row in sorted(rows, key=lambda r: layer_sort_key(str(r["layer"])))]
    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(max(10.0, 0.36 * len(layers)), 5.2))
    ax.plot(x, pruned, marker="o", label="Pruned-mask IoU")
    ax.plot(x, keep, marker="s", label="Keep-mask IoU")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("IoU")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pruned_ratio_bars(plt: Any, rows: Sequence[Dict[str, Any]], path: str, title: str) -> None:
    if not rows:
        return
    runs = sorted({str(row["run"]) for row in rows})
    layers = sorted({str(row["layer"]) for row in rows}, key=layer_sort_key)
    values = {run: {layer: 0.0 for layer in layers} for run in runs}
    for row in rows:
        values[str(row["run"])][str(row["layer"])] = float(row["pruned_ratio"])
    x = np.arange(len(layers))
    width = 0.8 / max(1, len(runs))
    fig, ax = plt.subplots(figsize=(max(10.0, 0.4 * len(layers)), 5.6))
    for i, run in enumerate(runs):
        offset = (i - (len(runs) - 1) / 2) * width
        ax.bar(x + offset, [values[run][layer] for layer in layers], width=width, label=run)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Pruned Ratio")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def sample_flat_values(arrays: Sequence[np.ndarray], max_values: int, rng: np.random.RandomState) -> np.ndarray:
    if not arrays:
        return np.zeros((0,), dtype=np.float32)
    flat = np.concatenate([np.ravel(a).astype(np.float32, copy=False) for a in arrays])
    flat = flat[np.isfinite(flat)]
    if flat.size > max_values:
        idx = rng.choice(flat.size, size=max_values, replace=False)
        flat = flat[idx]
    return flat


def collect_pruned_ratio_values(masks: Dict[str, Dict[str, Any]]) -> np.ndarray:
    values = []
    for item in masks.values():
        valid = item["valid"]
        if np.any(valid):
            values.append(np.asarray([float((item["pruned"] & valid).sum() / valid.sum())], dtype=np.float32))
    return np.concatenate(values) if values else np.zeros((0,), dtype=np.float32)


def plot_zero_ratio_histogram(
    plt: Any,
    run_masks: Dict[str, Dict[str, Dict[str, Any]]],
    path: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for label, masks in run_masks.items():
        values = collect_pruned_ratio_values(masks)
        if values.size:
            ax.hist(values, bins=40, alpha=0.45, density=True, label=label)
    ax.set_xlabel("Per-Parameter-Tensor Pruned Ratio")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_importance(path: Optional[str]) -> Dict[str, np.ndarray]:
    if not path:
        return {}
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    if ext in (".json", ".jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
        if not isinstance(obj, dict):
            raise TypeError("Importance JSON must be a dict.")
        return {str(k): np.asarray(v) for k, v in obj.items()}
    obj = unwrap_state_dict(torch_load(path))
    return {str(k): tensor_to_numpy(v) for k, v in obj.items() if hasattr(v, "shape")}


def filter_importance(
    importance: Dict[str, np.ndarray],
    component: str,
    include_bias: bool,
) -> Dict[str, np.ndarray]:
    out = {}
    for key, arr in importance.items():
        key = canonical_key(key)
        if component_of_key(key) != component:
            continue
        if not include_bias and np.asarray(arr).ndim < 2:
            continue
        out[key] = np.asarray(arr, dtype=np.float32)
    return out


def importance_summary_rows(label: str, importance: Dict[str, np.ndarray], component: str) -> List[Dict[str, Any]]:
    rows = []
    for key, arr in sorted(importance.items()):
        vals = np.abs(np.asarray(arr, dtype=np.float32).reshape(-1))
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rows.append(
            {
                "run": label,
                "component": component,
                "parameter": key,
                "layer": layer_id_for_key(key, component),
                "num_values": int(vals.size),
                "mean_abs_importance": float(np.mean(vals)),
                "p50_abs_importance": float(np.percentile(vals, 50)),
                "p90_abs_importance": float(np.percentile(vals, 90)),
                "p99_abs_importance": float(np.percentile(vals, 99)),
                "max_abs_importance": float(np.max(vals)),
            }
        )
    return rows


def plot_importance_histogram(
    plt: Any,
    label_to_importance: Dict[str, Dict[str, np.ndarray]],
    path: str,
    title: str,
    max_values: int,
    rng: np.random.RandomState,
) -> None:
    if not any(label_to_importance.values()):
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for label, tensors in label_to_importance.items():
        values = sample_flat_values([np.abs(v) for v in tensors.values()], max_values, rng)
        if values.size:
            values = np.log10(values + 1e-12)
            ax.hist(values, bins=80, alpha=0.45, density=True, label=label)
    ax.set_xlabel("log10(abs importance + 1e-12)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def overall_summary(layer_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = sum(int(row["valid_parameters"]) for row in layer_rows)
    if not layer_rows or valid == 0:
        return {}
    first = layer_rows[0]
    label_keys = [key for key in first if key.endswith("_pruned_count")]
    out = {
        "comparison": first["comparison"],
        "component": first["component"],
        "valid_parameters": valid,
        "mean_layer_pruned_iou": float(np.mean([float(row["pruned_mask_iou"]) for row in layer_rows])),
        "mean_layer_keep_iou": float(np.mean([float(row["keep_mask_iou"]) for row in layer_rows])),
    }
    total_pruned_inter = sum(int(row["pruned_intersection"]) for row in layer_rows)
    total_pruned_union = sum(int(row["pruned_union"]) for row in layer_rows)
    total_keep_inter = sum(int(row["keep_intersection"]) for row in layer_rows)
    total_keep_union = sum(int(row["keep_union"]) for row in layer_rows)
    out["global_pruned_mask_iou"] = float(total_pruned_inter / total_pruned_union) if total_pruned_union else 1.0
    out["global_keep_mask_iou"] = float(total_keep_inter / total_keep_union) if total_keep_union else 1.0
    for key in label_keys:
        label = key[: -len("_pruned_count")]
        total_pruned = sum(int(row[key]) for row in layer_rows)
        out[label + "_global_pruned_ratio"] = float(total_pruned / valid)
        out[label + "_global_keep_ratio"] = float(1.0 - total_pruned / valid)
    return out


def main() -> None:
    args = parse_args()
    if args.zero_tol < 0:
        raise ValueError("--zero_tol must be non-negative")
    ensure_dir(args.out_dir)
    rng = np.random.RandomState(args.seed)

    print("Loading checkpoints...")
    base = load_state_dict(args.base_ckpt)
    c4_t5 = load_state_dict(args.c4_t5_ckpt)
    cc3m_multi = load_state_dict(args.cc3m_multimodal_ckpt)
    cc3m_img = load_state_dict(args.cc3m_image_vit_ckpt)
    diagnostics = [
        checkpoint_diagnostics("base", base, args.include_bias),
        checkpoint_diagnostics("c4_t5", c4_t5, args.include_bias),
        checkpoint_diagnostics("cc3m_multimodal", cc3m_multi, args.include_bias),
        checkpoint_diagnostics("cc3m_image_vit", cc3m_img, args.include_bias),
    ]
    with open(os.path.join(args.out_dir, "checkpoint_key_diagnostics.json"), "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    for item in diagnostics:
        print(
            "%s: tensors=%d t5_candidates=%d vit_candidates=%d"
            % (
                item["name"],
                item["num_tensors"],
                item["num_t5_candidate_tensors"],
                item["num_vit_candidate_tensors"],
            )
        )

    mask_sets = {
        "c4_t5": {
            "t5": infer_masks(base, c4_t5, "t5", args.include_bias, args.zero_tol),
        },
        "cc3m_multimodal": {
            "t5": infer_masks(base, cc3m_multi, "t5", args.include_bias, args.zero_tol),
            "vit": infer_masks(base, cc3m_multi, "vit", args.include_bias, args.zero_tol),
        },
        "cc3m_image_vit": {
            "vit": infer_masks(base, cc3m_img, "vit", args.include_bias, args.zero_tol),
        },
    }
    for label, comps in mask_sets.items():
        for component, masks in comps.items():
            print("%s %s tensors: %d" % (label, component, len(masks)))

    required_counts = {
        "c4_t5/t5": len(mask_sets["c4_t5"]["t5"]),
        "cc3m_multimodal/t5": len(mask_sets["cc3m_multimodal"]["t5"]),
        "cc3m_multimodal/vit": len(mask_sets["cc3m_multimodal"]["vit"]),
        "cc3m_image_vit/vit": len(mask_sets["cc3m_image_vit"]["vit"]),
    }
    if any(count == 0 for count in required_counts.values()):
        missing = [name for name, count in required_counts.items() if count == 0]
        raise SystemExit(
            "No comparable pruning masks found for: %s. "
            "See checkpoint_key_diagnostics.json in --out_dir for key-prefix examples. "
            "Common causes: wrong checkpoint path, wrapper-only checkpoint, or a checkpoint "
            "that does not contain the requested module."
            % ", ".join(missing)
        )

    param_rows: List[Dict[str, Any]] = []
    layer_compare_rows: List[Dict[str, Any]] = []
    run_layer_rows: List[Dict[str, Any]] = []

    for label, component in (
        ("c4_t5", "t5"),
        ("cc3m_multimodal", "t5"),
        ("cc3m_multimodal", "vit"),
        ("cc3m_image_vit", "vit"),
    ):
        run_layer_rows.extend(summarize_run_masks(label, mask_sets[label][component], component))

    rows, layers = compare_mask_sets(
        "c4_t5",
        mask_sets["c4_t5"]["t5"],
        "cc3m_multimodal",
        mask_sets["cc3m_multimodal"]["t5"],
        "t5",
    )
    param_rows.extend(rows)
    layer_compare_rows.extend(layers)

    rows, layers = compare_mask_sets(
        "cc3m_image_vit",
        mask_sets["cc3m_image_vit"]["vit"],
        "cc3m_multimodal",
        mask_sets["cc3m_multimodal"]["vit"],
        "vit",
    )
    param_rows.extend(rows)
    layer_compare_rows.extend(layers)

    summary_rows = [row for row in (overall_summary([r for r in layer_compare_rows if r["comparison"] == comp]) for comp in sorted({r["comparison"] for r in layer_compare_rows})) if row]

    write_csv(os.path.join(args.out_dir, "mask_iou_by_parameter.csv"), param_rows)
    write_csv(os.path.join(args.out_dir, "mask_iou_by_layer.csv"), layer_compare_rows)
    write_csv(os.path.join(args.out_dir, "layer_keep_pruned_ratios.csv"), run_layer_rows)
    write_csv(os.path.join(args.out_dir, "mask_iou_summary.csv"), summary_rows)

    importance_inputs = {
        "c4_t5": load_importance(args.c4_t5_importance),
        "cc3m_multimodal": load_importance(args.cc3m_multimodal_importance),
        "cc3m_image_vit": load_importance(args.cc3m_image_vit_importance),
    }
    importance_rows: List[Dict[str, Any]] = []
    filtered_importance: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for label, raw in importance_inputs.items():
        filtered_importance[label] = {
            "t5": filter_importance(raw, "t5", args.include_bias),
            "vit": filter_importance(raw, "vit", args.include_bias),
        }
        for component in ("t5", "vit"):
            importance_rows.extend(
                importance_summary_rows(label, filtered_importance[label][component], component)
            )
    write_csv(os.path.join(args.out_dir, "importance_summary_by_parameter.csv"), importance_rows)

    with open(os.path.join(args.out_dir, "mask_compare_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "base_ckpt": os.path.abspath(args.base_ckpt),
                "c4_t5_ckpt": os.path.abspath(args.c4_t5_ckpt),
                "cc3m_multimodal_ckpt": os.path.abspath(args.cc3m_multimodal_ckpt),
                "cc3m_image_vit_ckpt": os.path.abspath(args.cc3m_image_vit_ckpt),
                "zero_tol": args.zero_tol,
                "include_bias": bool(args.include_bias),
                "definition": {
                    "valid": "base checkpoint parameter abs(value) > zero_tol",
                    "pruned": "valid and pruned checkpoint abs(value) <= zero_tol",
                    "keep": "valid and pruned checkpoint abs(value) > zero_tol",
                },
                "summary": summary_rows,
                "checkpoint_key_diagnostics": diagnostics,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    plt = setup_matplotlib()
    plot_layer_keep_ratios(
        plt,
        [row for row in run_layer_rows if row["component"] == "t5"],
        os.path.join(args.out_dir, "t5_layer_keep_ratio.png"),
        "T5 Layer Keep Ratio: C4 Text-Only vs CC3M Multimodal",
    )
    plot_pruned_ratio_bars(
        plt,
        [row for row in run_layer_rows if row["component"] == "t5"],
        os.path.join(args.out_dir, "t5_layer_pruned_ratio_bars.png"),
        "T5 Layer Pruned Ratio",
    )
    plot_layer_iou(
        plt,
        [row for row in layer_compare_rows if row["comparison"] == "c4_t5_vs_cc3m_multimodal"],
        os.path.join(args.out_dir, "t5_c4_vs_cc3m_multimodal_mask_iou_by_layer.png"),
        "T5 Mask IoU by Layer: C4 Text-Only vs CC3M Multimodal",
    )
    plot_zero_ratio_histogram(
        plt,
        {
            "c4_t5": mask_sets["c4_t5"]["t5"],
            "cc3m_multimodal": mask_sets["cc3m_multimodal"]["t5"],
        },
        os.path.join(args.out_dir, "t5_pruned_ratio_distribution.png"),
        "T5 Per-Parameter Pruned Ratio Distribution",
    )

    plot_layer_keep_ratios(
        plt,
        [row for row in run_layer_rows if row["component"] == "vit"],
        os.path.join(args.out_dir, "vit_layer_keep_ratio.png"),
        "ViT Layer Keep Ratio: CC3M Image-Only vs CC3M Multimodal",
    )
    plot_pruned_ratio_bars(
        plt,
        [row for row in run_layer_rows if row["component"] == "vit"],
        os.path.join(args.out_dir, "vit_layer_pruned_ratio_bars.png"),
        "ViT Layer Pruned Ratio",
    )
    plot_layer_iou(
        plt,
        [row for row in layer_compare_rows if row["comparison"] == "cc3m_image_vit_vs_cc3m_multimodal"],
        os.path.join(args.out_dir, "vit_image_vs_cc3m_multimodal_mask_iou_by_layer.png"),
        "ViT Mask IoU by Layer: CC3M Image-Only vs CC3M Multimodal",
    )
    plot_zero_ratio_histogram(
        plt,
        {
            "cc3m_image_vit": mask_sets["cc3m_image_vit"]["vit"],
            "cc3m_multimodal": mask_sets["cc3m_multimodal"]["vit"],
        },
        os.path.join(args.out_dir, "vit_pruned_ratio_distribution.png"),
        "ViT Per-Parameter Pruned Ratio Distribution",
    )

    plot_importance_histogram(
        plt,
        {
            "c4_t5": filtered_importance["c4_t5"]["t5"],
            "cc3m_multimodal": filtered_importance["cc3m_multimodal"]["t5"],
        },
        os.path.join(args.out_dir, "t5_importance_distribution.png"),
        "T5 Importance Score Distribution",
        args.max_hist_values,
        rng,
    )
    plot_importance_histogram(
        plt,
        {
            "cc3m_image_vit": filtered_importance["cc3m_image_vit"]["vit"],
            "cc3m_multimodal": filtered_importance["cc3m_multimodal"]["vit"],
        },
        os.path.join(args.out_dir, "vit_importance_distribution.png"),
        "ViT Importance Score Distribution",
        args.max_hist_values,
        rng,
    )

    print("[OK] wrote outputs to:", os.path.abspath(args.out_dir))
    for row in summary_rows:
        print(
            "%s %s: pruned IoU=%.4f keep IoU=%.4f"
            % (
                row["comparison"],
                row["component"],
                row["global_pruned_mask_iou"],
                row["global_keep_mask_iou"],
            )
        )


if __name__ == "__main__":
    main()
