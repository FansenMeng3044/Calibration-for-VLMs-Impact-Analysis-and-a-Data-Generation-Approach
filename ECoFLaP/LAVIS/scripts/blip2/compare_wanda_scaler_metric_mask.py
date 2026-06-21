#!/usr/bin/env python3
"""Compare Wanda scaler_row, W_metric, and masks between two recorded runs.

Inputs are directories produced by record_wanda_input_importance.py.  The
comparison is module-aligned: the same Linear layer name in run A and run B is
compared directly, which matches the granularity used by Wanda pruning.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare recorded Wanda scaler_row, W_metric, and masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run_a_dir", required=True)
    parser.add_argument("--run_b_dir", required=True)
    parser.add_argument("--label_a", default="run_a")
    parser.add_argument("--label_b", default="run_b")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--component",
        choices=["all", "t5", "t5_encoder", "t5_decoder", "vit"],
        default="all",
        help="Restrict comparison to one component family.",
    )
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument(
        "--position_plot_count",
        type=int,
        default=12,
        help="Number of most-different modules to render as position-level plots. Set 0 to disable.",
    )
    parser.add_argument(
        "--position_module_regex",
        default="",
        help="Optional regex over module_name/role/component to choose modules for position plots.",
    )
    parser.add_argument(
        "--position_downsample",
        type=int,
        default=256,
        help="Maximum side length used when block-downsampling large matrix heatmaps.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run(run_dir: str) -> Tuple[Dict[str, Dict[str, Any]], Any]:
    module_path = os.path.join(run_dir, "wanda_linear_input_importance_by_module.json")
    arrays_path = os.path.join(run_dir, "wanda_linear_input_importance_arrays.npz")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(module_path)
    if not os.path.isfile(arrays_path):
        raise FileNotFoundError(arrays_path)
    rows = load_json(module_path)
    if not isinstance(rows, list):
        raise ValueError("%s must contain a JSON list." % module_path)
    return {str(row["module_name"]): row for row in rows}, np.load(arrays_path)


def component_allowed(component: str, selected: str) -> bool:
    if selected == "all":
        return True
    if selected == "t5":
        return component in {"t5_encoder", "t5_decoder"}
    return component == selected


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def l1_mean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.reshape(-1).astype(np.float64) - b.reshape(-1).astype(np.float64))))


def l2_norm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.reshape(-1).astype(np.float64) - b.reshape(-1).astype(np.float64)))


def rankdata(values: np.ndarray) -> np.ndarray:
    values = values.reshape(-1).astype(np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.size < 2 or b.size < 2:
        return float("nan")
    ar = rankdata(a)
    br = rankdata(b)
    ar = ar - ar.mean()
    br = br - br.mean()
    denom = float(np.linalg.norm(ar) * np.linalg.norm(br))
    if denom == 0:
        return float("nan")
    return float(np.dot(ar, br) / denom)


def vector_metrics(prefix: str, a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    return {
        prefix + "_cosine": cosine(a, b),
        prefix + "_spearman": spearman(a, b),
        prefix + "_l1_mean": l1_mean(a, b),
        prefix + "_l2": l2_norm(a, b),
    }


def mask_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    a_bool = a.reshape(-1).astype(bool)
    b_bool = b.reshape(-1).astype(bool)
    keep_a = ~a_bool
    keep_b = ~b_bool
    pruned_inter = int(np.logical_and(a_bool, b_bool).sum())
    pruned_union = int(np.logical_or(a_bool, b_bool).sum())
    keep_inter = int(np.logical_and(keep_a, keep_b).sum())
    keep_union = int(np.logical_or(keep_a, keep_b).sum())
    return {
        "mask_available": True,
        "pruned_intersection": pruned_inter,
        "pruned_union": pruned_union,
        "pruned_iou": float(pruned_inter / pruned_union) if pruned_union else float("nan"),
        "keep_intersection": keep_inter,
        "keep_union": keep_union,
        "keep_iou": float(keep_inter / keep_union) if keep_union else float("nan"),
        "pruned_fraction_a": float(a_bool.mean()) if a_bool.size else 0.0,
        "pruned_fraction_b": float(b_bool.mean()) if b_bool.size else 0.0,
        "pruned_overlap_fraction_a": float(pruned_inter / max(int(a_bool.sum()), 1)),
        "pruned_overlap_fraction_b": float(pruned_inter / max(int(b_bool.sum()), 1)),
    }


def finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def weighted_mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
    num = 0.0
    den = 0.0
    for row in rows:
        value = row.get(key)
        if value is None or not math.isfinite(float(value)):
            continue
        weight = float(row.get("weight_numel", 1))
        num += float(value) * weight
        den += weight
    return float(num / den) if den else float("nan")


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
        writer.writerows(rows)


def write_json(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def get_array(arrays: Any, key: str) -> Optional[np.ndarray]:
    return arrays[key] if key in arrays.files else None


def array_percentile_clip(x: np.ndarray, low: float = 1.0, high: float = 99.0) -> Tuple[float, float]:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, low))
    hi = float(np.percentile(finite, high))
    if lo == hi:
        hi = lo + 1e-12
    return lo, hi


def downsample_mean(x: np.ndarray, max_side: int) -> np.ndarray:
    if x.ndim != 2:
        return x
    rows, cols = x.shape
    if rows <= max_side and cols <= max_side:
        return x
    row_bins = min(rows, max_side)
    col_bins = min(cols, max_side)
    row_edges = np.linspace(0, rows, row_bins + 1, dtype=np.int64)
    col_edges = np.linspace(0, cols, col_bins + 1, dtype=np.int64)
    out = np.zeros((row_bins, col_bins), dtype=np.float32)
    for i in range(row_bins):
        r0, r1 = int(row_edges[i]), int(row_edges[i + 1])
        for j in range(col_bins):
            c0, c1 = int(col_edges[j]), int(col_edges[j + 1])
            block = x[r0:r1, c0:c1]
            out[i, j] = float(block.mean()) if block.size else 0.0
    return out


def make_position_safe_name(row: Dict[str, Any], index: int) -> str:
    text = "%03d_%s_L%s_%s" % (
        index,
        str(row.get("component", "")),
        str(row.get("layer", "")),
        str(row.get("role", "")),
    )
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def compare_modules(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows_a, arrays_a = load_run(args.run_a_dir)
    rows_b, arrays_b = load_run(args.run_b_dir)
    module_rows: List[Dict[str, Any]] = []

    for name in sorted(set(rows_a) & set(rows_b)):
        row_a = rows_a[name]
        row_b = rows_b[name]
        component = str(row_a.get("component", row_b.get("component", "")))
        if not component_allowed(component, args.component):
            continue
        key = safe_key(name)
        scaler_a = get_array(arrays_a, key + ".wanda_scaler")
        scaler_b = get_array(arrays_b, key + ".wanda_scaler")
        if scaler_a is None or scaler_b is None or scaler_a.shape != scaler_b.shape:
            continue

        out: Dict[str, Any] = {
            "component": component,
            "layer": int(row_a.get("layer", row_b.get("layer", -1))),
            "role": row_a.get("role", ""),
            "module_name": name,
            "weight_shape": row_a.get("weight_shape", ""),
            "out_dim": int(row_a.get("out_dim", 0)),
            "in_dim": int(row_a.get("in_dim", 0)),
            "weight_numel": int(row_a.get("weight_numel", 0)),
            "label_a": args.label_a,
            "label_b": args.label_b,
        }
        out.update(vector_metrics("scaler_row", scaler_a, scaler_b))

        metric_a = get_array(arrays_a, key + ".wanda_metric")
        metric_b = get_array(arrays_b, key + ".wanda_metric")
        if metric_a is not None and metric_b is not None and metric_a.shape == metric_b.shape:
            out["w_metric_source"] = "full_wanda_metric"
            out.update(vector_metrics("w_metric", metric_a, metric_b))
        else:
            metric_a = get_array(arrays_a, key + ".wanda_importance_col_mean")
            metric_b = get_array(arrays_b, key + ".wanda_importance_col_mean")
            if metric_a is not None and metric_b is not None and metric_a.shape == metric_b.shape:
                out["w_metric_source"] = "wanda_importance_col_mean_proxy"
                out.update(vector_metrics("w_metric", metric_a, metric_b))
            else:
                out["w_metric_source"] = "missing"

        mask_a = get_array(arrays_a, key + ".wanda_pruned_mask")
        mask_b = get_array(arrays_b, key + ".wanda_pruned_mask")
        if mask_a is not None and mask_b is not None and mask_a.shape == mask_b.shape:
            out.update(mask_metrics(mask_a, mask_b))
        else:
            out.update(
                {
                    "mask_available": False,
                    "pruned_iou": float("nan"),
                    "keep_iou": float("nan"),
                    "pruned_fraction_a": float("nan"),
                    "pruned_fraction_b": float("nan"),
                }
            )
        module_rows.append(out)

    layer_groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in module_rows:
        layer_groups.setdefault((str(row["component"]), int(row["layer"])), []).append(row)

    layer_rows: List[Dict[str, Any]] = []
    for (component, layer), rows in sorted(layer_groups.items(), key=lambda item: (item[0][0], item[0][1])):
        layer_row: Dict[str, Any] = {
            "component": component,
            "layer": layer,
            "modules": len(rows),
            "weight_numel": int(sum(int(row.get("weight_numel", 0)) for row in rows)),
            "scaler_row_cosine": weighted_mean(rows, "scaler_row_cosine"),
            "scaler_row_spearman": weighted_mean(rows, "scaler_row_spearman"),
            "scaler_row_l1_mean": weighted_mean(rows, "scaler_row_l1_mean"),
            "w_metric_cosine": weighted_mean(rows, "w_metric_cosine"),
            "w_metric_spearman": weighted_mean(rows, "w_metric_spearman"),
            "w_metric_l1_mean": weighted_mean(rows, "w_metric_l1_mean"),
            "mask_modules": sum(1 for row in rows if row.get("mask_available")),
        }
        pruned_inter = sum(int(row.get("pruned_intersection", 0)) for row in rows if row.get("mask_available"))
        pruned_union = sum(int(row.get("pruned_union", 0)) for row in rows if row.get("mask_available"))
        keep_inter = sum(int(row.get("keep_intersection", 0)) for row in rows if row.get("mask_available"))
        keep_union = sum(int(row.get("keep_union", 0)) for row in rows if row.get("mask_available"))
        layer_row["pruned_iou"] = float(pruned_inter / pruned_union) if pruned_union else float("nan")
        layer_row["keep_iou"] = float(keep_inter / keep_union) if keep_union else float("nan")
        layer_row["pruned_fraction_a"] = finite_mean(row.get("pruned_fraction_a", float("nan")) for row in rows)
        layer_row["pruned_fraction_b"] = finite_mean(row.get("pruned_fraction_b", float("nan")) for row in rows)
        layer_rows.append(layer_row)

    return module_rows, layer_rows


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
        ("scaler_row_cosine", "scaler_row cosine", "layer_scaler_row_cosine.png"),
        ("w_metric_cosine", "W_metric cosine", "layer_w_metric_cosine.png"),
        ("pruned_iou", "Pruned mask IoU", "layer_pruned_mask_iou.png"),
        ("keep_iou", "Keep mask IoU", "layer_keep_mask_iou.png"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 6))
        for component in sorted({str(row["component"]) for row in layer_rows}):
            rows = [row for row in layer_rows if row["component"] == component]
            rows.sort(key=lambda row: int(row["layer"]))
            ax.plot([row["layer"] for row in rows], [row.get(metric, float("nan")) for row in rows], marker="o", label=component)
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

    mask_values = [float(row["pruned_iou"]) for row in module_rows if row.get("mask_available") and math.isfinite(float(row.get("pruned_iou", float("nan"))))]
    if mask_values:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(mask_values, bins=40, alpha=0.85)
        ax.set_xlabel("Per-module pruned mask IoU")
        ax.set_ylabel("Modules")
        ax.set_title("Final Wanda Mask Difference Distribution")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = os.path.join(out_dir, "module_pruned_mask_iou_hist.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)

    return paths


def select_position_rows(args: argparse.Namespace, module_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = list(module_rows)
    if args.position_module_regex:
        pattern = re.compile(args.position_module_regex)
        rows = [
            row
            for row in rows
            if pattern.search(
                "%s %s %s"
                % (str(row.get("module_name", "")), str(row.get("role", "")), str(row.get("component", "")))
            )
        ]
    rows = [
        row
        for row in rows
        if row.get("mask_available") and math.isfinite(float(row.get("pruned_iou", float("nan"))))
    ]
    rows.sort(key=lambda row: float(row.get("pruned_iou", 1.0)))
    return rows[: max(int(args.position_plot_count), 0)]


def make_position_plots(args: argparse.Namespace, module_rows: Sequence[Dict[str, Any]]) -> List[str]:
    if args.position_plot_count <= 0:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print("[WARN] matplotlib unavailable, skipping position plots: %s" % exc)
        return []

    _rows_a, arrays_a = load_run(args.run_a_dir)
    _rows_b, arrays_b = load_run(args.run_b_dir)
    selected = select_position_rows(args, module_rows)
    if not selected:
        return []

    position_dir = os.path.join(args.out_dir, "position_level")
    ensure_dir(position_dir)
    paths: List[str] = []
    manifest: List[Dict[str, Any]] = []

    for index, row in enumerate(selected, start=1):
        module_name = str(row["module_name"])
        key = safe_key(module_name)
        prefix = make_position_safe_name(row, index)

        scaler_a = get_array(arrays_a, key + ".wanda_scaler")
        scaler_b = get_array(arrays_b, key + ".wanda_scaler")
        if scaler_a is not None and scaler_b is not None and scaler_a.shape == scaler_b.shape:
            x = np.arange(scaler_a.reshape(-1).shape[0])
            diff = np.abs(scaler_a.reshape(-1).astype(np.float64) - scaler_b.reshape(-1).astype(np.float64))
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            axes[0].plot(x, scaler_a.reshape(-1), linewidth=0.8, label=args.label_a)
            axes[0].plot(x, scaler_b.reshape(-1), linewidth=0.8, label=args.label_b, alpha=0.8)
            axes[0].set_ylabel("scaler_row")
            axes[0].legend()
            axes[0].grid(True, alpha=0.2)
            axes[1].plot(x, diff, linewidth=0.8, color="tab:red")
            axes[1].set_xlabel("Input column index")
            axes[1].set_ylabel("|A - B|")
            axes[1].grid(True, alpha=0.2)
            fig.suptitle("%s\n%s" % (module_name, "scaler_row position difference"))
            fig.tight_layout()
            path = os.path.join(position_dir, prefix + "_scaler_row_positions.png")
            fig.savefig(path, dpi=200)
            plt.close(fig)
            paths.append(path)

        mask_a = get_array(arrays_a, key + ".wanda_pruned_mask")
        mask_b = get_array(arrays_b, key + ".wanda_pruned_mask")
        if mask_a is not None and mask_b is not None and mask_a.shape == mask_b.shape and mask_a.ndim == 2:
            a = mask_a.astype(bool)
            b = mask_b.astype(bool)
            disagree = np.logical_xor(a, b).astype(np.float32)
            panels = [
                (downsample_mean(a.astype(np.float32), args.position_downsample), args.label_a + " pruned fraction"),
                (downsample_mean(b.astype(np.float32), args.position_downsample), args.label_b + " pruned fraction"),
                (downsample_mean(disagree, args.position_downsample), "A/B mask disagreement"),
            ]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            for ax, (mat, title) in zip(axes, panels):
                im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0, cmap="viridis")
                ax.set_title(title)
                ax.set_xlabel("Input columns")
                ax.set_ylabel("Output rows")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle("%s\n%s" % (module_name, "final Wanda mask position difference"))
            fig.tight_layout()
            path = os.path.join(position_dir, prefix + "_mask_matrix_positions.png")
            fig.savefig(path, dpi=200)
            plt.close(fig)
            paths.append(path)

        metric_a = get_array(arrays_a, key + ".wanda_metric")
        metric_b = get_array(arrays_b, key + ".wanda_metric")
        if metric_a is not None and metric_b is not None and metric_a.shape == metric_b.shape and metric_a.ndim == 2:
            a = downsample_mean(metric_a.astype(np.float32), args.position_downsample)
            b = downsample_mean(metric_b.astype(np.float32), args.position_downsample)
            d = downsample_mean(np.abs(metric_a.astype(np.float32) - metric_b.astype(np.float32)), args.position_downsample)
            vmin, vmax = array_percentile_clip(np.concatenate([a.reshape(-1), b.reshape(-1)]))
            dmin, dmax = array_percentile_clip(d, 0.0, 99.0)
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            for ax, mat, title, lo, hi in [
                (axes[0], a, args.label_a + " W_metric", vmin, vmax),
                (axes[1], b, args.label_b + " W_metric", vmin, vmax),
                (axes[2], d, "|A - B| W_metric", dmin, dmax),
            ]:
                im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=lo, vmax=hi, cmap="magma")
                ax.set_title(title)
                ax.set_xlabel("Input columns")
                ax.set_ylabel("Output rows")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle("%s\n%s" % (module_name, "W_metric matrix position difference"))
            fig.tight_layout()
            path = os.path.join(position_dir, prefix + "_w_metric_matrix_positions.png")
            fig.savefig(path, dpi=200)
            plt.close(fig)
            paths.append(path)

        manifest.append(
            {
                "module_name": module_name,
                "component": row.get("component"),
                "layer": row.get("layer"),
                "role": row.get("role"),
                "pruned_iou": row.get("pruned_iou"),
                "scaler_row_cosine": row.get("scaler_row_cosine"),
                "w_metric_cosine": row.get("w_metric_cosine"),
                "has_full_w_metric": bool(
                    get_array(arrays_a, key + ".wanda_metric") is not None
                    and get_array(arrays_b, key + ".wanda_metric") is not None
                ),
            }
        )

    write_json(os.path.join(position_dir, "position_level_manifest.json"), manifest)
    return paths


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    module_rows, layer_rows = compare_modules(args)

    module_csv = os.path.join(args.out_dir, "wanda_module_scaler_metric_mask_comparison.csv")
    layer_csv = os.path.join(args.out_dir, "wanda_layer_scaler_metric_mask_comparison.csv")
    write_csv(module_csv, module_rows)
    write_csv(layer_csv, layer_rows)
    write_json(os.path.join(args.out_dir, "wanda_module_scaler_metric_mask_comparison.json"), module_rows)
    write_json(os.path.join(args.out_dir, "wanda_layer_scaler_metric_mask_comparison.json"), layer_rows)

    plot_paths: List[str] = []
    if not args.no_plots:
        plot_paths = make_plots(args.out_dir, layer_rows, module_rows)
        plot_paths.extend(make_position_plots(args, module_rows))

    print("[OK] matched modules:", len(module_rows))
    print("[OK] mask modules:", sum(1 for row in module_rows if row.get("mask_available")))
    print("[OK] wrote:", module_csv)
    print("[OK] wrote:", layer_csv)
    for path in plot_paths:
        print("[OK] plot:", path)


if __name__ == "__main__":
    main()
