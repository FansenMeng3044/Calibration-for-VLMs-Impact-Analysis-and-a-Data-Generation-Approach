#!/usr/bin/env python3
"""Plot semantic similarity and layer-wise L2 drift in one paper figure.

This script only consumes existing CSV results. It does not run extraction,
forward passes, or model loading.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from split_joint_analysis_common import ensure_dir, setup_matplotlib


HEATMAP_COLORS = ["#f7fffc", "#ddf8ee", "#a9ecd9", "#62d4c6", "#2aa7b8"]
LINE_COLORS = ["#0072B2", "#009E73", "#56B4E9", "#CC79A7", "#E69F00"]
LINE_MARKERS = ["o", "s", "^", "D", "P"]
CALIB_ORDER = ["MMBench", "MMMU", "OKVQA", "mathvista", "MathVista", "cc3m", "CC3M"]
EVAL_ORDER = ["MMBench", "MMMU", "OKVQA", "mathvista", "MathVista"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine semantic heatmap with OKVQA/MMBench T5 layer L2 curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--semantic_dir",
        default="/data/data2/mfs/llm_embedding_fidelity_fourbench/semantic_dense/semantic_both",
        help="Directory containing calib_eval_semantic_similarity_<part>.csv, or the CSV itself.",
    )
    parser.add_argument(
        "--okvqa_dir",
        default="/data/data2/mfs/t5_layer_fidelity_fourbench/OKVQA/t5_layer_both",
        help="Directory containing t5_layer_fidelity_<part>.csv, or the CSV itself.",
    )
    parser.add_argument(
        "--mmbench_dir",
        default="/data/data2/mfs/t5_layer_fidelity_fourbench/MMBench/t5_layer_both",
        help="Directory containing t5_layer_fidelity_<part>.csv, or the CSV itself.",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--part", choices=["both", "visual", "text"], default="both")
    parser.add_argument("--semantic_metric", default="centroid_cosine")
    parser.add_argument("--line_metric", default="rel_l2_to_dense_mean")
    parser.add_argument("--fig_name", default="semantic_heatmap_okvqa_mmbench_l2_combined")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_semantic_csv(path: str, part: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        return path
    candidates = [
        os.path.join(path, "calib_eval_semantic_similarity_%s.csv" % part),
        os.path.join(path, "semantic_similarity_%s.csv" % part),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("Could not find semantic CSV under %s" % path)


def resolve_layer_csv(path: str, part: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        return path
    candidates = [
        os.path.join(path, "t5_layer_fidelity_%s.csv" % part),
        os.path.join(path, "t5_layer_fidelity.csv"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("Could not find T5 layer fidelity CSV under %s" % path)


def ordered_labels(labels: Iterable[str], preferred: Sequence[str]) -> List[str]:
    seen = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    preferred_present = [label for label in preferred if label in seen]
    return preferred_present + [label for label in seen if label not in preferred_present]


def read_semantic_matrix(path: str, metric: str) -> Tuple[List[str], List[str], np.ndarray]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError("Empty semantic CSV: %s" % path)

    row_key = "calibration"
    col_key = "eval" if "eval" in rows[0] else "reference"
    if metric not in rows[0]:
        raise KeyError("Metric %r not found in %s. Available: %s" % (metric, path, sorted(rows[0].keys())))

    row_labels = ordered_labels((row[row_key] for row in rows), CALIB_ORDER)
    col_labels = ordered_labels((row[col_key] for row in rows), EVAL_ORDER)
    matrix = np.full((len(row_labels), len(col_labels)), np.nan, dtype=np.float64)
    row_index = {label: i for i, label in enumerate(row_labels)}
    col_index = {label: i for i, label in enumerate(col_labels)}
    for row in rows:
        matrix[row_index[row[row_key]], col_index[row[col_key]]] = float(row[metric])
    return row_labels, col_labels, matrix


def read_layer_series(path: str, metric: str) -> Dict[str, List[Tuple[int, float]]]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError("Empty layer CSV: %s" % path)
    if metric not in rows[0]:
        raise KeyError("Metric %r not found in %s. Available: %s" % (metric, path, sorted(rows[0].keys())))

    series: Dict[str, List[Tuple[int, float]]] = {}
    for row in rows:
        label = row["calibration"]
        series.setdefault(label, []).append((int(row["layer"]), float(row[metric])))
    for label in series:
        series[label].sort(key=lambda item: item[0])
    return series


def make_heatmap_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("soft_blue_green", HEATMAP_COLORS)


def draw_heatmap(ax, fig, row_labels: Sequence[str], col_labels: Sequence[str], matrix: np.ndarray):
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ValueError("Semantic matrix has no finite values.")
    im = ax.imshow(
        matrix,
        cmap=make_heatmap_cmap(),
        vmin=float(np.nanmin(matrix)),
        vmax=float(np.nanmax(matrix)),
        aspect="auto",
    )
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=32, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Evaluation Dataset")
    ax.set_ylabel("Calibration Dataset")
    ax.set_title("(a) Semantic Similarity", pad=10)

    vmin = float(np.nanmin(matrix))
    vmax = float(np.nanmax(matrix))
    span = max(vmax - vmin, 1e-12)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            color = "white" if (value - vmin) / span > 0.72 else "#183744"
            ax.text(j, i, "%.3f" % value, ha="center", va="center", fontsize=8.5, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("Centroid Cosine")


def draw_l2_panel(ax, series: Dict[str, List[Tuple[int, float]]], title: str, show_xlabel: bool) -> Tuple[List[object], List[str]]:
    labels = ordered_labels(series.keys(), CALIB_ORDER)
    handles = []
    plotted_labels = []
    for idx, label in enumerate(labels):
        points = series[label]
        if not points:
            continue
        handle = ax.plot(
            [x for x, _ in points],
            [y for _, y in points],
            color=LINE_COLORS[idx % len(LINE_COLORS)],
            marker=LINE_MARKERS[idx % len(LINE_MARKERS)],
            linewidth=2.15,
            markersize=4.2,
            label=label,
        )[0]
        handles.append(handle)
        plotted_labels.append(label)
    ax.set_title(title, pad=9)
    ax.set_ylabel("Relative L2 to Dense")
    if show_xlabel:
        ax.set_xlabel("T5 Encoder Layer")
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.grid(True, alpha=0.26, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return handles, plotted_labels


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    semantic_csv = resolve_semantic_csv(args.semantic_dir, args.part)
    okvqa_csv = resolve_layer_csv(args.okvqa_dir, args.part)
    mmbench_csv = resolve_layer_csv(args.mmbench_dir, args.part)

    row_labels, col_labels, semantic_matrix = read_semantic_matrix(semantic_csv, args.semantic_metric)
    okvqa_series = read_layer_series(okvqa_csv, args.line_metric)
    mmbench_series = read_layer_series(mmbench_csv, args.line_metric)

    plt = setup_matplotlib()
    if plt is None:
        raise RuntimeError("matplotlib is required to draw this combined figure.")

    fig = plt.figure(figsize=(13.2, 5.7))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.55], height_ratios=[1.0, 1.0], wspace=0.27, hspace=0.34)
    ax_heat = fig.add_subplot(grid[:, 0])
    ax_okvqa = fig.add_subplot(grid[0, 1])
    ax_mmbench = fig.add_subplot(grid[1, 1], sharex=ax_okvqa)

    draw_heatmap(ax_heat, fig, row_labels, col_labels, semantic_matrix)
    handles, labels = draw_l2_panel(ax_okvqa, okvqa_series, "(b) OKVQA Layer-wise L2 Drift", show_xlabel=False)
    draw_l2_panel(ax_mmbench, mmbench_series, "(c) MMBench Layer-wise L2 Drift", show_xlabel=True)

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.70, 1.015),
            ncol=min(len(labels), 5),
            frameon=False,
            columnspacing=1.35,
            handlelength=2.0,
        )

    fig.subplots_adjust(top=0.88, left=0.07, right=0.97, bottom=0.13)
    for ext in ("png", "pdf"):
        out_path = os.path.join(args.out_dir, "%s.%s" % (args.fig_name, ext))
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        print("[OK] plot:", out_path)
    plt.close(fig)
    print("[OK] semantic CSV:", semantic_csv)
    print("[OK] OKVQA layer CSV:", okvqa_csv)
    print("[OK] MMBench layer CSV:", mmbench_csv)


if __name__ == "__main__":
    main()
