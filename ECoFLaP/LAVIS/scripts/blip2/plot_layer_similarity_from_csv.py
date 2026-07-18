#!/usr/bin/env python3
"""Replot encoder layer-similarity curves from an existing CSV.

This is a paper-figure helper for results already produced by
``visualize_layer_activation_similarity.py``. It does not load the model or run
forward passes; it only consumes ``per_layer_similarity_to_dense.csv`` and emits
encoder-only figures with the same blue/green visual language used by the other
paper plots.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Sequence, Tuple

from split_joint_analysis_common import ensure_dir, setup_matplotlib


DEFAULT_CSV = (
    "/data/data2/mfs/split_joint_analysis_wanda_cc3m_025350/"
    "layer_similarity/per_layer_similarity_to_dense.csv"
)

PALETTES = {
    # Stronger than the old blue/green while still living in the paper palette.
    "blue_gold": ["#1D5D9B", "#E69F00", "#56B4E9", "#009E73", "#CC79A7"],
    # Best choice when you want to stay strictly blue-green.
    "deep_blue_teal": ["#174A7C", "#00A087", "#56B4E9", "#2AA7B8", "#4A9DAE"],
    # Colorblind-safe, very clear in print.
    "blue_vermillion": ["#0072B2", "#D55E00", "#009E73", "#56B4E9", "#CC79A7"],
    # Softer, closer to the semantic heatmap.
    "paper_teal": ["#2A6F97", "#4A9D8F", "#61A5C2", "#89C2B8", "#B279A2"],
}
MODEL_MARKERS = {
    "split": "o",
    "joint": "s",
    "merged": "o",
    "multimodal": "s",
}
FALLBACK_MARKERS = ["o", "s", "^", "D", "P"]
MODEL_LINESTYLES = {
    "split": "-",
    "joint": "--",
    "merged": "-",
    "multimodal": "--",
}

TOKEN_GROUP_TITLES = {
    "visual": "Visual Prefix",
    "text": "Text Tokens",
}
METRIC_TITLES = {
    "rel_l2": "Relative L2 Drift",
    "cosine": "Cosine Similarity",
    "centered_cosine": "Centered Cosine Similarity",
}
METRIC_YLABELS = {
    "rel_l2": "Relative L2 to Dense",
    "cosine": "Cosine to Dense",
    "centered_cosine": "Centered Cosine to Dense",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot encoder-only layer similarity curves from per_layer_similarity_to_dense.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to per_layer_similarity_to_dense.csv.")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Defaults to <csv_dir>/paper_encoder_similarity.",
    )
    parser.add_argument(
        "--metrics",
        default="rel_l2",
        help="Comma-separated metric names. Uses <metric>_mean columns.",
    )
    parser.add_argument(
        "--token_groups",
        default="visual,text",
        help="Comma-separated encoder token groups to draw.",
    )
    parser.add_argument(
        "--model_order",
        default="split,joint",
        help="Preferred comma-separated model order; missing models are ignored.",
    )
    parser.add_argument("--xlabel", default="Layer Index")
    parser.add_argument("--fig_prefix", default="encoder_similarity")
    parser.add_argument(
        "--palette",
        choices=sorted(PALETTES),
        default="blue_gold",
        help="Line/bar color pair. blue_gold is the most visibly separated.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no_combined", action="store_true", help="Only write individual figures.")
    return parser.parse_args()


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def pretty_model(label: str) -> str:
    mapping = {
        "split": "Split",
        "joint": "Joint",
        "merged": "Split",
        "multimodal": "Joint",
    }
    return mapping.get(label.casefold(), label)


def ordered_models(rows: Sequence[Dict[str, str]], preferred: Sequence[str]) -> List[str]:
    present: List[str] = []
    for row in rows:
        model = row.get("model", "")
        if model and model not in present:
            present.append(model)
    order: List[str] = []
    for wanted in preferred:
        for model in present:
            if model.casefold() == wanted.casefold() and model not in order:
                order.append(model)
    for model in present:
        if model not in order:
            order.append(model)
    return order


def metric_column(metric: str, rows: Sequence[Dict[str, str]]) -> str:
    column = "%s_mean" % metric
    if not rows:
        raise ValueError("No rows loaded.")
    if column not in rows[0]:
        raise KeyError(
            "Column %r not found. Available columns: %s"
            % (column, ", ".join(sorted(rows[0].keys())))
        )
    return column


def collect_series(
    rows: Sequence[Dict[str, str]],
    metric: str,
    token_group: str,
    models: Sequence[str],
) -> Dict[str, List[Tuple[int, float]]]:
    column = metric_column(metric, rows)
    series: Dict[str, List[Tuple[int, float]]] = {model: [] for model in models}
    for row in rows:
        if row.get("part") != "encoder":
            continue
        if row.get("token_group") != token_group:
            continue
        model = row.get("model", "")
        if model not in series:
            continue
        series[model].append((int(row["block"]), float(row[column])))
    return {model: sorted(points) for model, points in series.items() if points}


def palette_slot(model: str, index: int) -> int:
    lowered = model.casefold()
    if "split" in lowered or "merged" in lowered:
        return 0
    if "joint" in lowered or "multimodal" in lowered:
        return 1
    return index


def color_for_model(model: str, index: int, palette: str) -> str:
    colors = PALETTES[palette]
    return colors[palette_slot(model, index) % len(colors)]


def marker_for_model(model: str, index: int) -> str:
    lowered = model.casefold()
    for key, marker in MODEL_MARKERS.items():
        if key in lowered:
            return marker
    return FALLBACK_MARKERS[index % len(FALLBACK_MARKERS)]


def linestyle_for_model(model: str) -> str:
    lowered = model.casefold()
    for key, linestyle in MODEL_LINESTYLES.items():
        if key in lowered:
            return linestyle
    return "-"


def style_axis(ax, xlabel: str, metric: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(METRIC_YLABELS.get(metric, metric))
    ax.grid(True, alpha=0.24, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_one(
    plt,
    rows: Sequence[Dict[str, str]],
    models: Sequence[str],
    metric: str,
    token_group: str,
    xlabel: str,
    path: str,
    dpi: int,
    palette: str,
) -> None:
    series = collect_series(rows, metric, token_group, models)
    if not series:
        print("[WARN] no encoder %s rows for metric %s" % (token_group, metric))
        return

    fig, ax = plt.subplots(figsize=(5.2, 3.45))
    for idx, model in enumerate(models):
        points = series.get(model, [])
        if not points:
            continue
        ax.plot(
            [x for x, _ in points],
            [y for _, y in points],
            color=color_for_model(model, idx, palette),
            linestyle=linestyle_for_model(model),
            marker=marker_for_model(model, idx),
            linewidth=1.8,
            markersize=4.2,
            markeredgewidth=0.8,
            label=pretty_model(model),
        )
    title = "Encoder %s %s" % (
        TOKEN_GROUP_TITLES.get(token_group, token_group),
        METRIC_TITLES.get(metric, metric),
    )
    ax.set_title(title, pad=8)
    style_axis(ax, xlabel, metric)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("[OK] plot:", path)


def plot_combined(
    plt,
    rows: Sequence[Dict[str, str]],
    models: Sequence[str],
    metrics: Sequence[str],
    token_groups: Sequence[str],
    xlabel: str,
    path: str,
    dpi: int,
    palette: str,
) -> None:
    if not metrics or not token_groups:
        return
    line_metric = metrics[0]
    ncols = len(token_groups)
    fig, axes = plt.subplots(2, ncols, figsize=(5.25 * ncols, 6.7), squeeze=False)
    legend_handles = []
    legend_labels = []

    # The bottom-row bars share one y-axis range across token groups. This makes
    # cross-panel magnitude differences visually honest.
    bar_values: List[float] = []
    for token_group in token_groups:
        series = collect_series(rows, line_metric, token_group, models)
        for points in series.values():
            bar_values.extend([value for _, value in points])
    bar_ymax = max(bar_values) * 1.12 if bar_values else None

    for col_idx, token_group in enumerate(token_groups):
        series = collect_series(rows, line_metric, token_group, models)

        ax = axes[0][col_idx]
        for idx, model in enumerate(models):
            points = series.get(model, [])
            if not points:
                continue
            handle = ax.plot(
                [x for x, _ in points],
                [y for _, y in points],
                color=color_for_model(model, idx, palette),
                linestyle=linestyle_for_model(model),
                marker=marker_for_model(model, idx),
                linewidth=1.75,
                markersize=3.9,
                markeredgewidth=0.8,
                label=pretty_model(model),
            )[0]
            label = pretty_model(model)
            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)
        ax.set_title(
            "%s / %s"
            % (
                TOKEN_GROUP_TITLES.get(token_group, token_group),
                METRIC_TITLES.get(line_metric, line_metric),
            ),
            pad=7,
        )
        style_axis(ax, xlabel, line_metric)

        ax_bar = axes[1][col_idx]
        present_models = [model for model in models if series.get(model)]
        width = 0.34 if len(present_models) <= 2 else 0.72 / max(len(present_models), 1)
        for model_idx, model in enumerate(present_models):
            points = series[model]
            offset = (model_idx - (len(present_models) - 1) / 2.0) * width
            ax_bar.bar(
                [x + offset for x, _ in points],
                [y for _, y in points],
                width=width,
                color=color_for_model(model, models.index(model), palette),
                alpha=0.82,
                edgecolor=color_for_model(model, models.index(model), palette),
                linewidth=0.45,
                label=pretty_model(model),
            )
        ax_bar.set_title(
            "%s / %s (Bars)"
            % (
                TOKEN_GROUP_TITLES.get(token_group, token_group),
                METRIC_TITLES.get(line_metric, line_metric),
            ),
            pad=7,
        )
        style_axis(ax_bar, xlabel, line_metric)
        if bar_ymax is not None:
            ax_bar.set_ylim(0, bar_ymax)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(legend_handles),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("[OK] plot:", path)


def main() -> None:
    args = parse_args()
    csv_path = os.path.abspath(os.path.expanduser(args.csv))
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(csv_path), "paper_encoder_similarity")
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    ensure_dir(out_dir)

    rows = read_rows(csv_path)
    metrics = split_csv(args.metrics)
    token_groups = split_csv(args.token_groups)
    models = ordered_models(rows, split_csv(args.model_order))

    plt = setup_matplotlib()
    if plt is None:
        raise SystemExit("matplotlib is required for plotting.")

    for metric in metrics:
        metric_column(metric, rows)
        for token_group in token_groups:
            out_path = os.path.join(
                out_dir,
                "%s_encoder_%s_%s.png" % (args.fig_prefix, token_group, metric),
            )
            plot_one(
                plt,
                rows,
                models,
                metric,
                token_group,
                args.xlabel,
                out_path,
                args.dpi,
                args.palette,
            )

    if not args.no_combined:
        plot_combined(
            plt,
            rows,
            models,
            metrics,
            token_groups,
            args.xlabel,
            os.path.join(out_dir, "%s_encoder_2x2.png" % args.fig_prefix),
            args.dpi,
            args.palette,
        )

    print("[OK] wrote encoder-only figures to:", out_dir)


if __name__ == "__main__":
    main()
