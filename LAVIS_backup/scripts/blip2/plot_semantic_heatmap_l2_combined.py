#!/usr/bin/env python3
"""Plot semantic similarity and layer-wise L2 drift as separate paper figures.

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


HEATMAP_COLORS = ["#FFC6BC", "#FFD8D2", "#F1E5E5", "#D5E8F2", "#A5CDE2", "#5FA3C2"]
LINE_COLORS = ["#F08A7F", "#5FA3C2", "#FFC6BC", "#A5CDE2", "#B9A7EA"]
LINE_LABEL_COLORS = {
    "okvqa": "#FF6FB3",
    "cc3m": "#3F6FB5",
}
LINE_MARKERS = ["o", "s", "^", "D", "P"]
OUTPUT_EXTENSIONS = ("svg", "pdf")
# Prefer an actual bold face file. YaHei/SimHei have no matplotlib "bold" weight,
# so requesting fontweight="bold" only triggers findfont warnings and falls back to 400.
PAPER_FONT_FAMILY = ["Microsoft YaHei Bold", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
PAPER_BOLD_FONT_FILES = [
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\msyhbd.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]
CALIB_ORDER = ["MMBench", "MMMU", "OKVQA", "mathvista", "MathVista", "cc3m", "CC3M"]
EVAL_ORDER = ["MMBench", "MMMU", "OKVQA", "mathvista", "MathVista"]
DISPLAY_LABELS = {
    "mmbench": "MMBench",
    "mmmu": "MMMU",
    "okvqa": "OKVQA",
    "mathvista": "MathVista",
    "cc3m": "CC3M",
}
LOCAL_SEMANTIC_CSV = r"E:\1study\calibration\calib_eval_semantic_similarity_both.csv"
LOCAL_OKVQA_CSV = r"E:\1study\calibration\okvqat5_decoder_layer_fidelity.csv"
LOCAL_MMBENCH_CSV = r"E:\1study\calibration\mmbencht5_decoder_layer_fidelity.csv"
LOCAL_OUT_DIR = r"E:\1study\calibration\paper_figures_semantic_l2"
REMOTE_SEMANTIC_DIR = "/data/data2/mfs/llm_embedding_fidelity_fourbench/semantic_dense/semantic_both"
REMOTE_OKVQA_DIR = "/data/data2/mfs/t5_layer_fidelity_fourbench/OKVQA/t5_layer_both"
REMOTE_MMBENCH_DIR = "/data/data2/mfs/t5_layer_fidelity_fourbench/MMBench/t5_layer_both"


def first_existing(*paths: str) -> str:
    for path in paths:
        if os.path.exists(os.path.abspath(os.path.expanduser(path))):
            return path
    return paths[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot separate semantic heatmap and OKVQA/MMBench T5 layer L2 curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--semantic_dir",
        default=first_existing(LOCAL_SEMANTIC_CSV, REMOTE_SEMANTIC_DIR),
        help="Directory containing calib_eval_semantic_similarity_<part>.csv, or the CSV itself.",
    )
    parser.add_argument(
        "--okvqa_dir",
        default=first_existing(LOCAL_OKVQA_CSV, REMOTE_OKVQA_DIR),
        help="Directory containing t5_layer_fidelity_<part>.csv, or the CSV itself.",
    )
    parser.add_argument(
        "--mmbench_dir",
        default=first_existing(LOCAL_MMBENCH_CSV, REMOTE_MMBENCH_DIR),
        help="Directory containing t5_layer_fidelity_<part>.csv, or the CSV itself.",
    )
    parser.add_argument("--out_dir", default=LOCAL_OUT_DIR)
    parser.add_argument("--part", choices=["both", "visual", "text"], default="both")
    parser.add_argument("--semantic_metric", default="centroid_cosine")
    parser.add_argument("--line_metric", default="rel_l2_to_dense_mean")
    parser.add_argument("--fig_name", default="semantic_heatmap_okvqa_mmbench_l2")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also write one combined figure with the semantic heatmap and two L2 panels.",
    )
    parser.add_argument(
        "--combined_only",
        action="store_true",
        help="Write only the combined figure and skip the three separate figures.",
    )
    return parser.parse_args()


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig, out_dir: str, fig_name: str, dpi: int) -> None:
    ensure_dir(out_dir)
    for ext in OUTPUT_EXTENSIONS:
        out_path = os.path.join(out_dir, "%s.%s" % (fig_name, ext))
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print("[OK] plot:", out_path)


def resolve_paper_font_family() -> List[str]:
    from matplotlib import font_manager

    for path in PAPER_BOLD_FONT_FILES:
        if not os.path.exists(path):
            continue
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            return [name, "DejaVu Sans"]
        except (OSError, RuntimeError, ValueError):
            continue

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in PAPER_FONT_FAMILY:
        if name in available:
            return [name, "DejaVu Sans"]
    return ["DejaVu Sans"]


def configure_paper_font(plt) -> None:
    family = resolve_paper_font_family()
    PAPER_FONT_FAMILY[:] = family
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": family,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.unicode_minus": False,
        }
    )


def apply_axis_font(ax) -> None:
    ax.title.set_fontfamily(PAPER_FONT_FAMILY)
    ax.xaxis.label.set_fontfamily(PAPER_FONT_FAMILY)
    ax.yaxis.label.set_fontfamily(PAPER_FONT_FAMILY)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(PAPER_FONT_FAMILY)


def legend_kwargs(loc: str = "best") -> Dict[str, object]:
    return {
        "frameon": False,
        "loc": loc,
        "prop": {
            "family": PAPER_FONT_FAMILY,
            "size": 10,
        },
    }


def color_for_label(label: str, index: int) -> str:
    lowered = pretty_label(label).casefold()
    if lowered in LINE_LABEL_COLORS:
        return LINE_LABEL_COLORS[lowered]
    return LINE_COLORS[index % len(LINE_COLORS)]


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


def pretty_label(label: str) -> str:
    text = str(label).strip()
    lowered = text.casefold()
    for prefix in ("eval_", "calib_", "calibration_", "reference_"):
        if lowered.startswith(prefix):
            text = text[len(prefix) :]
            lowered = text.casefold()
            break
    return DISPLAY_LABELS.get(lowered, text)


def ordered_labels(labels: Iterable[str], preferred: Sequence[str]) -> List[str]:
    seen = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    preferred_rank = {pretty_label(label).casefold(): idx for idx, label in enumerate(preferred)}
    return sorted(
        seen,
        key=lambda label: (
            preferred_rank.get(pretty_label(label).casefold(), len(preferred_rank)),
            seen.index(label),
        ),
    )


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
    ax.set_xticklabels([])
    for col_idx, label in enumerate(col_labels):
        ax.text(
            col_idx + 0.32,
            -0.045,
            pretty_label(label),
            transform=ax.get_xaxis_transform(),
            rotation=32,
            ha="center",
            va="top",
            fontsize=10.5,
            fontfamily=PAPER_FONT_FAMILY,
            clip_on=False,
        )
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([pretty_label(label) for label in row_labels])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Semantic Similarity", pad=10)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        -0.14,
        1.0,
        "Calibration\nSet",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        fontfamily=PAPER_FONT_FAMILY,
    )
    ax.text(
        -0.14,
        -0.18,
        "Evaluation Set",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        fontfamily=PAPER_FONT_FAMILY,
    )

    vmin = float(np.nanmin(matrix))
    vmax = float(np.nanmax(matrix))
    span = max(vmax - vmin, 1e-12)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            color = "white" if (value - vmin) / span > 0.72 else "#183744"
            ax.text(
                j,
                i,
                "%.3f" % value,
                ha="center",
                va="center",
                fontsize=8.5,
                color=color,
                fontfamily=PAPER_FONT_FAMILY,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("Centroid Cosine")
    cbar.ax.yaxis.label.set_fontfamily(PAPER_FONT_FAMILY)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily(PAPER_FONT_FAMILY)
    apply_axis_font(ax)


def draw_l2_panel(
    ax,
    series: Dict[str, List[Tuple[int, float]]],
    title: str,
    show_xlabel: bool,
    show_legend: bool = True,
) -> Tuple[List[object], List[str]]:
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
            color=color_for_label(label, idx),
            marker=LINE_MARKERS[idx % len(LINE_MARKERS)],
            linewidth=1.55,
            markersize=3.7,
            markeredgewidth=0.8,
            label=pretty_label(label),
        )[0]
        handles.append(handle)
        plotted_labels.append(pretty_label(label))
    ax.set_title(title, pad=9)
    ax.set_ylabel("Relative L2 to Dense")
    if show_xlabel:
        ax.set_xlabel("T5 Decoder Layer", labelpad=7)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.grid(True, alpha=0.26, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if handles and show_legend:
        ax.legend(
            handles,
            plotted_labels,
            ncol=min(len(plotted_labels), 3),
            columnspacing=0.9,
            handlelength=1.5,
            **legend_kwargs("upper right"),
        )
    apply_axis_font(ax)
    return handles, plotted_labels


def draw_combined_figure(
    plt,
    args: argparse.Namespace,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    semantic_matrix: np.ndarray,
    okvqa_series: Dict[str, List[Tuple[int, float]]],
    mmbench_series: Dict[str, List[Tuple[int, float]]],
) -> None:
    fig = plt.figure(figsize=(12.2, 5.5))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.35],
        height_ratios=[1.0, 1.0],
        left=0.07,
        right=0.98,
        bottom=0.13,
        top=0.86,
        wspace=0.30,
        hspace=0.36,
    )

    ax_heat = fig.add_subplot(gs[:, 0])
    ax_okvqa = fig.add_subplot(gs[0, 1])
    ax_mmbench = fig.add_subplot(gs[1, 1])

    draw_heatmap(ax_heat, fig, row_labels, col_labels, semantic_matrix)
    draw_l2_panel(
        ax_okvqa,
        okvqa_series,
        "T5 Decoder L2 Drift on OK-VQA",
        show_xlabel=False,
        show_legend=True,
    )
    draw_l2_panel(
        ax_mmbench,
        mmbench_series,
        "T5 Decoder L2 Drift on MMBench",
        show_xlabel=True,
        show_legend=True,
    )

    save_figure(fig, args.out_dir, "%s_combined" % args.fig_name, args.dpi)
    plt.close(fig)


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
        raise RuntimeError("matplotlib is required to draw these figures.")
    configure_paper_font(plt)

    if args.combined or args.combined_only:
        draw_combined_figure(
            plt,
            args,
            row_labels,
            col_labels,
            semantic_matrix,
            okvqa_series,
            mmbench_series,
        )
        if args.combined_only:
            print("[OK] semantic CSV:", semantic_csv)
            print("[OK] OKVQA layer CSV:", okvqa_csv)
            print("[OK] MMBench layer CSV:", mmbench_csv)
            return

    fig_heat, ax_heat = plt.subplots(figsize=(5.8, 5.0))
    draw_heatmap(ax_heat, fig_heat, row_labels, col_labels, semantic_matrix)
    fig_heat.subplots_adjust(left=0.20, right=0.90, bottom=0.22, top=0.90)
    save_figure(fig_heat, args.out_dir, "%s_semantic_heatmap" % args.fig_name, args.dpi)
    plt.close(fig_heat)

    fig_okvqa, ax_okvqa = plt.subplots(figsize=(6.4, 3.8))
    draw_l2_panel(ax_okvqa, okvqa_series, "OKVQA Decoder Layer-wise L2 Drift", show_xlabel=True)
    fig_okvqa.tight_layout()
    save_figure(fig_okvqa, args.out_dir, "%s_okvqa_l2" % args.fig_name, args.dpi)
    plt.close(fig_okvqa)

    fig_mmbench, ax_mmbench = plt.subplots(figsize=(6.4, 3.8))
    draw_l2_panel(ax_mmbench, mmbench_series, "MMBench Decoder Layer-wise L2 Drift", show_xlabel=True)
    fig_mmbench.tight_layout()
    save_figure(fig_mmbench, args.out_dir, "%s_mmbench_l2" % args.fig_name, args.dpi)
    plt.close(fig_mmbench)

    print("[OK] semantic CSV:", semantic_csv)
    print("[OK] OKVQA layer CSV:", okvqa_csv)
    print("[OK] MMBench layer CSV:", mmbench_csv)


if __name__ == "__main__":
    main()
