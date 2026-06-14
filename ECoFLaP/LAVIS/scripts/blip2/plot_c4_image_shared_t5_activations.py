#!/usr/bin/env python3
"""Jointly visualize C4 text and image-query activations in T5 input space.

The image NPZ must be produced with:
  --input_mode vit_image_only --record_shared_t5_space

The text NPZ must be produced with:
  --input_mode t5_c4_text
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Tuple

import numpy as np


IMAGE_VECTOR_KEY = "t5_visual_query_first_block_input_tokens"
IMAGE_ROW_KEY = "t5_visual_query_token_row_index"
IMAGE_POSITION_KEY = "t5_visual_query_token_position"
TEXT_VECTOR_KEY = "t5_text_first_block_input_tokens"
TEXT_ROW_KEY = "t5_text_token_row_index"
TEXT_POSITION_KEY = "t5_text_token_position"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create joint t-SNE, histogram, and neuron-coverage plots for image "
            "visual-query tokens and C4 text tokens in the shared T5 input space."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image_npz", required=True)
    parser.add_argument("--text_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_points_per_modality", type=int, default=5000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--histogram_bins", type=int, default=80)
    parser.add_argument(
        "--coverage_thresholds",
        default="0.5,1,1.5,2,2.5,3,3.5,4,5,6",
        help="Comma-separated absolute z-score thresholds for the coverage curve.",
    )
    parser.add_argument(
        "--coverage_threshold",
        type=float,
        default=3.0,
        help="Absolute z-score threshold used for coverage bars and overlap.",
    )
    parser.add_argument(
        "--coverage_max_tokens_per_modality",
        type=int,
        default=5000,
        help="Equal token count cap used for each modality in coverage analysis.",
    )
    return parser.parse_args()


def setup_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except (ImportError, RuntimeError) as exc:
        raise SystemExit(
            "A compatible matplotlib/NumPy installation is required. "
            "Install the project requirements."
        ) from exc


def load_tokens(
    path: str,
    vector_key: str,
    row_key: str,
    position_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = [
            key
            for key in (vector_key, row_key, position_key)
            if key not in data
        ]
        if missing:
            raise KeyError(
                "%s is missing keys %s. Re-run the activation recorder with "
                "the required input mode."
                % (path, ", ".join(missing))
            )
        vectors = np.asarray(data[vector_key], dtype=np.float32)
        rows = np.asarray(data[row_key], dtype=np.int64)
        positions = np.asarray(data[position_key], dtype=np.int64)

    if vectors.ndim != 2:
        raise ValueError("%s must be 2D, got %s" % (vector_key, vectors.shape))
    if len(vectors) != len(rows) or len(vectors) != len(positions):
        raise ValueError(
            "Token vectors and metadata have inconsistent lengths in %s" % path
        )
    return vectors, rows, positions


def select_indices(
    num_points: int,
    max_points: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    if num_points <= max_points:
        return np.arange(num_points)
    return np.sort(rng.choice(num_points, size=max_points, replace=False))


def finite_vectors(vectors: np.ndarray) -> np.ndarray:
    return np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)


def parse_thresholds(value: str) -> np.ndarray:
    try:
        thresholds = np.asarray(
            [float(item.strip()) for item in value.split(",") if item.strip()],
            dtype=np.float32,
        )
    except ValueError as exc:
        raise ValueError("--coverage_thresholds must contain numbers.") from exc
    if thresholds.size == 0 or np.any(thresholds < 0):
        raise ValueError("--coverage_thresholds must contain non-negative values.")
    return np.unique(np.sort(thresholds))


def compute_neuron_coverage(
    image_vectors: np.ndarray,
    text_vectors: np.ndarray,
    thresholds: np.ndarray,
) -> Dict[str, Any]:
    """Compute per-dimension coverage after joint per-neuron z-score scaling."""
    if image_vectors.ndim != 2 or text_vectors.ndim != 2:
        raise ValueError("Coverage vectors must be two-dimensional.")
    if image_vectors.shape[1] != text_vectors.shape[1]:
        raise ValueError("Coverage vectors must have the same hidden size.")
    if len(image_vectors) != len(text_vectors):
        raise ValueError(
            "Coverage requires equal token counts per modality, got %d and %d."
            % (len(image_vectors), len(text_vectors))
        )

    image = finite_vectors(image_vectors).astype(np.float64, copy=False)
    text = finite_vectors(text_vectors).astype(np.float64, copy=False)
    combined = np.concatenate([image, text], axis=0)
    joint_mean = np.mean(combined, axis=0)
    joint_std = np.std(combined, axis=0)
    joint_std = np.where(joint_std > 1e-12, joint_std, 1.0)
    image_max_abs_z = np.max(np.abs((image - joint_mean) / joint_std), axis=0)
    text_max_abs_z = np.max(np.abs((text - joint_mean) / joint_std), axis=0)

    curve = []
    for threshold in thresholds:
        image_covered = image_max_abs_z > float(threshold)
        text_covered = text_max_abs_z > float(threshold)
        union = image_covered | text_covered
        intersection = image_covered & text_covered
        union_count = int(np.sum(union))
        intersection_count = int(np.sum(intersection))
        curve.append(
            {
                "threshold": float(threshold),
                "image_count": int(np.sum(image_covered)),
                "text_count": int(np.sum(text_covered)),
                "union_count": union_count,
                "intersection_count": intersection_count,
                "image_rate": float(np.mean(image_covered)),
                "text_rate": float(np.mean(text_covered)),
                "union_rate": float(np.mean(union)),
                "intersection_rate": float(np.mean(intersection)),
                "jaccard": (
                    float(intersection_count / union_count)
                    if union_count
                    else 1.0
                ),
            }
        )
    return {
        "hidden_size": int(image.shape[1]),
        "tokens_per_modality": int(len(image)),
        "image_max_abs_z": image_max_abs_z,
        "text_max_abs_z": text_max_abs_z,
        "curve": curve,
    }


def write_tsne_csv(
    path: str,
    image_xy: np.ndarray,
    text_xy: np.ndarray,
    image_rows: np.ndarray,
    image_positions: np.ndarray,
    text_rows: np.ndarray,
    text_positions: np.ndarray,
) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "row_index",
                "token_position",
                "tsne_dimension_1",
                "tsne_dimension_2",
            ],
        )
        writer.writeheader()
        groups = (
            ("Image Visual Query", image_xy, image_rows, image_positions),
            ("C4 Text", text_xy, text_rows, text_positions),
        )
        for source, coordinates, rows, positions in groups:
            for row, position, xy in zip(rows, positions, coordinates):
                writer.writerow(
                    {
                        "source": source,
                        "row_index": int(row),
                        "token_position": int(position),
                        "tsne_dimension_1": float(xy[0]),
                        "tsne_dimension_2": float(xy[1]),
                    }
                )


def make_histogram(
    out_dir: str,
    image_vectors: np.ndarray,
    text_vectors: np.ndarray,
    num_bins: int,
) -> Tuple[str, str]:
    image_values = finite_vectors(image_vectors).reshape(-1)
    text_values = finite_vectors(text_vectors).reshape(-1)
    combined = np.concatenate([image_values, text_values])
    lower, upper = np.percentile(combined, [0.5, 99.5])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        lower = float(np.min(combined))
        upper = float(np.max(combined))
    if lower >= upper:
        lower -= 0.5
        upper += 0.5

    edges = np.linspace(lower, upper, num_bins + 1)
    image_density, _ = np.histogram(image_values, bins=edges, density=True)
    text_density, _ = np.histogram(text_values, bins=edges, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.hist(
        image_values,
        bins=edges,
        density=True,
        color="#8FC5DA",
        edgecolor="#567B89",
        linewidth=0.35,
        alpha=0.62,
        label="Image Visual Query",
    )
    ax.hist(
        text_values,
        bins=edges,
        density=True,
        color="#93CD81",
        edgecolor="#527A48",
        linewidth=0.35,
        alpha=0.58,
        label="C4 Text",
    )
    ax.axvline(
        float(np.mean(image_values)),
        color="#416D7E",
        linestyle="--",
        linewidth=1.3,
        label="Image Mean",
    )
    ax.axvline(
        float(np.mean(text_values)),
        color="#477A3F",
        linestyle="--",
        linewidth=1.3,
        label="C4 Mean",
    )
    ax.set_xlim(lower, upper)
    ax.set_xlabel("Activation Value")
    ax.set_ylabel("Density")
    ax.set_title(
        "Image Visual-Query and C4 Text Activation Distributions\n"
        "(Shared T5 First-Block Input Space)"
    )
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "image_c4_shared_t5_histogram.png")
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(out_dir, "image_c4_shared_t5_histogram.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bin_left",
                "bin_right",
                "bin_center",
                "image_density",
                "c4_text_density",
            ],
        )
        writer.writeheader()
        for index in range(num_bins):
            writer.writerow(
                {
                    "bin_left": float(edges[index]),
                    "bin_right": float(edges[index + 1]),
                    "bin_center": float(centers[index]),
                    "image_density": float(image_density[index]),
                    "c4_text_density": float(text_density[index]),
                }
            )
    return plot_path, csv_path


def make_neuron_coverage_plots(
    out_dir: str,
    coverage: Dict[str, Any],
    fixed_threshold: float,
) -> Tuple[Tuple[str, ...], Dict[str, Any]]:
    """Write threshold curve, fixed-threshold bars, overlap, and CSV files."""
    curve = coverage["curve"]
    hidden_size = int(coverage["hidden_size"])
    image_max = np.asarray(coverage["image_max_abs_z"])
    text_max = np.asarray(coverage["text_max_abs_z"])
    image_covered = image_max > fixed_threshold
    text_covered = text_max > fixed_threshold
    both = image_covered & text_covered
    image_only = image_covered & ~text_covered
    text_only = text_covered & ~image_covered
    neither = ~image_covered & ~text_covered
    union = image_covered | text_covered
    union_count = int(np.sum(union))
    both_count = int(np.sum(both))

    curve_csv_path = os.path.join(
        out_dir, "image_c4_neuron_coverage_curve.csv"
    )
    curve_fields = [
        "threshold",
        "image_count",
        "text_count",
        "union_count",
        "intersection_count",
        "image_rate",
        "text_rate",
        "union_rate",
        "intersection_rate",
        "jaccard",
    ]
    with open(curve_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_fields)
        writer.writeheader()
        writer.writerows(curve)

    neuron_csv_path = os.path.join(
        out_dir, "image_c4_neuron_coverage_by_dimension.csv"
    )
    with open(neuron_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "neuron_index",
                "image_max_abs_z",
                "c4_text_max_abs_z",
                "image_covered",
                "c4_text_covered",
                "coverage_category",
            ],
        )
        writer.writeheader()
        for index in range(hidden_size):
            if both[index]:
                category = "both"
            elif image_only[index]:
                category = "image_only"
            elif text_only[index]:
                category = "c4_text_only"
            else:
                category = "neither"
            writer.writerow(
                {
                    "neuron_index": index,
                    "image_max_abs_z": float(image_max[index]),
                    "c4_text_max_abs_z": float(text_max[index]),
                    "image_covered": bool(image_covered[index]),
                    "c4_text_covered": bool(text_covered[index]),
                    "coverage_category": category,
                }
            )

    plt = setup_matplotlib()
    thresholds = np.asarray([item["threshold"] for item in curve])
    image_rates = 100.0 * np.asarray([item["image_rate"] for item in curve])
    text_rates = 100.0 * np.asarray([item["text_rate"] for item in curve])
    union_rates = 100.0 * np.asarray([item["union_rate"] for item in curve])

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.plot(
        thresholds,
        image_rates,
        marker="o",
        markersize=4,
        linewidth=1.8,
        color="#4C91AC",
        label="Image Visual Query",
    )
    ax.plot(
        thresholds,
        text_rates,
        marker="s",
        markersize=4,
        linewidth=1.8,
        color="#5B9B52",
        label="C4 Text",
    )
    ax.plot(
        thresholds,
        union_rates,
        marker="^",
        markersize=4,
        linewidth=1.5,
        linestyle="--",
        color="#7A6F9B",
        label="Union",
    )
    ax.set_xlabel("Coverage Threshold (Absolute Joint Z-Score)")
    ax.set_ylabel("Covered Neurons (%)")
    ax.set_ylim(0, 102)
    ax.set_title(
        "Neuron Coverage Across Activation Thresholds\n"
        "(Shared T5 First-Block Input Dimensions)"
    )
    ax.legend(frameon=True)
    ax.grid(alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    curve_plot_path = os.path.join(
        out_dir, "image_c4_neuron_coverage_curve.png"
    )
    fig.savefig(curve_plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    bar_labels = ["Image", "C4 Text", "Both", "Union"]
    bar_counts = [
        int(np.sum(image_covered)),
        int(np.sum(text_covered)),
        both_count,
        union_count,
    ]
    bar_rates = 100.0 * np.asarray(bar_counts, dtype=np.float64) / hidden_size
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    bars = ax.bar(
        bar_labels,
        bar_rates,
        color=["#8FC5DA", "#93CD81", "#8B9E78", "#9B91B8"],
        edgecolor=["#567B89", "#527A48", "#596B4C", "#655D7A"],
        linewidth=0.7,
    )
    for bar, rate, count in zip(bars, bar_rates, bar_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(rate + 1.5, 100.0),
            "%.1f%%\n(%d)" % (rate, count),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, 108)
    ax.set_ylabel("Covered Neurons (%)")
    ax.set_title(
        "Neuron Coverage at |z| > %.2f\n"
        "(%d Shared T5 Input Dimensions)" % (fixed_threshold, hidden_size)
    )
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    bar_plot_path = os.path.join(
        out_dir, "image_c4_neuron_coverage_bar.png"
    )
    fig.savefig(bar_plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    overlap_labels = ["Image Only", "C4 Only", "Both", "Neither"]
    overlap_counts = [
        int(np.sum(image_only)),
        int(np.sum(text_only)),
        both_count,
        int(np.sum(neither)),
    ]
    overlap_rates = (
        100.0 * np.asarray(overlap_counts, dtype=np.float64) / hidden_size
    )
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    bars = ax.bar(
        overlap_labels,
        overlap_rates,
        color=["#8FC5DA", "#93CD81", "#8B9E78", "#C9C9C9"],
        edgecolor=["#567B89", "#527A48", "#596B4C", "#777777"],
        linewidth=0.7,
    )
    for bar, rate, count in zip(bars, overlap_rates, overlap_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(rate + 1.5, 100.0),
            "%.1f%%\n(%d)" % (rate, count),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, 108)
    ax.set_ylabel("All Neurons (%)")
    ax.set_title(
        "Neuron Coverage Overlap at |z| > %.2f\n"
        "(Image Visual Query vs. C4 Text)" % fixed_threshold
    )
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    overlap_plot_path = os.path.join(
        out_dir, "image_c4_neuron_coverage_overlap.png"
    )
    fig.savefig(overlap_plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fixed_summary = {
        "threshold": float(fixed_threshold),
        "hidden_size": hidden_size,
        "tokens_per_modality": int(coverage["tokens_per_modality"]),
        "image_covered_count": int(np.sum(image_covered)),
        "image_covered_rate": float(np.mean(image_covered)),
        "c4_text_covered_count": int(np.sum(text_covered)),
        "c4_text_covered_rate": float(np.mean(text_covered)),
        "both_count": both_count,
        "both_rate": float(np.mean(both)),
        "image_only_count": int(np.sum(image_only)),
        "image_only_rate": float(np.mean(image_only)),
        "c4_text_only_count": int(np.sum(text_only)),
        "c4_text_only_rate": float(np.mean(text_only)),
        "neither_count": int(np.sum(neither)),
        "neither_rate": float(np.mean(neither)),
        "union_count": union_count,
        "union_rate": float(np.mean(union)),
        "jaccard": float(both_count / union_count) if union_count else 1.0,
    }
    paths = (
        curve_plot_path,
        bar_plot_path,
        overlap_plot_path,
        curve_csv_path,
        neuron_csv_path,
    )
    return paths, fixed_summary


def main() -> None:
    args = parse_args()
    if args.max_points_per_modality < 2:
        raise ValueError("--max_points_per_modality must be >= 2")
    if args.perplexity <= 0:
        raise ValueError("--perplexity must be > 0")
    if args.histogram_bins < 2:
        raise ValueError("--histogram_bins must be >= 2")
    if args.coverage_threshold < 0:
        raise ValueError("--coverage_threshold must be >= 0")
    if args.coverage_max_tokens_per_modality < 2:
        raise ValueError("--coverage_max_tokens_per_modality must be >= 2")
    coverage_thresholds = np.unique(
        np.append(parse_thresholds(args.coverage_thresholds), args.coverage_threshold)
    )

    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
    except (ImportError, RuntimeError) as exc:
        raise SystemExit(
            "A compatible scikit-learn/SciPy/NumPy installation is required. "
            "Install the project requirements, including numpy<2."
        ) from exc

    image_vectors, image_rows, image_positions = load_tokens(
        os.path.abspath(args.image_npz),
        IMAGE_VECTOR_KEY,
        IMAGE_ROW_KEY,
        IMAGE_POSITION_KEY,
    )
    text_vectors, text_rows, text_positions = load_tokens(
        os.path.abspath(args.text_npz),
        TEXT_VECTOR_KEY,
        TEXT_ROW_KEY,
        TEXT_POSITION_KEY,
    )
    if image_vectors.shape[1] != text_vectors.shape[1]:
        raise ValueError(
            "Image and text hidden sizes differ: %d vs %d. They must come from "
            "the same BLIP2-T5 model configuration."
            % (image_vectors.shape[1], text_vectors.shape[1])
        )

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(args.random_state)
    image_selected = select_indices(
        len(image_vectors), args.max_points_per_modality, rng
    )
    text_selected = select_indices(
        len(text_vectors), args.max_points_per_modality, rng
    )
    selected_image_vectors = finite_vectors(image_vectors[image_selected])
    selected_text_vectors = finite_vectors(text_vectors[text_selected])
    features = np.concatenate(
        [selected_image_vectors, selected_text_vectors], axis=0
    )
    features = StandardScaler().fit_transform(features)
    perplexity = min(float(args.perplexity), float(len(features) - 1))
    perplexity = max(perplexity, 1.0)
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate=200.0,
        random_state=args.random_state,
    ).fit_transform(features)

    num_image = len(image_selected)
    image_xy = coordinates[:num_image]
    text_xy = coordinates[num_image:]
    selected_image_rows = image_rows[image_selected]
    selected_image_positions = image_positions[image_selected]
    selected_text_rows = text_rows[text_selected]
    selected_text_positions = text_positions[text_selected]

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    ax.scatter(
        image_xy[:, 0],
        image_xy[:, 1],
        s=10,
        c="#B9DCEA",
        edgecolors="#5F7F8D",
        linewidths=0.2,
        alpha=0.62,
        label="Image Visual Query",
    )
    ax.scatter(
        text_xy[:, 0],
        text_xy[:, 1],
        s=10,
        c="#A9D99B",
        edgecolors="#527A48",
        linewidths=0.2,
        alpha=0.62,
        label="C4 Text",
    )
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title(
        "Image and C4 Token Activation Representations\n"
        "(Shared T5 First-Block Input Space; Image=%d, C4=%d)"
        % (len(image_xy), len(text_xy))
    )
    ax.legend(frameon=True)
    ax.grid(alpha=0.18, linewidth=0.6)
    fig.tight_layout()
    tsne_plot_path = os.path.join(out_dir, "image_c4_shared_t5_tsne.png")
    fig.savefig(tsne_plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    tsne_csv_path = os.path.join(out_dir, "image_c4_shared_t5_tsne.csv")
    write_tsne_csv(
        tsne_csv_path,
        image_xy,
        text_xy,
        selected_image_rows,
        selected_image_positions,
        selected_text_rows,
        selected_text_positions,
    )
    histogram_plot_path, histogram_csv_path = make_histogram(
        out_dir,
        selected_image_vectors,
        selected_text_vectors,
        args.histogram_bins,
    )
    coverage_token_count = min(
        len(image_vectors),
        len(text_vectors),
        args.coverage_max_tokens_per_modality,
    )
    coverage_rng = np.random.RandomState(args.random_state + 1)
    coverage_image_indices = select_indices(
        len(image_vectors), coverage_token_count, coverage_rng
    )
    coverage_text_indices = select_indices(
        len(text_vectors), coverage_token_count, coverage_rng
    )
    coverage = compute_neuron_coverage(
        image_vectors[coverage_image_indices],
        text_vectors[coverage_text_indices],
        coverage_thresholds,
    )
    coverage_paths, coverage_summary = make_neuron_coverage_plots(
        out_dir,
        coverage,
        args.coverage_threshold,
    )
    print(
        "Neuron coverage at |z| > %.2f: Image=%.2f%% C4=%.2f%% "
        "Both=%.2f%% Jaccard=%.4f"
        % (
            args.coverage_threshold,
            100.0 * coverage_summary["image_covered_rate"],
            100.0 * coverage_summary["c4_text_covered_rate"],
            100.0 * coverage_summary["both_rate"],
            coverage_summary["jaccard"],
        )
    )

    summary: Dict[str, Any] = {
        "image_npz": os.path.abspath(args.image_npz),
        "text_npz": os.path.abspath(args.text_npz),
        "hidden_size": int(image_vectors.shape[1]),
        "image_tokens_total": int(len(image_vectors)),
        "image_tokens_plotted": int(len(image_selected)),
        "c4_text_tokens_total": int(len(text_vectors)),
        "c4_text_tokens_plotted": int(len(text_selected)),
        "perplexity": float(perplexity),
        "random_state": int(args.random_state),
        "neuron_coverage": {
            "definition": (
                "A T5 input dimension is covered when its maximum absolute "
                "jointly standardized activation exceeds the threshold."
            ),
            "normalization": (
                "Per-dimension mean and standard deviation from balanced image "
                "and C4 token samples."
            ),
            "thresholds": [float(value) for value in coverage_thresholds],
            "fixed_threshold_summary": coverage_summary,
        },
    }
    summary_path = os.path.join(out_dir, "image_c4_shared_t5_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(
        "Joint t-SNE points: Image=%d/%d C4=%d/%d"
        % (
            len(image_selected),
            len(image_vectors),
            len(text_selected),
            len(text_vectors),
        )
    )
    for path in (
        tsne_plot_path,
        tsne_csv_path,
        histogram_plot_path,
        histogram_csv_path,
        *coverage_paths,
        summary_path,
    ):
        print("[OK] wrote:", path)


if __name__ == "__main__":
    main()
