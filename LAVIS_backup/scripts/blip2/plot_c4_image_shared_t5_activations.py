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
            "Create one joint t-SNE and histogram for image visual-query tokens "
            "and C4 text tokens in the shared T5 input space."
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


def main() -> None:
    args = parse_args()
    if args.max_points_per_modality < 2:
        raise ValueError("--max_points_per_modality must be >= 2")
    if args.perplexity <= 0:
        raise ValueError("--perplexity must be > 0")
    if args.histogram_bins < 2:
        raise ValueError("--histogram_bins must be >= 2")

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
        summary_path,
    ):
        print("[OK] wrote:", path)


if __name__ == "__main__":
    main()
