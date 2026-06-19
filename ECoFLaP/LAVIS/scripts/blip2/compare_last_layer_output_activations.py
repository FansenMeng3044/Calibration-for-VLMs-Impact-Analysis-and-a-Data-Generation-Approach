#!/usr/bin/env python3
"""Compare saved last-layer output activations within and across datasets.

This script consumes the NPZ files produced by
record_multimodal_per_sample_ffn_activations.py or
record_unimodal_per_sample_activations.py.  It compares the per-sample
``mean_abs_per_neuron`` vectors for the final block-output layer of one
component/token group, e.g.:

  - T5 encoder text positions: component=t5_encoder, token_group=text
  - T5 encoder visual-query positions: component=t5_encoder, token_group=visual
  - ViT image tokens: component=vit, token_group=all

It makes within-dataset similarity heatmaps and cross-dataset PCA/t-SNE plots.
Only runs with the same feature dimension are combined.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare final-layer output activation vectors across datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="One or more LABEL=OUT_DIR_OR_NPZ entries.",
    )
    parser.add_argument(
        "--component",
        required=True,
        choices=["vit", "qformer", "t5_encoder", "t5_decoder"],
        help="Model component to compare.",
    )
    parser.add_argument(
        "--token_group",
        default=None,
        help="Token group: all, text, visual, or decoder. Defaults by component.",
    )
    parser.add_argument(
        "--array_key",
        default=None,
        help="Exact NPZ key to use. If omitted, the final matching layer is selected automatically.",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_samples_per_run", type=int, default=None)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--no_standardize",
        action="store_true",
        help="Do not joint-zscore features before PCA/t-SNE and centroid distances.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def default_group(component: str) -> str:
    if component == "t5_decoder":
        return "decoder"
    if component == "t5_encoder":
        return "text"
    return "all"


def parse_run(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise ValueError("--runs entries must be LABEL=PATH, got: %s" % spec)
    label, path = spec.split("=", 1)
    label = label.strip()
    path = os.path.abspath(os.path.expanduser(path.strip().strip('"')))
    if not label:
        raise ValueError("Empty run label in %s" % spec)
    return label, path


def find_npz(path: str) -> str:
    if os.path.isfile(path):
        if not path.lower().endswith(".npz"):
            raise ValueError("Expected an .npz file, got: %s" % path)
        return path
    candidates = [
        "whole_model_per_sample_activation_arrays.npz",
        "unimodal_per_sample_activation_arrays.npz",
    ]
    for name in candidates:
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate):
            return candidate
    npzs = [os.path.join(path, name) for name in os.listdir(path) if name.lower().endswith(".npz")]
    if len(npzs) == 1:
        return npzs[0]
    raise FileNotFoundError("Could not uniquely find an activation .npz under: %s" % path)


def component_tokens(component: str) -> List[str]:
    if component == "t5_encoder":
        return ["t5", "encoder"]
    if component == "t5_decoder":
        return ["t5", "decoder"]
    return [component]


def extract_layer_from_key(key: str) -> Optional[int]:
    for pattern in (r"__block__(\d+)__", r"__layer__(\d+)__", r"__ffn__(\d+)__"):
        match = re.search(pattern, key)
        if match:
            return int(match.group(1))
    return None


def select_array_key(keys: Sequence[str], component: str, token_group: str, exact: Optional[str]) -> str:
    if exact:
        if exact not in keys:
            raise KeyError("Exact array key not found: %s" % exact)
        return exact

    required = component_tokens(component)
    candidates = []
    for key in keys:
        parts = key.split("__")
        if not key.endswith("__mean_abs_per_neuron"):
            continue
        if "output" not in parts:
            continue
        if token_group not in parts:
            continue
        if not all(token in parts for token in required):
            continue
        layer = extract_layer_from_key(key)
        if layer is None:
            continue
        candidates.append((layer, key))
    if not candidates:
        raise KeyError(
            "No final-layer output mean_abs_per_neuron key found for component=%s token_group=%s."
            % (component, token_group)
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][1]


def load_run(label: str, path: str, component: str, token_group: str, array_key: Optional[str], max_samples: Optional[int]) -> Dict[str, object]:
    npz_path = find_npz(path)
    data = np.load(npz_path)
    key = select_array_key(list(data.keys()), component, token_group, array_key)
    matrix = np.asarray(data[key], dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Expected 2D [samples, features] array for %s:%s, got %s." % (label, key, matrix.shape))
    if max_samples is not None:
        matrix = matrix[:max_samples]
    return {
        "label": label,
        "path": path,
        "npz_path": npz_path,
        "array_key": key,
        "layer": extract_layer_from_key(key),
        "matrix": matrix,
    }


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "run"


def cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(centered, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return (centered / denom) @ (centered / denom).T


def setup_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("[WARN] matplotlib unavailable; CSV summaries will still be written: %s" % exc)
        return None
    return plt


def plot_heatmap(plt, matrix: np.ndarray, title: str, path: str, xlabel: str = "Sample", ylabel: str = "Sample") -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def standardize_joint(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (x - mean) / std


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def maybe_tsne_2d(x: np.ndarray, perplexity: float, random_state: int) -> Optional[np.ndarray]:
    if x.shape[0] < 4:
        return None
    try:
        from sklearn.manifold import TSNE
    except Exception:
        return None
    usable_perplexity = min(float(perplexity), max(2.0, (x.shape[0] - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=usable_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(x)


def plot_embedding(plt, coords: np.ndarray, labels: Sequence[str], title: str, path: str) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    labels_arr = np.asarray(labels)
    for label in sorted(set(labels)):
        mask = labels_arr == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=36, alpha=0.75, label=label)
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
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


def pairwise_centroid_rows(runs: Sequence[Dict[str, object]], standardized_mats: Dict[str, np.ndarray]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    labels = [str(run["label"]) for run in runs]
    centroids = {label: standardized_mats[label].mean(axis=0) for label in labels}
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            a = centroids[left]
            b = centroids[right]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            cosine = 0.0 if denom == 0 else float(np.dot(a, b) / denom)
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "euclidean_centroid_distance": float(np.linalg.norm(a - b)),
                    "cosine_centroid_similarity": cosine,
                    "left_samples": int(standardized_mats[left].shape[0]),
                    "right_samples": int(standardized_mats[right].shape[0]),
                    "feature_dim": int(standardized_mats[left].shape[1]),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    token_group = args.token_group or default_group(args.component)
    ensure_dir(args.out_dir)

    runs = [
        load_run(label, path, args.component, token_group, args.array_key, args.max_samples_per_run)
        for label, path in (parse_run(spec) for spec in args.runs)
    ]

    plt = setup_matplotlib()
    summary_rows: List[Dict[str, object]] = []
    for run in runs:
        label = str(run["label"])
        matrix = np.asarray(run["matrix"], dtype=np.float32)
        sim = cosine_similarity(matrix)
        plot_heatmap(
            plt,
            sim,
            "%s within-dataset cosine similarity\n%s" % (label, run["array_key"]),
            os.path.join(args.out_dir, "%s_within_cosine_similarity.png" % safe_name(label)),
        )
        plot_heatmap(
            plt,
            matrix,
            "%s final-layer mean abs activation\n%s" % (label, run["array_key"]),
            os.path.join(args.out_dir, "%s_final_layer_activation_heatmap.png" % safe_name(label)),
            xlabel="Neuron dimension",
            ylabel="Sample",
        )
        per_sample_mean = matrix.mean(axis=1)
        summary_rows.append(
            {
                "label": label,
                "npz_path": run["npz_path"],
                "array_key": run["array_key"],
                "layer": run["layer"],
                "samples": int(matrix.shape[0]),
                "feature_dim": int(matrix.shape[1]),
                "per_sample_mean_mean": float(per_sample_mean.mean()),
                "per_sample_mean_std": float(per_sample_mean.std()),
                "per_sample_mean_p10": float(np.percentile(per_sample_mean, 10)),
                "per_sample_mean_p50": float(np.percentile(per_sample_mean, 50)),
                "per_sample_mean_p90": float(np.percentile(per_sample_mean, 90)),
                "within_cosine_mean": float(sim[np.triu_indices_from(sim, k=1)].mean()) if sim.shape[0] > 1 else 1.0,
                "within_cosine_std": float(sim[np.triu_indices_from(sim, k=1)].std()) if sim.shape[0] > 1 else 0.0,
            }
        )

    dims = sorted({int(np.asarray(run["matrix"]).shape[1]) for run in runs})
    for dim in dims:
        same_dim_runs = [run for run in runs if int(np.asarray(run["matrix"]).shape[1]) == dim]
        if len(same_dim_runs) < 2:
            continue
        x = np.concatenate([np.asarray(run["matrix"], dtype=np.float32) for run in same_dim_runs], axis=0)
        labels: List[str] = []
        standardized_mats: Dict[str, np.ndarray] = {}
        start = 0
        x_for_embedding = standardize_joint(x) if not args.no_standardize else x
        for run in same_dim_runs:
            label = str(run["label"])
            count = int(np.asarray(run["matrix"]).shape[0])
            labels.extend([label] * count)
            standardized_mats[label] = x_for_embedding[start : start + count]
            start += count

        if plt is not None:
            coords = pca_2d(x_for_embedding)
            plot_embedding(
                plt,
                coords,
                labels,
                "%s %s final-layer output PCA (dim=%d)" % (args.component, token_group, dim),
                os.path.join(args.out_dir, "%s_%s_dim%d_pca.png" % (args.component, token_group, dim)),
            )
            tsne = maybe_tsne_2d(x_for_embedding, args.tsne_perplexity, args.random_state)
            if tsne is not None:
                plot_embedding(
                    plt,
                    tsne,
                    labels,
                    "%s %s final-layer output t-SNE (dim=%d)" % (args.component, token_group, dim),
                    os.path.join(args.out_dir, "%s_%s_dim%d_tsne.png" % (args.component, token_group, dim)),
                )
        write_csv(
            os.path.join(args.out_dir, "%s_%s_dim%d_centroid_distances.csv" % (args.component, token_group, dim)),
            pairwise_centroid_rows(same_dim_runs, standardized_mats),
        )

    write_csv(os.path.join(args.out_dir, "last_layer_output_activation_summary.csv"), summary_rows)
    print("[OK] compared %d runs for component=%s token_group=%s" % (len(runs), args.component, token_group))
    print("[OK] wrote outputs to:", os.path.abspath(args.out_dir))
    for row in summary_rows:
        print(
            "%s: key=%s shape=(%s,%s)"
            % (row["label"], row["array_key"], row["samples"], row["feature_dim"])
        )


if __name__ == "__main__":
    main()
