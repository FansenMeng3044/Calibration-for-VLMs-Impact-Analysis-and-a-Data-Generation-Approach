#!/usr/bin/env python3
"""Visualize dense-vs-split-vs-joint last-layer output activation drift.

This script consumes activation NPZ files produced by
``record_multimodal_per_sample_ffn_activations.py``.  It assumes the same
forward samples were recorded for three model states:

  - dense: full-precision reference model
  - split: unimodal split-pruned checkpoint after merge
  - joint: multimodal jointly-pruned checkpoint

For each requested component/token group, it writes:

  - paired per-sample cosine/L2 CSV against dense
  - dense-vs-pruned cosine boxplot and violin plot
  - delta histogram: cos(dense, split) - cos(dense, joint)
  - PCA/t-SNE plots across dense/split/joint activations
  - abs(split - dense) and abs(joint - dense) heatmaps
  - a component-summary CSV/bar plot
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_COMPONENTS = (
    "vit:all",
    "qformer:all",
    "t5_encoder:visual",
    "t5_encoder:text",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare dense/split/joint final-layer output activations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dense", required=True, help="Dense activation out_dir or NPZ.")
    parser.add_argument("--split", required=True, help="Split-merge activation out_dir or NPZ.")
    parser.add_argument("--joint", required=True, help="Joint-pruned activation out_dir or NPZ.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--components",
        nargs="+",
        default=list(DEFAULT_COMPONENTS),
        help=(
            "Component specs as component:token_group. Examples: "
            "vit:all qformer:all t5_encoder:visual t5_encoder:text t5_decoder:decoder"
        ),
    )
    parser.add_argument(
        "--array_key",
        default=None,
        help="Exact NPZ key. Use only when --components contains one item.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--cosine_mode",
        choices=["centered", "raw"],
        default="centered",
        help="Cosine variant used for main plots. CSV always contains both.",
    )
    parser.add_argument(
        "--no_standardize_embedding",
        action="store_true",
        help="Do not joint-zscore features before PCA/t-SNE.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a requested component/token group is missing. Default is warn and skip.",
    )
    parser.add_argument("--heatmap_percentile", type=float, default=99.0)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "run"


def parse_component_spec(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise ValueError("Component spec must be component:token_group, got %r" % spec)
    component, token_group = spec.split(":", 1)
    component = component.strip()
    token_group = token_group.strip()
    if component not in {"vit", "qformer", "t5_encoder", "t5_decoder"}:
        raise ValueError("Unsupported component in %r" % spec)
    if not token_group:
        raise ValueError("Empty token group in %r" % spec)
    return component, token_group


def find_npz(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path.strip().strip('"')))
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
    if os.path.isdir(path):
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
    candidates: List[Tuple[int, str]] = []
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
            "No final-layer output mean_abs_per_neuron key for component=%s token_group=%s."
            % (component, token_group)
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][1]


def load_matrix(path: str, component: str, token_group: str, exact_key: Optional[str], max_samples: Optional[int]) -> Tuple[np.ndarray, str, str]:
    npz_path = find_npz(path)
    data = np.load(npz_path)
    key = select_array_key(list(data.keys()), component, token_group, exact_key)
    matrix = np.asarray(data[key], dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Expected 2D array for %s:%s, got %s" % (npz_path, key, matrix.shape))
    if max_samples is not None:
        matrix = matrix[:max_samples]
    return matrix, key, npz_path


def setup_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("[WARN] matplotlib unavailable; CSV files will still be written: %s" % exc)
        return None
    return plt


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
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
        writer.writerows(rows)


def normalize_rows(x: np.ndarray, centered: bool) -> np.ndarray:
    y = x.astype(np.float64, copy=True)
    if centered:
        y -= y.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(y, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return y / denom


def paired_cosine(a: np.ndarray, b: np.ndarray, centered: bool) -> np.ndarray:
    an = normalize_rows(a, centered=centered)
    bn = normalize_rows(b, centered=centered)
    return np.sum(an * bn, axis=1)


def paired_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a.astype(np.float64) - b.astype(np.float64), axis=1)


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


def plot_similarity_box_violin(plt, split_scores: np.ndarray, joint_scores: np.ndarray, title: str, prefix: str) -> None:
    if plt is None:
        return
    labels = ["dense vs split", "dense vs joint"]
    data = [split_scores, joint_scores]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel("Paired cosine similarity")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(prefix + "_paired_cosine_boxplot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_alpha(0.65)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel("Paired cosine similarity")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(prefix + "_paired_cosine_violin.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_delta_histogram(plt, delta: np.ndarray, title: str, path: str) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(delta, bins=40, alpha=0.82, color="#4C78A8", edgecolor="white")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(float(np.mean(delta)), color="#F58518", linestyle="-", linewidth=1.5, label="mean")
    ax.set_title(title)
    ax.set_xlabel("cos(dense, split) - cos(dense, joint)")
    ax.set_ylabel("Samples")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_component_bar(plt, rows: Sequence[Dict[str, object]], path: str) -> None:
    if plt is None or not rows:
        return
    labels = [str(row["component_token_group"]) for row in rows]
    split = np.asarray([float(row["main_cos_dense_split_mean"]) for row in rows])
    joint = np.asarray([float(row["main_cos_dense_joint_mean"]) for row in rows])
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8, 1.25 * len(labels)), 5.2))
    ax.bar(x - width / 2, split, width, label="dense vs split", color="#4C78A8")
    ax.bar(x + width / 2, joint, width, label="dense vs joint", color="#F58518")
    ax.set_ylabel("Mean paired cosine similarity")
    ax.set_title("Dense-reference final-layer similarity by component")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_embedding(plt, coords: np.ndarray, labels: Sequence[str], title: str, path: str) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    labels_arr = np.asarray(labels)
    colors = {"dense": "#4C78A8", "split": "#54A24B", "joint": "#E45756"}
    for label in ("dense", "split", "joint"):
        mask = labels_arr == label
        if np.any(mask):
            ax.scatter(coords[mask, 0], coords[mask, 1], s=32, alpha=0.78, label=label, color=colors[label])
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_with_pairs(plt, coords: np.ndarray, n: int, title: str, path: str, max_lines: int = 128) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    dense = coords[:n]
    split = coords[n : 2 * n]
    joint = coords[2 * n : 3 * n]
    stride = max(1, int(np.ceil(n / max_lines)))
    for idx in range(0, n, stride):
        ax.plot([dense[idx, 0], split[idx, 0]], [dense[idx, 1], split[idx, 1]], color="#54A24B", alpha=0.25, linewidth=0.8)
        ax.plot([dense[idx, 0], joint[idx, 0]], [dense[idx, 1], joint[idx, 1]], color="#E45756", alpha=0.25, linewidth=0.8)
    ax.scatter(dense[:, 0], dense[:, 1], s=30, alpha=0.8, label="dense", color="#4C78A8")
    ax.scatter(split[:, 0], split[:, 1], s=24, alpha=0.75, label="split", color="#54A24B")
    ax.scatter(joint[:, 0], joint[:, 1], s=24, alpha=0.75, label="joint", color="#E45756")
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_absdiff_heatmaps(plt, split_diff: np.ndarray, joint_diff: np.ndarray, title: str, path: str, percentile: float) -> None:
    if plt is None:
        return
    vmax = float(np.percentile(np.concatenate([split_diff.ravel(), joint_diff.ravel()]), percentile))
    if vmax <= 0:
        vmax = None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    ims = []
    ims.append(axes[0].imshow(split_diff, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=vmax))
    ims.append(axes[1].imshow(joint_diff, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=vmax))
    axes[0].set_title("|split - dense|")
    axes[1].set_title("|joint - dense|")
    axes[0].set_ylabel("Sample")
    for ax in axes:
        ax.set_xlabel("Neuron dimension")
    fig.suptitle(title)
    fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.82, label="Absolute activation difference")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def rows_for_component(
    dense: np.ndarray,
    split: np.ndarray,
    joint: np.ndarray,
    main_mode: str,
    component_group: str,
    key: str,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    n = min(dense.shape[0], split.shape[0], joint.shape[0])
    dense = dense[:n]
    split = split[:n]
    joint = joint[:n]

    raw_split = paired_cosine(dense, split, centered=False)
    raw_joint = paired_cosine(dense, joint, centered=False)
    centered_split = paired_cosine(dense, split, centered=True)
    centered_joint = paired_cosine(dense, joint, centered=True)
    l2_split = paired_l2(dense, split)
    l2_joint = paired_l2(dense, joint)

    if main_mode == "centered":
        main_split = centered_split
        main_joint = centered_joint
    else:
        main_split = raw_split
        main_joint = raw_joint
    delta = main_split - main_joint

    per_sample_rows: List[Dict[str, object]] = []
    for i in range(n):
        per_sample_rows.append(
            {
                "component_token_group": component_group,
                "array_key": key,
                "sample_index": i,
                "cos_raw_dense_split": float(raw_split[i]),
                "cos_raw_dense_joint": float(raw_joint[i]),
                "cos_centered_dense_split": float(centered_split[i]),
                "cos_centered_dense_joint": float(centered_joint[i]),
                "main_cos_dense_split": float(main_split[i]),
                "main_cos_dense_joint": float(main_joint[i]),
                "delta_split_minus_joint": float(delta[i]),
                "l2_dense_split": float(l2_split[i]),
                "l2_dense_joint": float(l2_joint[i]),
                "split_closer_by_main_cos": int(delta[i] > 0),
            }
        )

    summary = {
        "component_token_group": component_group,
        "array_key": key,
        "samples": int(n),
        "feature_dim": int(dense.shape[1]),
        "main_cosine_mode": main_mode,
        "main_cos_dense_split_mean": float(main_split.mean()),
        "main_cos_dense_split_median": float(np.median(main_split)),
        "main_cos_dense_joint_mean": float(main_joint.mean()),
        "main_cos_dense_joint_median": float(np.median(main_joint)),
        "delta_split_minus_joint_mean": float(delta.mean()),
        "delta_split_minus_joint_median": float(np.median(delta)),
        "fraction_split_closer": float(np.mean(delta > 0)),
        "l2_dense_split_mean": float(l2_split.mean()),
        "l2_dense_joint_mean": float(l2_joint.mean()),
    }
    return per_sample_rows, summary


def main() -> None:
    args = parse_args()
    if args.array_key and len(args.components) != 1:
        raise ValueError("--array_key can only be used with a single --components entry.")

    ensure_dir(args.out_dir)
    plt = setup_matplotlib()

    all_component_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    skipped: List[str] = []

    for spec in args.components:
        component, token_group = parse_component_spec(spec)
        component_group = "%s_%s" % (component, token_group)
        out_dir = os.path.join(args.out_dir, safe_name(component_group))
        ensure_dir(out_dir)
        exact = args.array_key if len(args.components) == 1 else None

        try:
            dense, dense_key, dense_npz = load_matrix(args.dense, component, token_group, exact, args.max_samples)
            split, split_key, split_npz = load_matrix(args.split, component, token_group, exact, args.max_samples)
            joint, joint_key, joint_npz = load_matrix(args.joint, component, token_group, exact, args.max_samples)
        except Exception as exc:
            msg = "[WARN] skip %s: %s" % (component_group, exc)
            if args.strict:
                raise
            print(msg)
            skipped.append(msg)
            continue

        if not (dense.shape[1] == split.shape[1] == joint.shape[1]):
            msg = "[WARN] skip %s: feature dims differ dense=%s split=%s joint=%s" % (
                component_group,
                dense.shape,
                split.shape,
                joint.shape,
            )
            if args.strict:
                raise ValueError(msg)
            print(msg)
            skipped.append(msg)
            continue

        n = min(dense.shape[0], split.shape[0], joint.shape[0])
        dense = dense[:n]
        split = split[:n]
        joint = joint[:n]

        key = dense_key
        if split_key != dense_key or joint_key != dense_key:
            print("[WARN] key differs for %s: dense=%s split=%s joint=%s" % (component_group, dense_key, split_key, joint_key))

        per_sample_rows, summary = rows_for_component(dense, split, joint, args.cosine_mode, component_group, key)
        summary.update(
            {
                "dense_npz": dense_npz,
                "split_npz": split_npz,
                "joint_npz": joint_npz,
            }
        )
        all_component_rows.extend(per_sample_rows)
        summary_rows.append(summary)
        write_csv(os.path.join(out_dir, "paired_dense_reference_similarity.csv"), per_sample_rows)

        main_split = np.asarray([row["main_cos_dense_split"] for row in per_sample_rows], dtype=np.float64)
        main_joint = np.asarray([row["main_cos_dense_joint"] for row in per_sample_rows], dtype=np.float64)
        delta = main_split - main_joint

        title = "%s last-layer output similarity to dense\n%s" % (component_group, key)
        prefix = os.path.join(out_dir, safe_name(component_group))
        plot_similarity_box_violin(plt, main_split, main_joint, title, prefix)
        plot_delta_histogram(
            plt,
            delta,
            "%s dense-reference cosine delta\npositive means split is closer to dense" % component_group,
            os.path.join(out_dir, safe_name(component_group) + "_delta_histogram.png"),
        )
        plot_absdiff_heatmaps(
            plt,
            np.abs(split - dense),
            np.abs(joint - dense),
            "%s absolute final-layer activation difference from dense" % component_group,
            os.path.join(out_dir, safe_name(component_group) + "_absdiff_heatmaps.png"),
            args.heatmap_percentile,
        )

        x = np.concatenate([dense, split, joint], axis=0).astype(np.float32)
        labels = ["dense"] * n + ["split"] * n + ["joint"] * n
        x_embed = x if args.no_standardize_embedding else standardize_joint(x)
        if plt is not None:
            pca = pca_2d(x_embed)
            plot_embedding(
                plt,
                pca,
                labels,
                "%s PCA of final-layer outputs" % component_group,
                os.path.join(out_dir, safe_name(component_group) + "_pca.png"),
            )
            plot_embedding_with_pairs(
                plt,
                pca,
                n,
                "%s PCA with same-sample shifts" % component_group,
                os.path.join(out_dir, safe_name(component_group) + "_pca_paired_shifts.png"),
            )
            tsne = maybe_tsne_2d(x_embed, args.tsne_perplexity, args.random_state)
            if tsne is not None:
                plot_embedding(
                    plt,
                    tsne,
                    labels,
                    "%s t-SNE of final-layer outputs" % component_group,
                    os.path.join(out_dir, safe_name(component_group) + "_tsne.png"),
                )
                plot_embedding_with_pairs(
                    plt,
                    tsne,
                    n,
                    "%s t-SNE with same-sample shifts" % component_group,
                    os.path.join(out_dir, safe_name(component_group) + "_tsne_paired_shifts.png"),
                )

    write_csv(os.path.join(args.out_dir, "all_components_paired_similarity.csv"), all_component_rows)
    write_csv(os.path.join(args.out_dir, "component_summary.csv"), summary_rows)
    plot_component_bar(plt, summary_rows, os.path.join(args.out_dir, "component_summary_bar.png"))
    if skipped:
        with open(os.path.join(args.out_dir, "skipped_components.txt"), "w", encoding="utf-8") as handle:
            for msg in skipped:
                handle.write(msg + "\n")

    print("[OK] wrote dense/split/joint visualization outputs to:", os.path.abspath(args.out_dir))
    for row in summary_rows:
        print(
            "%s: split_mean=%.6f joint_mean=%.6f delta=%.6f split_closer=%.1f%%"
            % (
                row["component_token_group"],
                row["main_cos_dense_split_mean"],
                row["main_cos_dense_joint_mean"],
                row["delta_split_minus_joint_mean"],
                100.0 * row["fraction_split_closer"],
            )
        )


if __name__ == "__main__":
    main()
