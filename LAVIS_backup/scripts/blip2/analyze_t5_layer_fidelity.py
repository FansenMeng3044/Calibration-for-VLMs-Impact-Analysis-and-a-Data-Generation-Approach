#!/usr/bin/env python3
"""Analyze layer-wise T5 encoder hidden-state fidelity to a dense model.

Inputs are directories or .npz files produced by extract_t5_layer_hidden_states.py.
For one fixed evaluation dataset, compare every calibration-pruned checkpoint to
the dense checkpoint at each T5 encoder layer and draw one curve per calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Sequence

import numpy as np

from split_joint_analysis_common import ensure_dir, parse_labeled_paths, setup_matplotlib, write_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot layer-wise T5 hidden-state fidelity curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dense", required=True, help="Dense extraction directory or t5_layer_hidden_states.npz.")
    p.add_argument("--emb", action="append", required=True, metavar="LABEL=DIR_OR_NPZ")
    p.add_argument("--part", choices=["both", "visual", "text"], default="both")
    p.add_argument("--eval_label", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def find_npz(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        return path
    cand = os.path.join(path, "t5_layer_hidden_states.npz")
    if os.path.isfile(cand):
        return cand
    raise FileNotFoundError("No t5_layer_hidden_states.npz under %s" % path)


def load_layers(path: str, part: str) -> np.ndarray:
    data = np.load(find_npz(path))
    key = "layer_%s" % part
    if key not in data:
        available = ", ".join(k for k in data.keys() if k.startswith("layer_"))
        raise KeyError("Missing %s in %s. Available: %s" % (key, path, available))
    arr = data[key].astype(np.float64)
    if arr.ndim != 3:
        raise ValueError("%s must have shape [samples, layers, hidden], got %s" % (key, arr.shape))
    return arr


def unit(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def summarize_to_dense(label: str, dense: np.ndarray, pruned: np.ndarray) -> List[Dict[str, object]]:
    if pruned.shape != dense.shape:
        raise ValueError("%s shape %s != dense %s; same eval rows/order are required." % (label, pruned.shape, dense.shape))
    dense_u = unit(dense)
    pruned_u = unit(pruned)
    rows: List[Dict[str, object]] = []
    num_layers = dense.shape[1]
    for layer in range(num_layers):
        d = dense[:, layer, :]
        p = pruned[:, layer, :]
        cos = np.sum(dense_u[:, layer, :] * pruned_u[:, layer, :], axis=1)
        rel = np.linalg.norm(p - d, axis=1) / np.clip(np.linalg.norm(d, axis=1), 1e-8, None)
        rows.append(
            {
                "calibration": label,
                "layer": layer,
                "n": int(dense.shape[0]),
                "cos_to_dense_mean": float(cos.mean()),
                "cos_to_dense_median": float(np.median(cos)),
                "cos_to_dense_std": float(cos.std()),
                "rel_l2_to_dense_mean": float(rel.mean()),
                "rel_l2_to_dense_median": float(np.median(rel)),
                "rel_l2_to_dense_std": float(rel.std()),
            }
        )
    return rows


def plot_metric(plt, rows: Sequence[Dict[str, object]], metric: str, ylabel: str, title: str, path: str) -> None:
    if plt is None:
        return
    labels = []
    for row in rows:
        label = str(row["calibration"])
        if label not in labels:
            labels.append(label)
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for label in labels:
        subset = [row for row in rows if row["calibration"] == label]
        subset.sort(key=lambda row: int(row["layer"]))
        ax.plot(
            [int(row["layer"]) for row in subset],
            [float(row[metric]) for row in subset],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            label=label,
        )
    ax.set_xlabel("T5 encoder layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    dense = load_layers(args.dense, args.part)
    embs = parse_labeled_paths(args.emb)

    rows: List[Dict[str, object]] = []
    for label, path in embs.items():
        rows.extend(summarize_to_dense(label, dense, load_layers(path, args.part)))

    csv_path = os.path.join(args.out_dir, "t5_layer_fidelity_%s.csv" % args.part)
    write_csv(csv_path, rows)

    eval_title = args.eval_label or "eval"
    plt = None if args.no_plots else setup_matplotlib()
    if plt is not None:
        plot_metric(
            plt,
            rows,
            "cos_to_dense_mean",
            "Mean cosine to dense",
            "%s: T5 layer-wise hidden-state fidelity (%s)" % (eval_title, args.part),
            os.path.join(args.out_dir, "t5_layer_fidelity_%s.png" % args.part),
        )
        plot_metric(
            plt,
            rows,
            "rel_l2_to_dense_mean",
            "Mean relative L2 to dense",
            "%s: T5 layer-wise relative drift (%s)" % (eval_title, args.part),
            os.path.join(args.out_dir, "t5_layer_rel_l2_%s.png" % args.part),
        )

    final_rows = []
    last_layer = int(max(int(row["layer"]) for row in rows)) if rows else -1
    for row in rows:
        if int(row["layer"]) == last_layer:
            final_rows.append(dict(row))
    write_csv(os.path.join(args.out_dir, "t5_final_layer_fidelity_%s.csv" % args.part), final_rows)

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "part": args.part,
                "eval_label": args.eval_label,
                "num_layers": int(dense.shape[1]),
                "num_samples": int(dense.shape[0]),
                "hidden": int(dense.shape[2]),
                "calibrations": list(embs.keys()),
                "csv": os.path.abspath(csv_path),
            },
            handle,
            indent=2,
        )
    print("[OK] wrote layer-wise fidelity rows:", csv_path)
    if final_rows:
        print("[OK] final layer ranking:")
        for row in sorted(final_rows, key=lambda item: float(item["cos_to_dense_mean"]), reverse=True):
            print("  %-12s cos=%.6f rel_l2=%.6f" % (
                row["calibration"],
                float(row["cos_to_dense_mean"]),
                float(row["rel_l2_to_dense_mean"]),
            ))


if __name__ == "__main__":
    main()
