#!/usr/bin/env python3
"""Analyze teacher-forced T5 decoder hidden-state and compact logit fidelity."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from split_joint_analysis_common import ensure_dir, parse_labeled_paths, setup_matplotlib, write_csv


LINE_COLORS = ["#2aa7b8", "#19b89e", "#62d4c6", "#168ca1", "#79b86f", "#7bb7d8"]
LINE_MARKERS = ["o", "s", "^", "D", "P", "X"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare decoder hidden states and compact logits to dense.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dense", required=True, help="Dense extraction directory or t5_decoder_logits.npz.")
    p.add_argument("--emb", action="append", required=True, metavar="LABEL=DIR_OR_NPZ")
    p.add_argument("--eval_label", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def find_npz(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        return path
    cand = os.path.join(path, "t5_decoder_logits.npz")
    if os.path.isfile(cand):
        return cand
    raise FileNotFoundError("No t5_decoder_logits.npz under %s" % path)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(find_npz(path))
    required = [
        "decoder_layer",
        "gold_logprob",
        "gold_logit",
        "decoder_mask",
        "argmax_ids",
        "topk_ids",
        "sequence_nll",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError("Missing keys in %s: %s" % (path, ", ".join(missing)))
    return {key: data[key] for key in data.keys()}


def unit(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def masked_cosine_by_sample(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a = np.where(mask, a, 0.0)
    b = np.where(mask, b, 0.0)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom[denom == 0] = 1.0
    return np.sum(a * b, axis=1) / denom


def check_compatible(label: str, dense: Dict[str, np.ndarray], other: Dict[str, np.ndarray]) -> None:
    for key in ("decoder_layer", "gold_logprob", "gold_logit", "decoder_mask", "argmax_ids", "topk_ids"):
        if dense[key].shape != other[key].shape:
            raise ValueError("%s key %s shape %s != dense %s" % (label, key, other[key].shape, dense[key].shape))
    if not np.array_equal(dense["decoder_mask"], other["decoder_mask"]):
        raise ValueError("%s decoder_mask differs from dense; same eval rows/order and targets are required." % label)


def decoder_layer_rows(label: str, dense: Dict[str, np.ndarray], other: Dict[str, np.ndarray]) -> List[Dict[str, object]]:
    d = dense["decoder_layer"].astype(np.float64)
    p = other["decoder_layer"].astype(np.float64)
    du = unit(d)
    pu = unit(p)
    rows: List[Dict[str, object]] = []
    for layer in range(d.shape[1]):
        cos = np.sum(du[:, layer, :] * pu[:, layer, :], axis=1)
        rel = np.linalg.norm(p[:, layer, :] - d[:, layer, :], axis=1) / np.clip(
            np.linalg.norm(d[:, layer, :], axis=1), 1e-8, None
        )
        rows.append(
            {
                "calibration": label,
                "layer": layer,
                "n": int(d.shape[0]),
                "cos_to_dense_mean": float(cos.mean()),
                "cos_to_dense_median": float(np.median(cos)),
                "cos_to_dense_std": float(cos.std()),
                "rel_l2_to_dense_mean": float(rel.mean()),
                "rel_l2_to_dense_median": float(np.median(rel)),
                "rel_l2_to_dense_std": float(rel.std()),
            }
        )
    return rows


def topk_jaccard(dense_ids: np.ndarray, pruned_ids: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    k = min(dense_ids.shape[-1], pruned_ids.shape[-1])
    values: List[float] = []
    for sample in range(dense_ids.shape[0]):
        for pos in range(dense_ids.shape[1]):
            if not mask[sample, pos]:
                continue
            left = set(int(x) for x in dense_ids[sample, pos, :k])
            right = set(int(x) for x in pruned_ids[sample, pos, :k])
            union = len(left | right)
            values.append(float(len(left & right) / union) if union else 1.0)
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(np.median(arr))


def logit_rows(label: str, dense: Dict[str, np.ndarray], other: Dict[str, np.ndarray]) -> Dict[str, object]:
    mask = dense["decoder_mask"].astype(bool)
    dense_lp = dense["gold_logprob"].astype(np.float64)
    other_lp = other["gold_logprob"].astype(np.float64)
    dense_lg = dense["gold_logit"].astype(np.float64)
    other_lg = other["gold_logit"].astype(np.float64)
    valid_dense_lp = dense_lp[mask]
    valid_other_lp = other_lp[mask]
    valid_dense_lg = dense_lg[mask]
    valid_other_lg = other_lg[mask]
    lp_abs = np.abs(valid_other_lp - valid_dense_lp)
    lg_abs = np.abs(valid_other_lg - valid_dense_lg)
    top1_agree = (dense["argmax_ids"] == other["argmax_ids"])[mask]
    topk_mean, topk_median = topk_jaccard(dense["topk_ids"], other["topk_ids"], mask)
    seq_abs = np.abs(other["sequence_nll"].astype(np.float64) - dense["sequence_nll"].astype(np.float64))
    sample_cos = masked_cosine_by_sample(dense_lp, other_lp, mask)
    return {
        "calibration": label,
        "n": int(mask.shape[0]),
        "tokens": int(mask.sum()),
        "gold_logprob_dense_mean": float(valid_dense_lp.mean()) if valid_dense_lp.size else float("nan"),
        "gold_logprob_pruned_mean": float(valid_other_lp.mean()) if valid_other_lp.size else float("nan"),
        "gold_logprob_mae": float(lp_abs.mean()) if lp_abs.size else float("nan"),
        "gold_logprob_rmse": float(np.sqrt(np.mean(lp_abs ** 2))) if lp_abs.size else float("nan"),
        "gold_logprob_pearson": pearson(valid_dense_lp, valid_other_lp),
        "gold_logprob_cosine_mean": float(sample_cos.mean()) if sample_cos.size else float("nan"),
        "gold_logit_mae": float(lg_abs.mean()) if lg_abs.size else float("nan"),
        "sequence_nll_dense_mean": float(dense["sequence_nll"].mean()),
        "sequence_nll_pruned_mean": float(other["sequence_nll"].mean()),
        "sequence_nll_abs_diff_mean": float(seq_abs.mean()),
        "top1_agreement": float(top1_agree.mean()) if top1_agree.size else float("nan"),
        "topk_jaccard_mean": topk_mean,
        "topk_jaccard_median": topk_median,
    }


def plot_decoder_curves(plt, rows: Sequence[Dict[str, object]], metric: str, ylabel: str, title: str, path: str) -> None:
    if plt is None:
        return
    labels: List[str] = []
    for row in rows:
        label = str(row["calibration"])
        if label not in labels:
            labels.append(label)
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for idx, label in enumerate(labels):
        subset = [row for row in rows if row["calibration"] == label]
        subset.sort(key=lambda row: int(row["layer"]))
        ax.plot(
            [int(row["layer"]) for row in subset],
            [float(row[metric]) for row in subset],
            color=LINE_COLORS[idx % len(LINE_COLORS)],
            marker=LINE_MARKERS[idx % len(LINE_MARKERS)],
            linewidth=2.0,
            markersize=4.5,
            label=label,
        )
    ax.set_xlabel("T5 decoder layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_logit_bars(plt, rows: Sequence[Dict[str, object]], metric: str, ylabel: str, title: str, path: str, higher_is_better: bool) -> None:
    if plt is None:
        return
    ordered = sorted(rows, key=lambda row: float(row[metric]), reverse=higher_is_better)
    labels = [str(row["calibration"]) for row in ordered]
    vals = [float(row[metric]) for row in ordered]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar(range(len(labels)), vals, color="#2aa7b8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.28)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    dense = load_npz(args.dense)
    embs = parse_labeled_paths(args.emb)

    layer_rows: List[Dict[str, object]] = []
    logit_summary_rows: List[Dict[str, object]] = []
    for label, path in embs.items():
        other = load_npz(path)
        check_compatible(label, dense, other)
        layer_rows.extend(decoder_layer_rows(label, dense, other))
        logit_summary_rows.append(logit_rows(label, dense, other))

    layer_csv = os.path.join(args.out_dir, "t5_decoder_layer_fidelity.csv")
    logit_csv = os.path.join(args.out_dir, "t5_logit_fidelity.csv")
    write_csv(layer_csv, layer_rows)
    write_csv(logit_csv, logit_summary_rows)

    eval_title = args.eval_label or "eval"
    plt = None if args.no_plots else setup_matplotlib()
    if plt is not None:
        plot_decoder_curves(
            plt,
            layer_rows,
            "cos_to_dense_mean",
            "Mean cosine to dense",
            "%s: T5 decoder layer-wise hidden-state fidelity" % eval_title,
            os.path.join(args.out_dir, "t5_decoder_layer_fidelity.png"),
        )
        plot_decoder_curves(
            plt,
            layer_rows,
            "rel_l2_to_dense_mean",
            "Mean relative L2 to dense",
            "%s: T5 decoder layer-wise relative drift" % eval_title,
            os.path.join(args.out_dir, "t5_decoder_layer_rel_l2.png"),
        )
        plot_logit_bars(
            plt,
            logit_summary_rows,
            "gold_logprob_mae",
            "Mean absolute difference",
            "%s: gold-token logprob MAE to dense" % eval_title,
            os.path.join(args.out_dir, "t5_gold_logprob_mae.png"),
            higher_is_better=False,
        )
        plot_logit_bars(
            plt,
            logit_summary_rows,
            "top1_agreement",
            "Dense/pruned top-1 agreement",
            "%s: teacher-forced top-1 token agreement" % eval_title,
            os.path.join(args.out_dir, "t5_top1_agreement.png"),
            higher_is_better=True,
        )
        plot_logit_bars(
            plt,
            logit_summary_rows,
            "topk_jaccard_mean",
            "Mean top-k Jaccard",
            "%s: top-k token-set overlap" % eval_title,
            os.path.join(args.out_dir, "t5_topk_jaccard.png"),
            higher_is_better=True,
        )

    final_layer = int(max(row["layer"] for row in layer_rows)) if layer_rows else -1
    final_rows = [dict(row) for row in layer_rows if int(row["layer"]) == final_layer]
    write_csv(os.path.join(args.out_dir, "t5_decoder_final_layer_fidelity.csv"), final_rows)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "eval_label": args.eval_label,
                "num_samples": int(dense["decoder_layer"].shape[0]),
                "num_decoder_layers": int(dense["decoder_layer"].shape[1]),
                "hidden": int(dense["decoder_layer"].shape[2]),
                "target_len": int(dense["gold_logprob"].shape[1]),
                "top_k": int(dense["topk_ids"].shape[-1]),
                "calibrations": list(embs.keys()),
                "layer_csv": os.path.abspath(layer_csv),
                "logit_csv": os.path.abspath(logit_csv),
            },
            handle,
            indent=2,
        )
    print("[OK] wrote decoder layer fidelity:", layer_csv)
    print("[OK] wrote logit fidelity:", logit_csv)
    if final_rows:
        print("[OK] decoder final layer ranking:")
        for row in sorted(final_rows, key=lambda item: float(item["cos_to_dense_mean"]), reverse=True):
            print("  %-12s cos=%.6f rel_l2=%.6f" % (
                row["calibration"],
                float(row["cos_to_dense_mean"]),
                float(row["rel_l2_to_dense_mean"]),
            ))
    print("[OK] logit ranking by gold_logprob_mae (lower is better):")
    for row in sorted(logit_summary_rows, key=lambda item: float(item["gold_logprob_mae"])):
        print("  %-12s mae=%.6f top1=%.4f topk=%.4f" % (
            row["calibration"],
            float(row["gold_logprob_mae"]),
            float(row["top1_agreement"]),
            float(row["topk_jaccard_mean"]),
        ))


if __name__ == "__main__":
    main()
