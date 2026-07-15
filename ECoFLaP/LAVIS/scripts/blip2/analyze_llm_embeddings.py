#!/usr/bin/env python3
"""Two views of the LLM-input embedding for the Part 2 mechanism.

Consumes llm_input_embeddings.npz files from extract_llm_input_embeddings.py.

--mode semantic  (Visualization A)
    Dense-model embeddings for several DATASETS -> a dataset x dataset semantic
    similarity matrix (calibration vs eval). Answers: is a good calibration one
    whose DATA is semantically close to the eval sets? Also tests whether a
    *semantic* task-matching (diagonal) effect exists, even though the Wanda-
    statistic one did not.

--mode fidelity  (Visualization B)
    Dense + each pruned model on the SAME eval set -> per-calibration similarity
    of the LLM-input embedding to dense. Because text embeddings are unchanged by
    pruning, this isolates how faithfully each pruned model's VISUAL prefix (the
    image representation handed to the LLM) reproduces dense. Answers: does that
    fidelity predict accuracy?

Both modes correlate their quantity with the column-centered accuracy main
effect when --accuracy_csv is given, and emit CSV + PNG.

Usage (A):
  python scripts/blip2/analyze_llm_embeddings.py --mode semantic \
      --emb MMBench=/p/llm_emb/dense_on_MMBench ... --emb cc3m=/p/llm_emb/dense_on_cc3m \
      --evals MMBench,MMMU,OKVQA,mathvista --accuracy_csv /p/acc.csv \
      --out_dir /p/out/semantic

Usage (B):
  python scripts/blip2/analyze_llm_embeddings.py --mode fidelity \
      --dense /p/llm_emb/dense_on_eval \
      --emb MMBench=/p/llm_emb/MMBench_on_eval ... --emb cc3m=/p/llm_emb/cc3m_on_eval \
      --accuracy_csv /p/acc.csv --out_dir /p/out/fidelity
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from split_joint_analysis_common import ensure_dir, parse_labeled_paths, setup_matplotlib, write_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Semantic (A) and fidelity (B) views of LLM-input embeddings.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["semantic", "fidelity"], required=True)
    p.add_argument("--emb", action="append", required=True, metavar="LABEL=DIR_OR_NPZ")
    p.add_argument("--dense", default=None, help="[fidelity] dense embeddings on the eval set.")
    p.add_argument("--calibs", default=None, help="[semantic] comma-separated calibration labels. Defaults to labels not in --evals.")
    p.add_argument("--evals", default=None, help="[semantic] comma-separated eval dataset labels.")
    p.add_argument("--part", choices=["visual", "text", "both"], default="both",
                   help="Which LLM-input embedding to use. Use both/text for semantic similarity; visual is often most sensitive for pruning fidelity.")
    p.add_argument("--accuracy_csv", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def load_accuracy_effects(path: str) -> Tuple[Dict[str, float], List[str], np.ndarray, List[str]]:
    """Return label -> column-centered row effect from an accuracy matrix CSV.

    Expected format:
      calib,MMBench,MMMU,OKVQA,mathvista
      MMBench,52.53,25.29,33.33,34.95

    Column-centering removes benchmark scale, so the row effect measures whether
    a calibration is globally strong or weak across eval datasets.
    """
    with open(path, "r", encoding="utf-8") as handle:
        reader = list(csv.reader(handle))
    if not reader:
        raise ValueError("Empty accuracy CSV: %s" % path)
    evals = [h.strip() for h in reader[0][1:]]
    labels: List[str] = []
    matrix: List[List[float]] = []
    for row in reader[1:]:
        if not row or not row[0].strip():
            continue
        labels.append(row[0].strip())
        matrix.append([float(x) for x in row[1:1 + len(evals)]])
    mat = np.asarray(matrix, dtype=np.float64)
    col_centered = mat - mat.mean(axis=0, keepdims=True)
    row_effect = col_centered.mean(axis=1)
    return {labels[i]: float(row_effect[i]) for i in range(len(labels))}, evals, mat, labels


def find_npz(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        return path
    cand = os.path.join(path, "llm_input_embeddings.npz")
    if os.path.isfile(cand):
        return cand
    raise FileNotFoundError("No llm_input_embeddings.npz under %s" % path)


def load_emb(path: str, part: str) -> np.ndarray:
    data = np.load(find_npz(path))
    if part == "visual":
        return data["visual_prefix"].astype(np.float64)
    if part == "text":
        return data["text_embed"].astype(np.float64)
    return np.concatenate([data["visual_prefix"], data["text_embed"]], axis=1).astype(np.float64)


def unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float("nan") if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x)); r[order] = np.arange(1, len(x) + 1)
    _, inv, c = np.unique(x, return_inverse=True, return_counts=True)
    s = np.zeros(len(c)); np.add.at(s, inv, r)
    return (s / c)[inv]


def spearman(x, y):
    return pearson(rankdata(np.asarray(x, float)), rankdata(np.asarray(y, float)))


def split_labels(value: Optional[str], labels: Sequence[str], what: str) -> List[str]:
    if value is None:
        return []
    wanted = [x.strip() for x in value.split(",") if x.strip()]
    missing = [x for x in wanted if x not in labels]
    if missing:
        raise SystemExit("%s labels not found in --emb: %s" % (what, ", ".join(missing)))
    return wanted


def heatmap(plt, labels, mat, title, path, evals=None):
    if plt is None:
        return
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.15 * n + 2.5, 1.15 * n + 2))
    im = ax.imshow(mat, cmap="viridis", vmin=float(np.nanmin(mat)), vmax=float(np.nanmax(mat)))
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=8,
                        color="white" if v < (np.nanmin(mat) + np.nanmax(mat)) / 2 else "black")
    ax.set_ylabel("calibration"); ax.set_xlabel("eval reference")
    ax.set_title(title); fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def rect_heatmap(plt, row_labels, col_labels, mat, title, path):
    if plt is None:
        return
    nr = len(row_labels)
    nc = len(col_labels)
    fig, ax = plt.subplots(figsize=(1.1 * nc + 3.2, 0.62 * nr + 2.6))
    im = ax.imshow(mat, cmap="viridis", vmin=float(np.nanmin(mat)), vmax=float(np.nanmax(mat)))
    ax.set_xticks(range(nc)); ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(nr)); ax.set_yticklabels(row_labels)
    mid = (np.nanmin(mat) + np.nanmax(mat)) / 2.0
    for i in range(nr):
        for j in range(nc):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, "%.3f" % v, ha="center", va="center", fontsize=8,
                        color="white" if v < mid else "black")
    ax.set_ylabel("calibration dataset")
    ax.set_xlabel("evaluation/reference dataset")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_semantic(args, plt):
    embs = parse_labeled_paths(args.emb)
    labels = list(embs)
    arrays = {lab: load_emb(p, args.part) for lab, p in embs.items()}
    unit_arrays = {lab: unit(arrays[lab]) for lab in labels}
    centroids = {lab: unit(arrays[lab].mean(axis=0, keepdims=True))[0] for lab in labels}
    n = len(labels)
    sim = np.eye(n)
    for i in range(n):
        for j in range(n):
            sim[i, j] = float(np.dot(centroids[labels[i]], centroids[labels[j]]))

    rows = [{"calibration": labels[i], "reference": labels[j], "semantic_cosine": sim[i, j]}
            for i in range(n) for j in range(n)]
    write_csv(os.path.join(args.out_dir, "semantic_similarity_%s.csv" % args.part), rows)

    evals = split_labels(args.evals, labels, "--evals") if args.evals else labels
    if args.calibs:
        calibs = split_labels(args.calibs, labels, "--calibs")
    else:
        eval_set = set(evals)
        calibs = [lab for lab in labels if lab not in eval_set] or labels

    calib_eval = np.zeros((len(calibs), len(evals)), dtype=np.float64)
    calib_eval_rows = []
    for i, calib in enumerate(calibs):
        for j, ev in enumerate(evals):
            value = float(np.dot(centroids[calib], centroids[ev]))
            pair = unit_arrays[calib] @ unit_arrays[ev].T
            mean_pairwise = float(pair.mean())
            mean_max_calib_to_eval = float(pair.max(axis=1).mean())
            mean_max_eval_to_calib = float(pair.max(axis=0).mean())
            calib_eval[i, j] = value
            calib_eval_rows.append({
                "calibration": calib,
                "eval": ev,
                "centroid_cosine": value,
                "mean_pairwise_cosine": mean_pairwise,
                "mean_max_calib_to_eval": mean_max_calib_to_eval,
                "mean_max_eval_to_calib": mean_max_eval_to_calib,
            })
    write_csv(os.path.join(args.out_dir, "calib_eval_semantic_similarity_%s.csv" % args.part), calib_eval_rows)

    # mean semantic similarity of each calibration to the chosen eval datasets.
    mean_to_evals = {}
    for lab in calibs:
        vals = [float(np.dot(centroids[lab], centroids[e])) for e in evals]
        mean_to_evals[lab] = float(np.mean(vals)) if vals else float("nan")
    write_csv(os.path.join(args.out_dir, "mean_semantic_similarity_to_evals.csv"),
              [{"calibration": lab, "mean_sim_to_evals": mean_to_evals[lab]} for lab in calibs])

    if plt is not None and not args.no_plots:
        heatmap(plt, labels, sim, "Calibration vs eval semantic similarity (%s)" % args.part,
                os.path.join(args.out_dir, "semantic_similarity_%s.png" % args.part), evals)
        rect_heatmap(plt, calibs, evals, calib_eval,
                     "Calibration-to-eval semantic similarity (%s)" % args.part,
                     os.path.join(args.out_dir, "calib_eval_semantic_similarity_%s.png" % args.part))

    corr = None
    if args.accuracy_csv:
        effects, _, _, _ = load_accuracy_effects(args.accuracy_csv)
        shared = [l for l in calibs if l in effects]
        if len(shared) >= 3:
            xs = [mean_to_evals[l] for l in shared]; ys = [effects[l] for l in shared]
            corr = {"pearson": pearson(xs, ys), "spearman": spearman(xs, ys)}
            # diagonal (semantic task-match) vs off-diagonal accuracy handled by the matrix itself
            if plt is not None and not args.no_plots:
                fig, ax = plt.subplots(figsize=(7, 5.5))
                ax.scatter(xs, ys, s=140, color="#4C78A8", edgecolors="black", zorder=3)
                for l in shared:
                    ax.annotate(l, (mean_to_evals[l], effects[l]), fontsize=11, xytext=(7, 5),
                                textcoords="offset points")
                ax.set_xlabel("mean semantic similarity to eval sets")
                ax.set_ylabel("global accuracy effect"); ax.grid(True, alpha=0.28)
                ax.set_title("Is a calibration good because it is semantically close to eval?")
                fig.tight_layout()
                fig.savefig(os.path.join(args.out_dir, "semantic_vs_accuracy.png"), dpi=220, bbox_inches="tight")
                plt.close(fig)

    print("\n=== semantic similarity (%s) ===" % args.part)
    print("  mean similarity to eval sets:")
    for lab in sorted(mean_to_evals, key=mean_to_evals.get, reverse=True):
        print("    %-10s %.4f" % (lab, mean_to_evals[lab]))
    if corr:
        print("  r(semantic-closeness-to-eval, accuracy): pearson=%+.3f spearman=%+.3f"
              % (corr["pearson"], corr["spearman"]))
        print("  -> strong positive = semantic task-matching IS the driver; ~0 = it is not.")
    return {"labels": labels, "sim": sim, "mean_to_evals": mean_to_evals, "corr": corr}


def run_fidelity(args, plt):
    if not args.dense:
        raise SystemExit("[fidelity] --dense is required (dense embeddings on the eval set).")
    dense = load_emb(args.dense, args.part)          # [N, H]
    embs = parse_labeled_paths(args.emb)
    dU = unit(dense)

    rows = []
    for lab, path in embs.items():
        p = load_emb(path, args.part)
        if p.shape != dense.shape:
            raise SystemExit("[fidelity] %s shape %s != dense %s (same eval rows/order required)"
                             % (lab, p.shape, dense.shape))
        cos = np.sum(unit(p) * dU, axis=1)                          # per-sample cosine to dense
        rel = np.linalg.norm(p - dense, axis=1) / np.clip(np.linalg.norm(dense, axis=1), 1e-8, None)
        rows.append({"calibration": lab, "n": int(p.shape[0]),
                     "cos_to_dense_mean": float(cos.mean()), "cos_to_dense_median": float(np.median(cos)),
                     "rel_l2_to_dense_mean": float(rel.mean())})
    write_csv(os.path.join(args.out_dir, "llm_input_fidelity_%s.csv" % args.part), rows)

    corr = None
    effects = None
    if args.accuracy_csv:
        effects, _, _, _ = load_accuracy_effects(args.accuracy_csv)
        shared = [r for r in rows if r["calibration"] in effects]
        if len(shared) >= 3:
            xs = [r["cos_to_dense_mean"] for r in shared]; ys = [effects[r["calibration"]] for r in shared]
            corr = {"metric": "cos_to_dense", "pearson": pearson(xs, ys), "spearman": spearman(xs, ys)}

    if plt is not None and not args.no_plots:
        labels = [r["calibration"] for r in rows]
        vals = [r["cos_to_dense_mean"] for r in rows]
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels) + 2), 4.6))
        ax.bar(range(len(labels)), vals, color="#4C78A8")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("LLM-input embedding cosine to dense")
        ax.set_ylim(min(vals) - 0.01, 1.0)
        ax.set_title("Per-calibration visual-prefix fidelity to dense (%s)" % args.part)
        ax.grid(True, axis="y", alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "llm_input_fidelity_%s.png" % args.part), dpi=220, bbox_inches="tight")
        plt.close(fig)
        if effects is not None:
            shared = [r for r in rows if r["calibration"] in effects]
            fig, ax = plt.subplots(figsize=(7, 5.5))
            xs = [r["cos_to_dense_mean"] for r in shared]; ys = [effects[r["calibration"]] for r in shared]
            ax.scatter(xs, ys, s=140, color="#E45756", edgecolors="black", zorder=3)
            for r in shared:
                ax.annotate(r["calibration"], (r["cos_to_dense_mean"], effects[r["calibration"]]),
                            fontsize=11, xytext=(7, 5), textcoords="offset points")
            ax.set_xlabel("LLM-input fidelity to dense (cosine)")
            ax.set_ylabel("global accuracy effect"); ax.grid(True, alpha=0.28)
            ax.set_title("Does a faithful LLM-input embedding predict accuracy?")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, "fidelity_vs_accuracy.png"), dpi=220, bbox_inches="tight")
            plt.close(fig)

    print("\n=== LLM-input fidelity to dense (%s) ===" % args.part)
    for r in sorted(rows, key=lambda r: r["cos_to_dense_mean"], reverse=True):
        print("  %-10s cos=%.5f  rel_l2=%.5f" % (r["calibration"], r["cos_to_dense_mean"], r["rel_l2_to_dense_mean"]))
    if corr:
        print("  r(fidelity, accuracy): pearson=%+.3f spearman=%+.3f" % (corr["pearson"], corr["spearman"]))
        print("  -> strong positive = pruning fidelity of the image representation drives accuracy.")
    return {"rows": rows, "corr": corr}


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    plt = None if args.no_plots else setup_matplotlib()
    result = run_semantic(args, plt) if args.mode == "semantic" else run_fidelity(args, plt)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as h:
        def clean(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o
        json.dump({"mode": args.mode, "part": args.part,
                   "corr": result.get("corr"),
                   "mean_to_evals": result.get("mean_to_evals")}, h, indent=2, default=clean)
    print("\n[OK] wrote:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
