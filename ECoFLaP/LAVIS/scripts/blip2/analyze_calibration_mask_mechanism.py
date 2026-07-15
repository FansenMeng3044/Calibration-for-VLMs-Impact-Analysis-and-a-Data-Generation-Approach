#!/usr/bin/env python3
"""Part 2 -- why does the calibration dataset behave as a global main effect?

Given several joint-pruned checkpoints that differ ONLY in which dataset was
used as pruning calibration (e.g. MMBench / MMMU / OKVQA / mathvista / cc3m),
this reads the masks they actually produced -- GPU-free, just the zero pattern
-- and tests the mechanism proposed in Part 1:

  Wanda's mask is ~dominated by |W|, so calibration has weak leverage over it.
  What little the calibration does control is a coarse "representativeness"
  effect, not a task-specific one: the calibration whose mask sits CENTRAL among
  the others generalizes to every eval, while an outlier calibration (out-of-
  distribution data) is slightly worse everywhere. Matching calibration to the
  eval task does nothing.

Falsifiable predictions this script checks and visualizes:

  1. Weak leverage      -> all masks are highly overlapping (mean pairwise
                           overlap near 1). If they were near the random
                           baseline instead, calibration would have strong
                           leverage and the whole story is wrong.
  2. Central vs outlier -> the mask-space map (MDS of 1-overlap) has one tight
                           cluster with outliers; centrality is not uniform.
  3. Centrality -> acc   -> a checkpoint's mask centrality predicts its global
                           accuracy effect (pass --accuracy_csv). Positive
                           correlation = mechanism confirmed. If instead the
                           diagonal (calib==eval) drove accuracy, centrality
                           would NOT predict it.

Outputs per component (ViT and T5 handled separately, because different
calibrations use different images AND different text):
  - pairwise mask-overlap matrix (heatmap + CSV)
  - mask-space MDS map (central vs outlier)
  - per-layer cross-calibration disagreement curve
  - centrality-vs-accuracy scatter with Pearson r (if --accuracy_csv given)

Usage:
  python scripts/blip2/analyze_calibration_mask_mechanism.py \
      --ckpt MMBench=/path/joint_mmbench.pth \
      --ckpt MMMU=/path/joint_mmmu.pth \
      --ckpt OKVQA=/path/joint_okvqa.pth \
      --ckpt mathvista=/path/joint_mathvista.pth \
      --ckpt cc3m=/path/joint_cc3m.pth \
      --out_dir /path/out/part2_mask_mechanism \
      --accuracy_csv /path/accuracy_matrix.csv

accuracy_matrix.csv (rows = calibration, cols = eval benchmark):
  calib,MMBench,MMMU,OKVQA,mathvista
  MMBench,52.53,25.29,33.33,34.95
  ...
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from split_joint_analysis_common import (
    ensure_dir,
    load_state_dict,
    parse_labeled_paths,
    prunable_block_group,
    setup_matplotlib,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mask-space mechanism analysis across calibration datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", action="append", required=True, metavar="LABEL=PATH",
                        help="Repeatable. One joint-pruned checkpoint per calibration dataset.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--accuracy_csv", default=None,
                        help="Optional rows=calib, cols=eval accuracy matrix for the centrality-vs-accuracy test.")
    parser.add_argument("--centrality_component", choices=["t5", "vit"], default="t5",
                        help="Which component's centrality to correlate with accuracy.")
    return parser.parse_args()


# --------------------------------------------------------------------------
# masks as bit-packed vectors (keeps memory sane: T5-XL keep-mask ~ 0.25 GB packed)
# --------------------------------------------------------------------------
def extract_packed_masks(path: str) -> Dict[str, Dict[str, object]]:
    """Return per-tensor {model, submodel, block, packed_keep, popcount, numel}."""
    import torch

    state = load_state_dict(path)
    out: Dict[str, Dict[str, object]] = {}
    for name, tensor in state.items():
        try:
            ndim = tensor.dim()
        except AttributeError:
            continue
        info = prunable_block_group(name, ndim)
        if info is None:
            continue
        model, submodel, block, _ = info
        keep = (tensor != 0).cpu().numpy().reshape(-1)
        packed = np.packbits(keep)
        out[name] = {
            "model": model,
            "submodel": submodel,
            "block": block,
            "packed": packed,
            "popcount": int(keep.sum()),
            "numel": int(keep.size),
        }
    del state
    return out


def pair_overlap(
    a: Dict[str, Dict[str, object]],
    b: Dict[str, Dict[str, object]],
    model: str,
) -> Tuple[float, float, int]:
    """(keep_overlap, iou, kept_a) over all tensors of `model` shared by a and b."""
    inter = 0
    keep_a = 0
    keep_b = 0
    for name, ra in a.items():
        if ra["model"] != model or name not in b:
            continue
        rb = b[name]
        anded = np.bitwise_and(ra["packed"], rb["packed"])
        inter += int(np.unpackbits(anded).sum())
        keep_a += int(ra["popcount"])
        keep_b += int(rb["popcount"])
    if keep_a == 0:
        return float("nan"), float("nan"), 0
    union = keep_a + keep_b - inter
    return inter / keep_a, (inter / union if union else float("nan")), keep_a


def per_layer_disagreement(
    masks: Dict[str, Dict[str, Dict[str, object]]], model: str
) -> List[Dict[str, object]]:
    """Mean pairwise (1 - keep_overlap) at each block, across all calibrations."""
    labels = list(masks)
    # gather tensors by (submodel, block)
    by_block: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    any_ckpt = masks[labels[0]]
    for name, rec in any_ckpt.items():
        if rec["model"] == model:
            by_block[(rec["submodel"], rec["block"])].append(name)

    rows: List[Dict[str, object]] = []
    for (submodel, block), names in sorted(by_block.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        disagreements = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = masks[labels[i]], masks[labels[j]]
                inter = keep = 0
                for name in names:
                    if name in a and name in b:
                        inter += int(np.unpackbits(np.bitwise_and(a[name]["packed"], b[name]["packed"])).sum())
                        keep += int(a[name]["popcount"])
                if keep:
                    disagreements.append(1.0 - inter / keep)
        if disagreements:
            rows.append({
                "model": model, "submodel": submodel, "block": block,
                "mean_disagreement": float(np.mean(disagreements)),
                "max_disagreement": float(np.max(disagreements)),
            })
    return rows


def classical_mds(distance: np.ndarray) -> np.ndarray:
    """2D classical MDS via double-centering + top eigenvectors. No sklearn."""
    n = distance.shape[0]
    d2 = distance ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1]
    vals = vals[order][:2]
    vecs = vecs[:, order][:, :2]
    vals = np.clip(vals, a_min=0.0, a_max=None)
    return vecs * np.sqrt(vals)


def load_accuracy_effects(path: str) -> Tuple[Dict[str, float], List[str], np.ndarray, List[str]]:
    """Return (label->column-centered row effect, evals, raw matrix, calib order)."""
    with open(path, "r", encoding="utf-8") as handle:
        reader = list(_csv.reader(handle))
    header = reader[0]
    evals = [h.strip() for h in header[1:]]
    labels: List[str] = []
    matrix: List[List[float]] = []
    for row in reader[1:]:
        if not row or not row[0].strip():
            continue
        labels.append(row[0].strip())
        matrix.append([float(x) for x in row[1:1 + len(evals)]])
    mat = np.asarray(matrix, dtype=np.float64)
    col_centered = mat - mat.mean(axis=0, keepdims=True)  # remove benchmark scale
    row_effect = col_centered.mean(axis=1)                 # calibration global effect
    effects = {labels[i]: float(row_effect[i]) for i in range(len(labels))}
    return effects, evals, mat, labels


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def plot_overlap_heatmap(plt, labels, mat, title, path):
    if plt is None:
        return
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2.5, 1.1 * n + 2))
    im = ax.imshow(mat, cmap="viridis", vmin=np.nanmin(mat), vmax=1.0)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, "%.3f" % v, ha="center", va="center",
                        color="white" if v < (np.nanmin(mat) + 1.0) / 2 else "black", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="kept-weight overlap")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mds(plt, labels, coords, centrality, title, path):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    c = np.asarray([centrality[l] for l in labels])
    sc = ax.scatter(coords[:, 0], coords[:, 1], s=260, c=c, cmap="viridis",
                    edgecolors="black", linewidths=1.2, zorder=3)
    for i, label in enumerate(labels):
        ax.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=11,
                    xytext=(8, 6), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2")
    ax.grid(True, alpha=0.28)
    fig.colorbar(sc, ax=ax, shrink=0.82, label="centrality (mean overlap with others)")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_disagreement(plt, rows_by_model, path):
    if plt is None or not rows_by_model:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, rows in rows_by_model.items():
        rows = sorted(rows, key=lambda r: (r["submodel"], r["block"]))
        xs = list(range(len(rows)))
        ax.plot(xs, [r["mean_disagreement"] for r in rows], marker="o", markersize=3.5,
                linewidth=1.7, label=model)
    ax.set_xlabel("prunable block (encoder->decoder / ViT blocks in order)")
    ax.set_ylabel("mean pairwise mask disagreement  (1 - overlap)")
    ax.set_title("Where in the network do calibrations disagree?")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_centrality_accuracy(plt, labels, centrality, effects, r, path):
    if plt is None:
        return
    xs = np.asarray([centrality[l] for l in labels])
    ys = np.asarray([effects[l] for l in labels])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(xs, ys, s=140, color="#4C78A8", edgecolors="black", zorder=3)
    for i, label in enumerate(labels):
        ax.annotate(label, (xs[i], ys[i]), fontsize=11, xytext=(7, 5), textcoords="offset points")
    if len(labels) >= 2 and np.std(xs) > 0:
        b, a = np.polyfit(xs, ys, 1)
        xline = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(xline, b * xline + a, color="#E45756", linewidth=1.8,
                label="fit (Pearson r=%.2f)" % r)
        ax.legend()
    ax.set_xlabel("mask centrality  (mean overlap with other calibrations)")
    ax.set_ylabel("global accuracy effect  (col-centered, points)")
    ax.set_title("Does a central mask predict better pruning?")
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ckpts = parse_labeled_paths(args.ckpt)
    if len(ckpts) < 3:
        raise SystemExit("Need at least 3 checkpoints to talk about centrality.")
    ensure_dir(args.out_dir)
    labels = list(ckpts)

    print("Reading masks (GPU-free)...")
    masks: Dict[str, Dict[str, Dict[str, object]]] = {}
    for label, path in ckpts.items():
        if not os.path.isfile(path):
            raise FileNotFoundError("Checkpoint not found for %s: %s" % (label, path))
        print("  [%-10s] %s" % (label, path))
        masks[label] = extract_packed_masks(path)
        if not masks[label]:
            raise SystemExit("No prunable tensors in %s -- unexpected layout." % label)

    plt = setup_matplotlib()
    components = sorted({rec["model"] for rec in masks[labels[0]].values()})
    summary: Dict[str, object] = {"checkpoints": ckpts, "components": {}}
    centralities: Dict[str, Dict[str, float]] = {}

    for model in components:
        n = len(labels)
        overlap = np.eye(n)
        iou = np.eye(n)
        rows: List[Dict[str, object]] = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ov, io, _ = pair_overlap(masks[labels[i]], masks[labels[j]], model)
                overlap[i, j] = ov
                iou[i, j] = io
        for i in range(n):
            for j in range(n):
                rows.append({"model": model, "calib_a": labels[i], "calib_b": labels[j],
                             "keep_overlap": overlap[i, j], "iou": iou[i, j]})
        write_csv(os.path.join(args.out_dir, "overlap_%s.csv" % model), rows)

        # centrality = mean overlap with the OTHER calibrations
        centrality = {}
        for i, label in enumerate(labels):
            others = [overlap[i, j] for j in range(n) if j != i]
            centrality[label] = float(np.mean(others))
        centralities[model] = centrality

        # mask-space map
        dist = 1.0 - iou
        np.fill_diagonal(dist, 0.0)
        coords = classical_mds(dist)

        plot_overlap_heatmap(plt, labels, overlap,
                             "%s mask overlap between calibrations" % model.upper(),
                             os.path.join(args.out_dir, "overlap_heatmap_%s.png" % model))
        plot_mds(plt, labels, coords, centrality,
                 "%s mask space (central = representative)" % model.upper(),
                 os.path.join(args.out_dir, "mask_space_%s.png" % model))

        off = overlap[~np.eye(n, dtype=bool)]
        summary["components"][model] = {
            "mean_pairwise_overlap": float(np.nanmean(off)),
            "min_pairwise_overlap": float(np.nanmin(off)),
            "centrality": centrality,
            "most_central": max(centrality, key=centrality.get),
            "most_outlier": min(centrality, key=centrality.get),
        }

    # per-layer disagreement across all components
    rows_by_model = {model: per_layer_disagreement(masks, model) for model in components}
    flat = [r for rows in rows_by_model.values() for r in rows]
    write_csv(os.path.join(args.out_dir, "per_layer_disagreement.csv"), flat)
    plot_disagreement(plt, rows_by_model, os.path.join(args.out_dir, "per_layer_disagreement.png"))

    # centrality vs accuracy
    corr = None
    if args.accuracy_csv:
        effects, evals, mat, acc_labels = load_accuracy_effects(args.accuracy_csv)
        comp = args.centrality_component if args.centrality_component in centralities else components[0]
        shared = [l for l in labels if l in effects]
        if len(shared) >= 3:
            xs = np.asarray([centralities[comp][l] for l in shared])
            ys = np.asarray([effects[l] for l in shared])
            if np.std(xs) > 0 and np.std(ys) > 0:
                corr = float(np.corrcoef(xs, ys)[0, 1])
            plot_centrality_accuracy(plt, shared, centralities[comp], effects, corr if corr is not None else float("nan"),
                                     os.path.join(args.out_dir, "centrality_vs_accuracy_%s.png" % comp))
            summary["accuracy"] = {
                "component": comp,
                "row_effects_col_centered": effects,
                "centrality_accuracy_pearson_r": corr,
            }
        else:
            print("[WARN] accuracy labels do not match >=3 checkpoints; skipping scatter.")

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # ---- verdict ----
    print("\n=== weak-leverage check (are all masks nearly the same?) ===")
    for model in components:
        s = summary["components"][model]
        print("  %-4s mean pairwise overlap=%.4f  (min=%.4f)"
              % (model, s["mean_pairwise_overlap"], s["min_pairwise_overlap"]))
    print("  If these are near 1.0, calibration has weak leverage over the mask -- so a")
    print("  fine-grained task-matching effect has no room to exist, exactly as predicted.")

    print("\n=== central vs outlier calibration ===")
    for model in components:
        s = summary["components"][model]
        cen = s["centrality"]
        ranked = sorted(cen, key=cen.get, reverse=True)
        print("  %-4s  most central -> %s ;  most outlier -> %s"
              % (model, s["most_central"], s["most_outlier"]))
        print("        " + "  ".join("%s=%.4f" % (l, cen[l]) for l in ranked))

    if corr is not None:
        print("\n=== mechanism test: centrality predicts accuracy? ===")
        print("  Pearson r(mask centrality, global accuracy effect) = %+.3f  [%s]"
              % (corr, args.centrality_component))
        if corr > 0.5:
            print("  Positive: the more central (representative) a calibration's mask, the better")
            print("  the pruned model does on EVERY benchmark. That is a global main effect, not")
            print("  task-matching -- the mechanism from Part 1 holds. The best calibration is the")
            print("  most representative one, not the one matching the eval task.")
        elif corr < -0.5:
            print("  Strongly NEGATIVE: outlier calibrations do better. That contradicts the")
            print("  representativeness story -- rethink; look at what the outlier preserves.")
        else:
            print("  Weak: centrality alone does not explain the accuracy ordering. The main effect")
            print("  is real (from the table) but its driver is something other than mask centrality")
            print("  -- consider the ViT masks, or an activation-magnitude (not mask) analysis.")

    print("\n[OK] wrote:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
