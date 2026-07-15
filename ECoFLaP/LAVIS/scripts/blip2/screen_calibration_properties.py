#!/usr/bin/env python3
"""Part 2 -- screen every measurable calibration property against accuracy.

Centrality turned out to NOT predict accuracy (it was even mildly reversed: the
best calibration, OKVQA, is a mask-space outlier). So instead of guessing the
next single hypothesis, this screens ALL cheap per-dataset properties at once and
ranks them by how well they correlate with the accuracy main effect. Whatever
rises to the top is the lead worth chasing; a flat table means the cause is not
in the calibration statistic at all (go downstream).

Properties screened per dataset (from the extract_wanda_statistics NPZ + meta):
  - statistic centrality (t5 all / text / visual, vit)         [geometry]
  - mask centrality (t5, vit) if --ckpt given                  [geometry]
  - RMS activation scale (t5)                                  [scale]
  - channel-statistic kurtosis (t5)                            [tail]
  - top-1% channel energy fraction (t5)                        [tail]
  - visual / text / pad token fractions                        [composition]

Each is correlated with the column-centered accuracy main effect across datasets
using BOTH Pearson (linear) and Spearman (rank; robust with n=5). Output is a
ranked CSV so you can plot the winner.

Usage:
  python scripts/blip2/screen_calibration_properties.py \
      --stats MMBench=/p/stats/MMBench ... --stats cc3m=/p/stats/cc3m \
      --accuracy_csv /p/accuracy_matrix.csv \
      --out_dir /p/out/part2_property_screen \
      [--ckpt MMBench=/p/joint_mmbench.pth ...]   # optional, adds mask centrality
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from split_joint_analysis_common import ensure_dir, parse_labeled_paths, setup_matplotlib, write_csv
from analyze_calibration_statistics import (
    find_npz, load_meta, statistic_vectors, pairwise_similarity, centrality_of, structure_descriptors,
    T5_GROUPS,
)
from analyze_calibration_mask_mechanism import load_accuracy_effects, extract_packed_masks, pair_overlap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rank calibration properties by correlation with accuracy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stats", action="append", required=True, metavar="LABEL=DIR_OR_NPZ")
    p.add_argument("--accuracy_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ckpt", action="append", default=None, metavar="LABEL=PATH",
                   help="Optional joint checkpoints -> adds mask-centrality features.")
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def statistic_centrality(datasets, labels, component, group) -> Dict[str, float]:
    vecs = {lab: statistic_vectors(datasets[lab], component, group) for lab in labels}
    present = [l for l in labels if vecs[l]]
    if len(present) < 3:
        return {}
    sim, _ = pairwise_similarity({l: vecs[l] for l in present}, present)
    return centrality_of(sim, present)


def mask_centrality(ckpts: Dict[str, str], component: str) -> Dict[str, float]:
    labels = list(ckpts)
    masks = {l: extract_packed_masks(p) for l, p in ckpts.items()}
    n = len(labels)
    ov = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                ov[i, j] = pair_overlap(masks[labels[i]], masks[labels[j]], component)[0]
    return {labels[i]: float(np.mean([ov[i, j] for j in range(n) if j != i])) for i in range(n)}


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    stats = parse_labeled_paths(args.stats)
    labels = list(stats)

    datasets: Dict[str, Any] = {}
    metas: Dict[str, Any] = {}
    for lab, path in stats.items():
        npz = find_npz(path)
        datasets[lab] = np.load(npz)
        metas[lab] = load_meta(npz)

    effects, evals, mat, acc_labels = load_accuracy_effects(args.accuracy_csv)

    # ---- assemble features ----
    features: Dict[str, Dict[str, float]] = {lab: {} for lab in labels}

    for group in ("all", "text", "visual"):
        cen = statistic_centrality(datasets, labels, "t5", group)
        for lab, v in cen.items():
            features[lab]["stat_centrality_t5_%s" % group] = v
    cen_vit = statistic_centrality(datasets, labels, "vit", "all")
    for lab, v in cen_vit.items():
        features[lab]["stat_centrality_vit"] = v

    for lab in labels:
        sd = structure_descriptors(statistic_vectors(datasets[lab], "t5", "all"), metas[lab], "t5", "all")
        features[lab]["rms_activation_t5"] = sd["rms_activation"]
        features[lab]["channel_kurtosis_t5"] = sd["channel_kurtosis"]
        features[lab]["top1pct_energy_t5"] = sd["top1pct_energy_fraction"]
        counts = metas[lab].get("t5_token_counts", {})
        tot = sum(counts.get(g, 0.0) for g in T5_GROUPS)
        if tot > 0:
            features[lab]["visual_token_frac"] = counts.get("visual", 0.0) / tot
            features[lab]["text_token_frac"] = counts.get("text", 0.0) / tot
            features[lab]["pad_token_frac"] = counts.get("pad", 0.0) / tot

    if args.ckpt:
        ckpts = parse_labeled_paths(args.ckpt)
        for comp in ("t5", "vit"):
            mc = mask_centrality(ckpts, comp)
            for lab, v in mc.items():
                if lab in features:
                    features[lab]["mask_centrality_%s" % comp] = v

    # ---- feature table ----
    feat_names: List[str] = []
    for lab in labels:
        for k in features[lab]:
            if k not in feat_names:
                feat_names.append(k)
    feat_rows = []
    for lab in labels:
        row = {"dataset": lab, "accuracy_effect": effects.get(lab, float("nan"))}
        row.update({k: features[lab].get(k, float("nan")) for k in feat_names})
        feat_rows.append(row)
    write_csv(os.path.join(args.out_dir, "calibration_features.csv"), feat_rows)

    # ---- correlate each feature with accuracy effect ----
    shared = [lab for lab in labels if lab in effects]
    y = np.asarray([effects[lab] for lab in shared], dtype=np.float64)
    corr_rows = []
    for name in feat_names:
        xs = np.asarray([features[lab].get(name, np.nan) for lab in shared], dtype=np.float64)
        ok = np.isfinite(xs)
        if ok.sum() < 3 or np.nanstd(xs[ok]) == 0:
            continue
        pr = pearson(xs[ok], y[ok])
        sr = spearman(xs[ok], y[ok])
        corr_rows.append({
            "feature": name, "n": int(ok.sum()),
            "pearson_r": pr, "spearman_rho": sr,
            "abs_spearman": abs(sr) if np.isfinite(sr) else 0.0,
        })
    corr_rows.sort(key=lambda r: r["abs_spearman"], reverse=True)
    write_csv(os.path.join(args.out_dir, "property_accuracy_correlations.csv"), corr_rows)

    plt = setup_matplotlib()
    if plt is not None and not args.no_plots and corr_rows:
        names = [r["feature"] for r in corr_rows]
        sr = [r["spearman_rho"] for r in corr_rows]
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 2))
        colors = ["#4C78A8" if v >= 0 else "#E45756" for v in sr]
        ax.barh(range(len(names)), sr, color=colors)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis(); ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Spearman rho with accuracy effect")
        ax.set_title("Which calibration property predicts accuracy?")
        ax.set_xlim(-1, 1); ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "property_screen.png"), dpi=220, bbox_inches="tight")
        plt.close(fig)

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as h:
        json.dump({"accuracy_effects": effects, "correlations": corr_rows}, h, indent=2)

    print("=== calibration property screen (n=%d datasets) ===" % len(shared))
    print("  %-26s %8s %8s" % ("feature", "pearson", "spearman"))
    for r in corr_rows:
        print("  %-26s %+8.3f %+8.3f" % (r["feature"], r["pearson_r"], r["spearman_rho"]))

    print("\n=== verdict ===")
    if not corr_rows:
        print("  No usable features. Check inputs.")
    else:
        top = corr_rows[0]
        if abs(top["spearman_rho"]) >= 0.8:
            print("  Strongest predictor: %s (spearman %+.2f)." % (top["feature"], top["spearman_rho"]))
            print("  That property tracks accuracy across calibrations -- chase it. If it is a")
            print("  composition/scale/tail feature (not centrality), the cause is a property of the")
            print("  calibration DATA, not of where its mask lands in geometry.")
        else:
            print("  Nothing correlates strongly (top |spearman|=%.2f). The accuracy main effect is")
            print("  NOT explained by any cheap statistic-side property. The cause is downstream --")
            print("  in how each pruned model behaves on eval data. Run")
            print("  analyze_calibration_downstream_drift.py next.")
            print("  Strongest (weak) lead: %s (spearman %+.2f)." % (corr_rows[0]["feature"], corr_rows[0]["spearman_rho"]))

    print("\n[OK] wrote:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
